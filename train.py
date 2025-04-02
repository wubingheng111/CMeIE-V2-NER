import json
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

# Login using huggingface-url to download the dataset url:https://huggingface.co/datasets/wubingheng/CMeIE-V2-NER/tree/main

# 文件路径
train_file = "./CMeIE-V2_train.jsonl"
test_file = "./CMeIE-V2_dev.jsonl"


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(42)

# 数据加载和预处理
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error parsing line: {line}")
    return data

def process_data_for_re(data):
    """处理数据为关系抽取任务格式"""
    examples = []
    for item in data:
        text = item['text']
        for spo in item['spo_list']:
            subject = spo['subject']
            predicate = spo['predicate']
            
            # 处理object可能是字典的情况
            if isinstance(spo['object'], dict) and '@value' in spo['object']:
                object_ = spo['object']['@value']
            else:
                object_ = spo['object']
                
            examples.append({
                'text': text,
                'subject': subject,
                'object': object_,
                'relation': predicate
            })
    return examples

# 构建数据集类
class REDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 获取所有唯一的关系类型
        self.relations = sorted(list(set([ex['relation'] for ex in examples])))
        self.relation_to_id = {rel: i for i, rel in enumerate(self.relations)}
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        subject = example['subject']
        object_ = example['object']
        relation = example['relation']
        
        # 标记实体在文本中的位置
        text_marked = text.replace(subject, f"[E1]{subject}[/E1]", 1)
        text_marked = text_marked.replace(object_, f"[E2]{object_}[/E2]", 1)
        
        # BERT分词
        encoding = self.tokenizer(
            text_marked,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.relation_to_id[relation], dtype=torch.long)
        }

# 构建模型
class REModel(nn.Module):
    def __init__(self, num_relations, model_name='bert-base-chinese'):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_relations)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 训练函数
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算训练集F1
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_f1s.append(train_f1)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算验证集F1
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_f1s.append(val_f1)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_re_model.pt')
            print("保存最佳模型 ✓")
            
    return train_losses, val_losses, train_f1s, val_f1s

# 评估模型
def evaluate_model(model, test_dataloader, relation_list, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换ID为关系名称
    pred_relations = [relation_list[idx] for idx in all_preds]
    true_relations = [relation_list[idx] for idx in all_labels]
    
    # 计算分类报告
    report = classification_report(true_relations, pred_relations, digits=4)
    
    return report, all_preds, all_labels

# 可视化训练过程
def plot_training_history(train_losses, val_losses, train_f1s, val_f1s):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='train_loss')
    plt.plot(epochs, val_losses, 'r-', label='val_loss')
    plt.title('training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # F1分数图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1s, 'b-', label='train_f1')
    plt.plot(epochs, val_f1s, 'r-', label='val_f1')
    plt.title('training and validation F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_relation(model, tokenizer, text, subject, object_, relation_list, device, max_length=128):
    model.eval()
    
    # 标记实体
    text_marked = text.replace(subject, f"[E1]{subject}[/E1]", 1)
    text_marked = text_marked.replace(object_, f"[E2]{object_}[/E2]", 1)
    
    # 分词
    encoding = tokenizer(
        text_marked,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        _, pred = torch.max(logits, 1)
    
    predicted_relation = relation_list[pred.item()]
    
    # 计算所有关系的概率分布
    probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    relation_probs = {relation_list[i]: float(probs[i]) for i in range(len(relation_list))}
    
    return predicted_relation, relation_probs

# 保存与加载模型
def save_model_and_tokenizer(model, tokenizer, relation_to_id, output_dir='re_model'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存模型
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    # 保存关系映射
    with open(f"{output_dir}/relations.json", 'w', encoding='utf-8') as f:
        json.dump(relation_to_id, f, ensure_ascii=False, indent=2)
    
    print(f"模型和配置已保存到: {output_dir}")

def load_model_and_tokenizer(model_dir='re_model'):
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    # 加载关系映射
    with open(f"{model_dir}/relations.json", 'r', encoding='utf-8') as f:
        relation_to_id = json.load(f)
    
    relations = list(relation_to_id.keys())
    num_relations = len(relations)
    
    # 初始化模型
    model = REModel(num_relations=num_relations)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    
    return model, tokenizer, relations

def main():
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载和处理数据
    print("加载训练数据...")
    train_data = load_data(train_file)
    print(f"加载了 {len(train_data)} 条训练数据")
    
    print("加载测试数据...")
    test_data = load_data(test_file)
    print(f"加载了 {len(test_data)} 条测试数据")
    
    # 处理数据为关系抽取任务格式
    print("处理数据...")
    train_examples = process_data_for_re(train_data)
    test_examples = process_data_for_re(test_data)
    
    # 划分训练集和验证集
    train_examples, val_examples = train_test_split(
        train_examples, test_size=0.1, random_state=42
    )
    
    print(f"训练集: {len(train_examples)} 样本")
    print(f"验证集: {len(val_examples)} 样本")
    print(f"测试集: {len(test_examples)} 样本")
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 特殊标记
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = REDataset(train_examples, tokenizer)
    val_dataset = REDataset(val_examples, tokenizer)
    test_dataset = REDataset(test_examples, tokenizer)
    
    # 创建数据加载器
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 获取关系列表
    relations = train_dataset.relations
    num_relations = len(relations)
    print(f"识别的关系类型数量: {num_relations}")
    
    # 初始化模型
    model = REModel(num_relations)
    model.bert.resize_token_embeddings(len(tokenizer))
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * 3
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    
    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_dataloader, val_dataloader, 
        optimizer, scheduler, device, epochs=3
    )
    
    # 可视化训练过程
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_re_model.pt'))
    
    # 在测试集上评估
    print("\n在测试集上评估模型...")
    report, preds, labels = evaluate_model(model, test_dataloader, relations, device)
    print("\n分类报告:")
    print(report)
    
    # 保存模型
    save_model_and_tokenizer(model, tokenizer, train_dataset.relation_to_id)
    
    # 示例预测
    example_text = "患者有发热和呼吸道症状，确诊为新型肺炎"
    example_subject = "新型肺炎"
    example_object = "发热"
    
    predicted_relation, probs = predict_relation(
        model, tokenizer, example_text, example_subject, example_object, relations, device
    )
    
    print("\n示例预测:")
    print(f"文本: {example_text}")
    print(f"主体: {example_subject}")
    print(f"客体: {example_object}")
    print(f"预测关系: {predicted_relation}")
    
    # 显示前3个最可能的关系
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print("\n关系概率分布 (前3):")
    for rel, prob in sorted_probs[:3]:
        print(f"{rel}: {prob:.4f}")

if __name__ == "__main__":
    main()