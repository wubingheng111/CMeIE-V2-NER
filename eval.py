import json
import os
import torch
import numpy as np
import re
from torch import nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import spacy
from itertools import combinations

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

def extract_relations_from_data(file_path):
    """从训练数据中提取关系类型"""
    relations = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                for spo in data['spo_list']:
                    relations.add(spo['predicate'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
                continue
                
    relations = sorted(list(relations))
    relation_to_id = {rel: i for i, rel in enumerate(relations)}
    
    print(f"从数据中提取了 {len(relations)} 种关系类型")
    return relations, relation_to_id

def load_model_for_testing(model_path='best_re_model.pt'):
    """加载预训练模型并从训练数据中提取关系"""
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 从训练数据中提取关系
    train_file = "./CMeIE-V2/CMeIE-V2/CMeIE-V2_train.jsonl"
    relations, relation_to_id = extract_relations_from_data(train_file)
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 添加特殊标记
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # 初始化模型
    model = REModel(num_relations=len(relations))
    model.bert.resize_token_embeddings(len(tokenizer))
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"成功加载模型权重: {model_path}")
    
    return model, tokenizer, relations, device

def test_model_with_examples(model, tokenizer, relations, device, examples):
    """使用一系列样例测试模型"""
    results = []
    
    for example in examples:
        text = example['text']
        subject = example['subject']
        object_ = example['object']
        
        predicted_relation, probs = predict_relation(
            model, tokenizer, text, subject, object_, relations, device
        )
        
        # 获取前3个最可能的关系
        top_relations = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        results.append({
            'text': text,
            'subject': subject,
            'object': object_,
            'predicted_relation': predicted_relation,
            'top_relations': top_relations,
            'ground_truth': example.get('relation', '未知')
        })
    
    return results


def main():
    # 模型路径
    model_path = 'best_re_model.pt'
    # 加载模型
    model, tokenizer, relations, device = load_model_for_testing(model_path)
    
    custom_examples = [
        {
            'text': '出生后感染性肺炎可出现发热或体温不升，反应差等全身症状。 维持正常血气 有低氧血症时可根据病情和血气分析结果选用鼻导管、面罩、鼻塞式CPAP给氧，使血气维持在正常范围。',
            'subject': '失眠症',
            'object': '维持正常血气"'
        },
        {
            'text': '失眠症@引导意象和冥想指导患者以舒适、宁静的意象替代忧心忡忡的想法',
            'subject': '高血压',
            'object': '以舒适、宁静的意象替代忧心忡忡的想法'
        },
        {
            'text': '患者接受了阿司匹林治疗，有效缓解了心肌梗死的症状。',
            'subject': '阿司匹林',
            'object': '心肌梗死'
        }
    ]
    
    print("\n===== 测试自定义样例 =====")
    results = test_model_with_examples(model, tokenizer, relations, device, custom_examples)
    
    for i, result in enumerate(results):
        print(f"\n示例 {i+1}:")
        print(f"文本: {result['text']}")
        print(f"主体: {result['subject']}")
        print(f"客体: {result['object']}")
        print(f"预测关系: {result['predicted_relation']}")
        print("前3个最可能关系:")
        for rel, prob in result['top_relations']:
            print(f"  - {rel}: {prob:.4f}")
    
if __name__ == "__main__":
    main()