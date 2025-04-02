# 中文医学实体关系抽取模型
## 项目简介
该项目实现了基于BERT的中文医学实体关系抽取模型，用于从中文医学文本中自动识别实体之间的语义关系。模型通过特殊标记表示实体位置，并利用预训练语言模型对实体间关系进行分类。

## 数据集
项目使用CMeIE-V2数据集，这是一个中文医学领域的实体关系抽取数据集。数据集包含:

- 训练集 (CMeIE-V2_train.jsonl)
- 开发/测试集 (CMeIE-V2_dev.jsonl)
数据格式为JSONL，每行包含一个JSON对象，其中包含文本和主体-谓语-客体三元组列表。
- 下载目录(不要用dataset):https://huggingface.co/datasets/wubingheng/CMeIE-V2-NER/tree/main
## 模型架构
该模型基于以下架构:

利用预训练的中文BERT模型(bert-base-chinese)作为编码器
添加特殊标记[E1], [/E1], [E2], [/E2]标记句子中的实体
在BERT池化输出之上添加分类层进行关系预测
## 主要功能
### 数据处理:

从JSONL文件加载数据
提取主体-谓语-客体三元组
将实体在句子中标记并转换为BERT输入格式
模型训练:

### 支持批量训练
使用交叉熵损失函数
提供学习率调度器
自动保存最佳模型
模型评估:

### 计算宏平均F1分数
生成详细的分类报告
可视化训练和验证损失及F1分数
![image](https://github.com/user-attachments/assets/fc3ce856-b76f-4a7b-a0f4-5e215427d487)
预测功能:
## 使用方法
准备数据集文件:
确保CMeIE-V2_train.jsonl和CMeIE-V2_dev.jsonl在正确路径
训练完成后，可以查看:<p>
训练历史可视化图(training_history.png)<p>
详细分类报告模型将保存在re_model目录<p>
***python train.py*** 启动训练<p>
## 示例输出
训练后，模型可以预测实体间的关系，例如:<p>
***文本: 患者有发热和呼吸道症状，确诊为新型肺炎***<p>
***主体: 新型肺炎***<p>
***客体: 发热***<p>
***预测关系: 临床表现***<p>
![image](https://github.com/user-attachments/assets/143b1745-4068-4255-b98a-172e1ca53cf4)



