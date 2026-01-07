import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 加载模型和tokenizer
model_dir = "/data1/wph/nlp/hw1/final/new/bert_fraud_cls"  # 您的BERT模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)


# 定义评估函数
def evaluate(model, tokenizer, texts, labels, batch_size=64, max_length=256):
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    # 按批次进行推理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # 对文本进行编码
        enc = tokenizer(batch_texts, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}  # 将数据移到GPU

        # 进行预测
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits

        # 计算预测标签
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).cpu().numpy()

        # 存储预测结果
        all_preds.extend(preds)
        all_labels.extend(batch_labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1


# 加载对抗数据集
adv_df = pd.read_csv("test_adv_bert_attack.csv")  # roberta - 同义词
texts = adv_df['adv_text'].tolist()  # 对抗样本文本
labels = adv_df['label'].tolist()  # 对抗样本标签
# 执行评估
accuracy, precision, recall, f1 = evaluate(model, tokenizer, texts, labels)

# 输出评估结果
print(f">>>>>> Eval roberta synonym attack results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 加载对抗数据集
adv_df = pd.read_csv(
    "test_adv_bert_attack_sentence_roberta.csv")  # roberta - 句子
texts = adv_df['adv_text'].tolist()  # 对抗样本文本
labels = adv_df['label'].tolist()  # 对抗样本标签
# 执行评估
accuracy, precision, recall, f1 = evaluate(model, tokenizer, texts, labels)

# 输出评估结果
print(f">>>>>> Eval roberta sentence attack results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 加载对抗数据集
adv_df = pd.read_csv("test_adv_bert_attack_MACBERT.csv")  # MACBERT - 同义词
texts = adv_df['adv_text'].tolist()  # 对抗样本文本
labels = adv_df['label'].tolist()  # 对抗样本标签
# 执行评估
accuracy, precision, recall, f1 = evaluate(model, tokenizer, texts, labels)

# 输出评估结果
print(f">>>>>> Eval MACBERT synonym attack results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 加载对抗数据集
adv_df = pd.read_csv(
    "test_adv_bert_attack_sentence_MACBERT.csv")  # MACBERT - 句子
texts = adv_df['adv_text'].tolist()  # 对抗样本文本
labels = adv_df['label'].tolist()  # 对抗样本标签
# 执行评估
accuracy, precision, recall, f1 = evaluate(model, tokenizer, texts, labels)

# 输出评估结果
print(f">>>>>> Eval MACBERT sentence attack results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
