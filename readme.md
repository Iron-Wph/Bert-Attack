# Adversarial Data Rewriting for Fraud Dialogue Detection
````markdown
本项目围绕 **“对抗性数据改写在欺诈对话检测中的应用”** 展开研究，基于 **BERT-base-Chinese** 欺诈检测模型，系统性实现并评估了 **词级（Synonym-level）** 与 **句级（Sentence-level, Span-mask + Beam Search）** 两类文本对抗攻击方法，分析模型在高准确率条件下的鲁棒性与潜在脆弱性。

---

## 📁 项目结构

```text
.
├── bert_fraud_cls/                         # 训练好的 BERT 欺诈检测模型（Victim）
├── train.py                                # BERT-base-Chinese 微调训练脚本
├── eval.py                                 # 原始 / 对抗数据集评估脚本
├── asr.py                                  # 攻击成功率（ASR）统计脚本
├── bert_attack.py                          # 对抗样本生成主程序
├── train.log                               # 训练日志
├── eval.log                                # 评估日志
├── test_adv_bert_attack.csv                # 词级攻击（RoBERTa）
├── test_adv_bert_attack_MACBERT.csv        # 词级攻击（MacBERT）
├── test_adv_bert_attack_sentence_roberta.csv
├── test_adv_bert_attack_sentence_MACBERT.csv
└── README.md
````

---

## 🔍 研究问题

尽管基于预训练语言模型（如 BERT）的欺诈检测系统在标准测试集上可取得接近 **100% 的准确率**，但其在面对 **语义保持而表述发生变化的文本输入** 时，是否仍然具备足够的鲁棒性，仍值得深入研究。

本项目主要关注以下问题：

* 在语义基本不变的前提下，文本改写是否能够绕过欺诈检测模型？
* 不同对抗改写策略（词级 vs 句级）对模型性能的影响有何差异？
* 不同掩码语言模型（RoBERTa / MacBERT）在攻击效果上是否存在显著差别？

---

## 🧠 方法概述

### Victim 模型

* **模型**：BERT-base-Chinese
* **任务**：二分类（诈骗 / 非诈骗）
* **训练方式**：在清洗后的欺诈对话数据集上微调 1 个 epoch

### 对抗攻击方法

#### 1️⃣ 词级对抗攻击（Synonym-level，BERT-Attack）

* 基于 **BERT-Attack** 方法
* 通过对输入文本中关键词进行 mask，利用 MLM 生成替换候选
* 使用 Victim 模型对候选文本进行打分，选择最能降低原始标签置信度的替换
* 特点：改动小、语义保持性强、攻击效率较高

#### 2️⃣ 句级对抗攻击（Sentence-level，Span-mask + Beam Search）

* 基于 **BAE（BERT-based Adversarial Examples）** 思路
* 对多个重要 token span 进行 mask
* 使用 MLM 进行迭代填充，并结合 beam search 搜索最具对抗性的句子
* 特点：改写范围更大，句法结构变化明显，但计算成本较高

### 使用的掩码语言模型（MLM）

* `hfl/chinese-roberta-wwm-ext`
* `hfl/chinese-macbert-base`

---

## 🧪 实验环境

* **操作系统**：Linux
* **Python**：3.10
* **GPU**：NVIDIA GPU（通过 `CUDA_VISIBLE_DEVICES` 指定）
* **主要依赖**：

  * PyTorch
  * Transformers
  * NumPy
  * Pandas
* **运行方式**：离线加载本地模型（`HF_HUB_OFFLINE=1`）

---

## 📊 实验结果

### 1️⃣ 原始数据集性能

| 数据集            | Accuracy   | Precision | Recall | F1-score |
| ----------------- | ---------- | --------- | ------ | -------- |
| Original Test Set | **99.84%** | 99.92%    | 99.78% | 99.85%   |

该结果表明，BERT-base-Chinese 在标准测试集上具有极高的分类性能。

---

### 2️⃣ 对抗攻击成功率（ASR）

| 攻击方法       | MLM     | ASR（label=1） | ASR-all |
| -------------- | ------- | -------------- | ------- |
| Synonym        | RoBERTa | 23.28%         | 12.68%  |
| Synonym        | MacBERT | 22.92%         | 12.48%  |
| Sentence-level | RoBERTa | 9.58%          | 5.22%   |
| Sentence-level | MacBERT | 6.56%          | 3.57%   |

---

### 3️⃣ 对抗数据集上的模型性能

| 数据集            | Accuracy | Precision | Recall | F1-score |
| ----------------- | -------- | --------- | ------ | -------- |
| Original Test Set | 99.84%   | 99.92%    | 99.78% | 99.85%   |
| RoBERTa Synonym   | 87.28%   | 99.91%    | 76.71% | 86.79%   |
| RoBERTa Sentence  | 94.74%   | 99.92%    | 90.41% | 94.93%   |
| MacBERT Synonym   | 87.48%   | 99.91%    | 77.07% | 87.02%   |
| MacBERT Sentence  | 96.39%   | 99.92%    | 93.44% | 96.57%   |

---

## 📈 结果分析

* **词级替换方法**
  改动幅度小、自然性较好，但能够显著降低模型在诈骗样本上的召回率，说明模型对部分关键词高度依赖。

* **句级改写方法**
  改写更自然、句式变化更明显，但攻击成功率相对较低，且推理时间更长，更接近真实场景中的自然改写行为。

* **不同 MLM 的影响**
  RoBERTa 在攻击成功率上略高，而 MacBERT 在生成文本的通顺性与稳定性方面表现更好。

---

## ▶️ 使用说明

### 1️⃣ 训练欺诈检测模型

```bash
python train.py
```

### 2️⃣ 生成对抗样本

```bash
python bert_attack.py
```

### 3️⃣ 评估模型性能

```bash
python eval.py
```

### 4️⃣ 计算攻击成功率（ASR）

```bash
python asr.py
```

---

## 📌 总结

实验结果表明：
即使在标准测试集上表现接近完美的 BERT 欺诈检测模型，在面对对抗性文本改写时仍存在明显脆弱性。
对抗数据改写为评估与提升欺诈检测系统鲁棒性提供了一种有效且必要的手段。

---

## 📚 参考文献

* Jin et al., *TextFooler: A Model-Agnostic Textual Adversarial Attack*, AAAI 2020
* Li et al., *BERT-Attack: Adversarial Attack Against BERT Using BERT*, EMNLP 2020
* Garg & Ramakrishnan, *BAE: BERT-based Adversarial Examples*, EMNLP 2019
* Morris et al., *TextAttack: A Framework for Adversarial Attacks in NLP*, EMNLP 2020

---

## 👤 作者

**Peihong Wang**
SZU

