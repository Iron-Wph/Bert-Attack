import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
import numpy as np
import re
import pandas as pd
from typing import Tuple


def normalize_dialogue_to_one_line(text: str) -> str:
    s = str(text)
    s = re.sub(r"^音频内容：\s*", "", s)
    s = re.sub(r"\b(left|right)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    s = s.replace('"', "").strip()
    s = s.replace("\r", "\n")
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class DataProcessor:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def load_and_clean(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        print(df["specific_dialogue_content"][0])
        required_cols = ["specific_dialogue_content", "is_fraud"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"CSV 缺少必要列: {c}")

        df = df.dropna(subset=["specific_dialogue_content", "is_fraud"])

        # 标签：True/False -> 1/0
        df["label"] = df["is_fraud"].astype(str).apply(
            lambda x: 1 if x.strip().lower() == "true" else 0
        )

        df["text"] = df["specific_dialogue_content"].apply(
            normalize_dialogue_to_one_line)

        if "fraud_type" not in df.columns:
            df["fraud_type"] = "unknown"

        return df[["text", "label", "fraud_type"]].reset_index(drop=True)

    def get_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.load_and_clean(self.train_path)
        test_df = self.load_and_clean(self.test_path)

        print(f"[Data] 训练集大小: {len(train_df)}")
        print(f"[Data] 测试集大小: {len(test_df)}")
        print("[Data] 训练集标签分布:\n", train_df["label"].value_counts(dropna=False))

        return train_df, test_df


# ====== 你上面的 DataProcessor 直接 import 或复制到此处 ======
# from your_module import DataProcessor

# --------- 评估指标 ----------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # 手写避免额外依赖 sklearn（若你有 sklearn 也可以换成 sklearn 的）
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

# --------- 将 DataFrame 转成可被 Trainer 使用的数据集 ----------


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,   # 交给 DataCollatorWithPadding 动态 padding
        )
        enc["labels"] = label
        return enc


def main():
    # ====== 路径改这里 ======
    TRAIN_CSV = "/data1/wph/nlp/hw1/final/datasets/train_set.csv"
    TEST_CSV = "/data1/wph/nlp/hw1/final/datasets/test_set.csv"
    OUTPUT_DIR = "./bert_fraud_cls"

    # MODEL_NAME = "bert-base-chinese"
    MODEL_NAME = "/data1/wph/nlp/hw1/google-bert/bert-base-chinese"
    MAX_LEN = 256
    SEED = 42

    set_seed(SEED)

    # 1) 读数据
    dp = DataProcessor(TRAIN_CSV, TEST_CSV)
    train_df, test_df = dp.get_datasets()

    # 2) tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # 3) Dataset
    train_ds = SimpleTextDataset(
        train_df["text"], train_df["label"], tokenizer, max_length=MAX_LEN)
    eval_ds = SimpleTextDataset(
        test_df["text"],  test_df["label"],  tokenizer, max_length=MAX_LEN)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4) TrainingArguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        # 训练参数：按你机器调整
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),   # 有 CUDA 就自动 fp16
        report_to="none",                # 不上报 wandb
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6) train + eval + save
    trainer.train()
    metrics = trainer.evaluate()
    print("[Eval]", metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[Done] model saved to: {OUTPUT_DIR}")


def test_data_load():
    TRAIN_CSV = "/data1/wph/nlp/hw1/final/datasets/train_set.csv"
    TEST_CSV = "/data1/wph/nlp/hw1/final/datasets/test_set.csv"
    processor = DataProcessor(TRAIN_CSV, TEST_CSV)
    train_df, test_df = processor.get_datasets()

    print(test_df['text'][0])


if __name__ == "__main__":
    # test_data_load()
    main()
