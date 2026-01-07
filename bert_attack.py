from typing import List, Tuple, Dict, Any, Optional
import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM

# =========================
# 0) 强制离线（推荐）
# =========================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =========================
# 1) 你的数据清洗（去 left/right + 合并一句）
# =========================


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

        required_cols = ["specific_dialogue_content", "is_fraud"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"CSV 缺少必要列: {c}")

        df = df.dropna(subset=["specific_dialogue_content", "is_fraud"])

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


# =========================
# 2) Victim：BERT 二分类器（本地加载）
# =========================
class VictimClassifier:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(device)
        self.model.eval()

        # mask token（用于重要性评估）
        self.mask_token = self.tokenizer.mask_token or "[MASK]"

    @torch.no_grad()
    def predict_proba(self, text: str, max_length: int = 256) -> np.ndarray:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits  # [1, 2]
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        return probs  # [p0, p1]

    @torch.no_grad()
    def predict_proba_batch(self, texts, max_length: int = 256, batch_size: int = 64) -> np.ndarray:
        all_probs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)  # [N, 2]

    @torch.no_grad()
    def predict_label(self, text: str, max_length: int = 256) -> int:
        probs = self.predict_proba(text, max_length=max_length)
        return int(np.argmax(probs))


# =========================
# 3) Attacker：Masked LM（本地加载）
# =========================
class MLMAttacker:
    def __init__(self, mlm_dir: str, device: str = "cuda"):
        self.mlm_dir = mlm_dir
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            mlm_dir, use_fast=True, local_files_only=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            mlm_dir, local_files_only=True
        ).to(device)
        self.model.eval()

        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "攻击用 MLM tokenizer 没有 mask_token_id，无法进行 BERT-Attack。")

    @torch.no_grad()
    def topk_replacements(
        self,
        text: str,
        target_span: Tuple[int, int],
        top_k: int = 20,
        max_length: int = 256,
    ) -> List[str]:
        """
        给定原始 text 和要替换的字符 span=(l,r)，用 MLM 在 span 上做 mask，返回 top-k 候选 token（字符串）。
        注意：这里是“单 token 替换”版本，适合中文（一个词被分成多个 token 时会更难）。
        """
        l, r = target_span
        masked_text = text[:l] + self.tokenizer.mask_token + text[r:]

        enc = self.tokenizer(
            masked_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc).logits  # [1, seq, vocab]
        input_ids = enc["input_ids"][0]  # [seq]
        mask_pos = (input_ids == self.tokenizer.mask_token_id).nonzero(
            as_tuple=False)

        if mask_pos.numel() == 0:
            return []

        # 只取第一个 mask（我们构造时只有一个）
        pos = int(mask_pos[0].item())
        scores = out[0, pos]  # [vocab]
        topk = torch.topk(scores, k=top_k).indices.detach().cpu().tolist()

        cands = []
        for tid in topk:
            tok = self.tokenizer.convert_ids_to_tokens(int(tid))
            # 过滤掉特殊符号/子词前缀
            if tok in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.mask_token]:
                continue
            if tok.startswith("##"):
                continue
            # 过滤纯空白
            if tok.strip() == "":
                continue
            cands.append(tok)

        return cands

    @torch.no_grad()
    def fill_one_mask_topk(
        self,
        masked_text: str,
        top_k: int = 20,
        max_length: int = 256,
    ) -> List[str]:
        """
        输入包含至少一个 [MASK] 的句子，只对“第一个 mask”位置预测 top-k token。
        返回候选 token 列表（字符串）。
        """
        enc = self.tokenizer(
            masked_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits  # [1, seq, vocab]
        input_ids = enc["input_ids"][0]
        mask_pos = (input_ids == self.tokenizer.mask_token_id).nonzero(
            as_tuple=False)
        if mask_pos.numel() == 0:
            return []

        pos = int(mask_pos[0].item())
        scores = logits[0, pos]
        topk_ids = torch.topk(scores, k=top_k).indices.detach().cpu().tolist()

        cands = []
        for tid in topk_ids:
            tok = self.tokenizer.convert_ids_to_tokens(int(tid))
            if tok in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.mask_token]:
                continue
            if tok.startswith("##"):
                continue
            if tok.strip() == "":
                continue
            cands.append(tok)
        return cands


@dataclass
class SentenceRewriteConfig:
    max_length: int = 256
    # span mask 参数
    span_len: int = 3             # 每次改写覆盖的 token 数（2~4 常用）
    num_spans: int = 2            # 一句话里 mask 几个 span（建议 1~3）
    # 迭代填充 / beam
    fill_top_k: int = 20          # 每个 mask 的 top-k
    beam_size: int = 5            # 每步保留的候选句数量
    max_fill_steps: int = 16      # 防止死循环：最多填多少个 mask
    # victim-guided 选择
    victim_batch_size: int = 64


class SpanMaskSentenceRewriter:
    """
    整句改写（span mask + iterative filling）：
    - 选重要位置 -> mask span -> MLM 逐个 [MASK] 填充
    - 每步用 victim 选最能降低真标签概率（或翻转）的候选
    """

    def __init__(self, victim: VictimClassifier, mlm: MLMAttacker, cfg: SentenceRewriteConfig):
        self.victim = victim
        self.mlm = mlm
        self.cfg = cfg

    def _get_token_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        使用 victim 的 tokenizer 获取 offset_mapping，得到 token 的字符 span。
        比 jieba+find 更稳、更适合整句改写。
        """
        enc = self.victim.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
        )
        offsets = enc["offset_mapping"]
        spans = []
        for (l, r) in offsets:
            if l == r:
                continue  # 跳过 special tokens
            spans.append((int(l), int(r)))
        return spans

    def _importance_rank_tokens(self, text: str, true_label: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        token 级重要性：mask 某个 token span，观察 P(y_true) 的下降
        """
        base_p = float(self.victim.predict_proba(
            text, max_length=self.cfg.max_length)[true_label])
        spans = self._get_token_spans(text)

        masked_texts = []
        kept_spans = []
        for (l, r) in spans:
            # 过滤：太短/空白
            if r - l <= 0:
                continue
            if re.fullmatch(r"\s+", text[l:r] or ""):
                continue
            masked_texts.append(text[:l] + self.victim.mask_token + text[r:])
            kept_spans.append((l, r))

        if not masked_texts:
            return []

        probs = self.victim.predict_proba_batch(
            masked_texts,
            max_length=self.cfg.max_length,
            batch_size=self.cfg.victim_batch_size,
        )
        scores = base_p - probs[:, true_label]
        ranked = list(zip(kept_spans, scores.tolist()))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _mask_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        """
        将多个 span 替换为 [MASK]（为避免 offset 变化，倒序替换）
        """
        spans = sorted(spans, key=lambda x: x[0], reverse=True)
        out = text
        for (l, r) in spans:
            out = out[:l] + self.victim.mask_token + out[r:]
        return out

    def _expand_to_span(self, token_spans: List[Tuple[int, int]], idx: int, span_len: int) -> Tuple[int, int]:
        """
        将 token idx 扩展为连续 span_len 个 token 的字符范围
        """
        j = min(idx + span_len - 1, len(token_spans) - 1)
        l = token_spans[idx][0]
        r = token_spans[j][1]
        return (l, r)

    def rewrite_one(self, text: str, true_label: int) -> Dict[str, Any]:
        """
        返回与 word-level attack_one 类似的结构，便于统一保存 CSV
        """
        probs_before = self.victim.predict_proba(
            text, max_length=self.cfg.max_length)
        pred_before = int(np.argmax(probs_before))
        p_true_before = float(probs_before[true_label])

        # 1) 重要性排序（token级）
        ranked = self._importance_rank_tokens(text, true_label=true_label)
        if not ranked:
            return {
                "adv_text": text,
                "success": False,
                "num_changes": 0,
                "p_true_before": p_true_before,
                "p_true_after": p_true_before,
                "pred_before": pred_before,
                "pred_after": pred_before,
                "changes": [],
                "note": "no_ranked_tokens",
            }

        # 2) 选 num_spans 个重要 token，并扩展成 span_len
        token_spans = self._get_token_spans(text)
        # ranked 给的是 span(l,r)，但我们需要它在 token_spans 里的 idx；用最简单的近似匹配：
        # 找到 ranked span 在 token_spans 中的起点 idx
        chosen = []
        used_ranges = []
        for (span_lr, score) in ranked:
            if len(chosen) >= self.cfg.num_spans:
                break
            l, r = span_lr
            # 找 idx：第一个起点 l 相同或最接近的 token
            idx = None
            for k, (tl, tr) in enumerate(token_spans):
                if tl == l and tr == r:
                    idx = k
                    break
            if idx is None:
                continue
            big_span = self._expand_to_span(
                token_spans, idx, self.cfg.span_len)

            # 避免 span 重叠
            overlap = False
            for (ul, ur) in used_ranges:
                if not (big_span[1] <= ul or big_span[0] >= ur):
                    overlap = True
                    break
            if overlap:
                continue

            chosen.append(big_span)
            used_ranges.append(big_span)

        if not chosen:
            return {
                "adv_text": text,
                "success": False,
                "num_changes": 0,
                "p_true_before": p_true_before,
                "p_true_after": p_true_before,
                "pred_before": pred_before,
                "pred_after": pred_before,
                "changes": [],
                "note": "no_spans_chosen",
            }

        # 3) mask spans -> 得到 masked 句
        masked = self._mask_spans(text, chosen)

        # 4) 迭代填充（beam）
        beam = [masked]
        max_steps = self.cfg.max_fill_steps

        def has_mask(s: str) -> bool:
            return self.mlm.tokenizer.mask_token in s

        for step in range(max_steps):
            # 如果 beam 里已经没有 mask，停止
            if all(not has_mask(s) for s in beam):
                break

            new_candidates = []
            for s in beam:
                if not has_mask(s):
                    new_candidates.append(s)
                    continue

                # 对第一个 mask 预测 topk token
                toks = self.mlm.fill_one_mask_topk(
                    s, top_k=self.cfg.fill_top_k, max_length=self.cfg.max_length
                )
                if not toks:
                    new_candidates.append(s)  # 填不了就保留
                    continue

                # 生成替换后的句子（只替换第一个 mask）
                for tok in toks:
                    new_s = s.replace(self.mlm.tokenizer.mask_token, tok, 1)
                    new_candidates.append(new_s)

            # 5) 用 victim 评估，选 beam_size 个最“对抗”的
            probs = self.victim.predict_proba_batch(
                new_candidates,
                max_length=self.cfg.max_length,
                batch_size=self.cfg.victim_batch_size,
            )
            p_true = probs[:, true_label]
            pred = probs.argmax(axis=-1)

            # 优先：已翻转（pred != true_label）的排前面；其次 p_true 越小越好
            idxs = list(range(len(new_candidates)))
            # False(翻转) 更小 -> 更靠前
            idxs.sort(key=lambda i: (pred[i] == true_label, p_true[i]))

            beam = [new_candidates[i] for i in idxs[:self.cfg.beam_size]]

            # 如果 beam[0] 已经翻转，可以提前结束
            best0_probs = self.victim.predict_proba(
                beam[0], max_length=self.cfg.max_length)
            if int(np.argmax(best0_probs)) != true_label:
                break

        adv_text = beam[0]
        probs_after = self.victim.predict_proba(
            adv_text, max_length=self.cfg.max_length)
        pred_after = int(np.argmax(probs_after))
        p_true_after = float(probs_after[true_label])

        success = (pred_after != true_label)

        return {
            "adv_text": adv_text,
            "success": bool(success),
            "num_changes": len(chosen),  # 这里按 span 数计“改写点”
            "p_true_before": p_true_before,
            "p_true_after": p_true_after,
            "pred_before": pred_before,
            "pred_after": pred_after,
            "changes": chosen,  # 记录 span 范围，或你也可记录原/改写片段
            "note": "sentence_spanmask",
        }

# =========================
# 4) 中文“词”切分：无外部依赖版本（可用；若你装了 jieba，会更好）
# =========================


def tokenize_words_zh(text: str) -> List[str]:
    """
    轻量级中文分词：
    - 优先用 jieba（若可用）
    - 否则用“中文连续串/英文数字串/标点分割”的近似方案
    """
    try:
        import jieba  # type: ignore
        words = [w.strip() for w in jieba.lcut(text) if w.strip()]
        return words
    except Exception:
        # fallback：保留中文连续串/英文数字串
        parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text)
        return [p for p in parts if p.strip()]


def find_first_span(text: str, word: str) -> Optional[Tuple[int, int]]:
    """找到 word 在 text 中第一次出现的字符 span"""
    idx = text.find(word)
    if idx < 0:
        return None
    return (idx, idx + len(word))


# =========================
# 5) BERT-Attack 主逻辑
# =========================
@dataclass
class AttackConfig:
    max_length: int = 256
    top_k: int = 20
    max_mod_ratio: float = 0.15      # 最多改动词数占比
    min_word_len: int = 2            # 太短的词不改（可调）
    skip_patterns: Tuple[str, ...] = (
        r"^\d+$",                    # 纯数字
    )
    # 常见停用/代词（可扩充）
    stop_words: Tuple[str, ...] = (
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
        "这", "那", "这里", "那里", "这个", "那个",
        "的", "了", "在", "是", "就", "都", "还", "也", "啊", "呀", "吧", "吗",
    )


class BertAttackGenerator:
    def __init__(self, victim: VictimClassifier, mlm: MLMAttacker, cfg: AttackConfig):
        self.victim = victim
        self.mlm = mlm
        self.cfg = cfg

        self.stop_set = set(cfg.stop_words)

    def _is_skippable_word(self, w: str) -> bool:
        if w in self.stop_set:
            return True
        if len(w) < self.cfg.min_word_len:
            return True
        for pat in self.cfg.skip_patterns:
            if re.match(pat, w):
                return True
        # 过滤明显的标点/空白
        if re.fullmatch(r"\W+", w):
            return True
        return False

    def _word_importance_ranking(self, text: str, true_label: int) -> List[Tuple[str, float]]:
        """
        BERT-Attack 常见做法：用 victim 置信度下降估计词重要性
        score(w) = P(y|x) - P(y|x_mask(w))
        """
        base_probs = self.victim.predict_proba(
            text, max_length=self.cfg.max_length)
        base_p = float(base_probs[true_label])

        words = tokenize_words_zh(text)
        scored: List[Tuple[str, float]] = []

        for w in words:
            if self._is_skippable_word(w):
                continue
            span = find_first_span(text, w)
            if span is None:
                continue

            l, r = span
            masked_text = text[:l] + self.victim.mask_token + text[r:]
            new_probs = self.victim.predict_proba(
                masked_text, max_length=self.cfg.max_length)
            new_p = float(new_probs[true_label])

            scored.append((w, base_p - new_p))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def attack_one(self, text: str, true_label: int) -> Dict[str, Any]:
        """
        输出：
        - adv_text
        - success
        - num_changes
        - p_true_before / p_true_after
        - pred_before / pred_after
        - changes: [(old, new), ...]
        """
        pred_before = self.victim.predict_label(
            text, max_length=self.cfg.max_length)
        probs_before = self.victim.predict_proba(
            text, max_length=self.cfg.max_length)
        p_true_before = float(probs_before[true_label])

        # 重要性排序
        ranked = self._word_importance_ranking(text, true_label=true_label)
        if len(ranked) == 0:
            return {
                "adv_text": text,
                "success": False,
                "num_changes": 0,
                "p_true_before": p_true_before,
                "p_true_after": p_true_before,
                "pred_before": pred_before,
                "pred_after": pred_before,
                "changes": [],
                "note": "no_valid_words",
            }

        words = [w for w, _ in ranked]
        max_changes = max(
            1, int(math.ceil(len(words) * self.cfg.max_mod_ratio)))

        cur_text = text
        cur_p_true = p_true_before
        cur_pred = pred_before
        changes: List[Tuple[str, str]] = []

        for w in words:
            if len(changes) >= max_changes:
                break

            span = find_first_span(cur_text, w)
            if span is None:
                continue

            # 用 MLM 产生候选
            cands = self.mlm.topk_replacements(
                cur_text,
                target_span=span,
                top_k=self.cfg.top_k,
                max_length=self.cfg.max_length,
            )
            if not cands:
                continue

            # 过滤：不要与原词相同；不要太奇怪
            cands = [c for c in cands if c != w and c.strip() != ""]
            if not cands:
                continue

            # 选“最能降低真标签概率”的替换（或直接翻转）
            best_text = None
            best_p = cur_p_true
            best_pred = cur_pred
            best_cand = None

            l, r = span
            for c in cands:
                tmp_text = cur_text[:l] + c + cur_text[r:]
                tmp_probs = self.victim.predict_proba(
                    tmp_text, max_length=self.cfg.max_length)
                tmp_p_true = float(tmp_probs[true_label])
                tmp_pred = int(np.argmax(tmp_probs))

                # 优先翻转；否则选最小 p_true
                if tmp_pred != true_label:
                    best_text = tmp_text
                    best_p = tmp_p_true
                    best_pred = tmp_pred
                    best_cand = c
                    break

                if tmp_p_true < best_p:
                    best_text = tmp_text
                    best_p = tmp_p_true
                    best_pred = tmp_pred
                    best_cand = c

            if best_text is None or best_cand is None:
                continue

            # 如果没有带来任何下降，就跳过（保持改写质量）
            if best_p >= cur_p_true:
                continue

            # 应用替换
            cur_text = best_text
            cur_p_true = best_p
            cur_pred = best_pred
            changes.append((w, best_cand))

            # 成功则提前停止
            if cur_pred != true_label:
                break

        probs_after = self.victim.predict_proba(
            cur_text, max_length=self.cfg.max_length)
        pred_after = int(np.argmax(probs_after))
        p_true_after = float(probs_after[true_label])

        success = (pred_after != true_label)

        return {
            "adv_text": cur_text,
            "success": bool(success),
            "num_changes": len(changes),
            "p_true_before": p_true_before,
            "p_true_after": p_true_after,
            "pred_before": pred_before,
            "pred_after": pred_after,
            "changes": changes,
            "note": "",
        }


# =========================
# 6) 批量生成对抗数据并保存
# =========================
def generate_adversarial_dataset(
    input_csv: str,
    output_csv: str,
    victim_dir: str,
    mlm_dir: str,
    split: str = "test",
    device: str = "cuda",
    max_samples: Optional[int] = None,
    only_label: Optional[int] = None,   # 例如只攻击 label=1（诈骗）
    METHOD: str = "sentence",  # "synonym" 或 "sentence"（整句改写）
):
    """
    input_csv: 你的原始数据（至少含 specific_dialogue_content, is_fraud）
    output_csv: 保存对抗样本
    victim_dir: 你训练好的二分类模型目录（Trainer.save_model 的输出目录）
    mlm_dir: 本地 MLM 目录（如 roberta-wwm-ext / macbert）
    """
    dp = DataProcessor(train_path=input_csv, test_path=input_csv)
    df = dp.load_and_clean(input_csv)

    if only_label is not None:
        df = df[df["label"] == int(only_label)].reset_index(drop=True)

    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    print(
        f"[Attack] load {len(df)} samples from {input_csv} (only_label={only_label})")

    victim = VictimClassifier(victim_dir, device=device)
    mlm = MLMAttacker(mlm_dir, device=device)
    cfg = AttackConfig(
        max_length=256,
        top_k=20,
        max_mod_ratio=0.15,
        min_word_len=2,
    )
    attacker = BertAttackGenerator(victim, mlm, cfg)

    records = []
    succ = 0
    s_cfg = SentenceRewriteConfig(
        max_length=256,
        span_len=1,
        num_spans=8,
        fill_top_k=50,
        beam_size=8,
        max_fill_steps=32,
        victim_batch_size=256,
    )
    sentence_rewriter = SpanMaskSentenceRewriter(
        victim, mlm, s_cfg)

    for i, row in df.iterrows():
        text = row["text"]
        label = int(row["label"])

        # out = attacker.attack_one(text, true_label=label)
        if label == 1:
            if METHOD == "synonym":
                out = attacker.attack_one(text, true_label=label)
            else:
                # sentence-level
                # s_cfg = SentenceRewriteConfig(
                #     max_length=256,
                #     span_len=1,
                #     num_spans=8,
                #     fill_top_k=50,
                #     beam_size=8,
                #     max_fill_steps=32,
                #     victim_batch_size=64,
                # )
                # sentence_rewriter = SpanMaskSentenceRewriter(
                #     victim, mlm, s_cfg)
                out = sentence_rewriter.rewrite_one(text, true_label=label)
            attacked = True
            # out = attacker.attack_one(text, true_label=label)
            # attacked = True
        else:
            # label=0 不攻击，直接保留原文
            out = {
                "adv_text": text,
                "success": False,
                "num_changes": 0,
                "p_true_before": None,
                "p_true_after": None,
                "pred_before": None,
                "pred_after": None,
                "changes": [],
                "note": "skip_label0",
            }
            attacked = False

        succ += int(out["success"])

        records.append({
            "text": text,
            "label": label,
            "fraud_type": row.get("fraud_type", "unknown"),
            "adv_text": out["adv_text"],
            "success": out["success"],
            "num_changes": out["num_changes"],
            "p_true_before": out["p_true_before"],
            "p_true_after": out["p_true_after"],
            "pred_before": out["pred_before"],
            "pred_after": out["pred_after"],
            "changes": json.dumps(out["changes"], ensure_ascii=False),
            "note": out.get("note", ""),
            "attacked": attacked,
        })

        if (i + 1) % 50 == 0:
            print(f"[Attack] {i+1}/{len(df)}  ASR={succ/(i+1):.3f}")

    adv_df = pd.DataFrame(records)
    adv_df.to_csv(output_csv, index=False)
    print(f"[Attack] Saved: {output_csv}")
    print(
        f"[Attack] Final ASR={succ/len(df):.4f}  (success/total={succ}/{len(df)})")


# =========================
# 7) 入口：你改路径即可跑
# =========================
if __name__ == "__main__":
    # ====== 你需要改这四个路径 ======
    INPUT_CSV = "/data1/wph/nlp/hw1/final/datasets/test_set.csv"  # 你要攻击的数据（建议先攻击 test）

    VICTIM_DIR = "/data1/wph/nlp/hw1/final/new/bert_fraud_cls"  # 你训练保存的二分类模型目录
    # MLM_DIR = "hfl/chinese-roberta-wwm-ext"  # 你本地的 MLM 目录（示例）
    MLM_DIR = "hfl/chinese-macbert-base"
    OUTPUT_CSV = f"test_adv_bert_attack_MACBERT.csv"

    generate_adversarial_dataset(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        victim_dir=VICTIM_DIR,
        mlm_dir=MLM_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        METHOD="synonym",     # "synonym" 或 "sentence"
        max_samples=None,      # 可先填 200 做小跑
    )
