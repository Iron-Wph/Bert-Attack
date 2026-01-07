import pandas as pd

path = "test_adv_bert_attack_sentence_MACBERT.csv"
df = pd.read_csv(path)

pos = df[df["label"] == 1]
asr = pos["success"].astype(int).mean() if len(pos) > 0 else 0.0

print("pos_total =", len(pos))
print("pos_success =", pos["success"].astype(int).sum())
print("ASR_pos =", asr)
