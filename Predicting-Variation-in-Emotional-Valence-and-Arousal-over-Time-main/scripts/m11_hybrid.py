"""
M11: Hybrid ensemble of best TF-IDF (M5) and best DistilBERT (M10)

Combines the strongest TF-IDF and DistilBERT models using weighted
averaging. The ensemble weight w is selected on a held-out validation
split to maximise the mean composite Pearson correlation across valence
and arousal. Candidate weights are evaluated in increments of 0.1.

    y_hat = w * y_tfidf + (1 - w) * y_bert

Requires M5 and M10 to have been run first, producing:
  - outputs/submission_m5.csv   (test predictions)
  - outputs/submission_m10.csv  (test predictions)
  - outputs/val_m5.csv          (validation predictions + true labels)
  - outputs/val_m10.csv         (validation predictions + true labels)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TFIDF_TEST_PATH = ROOT / "outputs" / "submission_m5.csv"
BERT_TEST_PATH = ROOT / "outputs" / "submission_m10.csv"
TFIDF_VAL_PATH = ROOT / "outputs" / "val_m5.csv"
BERT_VAL_PATH = ROOT / "outputs" / "val_m10.csv"

OUT_PATH = ROOT / "outputs" / "submission_m11.csv"
OUT_PATH.parent.mkdir(exist_ok=True)


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

val_tfidf = pd.read_csv(TFIDF_VAL_PATH)
val_bert = pd.read_csv(BERT_VAL_PATH)

# Merge validation predictions on shared rows (user_id + text_id)
val = val_tfidf.merge(
    val_bert,
    on=["user_id", "text_id"],
    suffixes=("_tfidf", "_bert"),
)

# True labels (same in both files)
true_valence = val["true_valence_tfidf"].values
true_arousal = val["true_arousal_tfidf"].values


# ============================================================
# WEIGHT TUNING: MAXIMISE MEAN COMPOSITE PEARSON r
# ============================================================

candidate_weights = np.arange(0.0, 1.01, 0.1)

best_w = 0.5
best_score = -np.inf

print("Weight tuning on validation set:")
print(f"{'w':>5}  {'r_valence':>10}  {'r_arousal':>10}  {'mean_r':>8}")
print("-" * 40)

for w in candidate_weights:
    pred_val = w * val["pred_valence_tfidf"] + (1 - w) * val["pred_valence_bert"]
    pred_aro = w * val["pred_arousal_tfidf"] + (1 - w) * val["pred_arousal_bert"]

    r_valence, _ = pearsonr(true_valence, pred_val)
    r_arousal, _ = pearsonr(true_arousal, pred_aro)
    mean_r = (r_valence + r_arousal) / 2

    print(f"{w:5.1f}  {r_valence:10.4f}  {r_arousal:10.4f}  {mean_r:8.4f}")

    if mean_r > best_score:
        best_score = mean_r
        best_w = w

print(f"\nSelected weight: w = {best_w:.1f} (mean r = {best_score:.4f})")


# ============================================================
# COMBINE TEST PREDICTIONS
# ============================================================

test_tfidf = pd.read_csv(TFIDF_TEST_PATH)
test_bert = pd.read_csv(BERT_TEST_PATH)

df = test_tfidf.merge(
    test_bert,
    on=["user_id", "text_id"],
    suffixes=("_tfidf", "_bert"),
)

df["pred_valence"] = (
    best_w * df["pred_valence_tfidf"]
    + (1 - best_w) * df["pred_valence_bert"]
)
df["pred_arousal"] = (
    best_w * df["pred_arousal_tfidf"]
    + (1 - best_w) * df["pred_arousal_bert"]
)

submission = df[["user_id", "text_id", "pred_valence", "pred_arousal"]]
submission.to_csv(OUT_PATH, index=False)
print(f"Saved hybrid submission to {OUT_PATH}")
