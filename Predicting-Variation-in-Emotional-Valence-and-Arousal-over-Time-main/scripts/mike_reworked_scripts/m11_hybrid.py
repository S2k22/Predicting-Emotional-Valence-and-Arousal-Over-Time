"""M11: Hybrid ensemble of best TF-IDF (M5) and best DistilBERT (M10).

Weighted averaging: y_hat = w * y_tfidf + (1 - w) * y_bert
Weight w is selected on validation to maximise mean Pearson r.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from utils import OUTPUTS_DIR

OUTPUTS_DIR.mkdir(exist_ok=True)

# ---- Load validation predictions ----
val_tfidf = pd.read_csv(OUTPUTS_DIR / "val_m5.csv")
val_bert = pd.read_csv(OUTPUTS_DIR / "val_m10.csv")

val = val_tfidf.merge(val_bert, on=["user_id", "text_id"], suffixes=("_tfidf", "_bert"))
true_valence = val["true_valence_tfidf"].values
true_arousal = val["true_arousal_tfidf"].values

# ---- Weight tuning ----
best_w, best_score = 0.5, -np.inf

print(f"{'w':>5}  {'r_valence':>10}  {'r_arousal':>10}  {'mean_r':>8}")
print("-" * 40)

for w in np.arange(0.0, 1.01, 0.1):
    pred_v = w * val["pred_valence_tfidf"] + (1 - w) * val["pred_valence_bert"]
    pred_a = w * val["pred_arousal_tfidf"] + (1 - w) * val["pred_arousal_bert"]
    r_v, _ = pearsonr(true_valence, pred_v)
    r_a, _ = pearsonr(true_arousal, pred_a)
    mean_r = (r_v + r_a) / 2
    print(f"{w:5.1f}  {r_v:10.4f}  {r_a:10.4f}  {mean_r:8.4f}")
    if mean_r > best_score:
        best_score, best_w = mean_r, w

print(f"\nSelected weight: w = {best_w:.1f} (mean r = {best_score:.4f})")

# ---- Combine test predictions ----
test_tfidf = pd.read_csv(OUTPUTS_DIR / "submission_m5.csv")
test_bert = pd.read_csv(OUTPUTS_DIR / "submission_m10.csv")
df = test_tfidf.merge(test_bert, on=["user_id", "text_id"], suffixes=("_tfidf", "_bert"))

df["pred_valence"] = best_w * df["pred_valence_tfidf"] + (1 - best_w) * df["pred_valence_bert"]
df["pred_arousal"] = best_w * df["pred_arousal_tfidf"] + (1 - best_w) * df["pred_arousal_bert"]

df[["user_id", "text_id", "pred_valence", "pred_arousal"]].to_csv(
    OUTPUTS_DIR / "submission_m11.csv", index=False,
)
print(f"Saved hybrid submission to {OUTPUTS_DIR / 'submission_m11.csv'}")
