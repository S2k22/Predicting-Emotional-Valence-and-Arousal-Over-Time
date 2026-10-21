"""
M11: weighted average of the best TF-IDF model (M5) and the best DistilBERT
model (M10):

    y_hat = w * y_tfidf + (1 - w) * y_bert

w is chosen on the validation split from 0.0, 0.1, ..., 1.0 by maximising
the mean of the pooled Pearson r for valence and arousal. Requires that
M5 and M10 have been run first.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import OUTPUTS_DIR, save_predictions

KEYS = ["user_id", "text_id"]


def load_pair(prefix):
    tfidf = pd.read_csv(OUTPUTS_DIR / f"{prefix}_m5.csv")
    bert = pd.read_csv(OUTPUTS_DIR / f"{prefix}_m10.csv")
    return tfidf.merge(bert, on=KEYS, suffixes=("_tfidf", "_bert"))


def blend(df, w, dim):
    return w * df[f"pred_{dim}_tfidf"] + (1 - w) * df[f"pred_{dim}_bert"]


# Choose w on validation
val = load_pair("val")
best_w, best_score = None, -np.inf
print(f"{'w':>4}  {'r_valence':>9}  {'r_arousal':>9}  {'mean':>7}")
for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
    r_val = pearsonr(val["true_valence_tfidf"], blend(val, w, "valence"))[0]
    r_aro = pearsonr(val["true_arousal_tfidf"], blend(val, w, "arousal"))[0]
    mean_r = (r_val + r_aro) / 2
    print(f"{w:4.1f}  {r_val:9.4f}  {r_aro:9.4f}  {mean_r:7.4f}")
    if mean_r > best_score:
        best_w, best_score = w, mean_r
print(f"\nSelected w = {best_w:.1f}")

val_out = val[KEYS].assign(
    pred_valence=blend(val, best_w, "valence"),
    pred_arousal=blend(val, best_w, "arousal"),
    true_valence=val["true_valence_tfidf"],
    true_arousal=val["true_arousal_tfidf"],
)
save_predictions(val_out, "val_m11")

# Apply the chosen w to the test predictions
test = load_pair("submission")
test_out = test[KEYS].assign(
    pred_valence=blend(test, best_w, "valence"),
    pred_arousal=blend(test, best_w, "arousal"),
)
save_predictions(test_out, "submission_m11")
