"""
Evaluation metrics for longitudinal valence/arousal prediction.

These follow the official SemEval-2026 Task 2 (subtask 1) evaluation:

- r_within:    mean of per-user Pearson correlations. Users with fewer than
               two entries or constant gold labels are skipped; a user whose
               predictions are constant counts as r = 0.
- r_between:   Pearson correlation between per-user mean predictions and
               per-user mean gold labels.
- r_composite: tanh of the mean Fisher z of r_within and r_between.
- mae_composite: computed the same way from per-user MAE and between-user
               MAE, exactly as in the official script.
"""

import numpy as np
from scipy.stats import pearsonr


def longitudinal_metrics(user_ids, predictions, labels):
    users = np.asarray(user_ids)
    preds = np.asarray(predictions, dtype=float)
    labs = np.asarray(labels, dtype=float)

    within_r, within_mae, mean_pred, mean_lab = [], [], [], []
    for user in np.unique(users):
        mask = users == user
        p, y = preds[mask], labs[mask]
        mean_pred.append(p.mean())
        mean_lab.append(y.mean())
        within_mae.append(np.abs(p - y).mean())

        if mask.sum() < 2 or np.var(y) == 0.0:
            continue
        within_r.append(0.0 if np.var(p) == 0.0 else pearsonr(p, y)[0])

    r_within = float(np.mean(within_r))
    r_between = float(pearsonr(mean_pred, mean_lab)[0])
    r_composite = float(np.tanh((np.arctanh(r_within) + np.arctanh(r_between)) / 2))

    mae_within = float(np.mean(within_mae))
    mae_between = float(np.mean(np.abs(np.array(mean_pred) - np.array(mean_lab))))
    mae_composite = float(np.tanh((np.arctanh(mae_within) + np.arctanh(mae_between)) / 2))

    return {
        "r_between": r_between,
        "r_within": r_within,
        "r_composite": r_composite,
        "mae_composite": mae_composite,
    }
