"""
Score every model on the validation split and on the test set.

    python src/evaluate.py                       # scores outputs/
    python src/evaluate.py --pred-dir results/predictions --out-dir results

Validation files (val_m*.csv) already contain the gold labels. Test
predictions (submission_m*.csv) are joined with the labels in data/test.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from common import OUTPUTS_DIR, TEST_PATH
from metrics import longitudinal_metrics

MODELS = [f"m{i}" for i in range(1, 12)]


def score(df, label_cols):
    row = {}
    for dim in ("valence", "arousal"):
        m = longitudinal_metrics(df["user_id"], df[f"pred_{dim}"], df[label_cols[dim]])
        for key, value in m.items():
            row[f"{dim}_{key}"] = round(value, 4)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    test_labels = pd.read_csv(TEST_PATH)[["user_id", "text_id", "valence", "arousal"]]

    val_rows, test_rows = [], []
    for model in MODELS:
        val_path = args.pred_dir / f"val_{model}.csv"
        test_path = args.pred_dir / f"submission_{model}.csv"

        if val_path.exists():
            val = pd.read_csv(val_path)
            val_rows.append({"model": model.upper(),
                             **score(val, {"valence": "true_valence", "arousal": "true_arousal"})})
        if test_path.exists():
            test = pd.read_csv(test_path).merge(test_labels, on=["user_id", "text_id"])
            test_rows.append({"model": model.upper(),
                              **score(test, {"valence": "valence", "arousal": "arousal"})})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("validation_metrics", val_rows), ("test_metrics", test_rows)):
        table = pd.DataFrame(rows)
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"\n{name}")
        cols = ["model", "valence_r_composite", "arousal_r_composite"]
        print(table[cols].to_string(index=False) if not table.empty else "(no files found)")


if __name__ == "__main__":
    main()
