"""
Error analysis for the best single model (M10).

Reads val_m*.csv files and merges with training data to recover
text, is_words, collection_phase, and user_id metadata. Produces:

  1. Error-by-subgroup tables (text type, collection phase, text length)
  2. Concrete best/worst prediction examples for the paper
  3. Per-user analysis (hardest users, easiest users)
  4. Cross-model comparison on hardest examples

Main results table is produced separately by the official eval script.
Outputs are saved to  outputs/error_analysis/  as CSVs and a
plain-text report that can be pasted into a LaTeX document.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


# PATHS


# ---- CONFIGURE THESE TWO PATHS ----
DATA_DIR = Path(r"C:\Users\Mike\Desktop\Predicting-Variation-in-Emotional-Valence-and-Arousal-over-Time-main\datasets")
OUTPUTS = Path(r"C:\Users\Mike\Desktop\Predicting-Variation-in-Emotional-Valence-and-Arousal-over-Time-main\outputs")

TRAIN_PATH = DATA_DIR / "train.csv"
ANALYSIS_DIR = OUTPUTS / "error_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Use the best single model (M10) and the hybrid (M11) for
# detailed examples; compute summary metrics for all models.
MODEL_IDS = [f"m{i}" for i in range(1, 12)]
BEST_MODEL = "m10"   # best single model for detailed analysis
HYBRID_MODEL = "m11"  # ensemble



# LOAD DATA


train = pd.read_csv(TRAIN_PATH)
train["text"] = train["text"].fillna("")
train["user_id"] = train["user_id"].astype(str)
train["text_id"] = train["text_id"].astype(str)

# Add features needed for subgroup analysis
train["text_length"] = train["text"].str.len()
train["word_count"] = train["text"].str.split().str.len()

# Load all available val files
val_dfs = {}
for mid in MODEL_IDS:
    path = OUTPUTS / f"val_{mid}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["user_id"] = df["user_id"].astype(str)
        df["text_id"] = df["text_id"].astype(str)
        val_dfs[mid] = df
        print(f"Loaded {path.name}: {len(df)} rows")
    else:
        print(f"Skipping {path.name} (not found)")



# HELPER: COMPUTE METRICS (used by subgroup breakdowns below)


def compute_metrics(df):
    """Compute Pearson r, RMSE, and MAE for valence and arousal."""
    r_val, _ = pearsonr(df["true_valence"], df["pred_valence"])
    r_aro, _ = pearsonr(df["true_arousal"], df["pred_arousal"])

    rmse_val = np.sqrt(np.mean((df["true_valence"] - df["pred_valence"]) ** 2))
    rmse_aro = np.sqrt(np.mean((df["true_arousal"] - df["pred_arousal"]) ** 2))

    mae_val = np.mean(np.abs(df["true_valence"] - df["pred_valence"]))
    mae_aro = np.mean(np.abs(df["true_arousal"] - df["pred_arousal"]))

    return {
        "r_valence": r_val,
        "r_arousal": r_aro,
        "r_mean": (r_val + r_aro) / 2,
        "rmse_valence": rmse_val,
        "rmse_arousal": rmse_aro,
        "mae_valence": mae_val,
        "mae_arousal": mae_aro,
    }



# DETAILED ERROR ANALYSIS ON BEST MODEL


# Pick the best available model for detailed analysis
detail_model = BEST_MODEL if BEST_MODEL in val_dfs else list(val_dfs.keys())[-1]
val = val_dfs[detail_model].copy()

# Merge with training data to get text and metadata
val = val.merge(
    train[["user_id", "text_id", "text", "is_words", "collection_phase",
           "text_length", "word_count"]],
    on=["user_id", "text_id"],
    how="left",
)

matched = val["text"].notna().sum()
print(f"\nMerge result: {matched}/{len(val)} rows matched with training data")
if matched == 0:
    print("WARNING: No rows matched. Check that user_id/text_id formats align.")

# Compute errors
val["err_valence"] = val["pred_valence"] - val["true_valence"]
val["err_arousal"] = val["pred_arousal"] - val["true_arousal"]
val["abs_err_valence"] = val["err_valence"].abs()
val["abs_err_arousal"] = val["err_arousal"].abs()
val["abs_err_mean"] = (val["abs_err_valence"] + val["abs_err_arousal"]) / 2

# Label text type (handle bool, int, or float is_words)
val["is_words"] = pd.to_numeric(val["is_words"], errors="coerce").astype("Int64")
val["text_type"] = val["is_words"].map({0: "essay", 1: "word list"})
val["text"] = val["text"].fillna("")



# ERROR BY TEXT TYPE


def subgroup_metrics(df, group_col):
    rows = []
    for name, grp in df.groupby(group_col, dropna=True):
        m = compute_metrics(grp)
        m[group_col] = name
        m["n"] = len(grp)
        rows.append(m)
    if not rows:
        print(f"  WARNING: no valid groups found for '{group_col}'")
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(group_col)


by_text_type = subgroup_metrics(val, "text_type")
if not by_text_type.empty:
    by_text_type.to_csv(ANALYSIS_DIR / "error_by_text_type.csv", float_format="%.4f")
    print(f"\n=== ERROR BY TEXT TYPE ({detail_model.upper()}) ===")
    print(by_text_type.round(4).to_string())



# ERROR BY COLLECTION PHASE


if "collection_phase" in val.columns and val["collection_phase"].notna().any():
    by_phase = subgroup_metrics(val, "collection_phase")
    if not by_phase.empty:
        by_phase.to_csv(ANALYSIS_DIR / "error_by_phase.csv", float_format="%.4f")
        print(f"\n=== ERROR BY COLLECTION PHASE ({detail_model.upper()}) ===")
        print(by_phase.round(4).to_string())



# ERROR BY TEXT LENGTH BINS

if val["text_length"].notna().any():
    val["length_bin"] = pd.cut(
        val["text_length"],
        bins=[0, 50, 150, 300, 600, float("inf")],
        labels=["<50", "50-150", "150-300", "300-600", "600+"],
    )

    by_length = subgroup_metrics(val, "length_bin")
    if not by_length.empty:
        by_length.to_csv(ANALYSIS_DIR / "error_by_length.csv", float_format="%.4f")
        print(f"\n=== ERROR BY TEXT LENGTH ({detail_model.upper()}) ===")
        print(by_length.round(4).to_string())



# EXAMPLES FOR THE PAPER


def format_example(row, rank_label=""):
    """Format a single prediction example for display."""
    text_preview = row["text"][:120] + ("..." if len(row["text"]) > 120 else "")
    return (
        f"{rank_label}"
        f"  text_type: {row['text_type']}  |  "
        f"text_id: {row['text_id']}  |  user_id: {row['user_id']}\n"
        f"  text: \"{text_preview}\"\n"
        f"  valence  — true: {row['true_valence']:.2f}  pred: {row['pred_valence']:.2f}  "
        f"err: {row['err_valence']:+.2f}\n"
        f"  arousal  — true: {row['true_arousal']:.2f}  pred: {row['pred_arousal']:.2f}  "
        f"err: {row['err_arousal']:+.2f}\n"
        f"  mean |err|: {row['abs_err_mean']:.3f}\n"
    )


N_EXAMPLES = 5

# Best predictions (lowest mean absolute error)
best = val.nsmallest(N_EXAMPLES, "abs_err_mean")

# Worst predictions (highest mean absolute error)
worst = val.nlargest(N_EXAMPLES, "abs_err_mean")

# Worst per text type
worst_essay = val[val["is_words"] == 0].nlargest(N_EXAMPLES, "abs_err_mean")
worst_words = val[val["is_words"] == 1].nlargest(N_EXAMPLES, "abs_err_mean")

# Largest valence over-prediction / under-prediction
over_valence = val.nlargest(N_EXAMPLES, "err_valence")
under_valence = val.nsmallest(N_EXAMPLES, "err_valence")

# Largest arousal over-prediction / under-prediction
over_arousal = val.nlargest(N_EXAMPLES, "err_arousal")
under_arousal = val.nsmallest(N_EXAMPLES, "err_arousal")


report_lines = []
report_lines.append(f"ERROR ANALYSIS REPORT — {detail_model.upper()}")
report_lines.append(f"Validation set: {len(val)} samples\n")

sections = [
    ("BEST PREDICTIONS (lowest mean |error|)", best),
    ("WORST PREDICTIONS (highest mean |error|)", worst),
    ("WORST ESSAYS", worst_essay),
    ("WORST WORD LISTS", worst_words),
    ("LARGEST VALENCE OVER-PREDICTIONS", over_valence),
    ("LARGEST VALENCE UNDER-PREDICTIONS", under_valence),
    ("LARGEST AROUSAL OVER-PREDICTIONS", over_arousal),
    ("LARGEST AROUSAL UNDER-PREDICTIONS", under_arousal),
]

for title, subset in sections:
    report_lines.append(f"\n{'='*60}")
    report_lines.append(title)
    report_lines.append("=" * 60)
    for i, (_, row) in enumerate(subset.iterrows(), 1):
        report_lines.append(format_example(row, rank_label=f"[{i}]"))

report_text = "\n".join(report_lines)
print(f"\n{report_text}")

(ANALYSIS_DIR / "example_report.txt").write_text(report_text)



# SAVE FULL ERROR TABLE (for custom exploration)

val.to_csv(ANALYSIS_DIR / "full_errors.csv", index=False)



# PER-USER DIFFICULTY

user_stats = (
    val.groupby("user_id")
    .agg(
        n=("abs_err_mean", "size"),
        mean_abs_err=("abs_err_mean", "mean"),
        std_abs_err=("abs_err_mean", "std"),
        mean_true_valence=("true_valence", "mean"),
        mean_true_arousal=("true_arousal", "mean"),
    )
    .sort_values("mean_abs_err", ascending=False)
)

user_stats.to_csv(ANALYSIS_DIR / "per_user_difficulty.csv", float_format="%.4f")

print(f"\n=== HARDEST USERS ({detail_model.upper()}) ===")
print(user_stats.head(10).round(4).to_string())
print(f"\n=== EASIEST USERS ({detail_model.upper()}) ===")
print(user_stats.tail(10).round(4).to_string())



# CROSS-MODEL COMPARISON ON SHARED EXAMPLES


# Find the worst examples from the best model and see how other models
# did on those same samples — useful for showing ensemble improvement.

if len(val_dfs) > 1:
    worst_ids = worst[["user_id", "text_id"]].copy()
    worst_ids["user_id"] = worst_ids["user_id"].astype(str)
    worst_ids["text_id"] = worst_ids["text_id"].astype(str)

    comparison_rows = []
    for mid, vdf in val_dfs.items():
        vdf = vdf.copy()
        vdf["user_id"] = vdf["user_id"].astype(str)
        vdf["text_id"] = vdf["text_id"].astype(str)
        matched = worst_ids.merge(vdf, on=["user_id", "text_id"], how="inner")
        if len(matched) == 0:
            continue
        for _, row in matched.iterrows():
            comparison_rows.append({
                "model": mid.upper(),
                "user_id": row["user_id"],
                "text_id": row["text_id"],
                "pred_valence": row["pred_valence"],
                "pred_arousal": row["pred_arousal"],
                "true_valence": row["true_valence"],
                "true_arousal": row["true_arousal"],
                "abs_err_valence": abs(row["pred_valence"] - row["true_valence"]),
                "abs_err_arousal": abs(row["pred_arousal"] - row["true_arousal"]),
            })

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(ANALYSIS_DIR / "cross_model_worst_examples.csv",
                          index=False, float_format="%.4f")

        print(f"\n=== CROSS-MODEL VIEW OF WORST {detail_model.upper()} EXAMPLES ===")
        pivot = comparison.pivot_table(
            index=["text_id"],
            columns="model",
            values=["abs_err_valence", "abs_err_arousal"],
            aggfunc="first",
        ).round(4)
        print(pivot.to_string())


# SUMMARY

print(f"\n{'='*60}")
print(f"All outputs saved to: {ANALYSIS_DIR}/")
print(f"  error_by_text_type.csv      — breakdown by essay vs word list")
print(f"  error_by_phase.csv          — breakdown by collection phase")
print(f"  error_by_length.csv         — breakdown by text length")
print(f"  example_report.txt          — concrete examples for the paper")
print(f"  full_errors.csv             — full error table for {detail_model.upper()}")
print(f"  per_user_difficulty.csv     — per-user error stats")
print(f"  cross_model_worst_examples.csv — how all models did on hardest cases")
print(f"{'='*60}")