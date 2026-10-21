"""
M5: TF-IDF + all features + text-type split

Same features as M4 but trains separate models for essays (is_words=0)
and feeling-word lists (is_words=1). This tests whether text-type
specialization improves predictions.

Additionally saves validation predictions for hybrid weight tuning (M11).
"""

import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUT_PATH = ROOT / "outputs" / "submission_m5.csv"
VAL_PATH = ROOT / "outputs" / "val_m5.csv"

SEED = 42
VAL_SIZE = 0.15

OUT_PATH.parent.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()

    # --- Structural / intensity ---
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_words"] = df["is_words"].astype(int)
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["uppercase_count"] = df["text"].str.count(r"[A-Z]")
    df["uppercase_ratio"] = df["uppercase_count"] / (df["text_length"] + 1)

    # --- Temporal / study-design ---
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%m-%d-%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.sort_values(["user_id", "timestamp"])

    df["time_since_first"] = (
        df.groupby("user_id")["timestamp"]
        .transform(lambda x: (x - x.min()).dt.total_seconds() / 86400)
        .fillna(0)
    )
    df["time_since_previous"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )

    return df


train = add_features(train)
test = add_features(test)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = [
    # Structural / intensity
    "text_length",
    "word_count",
    "is_words",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "uppercase_ratio",
    # Temporal / study-design
    "collection_phase",
    "time_since_first",
    "time_since_previous",
    # User baselines
    "user_valence_mean",
    "user_arousal_mean",
]

INPUT_FEATURES = ["text"] + NUMERIC_FEATURES
TARGET_COLS = ["valence", "arousal"]


# ============================================================
# USER-GROUPED TRAIN / VALIDATION SPLIT
# ============================================================

def group_train_val_split(df, group_col="user_id", val_size=VAL_SIZE, seed=SEED):
    """Split rows so the same user_id cannot appear in both train and validation."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))

    train_split = df.iloc[train_idx].reset_index(drop=True)
    val_split = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_split[group_col]) & set(val_split[group_col])
    print(f"User overlap between train and validation: {len(overlap)}")
    assert len(overlap) == 0, "User leakage detected between train and validation"

    return train_split, val_split


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=20000,
            ), "text"),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return MultiOutputRegressor(
        Pipeline([
            ("features", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ])
    )


# ============================================================
# HELPER: COMPUTE USER BASELINES FROM A TRAINING SUBSET
# ============================================================

def attach_user_baselines(train_df, *other_dfs):
    """Compute user means from train_df and attach to all dataframes."""
    user_means = train_df.groupby("user_id")[["valence", "arousal"]].mean()
    user_means.columns = ["user_valence_mean", "user_arousal_mean"]

    global_val = train_df["valence"].mean()
    global_aro = train_df["arousal"].mean()

    result = []
    for df in (train_df, *other_dfs):
        df = df.copy()
        # Drop existing user baseline columns if present
        df = df.drop(
            columns=["user_valence_mean", "user_arousal_mean"],
            errors="ignore",
        )
        df = df.merge(user_means, on="user_id", how="left")
        df["user_valence_mean"] = df["user_valence_mean"].fillna(global_val)
        df["user_arousal_mean"] = df["user_arousal_mean"].fillna(global_aro)
        result.append(df)

    return result


# ============================================================
# TRAIN WITH TEXT-TYPE SPLIT: FULL DATA → TEST PREDICTIONS
# ============================================================

all_test_preds = []

for is_words_value in [0, 1]:
    train_part = train[train["is_words"] == is_words_value].copy()
    test_part = test[test["is_words"] == is_words_value].copy()

    train_part, test_part = attach_user_baselines(train_part, test_part)

    model = build_model()
    model.fit(train_part[INPUT_FEATURES], train_part[TARGET_COLS])
    preds = model.predict(test_part[INPUT_FEATURES])

    all_test_preds.append(pd.DataFrame({
        "user_id": test_part["user_id"].values,
        "text_id": test_part["text_id"].values,
        "pred_valence": preds[:, 0],
        "pred_arousal": preds[:, 1],
    }))

submission = pd.concat(all_test_preds, axis=0)
submission = test[["user_id", "text_id"]].merge(
    submission, on=["user_id", "text_id"], how="left",
)
submission.to_csv(OUT_PATH, index=False)
print(f"Saved submission to {OUT_PATH}")


# ============================================================
# VALIDATION PREDICTIONS FOR M11 WEIGHT TUNING
# ============================================================

all_val_preds = []

for is_words_value in [0, 1]:
    train_part = train[train["is_words"] == is_words_value].copy()

    train_split, val_split = group_train_val_split(train_part)

    train_split, val_split = attach_user_baselines(train_split, val_split)

    model = build_model()
    model.fit(train_split[INPUT_FEATURES], train_split[TARGET_COLS])
    val_preds = model.predict(val_split[INPUT_FEATURES])

    all_val_preds.append(pd.DataFrame({
        "user_id": val_split["user_id"].values,
        "text_id": val_split["text_id"].values,
        "pred_valence": val_preds[:, 0],
        "pred_arousal": val_preds[:, 1],
        "true_valence": val_split["valence"].values,
        "true_arousal": val_split["arousal"].values,
    }))

val_df = pd.concat(all_val_preds, axis=0)
val_df.to_csv(VAL_PATH, index=False)
print(f"Saved validation predictions to {VAL_PATH}")
