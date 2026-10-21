"""
Shared code for all models: paths, data loading, feature engineering,
user-grouped validation splits, user baselines, the TF-IDF pipeline and
helpers for saving predictions.

Nothing in this file needs PyTorch, so the TF-IDF models (M1-M5) run
without a GPU or a transformers installation.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Paths (override with environment variables if needed)
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET_COLS = ["valence", "arousal"]
SEED = 42
VAL_SIZE = 0.15


# ------------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------------

STRUCTURAL_FEATURES = [
    "text_length",
    "word_count",
    "is_words",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "uppercase_ratio",
]

TEMPORAL_FEATURES = [
    "collection_phase",
    "time_since_first",
    "time_since_previous",
]

USER_FEATURES = [
    "user_valence_mean",
    "user_arousal_mean",
]


def get_numeric_features(structural=False, temporal=False, user=False):
    """Return the list of numeric feature columns for the chosen groups."""
    features = []
    if structural:
        features += STRUCTURAL_FEATURES
    if temporal:
        features += TEMPORAL_FEATURES
    if user:
        features += USER_FEATURES
    return features


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_data():
    """Load train and test CSVs. Missing text is replaced by an empty string."""
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. See data/README.md for how to get the data."
            )
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")
    return train, test


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def add_structural_features(df):
    """Surface form and expressive intensity of each entry."""
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_words"] = df["is_words"].astype(int)
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["uppercase_count"] = df["text"].str.count(r"[A-Z]")
    df["uppercase_ratio"] = df["uppercase_count"] / (df["text_length"] + 1)
    return df


def add_temporal_features(df):
    """Days since the user's first entry and since their previous entry."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
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


def add_features(df, structural=True, temporal=True):
    """Return a copy of df with the requested feature groups added."""
    df = df.copy()
    if structural:
        df = add_structural_features(df)
    if temporal:
        df = add_temporal_features(df)
    return df


# ------------------------------------------------------------------
# User baselines
# ------------------------------------------------------------------

def attach_user_baselines(reference_df, *dfs):
    """
    Compute each user's mean valence and arousal from reference_df and
    attach them to every df in dfs. Users not present in reference_df get
    the global mean of reference_df. Only labels from reference_df are used.
    """
    user_means = reference_df.groupby("user_id")[TARGET_COLS].mean()
    user_means.columns = USER_FEATURES

    global_valence = reference_df["valence"].mean()
    global_arousal = reference_df["arousal"].mean()

    results = []
    for df in dfs:
        df = df.drop(columns=USER_FEATURES, errors="ignore")
        df = df.merge(user_means, on="user_id", how="left")
        df["user_valence_mean"] = df["user_valence_mean"].fillna(global_valence)
        df["user_arousal_mean"] = df["user_arousal_mean"].fillna(global_arousal)
        results.append(df)
    return results


# ------------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------------

def group_train_val_split(df, group_col="user_id", val_size=VAL_SIZE, seed=SEED):
    """Split rows so that no user appears in both train and validation."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))

    train_split = df.iloc[train_idx].reset_index(drop=True)
    val_split = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_split[group_col]) & set(val_split[group_col])
    assert not overlap, "User leakage detected between train and validation"
    return train_split, val_split


# ------------------------------------------------------------------
# TF-IDF + Ridge
# ------------------------------------------------------------------

def _tfidf():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000)


def build_tfidf_model(numeric_features=None):
    """TF-IDF (1-2 grams) + optional standardised numeric features -> Ridge."""
    if numeric_features:
        features = ColumnTransformer([
            ("text", _tfidf(), "text"),
            ("numeric", StandardScaler(), numeric_features),
        ])
    else:
        features = _tfidf()

    return MultiOutputRegressor(
        Pipeline([("features", features), ("regressor", Ridge(alpha=1.0))])
    )


def tfidf_input(df, numeric_features=None):
    """Input expected by build_tfidf_model."""
    if numeric_features:
        return df[["text"] + numeric_features]
    return df["text"]


# ------------------------------------------------------------------
# Saving predictions
# ------------------------------------------------------------------

def prediction_frame(df, preds, with_labels=False):
    out = pd.DataFrame({
        "user_id": df["user_id"].values,
        "text_id": df["text_id"].values,
        "pred_valence": preds[:, 0],
        "pred_arousal": preds[:, 1],
    })
    if with_labels:
        out["true_valence"] = df["valence"].values
        out["true_arousal"] = df["arousal"].values
    return out


def save_predictions(frame, name, test_order=None):
    """
    Save a prediction frame to outputs/<name>.csv. If test_order is given,
    rows are put back into the original test-set order.
    """
    if test_order is not None:
        frame = test_order[["user_id", "text_id"]].merge(
            frame, on=["user_id", "text_id"], how="left"
        )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    print(f"Saved {path}")
