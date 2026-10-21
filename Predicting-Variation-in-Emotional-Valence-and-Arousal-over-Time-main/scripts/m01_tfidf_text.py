"""
M1: TF-IDF text-only

Text representation only. TF-IDF unigrams and bigrams followed by
independent Ridge regressors for valence and arousal.
No engineered features, no text-type split.
"""

import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline



# PATH SETUP

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUT_PATH = ROOT / "outputs" / "submission_m1.csv"
VAL_PATH = ROOT / "outputs" / "val_m1.csv"

SEED = 42
VAL_SIZE = 0.15

OUT_PATH.parent.mkdir(exist_ok=True)



# LOAD DATA


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")



# TARGETS


TARGET_COLS = ["valence", "arousal"]

y_train = train[TARGET_COLS]



# USER-GROUPED TRAIN / VALIDATION SPLIT

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



# MODEL BUILDER


def build_model():
    return MultiOutputRegressor(
        Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=20000,
            )),
            ("regressor", Ridge(alpha=1.0)),
        ])
    )



# MODEL (full training data → test predictions)

model = build_model()
model.fit(train["text"], y_train)

preds = model.predict(test["text"])



# SAVE SUBMISSION


submission = pd.DataFrame({
    "user_id": test["user_id"],
    "text_id": test["text_id"],
    "pred_valence": preds[:, 0],
    "pred_arousal": preds[:, 1],
})

submission.to_csv(OUT_PATH, index=False)
print(f"Saved submission to {OUT_PATH}")



# VALIDATION PREDICTIONS


train_split, val_split = group_train_val_split(train)

val_model = build_model()
val_model.fit(train_split["text"], train_split[TARGET_COLS])
val_preds = val_model.predict(val_split["text"])

val_df = pd.DataFrame({
    "user_id": val_split["user_id"].values,
    "text_id": val_split["text_id"].values,
    "pred_valence": val_preds[:, 0],
    "pred_arousal": val_preds[:, 1],
    "true_valence": val_split["valence"].values,
    "true_arousal": val_split["arousal"].values,
})

val_df.to_csv(VAL_PATH, index=False)
print(f"Saved validation predictions to {VAL_PATH}")
