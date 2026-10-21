"""
Shared utilities for the valence/arousal prediction pipeline.

Provides reusable components for data loading, feature engineering,
train/val splitting, user baselines, PyTorch datasets and models,
training loops, and prediction saving.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# PATHS AND CONSTANTS


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUTPUTS_DIR = ROOT / "outputs"

BERT_MODEL_NAME = (
    "/fp/projects01/ec403/hf_models/"
    "models--distilbert-base-multilingual-cased/"
    "snapshots/45c032ab32cc946ad88a166f7cb282f58c753c2e"
)

TARGET_COLS = ["valence", "arousal"]
SEED = 42
VAL_SIZE = 0.15

# Feature group definitions
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
    """Build the list of numeric features from selected groups."""
    features = []
    if structural:
        features += STRUCTURAL_FEATURES
    if temporal:
        features += TEMPORAL_FEATURES
    if user:
        features += USER_FEATURES
    return features



# REPRODUCIBILITY


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# DATA LOADING


def load_data():
    """Load train and test CSVs with NaN text filled."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")
    return train, test



# FEATURE ENGINEERING


def add_structural_features(df):
    """Add structural/intensity features in place."""
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_words"] = df["is_words"].astype(int)
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["uppercase_count"] = df["text"].str.count(r"[A-Z]")
    df["uppercase_ratio"] = df["uppercase_count"] / (df["text_length"] + 1)
    return df


def add_temporal_features(df):
    """Add temporal/study-design features. Requires structural features first."""
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%m-%d-%Y %H:%M:%S", errors="coerce"
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


def add_features(df, structural=True, temporal=True):
    """Add engineered features to a dataframe."""
    df = df.copy()
    if structural:
        df = add_structural_features(df)
    if temporal:
        df = add_temporal_features(df)
    return df


# USER BASELINES


def attach_user_baselines(reference_df, *dfs):
    """
    Compute per-user mean valence/arousal from reference_df and attach
    to each df. Unseen users get the global mean from reference_df.
    """
    user_means = reference_df.groupby("user_id")[TARGET_COLS].mean()
    user_means.columns = ["user_valence_mean", "user_arousal_mean"]

    global_val = reference_df["valence"].mean()
    global_aro = reference_df["arousal"].mean()

    results = []
    for df in dfs:
        df = df.copy()
        df = df.drop(columns=USER_FEATURES, errors="ignore")
        df = df.merge(user_means, on="user_id", how="left")
        df["user_valence_mean"] = df["user_valence_mean"].fillna(global_val)
        df["user_arousal_mean"] = df["user_arousal_mean"].fillna(global_aro)
        results.append(df)
    return results



# TRAIN / VALIDATION SPLITTING


def group_train_val_split(df, group_col="user_id", val_size=VAL_SIZE, seed=SEED):
    """Split so no user appears in both train and validation."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))

    train_split = df.iloc[train_idx].reset_index(drop=True)
    val_split = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_split[group_col]) & set(val_split[group_col])
    assert len(overlap) == 0, "User leakage detected between train and validation"
    print(f"User overlap between train and validation: {len(overlap)}")

    return train_split, val_split



# SAVING HELPERS


def save_submission(test_df, preds, out_path):
    """Save test predictions in submission format."""
    submission = pd.DataFrame({
        "user_id": test_df["user_id"].values,
        "text_id": test_df["text_id"].values,
        "pred_valence": preds[:, 0],
        "pred_arousal": preds[:, 1],
    })
    out_path.parent.mkdir(exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")
    return submission


def save_val_predictions(val_df, val_preds, val_path):
    """Save validation predictions with true labels."""
    result = pd.DataFrame({
        "user_id": val_df["user_id"].values,
        "text_id": val_df["text_id"].values,
        "pred_valence": val_preds[:, 0],
        "pred_arousal": val_preds[:, 1],
        "true_valence": val_df["valence"].values,
        "true_arousal": val_df["arousal"].values,
    })
    result.to_csv(val_path, index=False)
    print(f"Saved validation predictions to {val_path}")
    return result


# TF-IDF MODEL BUILDER


def build_tfidf_model(numeric_features=None):
    """Build a TF-IDF + Ridge pipeline, optionally with numeric features."""
    if numeric_features:
        preprocessor = ColumnTransformer(transformers=[
            ("text", TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000,
            ), "text"),
            ("numeric", StandardScaler(), numeric_features),
        ])
    else:
        preprocessor = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000,
        )

    return MultiOutputRegressor(
        Pipeline([("features", preprocessor), ("regressor", Ridge(alpha=1.0))])
    )


def get_tfidf_input(df, numeric_features=None):
    """Return the appropriate input for a TF-IDF model."""
    if numeric_features:
        return df[["text"] + numeric_features]
    return df["text"]



# TRANSFORMER DATASET


class EmotionDataset(Dataset):
    """PyTorch dataset for text (+optional numeric) regression."""

    def __init__(self, texts, tokenizer, numeric=None, targets=None, max_length=128):
        self.texts = list(texts)
        self.numeric = numeric
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.numeric is not None:
            item["numeric"] = self.numeric[idx]
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item



# TRANSFORMER MODEL


class TransformerRegressor(nn.Module):
    """DistilBERT [CLS] + optional numeric → 2-output regression head."""

    def __init__(self, model_name, num_numeric_features=0):
        super().__init__()
        from transformers import AutoModel
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        input_size = hidden_size + num_numeric_features
        self.regressor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )
        self.has_numeric = num_numeric_features > 0

    def forward(self, input_ids, attention_mask, numeric=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        if self.has_numeric and numeric is not None:
            cls_embedding = torch.cat([cls_embedding, numeric], dim=1)
        return self.regressor(cls_embedding)



# TRANSFORMER TRAINING


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        preds = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch.get("numeric", torch.empty(0)).to(DEVICE) if "numeric" in batch else None,
        )
        loss = loss_fn(preds, batch["labels"].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch.get("numeric", torch.empty(0)).to(DEVICE) if "numeric" in batch else None,
            )
            loss = loss_fn(preds, batch["labels"].to(DEVICE))
            total_loss += loss.item()
    return total_loss / len(loader)


def predict(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch.get("numeric", torch.empty(0)).to(DEVICE) if "numeric" in batch else None,
            )
            all_preds.append(preds.cpu().numpy())
    return np.vstack(all_preds)


def train_bert_model(
    train_split, val_split, test_df, numeric_features=None,
    model_name=BERT_MODEL_NAME, max_length=128, batch_size=16,
    epochs=8, lr=1e-5, weight_decay=0.01,
):
    """
    Full training pipeline for a transformer model.
    Returns (test_preds, val_preds, model).
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare numeric tensors if needed
    train_num = val_num = test_num = None
    num_numeric = 0
    if numeric_features:
        num_numeric = len(numeric_features)
        scaler = StandardScaler()
        train_num = torch.tensor(
            scaler.fit_transform(train_split[numeric_features]), dtype=torch.float32
        )
        val_num = torch.tensor(
            scaler.transform(val_split[numeric_features]), dtype=torch.float32
        )
        test_num = torch.tensor(
            scaler.transform(test_df[numeric_features]), dtype=torch.float32
        )

    train_targets = torch.tensor(train_split[TARGET_COLS].values, dtype=torch.float32)
    val_targets = torch.tensor(val_split[TARGET_COLS].values, dtype=torch.float32)

    train_ds = EmotionDataset(train_split["text"], tokenizer, train_num, train_targets, max_length)
    val_ds = EmotionDataset(val_split["text"], tokenizer, val_num, val_targets, max_length)
    test_ds = EmotionDataset(test_df["text"], tokenizer, test_num, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TransformerRegressor(model_name, num_numeric).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    print(f"Training: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")

    for epoch in range(epochs):
        t_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        v_loss = evaluate_loss(model, val_loader, loss_fn)
        print(f"Epoch {epoch+1}/{epochs} | train loss: {t_loss:.4f} | val loss: {v_loss:.4f}")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    test_preds = predict(model, test_loader)
    val_preds = predict(model, val_loader)

    return test_preds, val_preds, model
