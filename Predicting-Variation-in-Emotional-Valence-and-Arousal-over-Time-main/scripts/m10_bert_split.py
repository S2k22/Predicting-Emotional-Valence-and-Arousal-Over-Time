"""
M10: DistilBERT + all features + text-type split

Same features as M9 but trains separate models for essays (is_words=0)
and feeling-word lists (is_words=1). Uses a 15% validation split within
each text type for early stopping. User baselines are recomputed from the
training split only within each fold to avoid leakage.

Additionally saves validation predictions for hybrid weight tuning (M11).
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUT_PATH = ROOT / "outputs" / "submission_m10.csv"
VAL_PATH = ROOT / "outputs" / "val_m10.csv"

MODEL_NAME = "/fp/projects01/ec403/hf_models/models--distilbert-base-multilingual-cased/snapshots/45c032ab32cc946ad88a166f7cb282f58c753c2e"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-5
WEIGHT_DECAY = 0.01
SEED = 42
VAL_SIZE = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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


# ============================================================
# HELPER: ATTACH USER BASELINES (computed from reference set)
# ============================================================

def attach_user_baselines(reference_df, *dfs):
    """Compute user means from reference_df and attach to all dfs."""
    user_means = reference_df.groupby("user_id")[["valence", "arousal"]].mean()
    user_means.columns = ["user_valence_mean", "user_arousal_mean"]

    global_val = reference_df["valence"].mean()
    global_aro = reference_df["arousal"].mean()

    results = []
    for df in dfs:
        df = df.copy()
        df = df.drop(
            columns=["user_valence_mean", "user_arousal_mean"],
            errors="ignore",
        )
        df = df.merge(user_means, on="user_id", how="left")
        df["user_valence_mean"] = df["user_valence_mean"].fillna(global_val)
        df["user_arousal_mean"] = df["user_arousal_mean"].fillna(global_aro)
        results.append(df)

    return results


# ============================================================
# DATASET
# ============================================================

class EmotionDataset(Dataset):
    def __init__(self, texts, numeric_features, tokenizer, targets=None):
        self.texts = list(texts)
        self.numeric_features = numeric_features
        self.tokenizer = tokenizer
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "numeric": self.numeric_features[idx],
        }
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item


# ============================================================
# MODEL
# ============================================================

class TransformerRegressor(nn.Module):
    def __init__(self, model_name, num_numeric_features):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size + num_numeric_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, numeric):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_embedding, numeric], dim=1)
        return self.regressor(combined)


# ============================================================
# TRAINING HELPERS
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        preds = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["numeric"].to(DEVICE),
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
                batch["numeric"].to(DEVICE),
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
                batch["numeric"].to(DEVICE),
            )
            all_preds.append(preds.cpu().numpy())
    return np.vstack(all_preds)


# ============================================================
# MAIN TRAINING LOOP: SEPARATE MODELS FOR ESSAYS AND WORD LISTS
# ============================================================

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_test_preds = []
all_val_preds = []

for is_words_value in [0, 1]:
    print("\n" + "=" * 60)
    print(f"Training model for is_words={is_words_value}")
    print("=" * 60)

    train_part = train[train["is_words"] == is_words_value].copy()
    test_part = test[test["is_words"] == is_words_value].copy()

    train_part = train_part.reset_index(drop=True)
    test_part = test_part.reset_index(drop=True)

    # Train / val split within this text type, grouped by user_id
    train_split, val_split = group_train_val_split(train_part)

    # Compute user baselines from training split only (no leakage)
    train_split, val_split, test_part = attach_user_baselines(
        train_split, train_split, val_split, test_part,
    )

    # Scale numeric features (fit on training split only)
    scaler = StandardScaler()

    train_numeric = torch.tensor(
        scaler.fit_transform(train_split[NUMERIC_FEATURES]), dtype=torch.float32
    )
    val_numeric = torch.tensor(
        scaler.transform(val_split[NUMERIC_FEATURES]), dtype=torch.float32
    )
    test_numeric = torch.tensor(
        scaler.transform(test_part[NUMERIC_FEATURES]), dtype=torch.float32
    )

    train_targets = torch.tensor(train_split[TARGET_COLS].values, dtype=torch.float32)
    val_targets = torch.tensor(val_split[TARGET_COLS].values, dtype=torch.float32)

    # Datasets and loaders
    train_dataset = EmotionDataset(train_split["text"], train_numeric, tokenizer, train_targets)
    val_dataset = EmotionDataset(val_split["text"], val_numeric, tokenizer, val_targets)
    test_dataset = EmotionDataset(test_part["text"], test_numeric, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = TransformerRegressor(
        MODEL_NAME, num_numeric_features=len(NUMERIC_FEATURES)
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None

    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        print(
            f"is_words={is_words_value} | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best validation loss for is_words={is_words_value}: {best_val_loss:.4f}")

    model.load_state_dict(best_state_dict)
    model.to(DEVICE)

    # Test predictions
    test_preds = predict(model, test_loader)
    all_test_preds.append(pd.DataFrame({
        "user_id": test_part["user_id"].values,
        "text_id": test_part["text_id"].values,
        "pred_valence": test_preds[:, 0],
        "pred_arousal": test_preds[:, 1],
    }))

    # Validation predictions (for M11 weight tuning)
    val_preds = predict(model, val_loader)
    all_val_preds.append(pd.DataFrame({
        "user_id": val_split["user_id"].values,
        "text_id": val_split["text_id"].values,
        "pred_valence": val_preds[:, 0],
        "pred_arousal": val_preds[:, 1],
        "true_valence": val_split["valence"].values,
        "true_arousal": val_split["arousal"].values,
    }))


# ============================================================
# SAVE TEST SUBMISSION
# ============================================================

submission = pd.concat(all_test_preds, axis=0)
submission = test[["user_id", "text_id"]].merge(
    submission, on=["user_id", "text_id"], how="left",
)

OUT_PATH.parent.mkdir(exist_ok=True)
submission.to_csv(OUT_PATH, index=False)
print(f"\nSaved submission to {OUT_PATH}")


# ============================================================
# SAVE VALIDATION PREDICTIONS FOR M11
# ============================================================

val_df = pd.concat(all_val_preds, axis=0)
val_df.to_csv(VAL_PATH, index=False)
print(f"Saved validation predictions to {VAL_PATH}")
