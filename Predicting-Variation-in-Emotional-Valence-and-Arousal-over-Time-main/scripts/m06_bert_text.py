"""
M6: DistilBERT text-only

Contextual text representation only. The final-layer [CLS] embedding from
multilingual DistilBERT is passed to a regression head. No engineered
features, no text-type split. Uses a 15% validation split for early stopping.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUT_PATH = ROOT / "outputs" / "submission_m6.csv"
VAL_PATH = ROOT / "outputs" / "val_m6.csv"

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
# TRAIN / VALIDATION SPLIT
# ============================================================

train_split, val_split = group_train_val_split(train)


# ============================================================
# DATASET (text-only, no numeric features)
# ============================================================

class EmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, targets=None):
        self.texts = list(texts)
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
        }
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item


# ============================================================
# MODEL (text-only — [CLS] → regression head)
# ============================================================

class TransformerRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_embedding)


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
            )
            all_preds.append(preds.cpu().numpy())
    return np.vstack(all_preds)


# ============================================================
# MAIN
# ============================================================

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_targets = torch.tensor(train_split[TARGET_COLS].values, dtype=torch.float32)
val_targets = torch.tensor(val_split[TARGET_COLS].values, dtype=torch.float32)

train_dataset = EmotionDataset(train_split["text"], tokenizer, train_targets)
val_dataset = EmotionDataset(val_split["text"], tokenizer, val_targets)
test_dataset = EmotionDataset(test["text"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TransformerRegressor(MODEL_NAME).to(DEVICE)

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
    print(f"Epoch {epoch + 1}/{EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print(f"Best validation loss: {best_val_loss:.4f}")

model.load_state_dict(best_state_dict)
model.to(DEVICE)

preds = predict(model, test_loader)


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "user_id": test["user_id"].values,
    "text_id": test["text_id"].values,
    "pred_valence": preds[:, 0],
    "pred_arousal": preds[:, 1],
})

OUT_PATH.parent.mkdir(exist_ok=True)
submission.to_csv(OUT_PATH, index=False)
print(f"\nSaved submission to {OUT_PATH}")


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

val_preds = predict(model, val_loader)

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
