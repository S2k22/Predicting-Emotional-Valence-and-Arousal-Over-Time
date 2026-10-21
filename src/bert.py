"""
DistilBERT regression model used by M6-M10.

The [CLS] embedding of multilingual DistilBERT is optionally concatenated
with standardised numeric features and passed to a small MLP that predicts
valence and arousal. The checkpoint with the lowest validation loss is kept.

Set MODEL_NAME to a local path if the machine has no internet access
(for example on an HPC cluster).
"""

import os
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import SEED, TARGET_COLS

MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-multilingual-cased")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters used for all reported runs
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.25


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, numeric=None, targets=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.numeric = numeric
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.numeric is not None:
            item["numeric"] = self.numeric[idx]
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item


class TransformerRegressor(nn.Module):
    def __init__(self, model_name, num_numeric=0):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden = self.transformer.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(hidden + num_numeric, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, numeric=None):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        features = out.last_hidden_state[:, 0, :]
        if numeric is not None:
            features = torch.cat([features, numeric], dim=1)
        return self.head(features)


def _forward(model, batch):
    numeric = batch["numeric"].to(DEVICE) if "numeric" in batch else None
    return model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), numeric)


def _predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(_forward(model, batch).cpu().numpy())
    return np.vstack(preds)


def _mean_loss(model, loader, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += loss_fn(_forward(model, batch), batch["labels"].to(DEVICE)).item()
    return total / len(loader)


def train_bert_model(train_split, val_split, test_df, numeric_features=None):
    """
    Fine-tune on train_split, keep the epoch with the lowest validation loss,
    and return (test_predictions, validation_predictions).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_num = val_num = test_num = None
    if numeric_features:
        scaler = StandardScaler().fit(train_split[numeric_features])
        to_tensor = lambda df: torch.tensor(scaler.transform(df[numeric_features]),
                                            dtype=torch.float32)
        train_num, val_num, test_num = map(to_tensor, (train_split, val_split, test_df))

    targets = lambda df: torch.tensor(df[TARGET_COLS].values, dtype=torch.float32)
    train_loader = DataLoader(
        EmotionDataset(train_split["text"], tokenizer, train_num, targets(train_split)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        EmotionDataset(val_split["text"], tokenizer, val_num, targets(val_split)),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        EmotionDataset(test_df["text"], tokenizer, test_num), batch_size=BATCH_SIZE,
    )

    model = TransformerRegressor(MODEL_NAME, len(numeric_features or [])).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps,
    )
    loss_fn = nn.MSELoss()

    print(f"Device: {DEVICE} | train {len(train_split)} | val {len(val_split)} | "
          f"test {len(test_df)}")
    best_loss, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(_forward(model, batch), batch["labels"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        val_loss = _mean_loss(model, val_loader, loss_fn)
        print(f"Epoch {epoch}/{EPOCHS} | train loss {train_loss / len(train_loader):.4f} "
              f"| val loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return _predict(model, test_loader), _predict(model, val_loader)
