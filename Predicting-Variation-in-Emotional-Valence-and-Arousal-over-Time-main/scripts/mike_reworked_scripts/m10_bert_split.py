"""M10: DistilBERT + all features + text-type split (essays vs word lists)."""

from utils import *

set_seed()
train, test = load_data()
train = add_features(train)
test = add_features(test)

NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

print(f"Using device: {DEVICE}")

all_test_preds = []
all_val_preds = []

for is_words_val in [0, 1]:
    print(f"\n{'='*60}\nTraining model for is_words={is_words_val}\n{'='*60}")

    train_part = train[train["is_words"] == is_words_val].reset_index(drop=True)
    test_part = test[test["is_words"] == is_words_val].reset_index(drop=True)

    train_split, val_split = group_train_val_split(train_part)
    train_split, val_split, test_part = attach_user_baselines(
        train_split, train_split, val_split, test_part,
    )

    test_preds, val_preds, _ = train_bert_model(
        train_split, val_split, test_part, numeric_features=NUMERIC,
    )

    all_test_preds.append(pd.DataFrame({
        "user_id": test_part["user_id"].values, "text_id": test_part["text_id"].values,
        "pred_valence": test_preds[:, 0], "pred_arousal": test_preds[:, 1],
    }))
    all_val_preds.append(pd.DataFrame({
        "user_id": val_split["user_id"].values, "text_id": val_split["text_id"].values,
        "pred_valence": val_preds[:, 0], "pred_arousal": val_preds[:, 1],
        "true_valence": val_split["valence"].values, "true_arousal": val_split["arousal"].values,
    }))

# Save test submission
submission = pd.concat(all_test_preds)
submission = test[["user_id", "text_id"]].merge(submission, on=["user_id", "text_id"], how="left")
OUTPUTS_DIR.mkdir(exist_ok=True)
submission.to_csv(OUTPUTS_DIR / "submission_m10.csv", index=False)
print(f"\nSaved submission to {OUTPUTS_DIR / 'submission_m10.csv'}")

# Save validation predictions for M11
pd.concat(all_val_preds).to_csv(OUTPUTS_DIR / "val_m10.csv", index=False)
print(f"Saved validation predictions to {OUTPUTS_DIR / 'val_m10.csv'}")
