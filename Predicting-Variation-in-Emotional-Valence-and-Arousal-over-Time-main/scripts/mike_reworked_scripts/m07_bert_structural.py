"""M7: DistilBERT + structural/intensity features."""

from utils import *

set_seed()
train, test = load_data()
train = add_features(train, structural=True, temporal=False)
test = add_features(test, structural=True, temporal=False)

NUMERIC = get_numeric_features(structural=True)

train_split, val_split = group_train_val_split(train)

print(f"Using device: {DEVICE}")

test_preds, val_preds, _ = train_bert_model(
    train_split, val_split, test, numeric_features=NUMERIC,
)

save_submission(test, test_preds, OUTPUTS_DIR / "submission_m7.csv")
save_val_predictions(val_split, val_preds, OUTPUTS_DIR / "val_m7.csv")
