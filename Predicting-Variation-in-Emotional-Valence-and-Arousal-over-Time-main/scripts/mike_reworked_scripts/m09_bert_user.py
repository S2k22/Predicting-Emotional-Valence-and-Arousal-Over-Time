"""M9: DistilBERT + structural/intensity + temporal/study-design + user baselines."""

from utils import *

set_seed()
train, test = load_data()
train = add_features(train)
test = add_features(test)

NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

train_split, val_split = group_train_val_split(train)
train_split, val_split, test = attach_user_baselines(
    train_split, train_split, val_split, test,
)

print(f"Using device: {DEVICE}")

test_preds, val_preds, _ = train_bert_model(
    train_split, val_split, test, numeric_features=NUMERIC,
)

save_submission(test, test_preds, OUTPUTS_DIR / "submission_m9.csv")
save_val_predictions(val_split, val_preds, OUTPUTS_DIR / "val_m9.csv")
