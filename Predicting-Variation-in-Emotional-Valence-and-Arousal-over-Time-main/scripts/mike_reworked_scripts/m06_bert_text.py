"""M6: DistilBERT text-only — [CLS] embedding → regression head."""

from utils import *

set_seed()
train, test = load_data()
train_split, val_split = group_train_val_split(train)

print(f"Using device: {DEVICE}")

test_preds, val_preds, _ = train_bert_model(train_split, val_split, test)

save_submission(test, test_preds, OUTPUTS_DIR / "submission_m6.csv")
save_val_predictions(val_split, val_preds, OUTPUTS_DIR / "val_m6.csv")
