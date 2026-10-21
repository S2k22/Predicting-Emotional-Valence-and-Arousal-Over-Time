"""M1: TF-IDF text-only — no engineered features, no text-type split."""

from utils import *

train, test = load_data()

# Full train → test
model = build_tfidf_model()
model.fit(train["text"], train[TARGET_COLS])
preds = model.predict(test["text"])
save_submission(test, preds, OUTPUTS_DIR / "submission_m1.csv")

# Validation
train_split, val_split = group_train_val_split(train)
val_model = build_tfidf_model()
val_model.fit(train_split["text"], train_split[TARGET_COLS])
val_preds = val_model.predict(val_split["text"])
save_val_predictions(val_split, val_preds, OUTPUTS_DIR / "val_m1.csv")
