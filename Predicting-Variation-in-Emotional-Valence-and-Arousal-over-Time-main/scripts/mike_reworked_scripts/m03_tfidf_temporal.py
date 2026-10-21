"""M3: TF-IDF + structural/intensity + temporal/study-design features."""

from utils import *

train, test = load_data()
train = add_features(train)
test = add_features(test)

NUMERIC = get_numeric_features(structural=True, temporal=True)

# Full train → test
model = build_tfidf_model(NUMERIC)
model.fit(get_tfidf_input(train, NUMERIC), train[TARGET_COLS])
preds = model.predict(get_tfidf_input(test, NUMERIC))
save_submission(test, preds, OUTPUTS_DIR / "submission_m3.csv")

# Validation
train_split, val_split = group_train_val_split(train)
val_model = build_tfidf_model(NUMERIC)
val_model.fit(get_tfidf_input(train_split, NUMERIC), train_split[TARGET_COLS])
val_preds = val_model.predict(get_tfidf_input(val_split, NUMERIC))
save_val_predictions(val_split, val_preds, OUTPUTS_DIR / "val_m3.csv")
