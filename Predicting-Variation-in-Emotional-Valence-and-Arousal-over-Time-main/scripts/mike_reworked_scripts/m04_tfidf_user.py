"""M4: TF-IDF + structural/intensity + temporal/study-design + user baselines."""

from utils import *

train, test = load_data()
train = add_features(train)
test = add_features(test)

NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

# Full train → test (user baselines from all training data)
train, test = attach_user_baselines(train, train, test)

model = build_tfidf_model(NUMERIC)
model.fit(get_tfidf_input(train, NUMERIC), train[TARGET_COLS])
preds = model.predict(get_tfidf_input(test, NUMERIC))
save_submission(test, preds, OUTPUTS_DIR / "submission_m4.csv")

# Validation (user baselines from training split only)
train_split, val_split = group_train_val_split(train)
train_split, val_split = attach_user_baselines(train_split, train_split, val_split)

val_model = build_tfidf_model(NUMERIC)
val_model.fit(get_tfidf_input(train_split, NUMERIC), train_split[TARGET_COLS])
val_preds = val_model.predict(get_tfidf_input(val_split, NUMERIC))
save_val_predictions(val_split, val_preds, OUTPUTS_DIR / "val_m4.csv")
