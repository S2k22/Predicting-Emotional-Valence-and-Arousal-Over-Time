"""M4: TF-IDF + structural/intensity + temporal + user baseline features."""

from common import (TARGET_COLS, add_features, attach_user_baselines, build_tfidf_model,
                    get_numeric_features, group_train_val_split, load_data,
                    prediction_frame, save_predictions, tfidf_input)

train, test = load_data()
train = add_features(train)
test = add_features(test)
NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

# Test: user baselines from all training data
train_full, test_full = attach_user_baselines(train, train, test)
model = build_tfidf_model(NUMERIC)
model.fit(tfidf_input(train_full, NUMERIC), train_full[TARGET_COLS])
preds = model.predict(tfidf_input(test_full, NUMERIC))
save_predictions(prediction_frame(test_full, preds), "submission_m4", test_order=test)

# Validation: user baselines from the training split only
train_split, val_split = group_train_val_split(train)
train_split, val_split = attach_user_baselines(train_split, train_split, val_split)
model = build_tfidf_model(NUMERIC)
model.fit(tfidf_input(train_split, NUMERIC), train_split[TARGET_COLS])
preds = model.predict(tfidf_input(val_split, NUMERIC))
save_predictions(prediction_frame(val_split, preds, with_labels=True), "val_m4")
