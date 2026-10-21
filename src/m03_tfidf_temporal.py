"""M3: TF-IDF + structural/intensity + temporal/study-design features."""

from common import (TARGET_COLS, add_features, build_tfidf_model, get_numeric_features,
                    group_train_val_split, load_data, prediction_frame, save_predictions,
                    tfidf_input)

train, test = load_data()
train = add_features(train, structural=True, temporal=True)
test = add_features(test, structural=True, temporal=True)
NUMERIC = get_numeric_features(structural=True, temporal=True)

# Train on all training data -> test predictions
model = build_tfidf_model(NUMERIC)
model.fit(tfidf_input(train, NUMERIC), train[TARGET_COLS])
preds = model.predict(tfidf_input(test, NUMERIC))
save_predictions(prediction_frame(test, preds), "submission_m3", test_order=test)

# Train on the training split -> validation predictions
train_split, val_split = group_train_val_split(train)
model = build_tfidf_model(NUMERIC)
model.fit(tfidf_input(train_split, NUMERIC), train_split[TARGET_COLS])
preds = model.predict(tfidf_input(val_split, NUMERIC))
save_predictions(prediction_frame(val_split, preds, with_labels=True), "val_m3")
