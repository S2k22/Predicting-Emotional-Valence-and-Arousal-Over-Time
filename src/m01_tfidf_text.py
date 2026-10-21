"""M1: TF-IDF text only. No engineered features, no text-type split."""

from common import (TARGET_COLS, build_tfidf_model, group_train_val_split,
                    load_data, prediction_frame, save_predictions)

train, test = load_data()

# Train on all training data -> test predictions
model = build_tfidf_model()
model.fit(train["text"], train[TARGET_COLS])
save_predictions(prediction_frame(test, model.predict(test["text"])), "submission_m1")

# Train on the training split -> validation predictions
train_split, val_split = group_train_val_split(train)
model = build_tfidf_model()
model.fit(train_split["text"], train_split[TARGET_COLS])
save_predictions(
    prediction_frame(val_split, model.predict(val_split["text"]), with_labels=True), "val_m1"
)
