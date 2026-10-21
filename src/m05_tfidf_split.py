"""
M5: TF-IDF + all features, with separate models for essays (is_words=0)
and feeling-word lists (is_words=1). Its validation predictions are used
to choose the M11 ensemble weight.
"""

import pandas as pd

from common import (TARGET_COLS, add_features, attach_user_baselines, build_tfidf_model,
                    get_numeric_features, group_train_val_split, load_data,
                    prediction_frame, save_predictions, tfidf_input)

train, test = load_data()
train = add_features(train)
test = add_features(test)
NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

test_parts, val_parts = [], []
for text_type in (0, 1):
    train_part = train[train["is_words"] == text_type]
    test_part = test[test["is_words"] == text_type]

    # Test: train on all training data of this text type
    tr, te = attach_user_baselines(train_part, train_part, test_part)
    model = build_tfidf_model(NUMERIC)
    model.fit(tfidf_input(tr, NUMERIC), tr[TARGET_COLS])
    test_parts.append(prediction_frame(te, model.predict(tfidf_input(te, NUMERIC))))

    # Validation: user-grouped split within this text type
    train_split, val_split = group_train_val_split(train_part)
    train_split, val_split = attach_user_baselines(train_split, train_split, val_split)
    model = build_tfidf_model(NUMERIC)
    model.fit(tfidf_input(train_split, NUMERIC), train_split[TARGET_COLS])
    preds = model.predict(tfidf_input(val_split, NUMERIC))
    val_parts.append(prediction_frame(val_split, preds, with_labels=True))

save_predictions(pd.concat(test_parts), "submission_m5", test_order=test)
save_predictions(pd.concat(val_parts), "val_m5")
