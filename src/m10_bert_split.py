"""
M10: DistilBERT + all features, with separate models for essays
(is_words=0) and feeling-word lists (is_words=1). Its validation
predictions are used to choose the M11 ensemble weight.
"""

import pandas as pd

from bert import set_seed, train_bert_model
from common import (add_features, attach_user_baselines, get_numeric_features,
                    group_train_val_split, load_data, prediction_frame, save_predictions)

set_seed()
train, test = load_data()
train = add_features(train)
test = add_features(test)
NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

test_parts, val_parts = [], []
for text_type in (0, 1):
    print(f"\n=== is_words = {text_type} ===")
    train_part = train[train["is_words"] == text_type].reset_index(drop=True)
    test_part = test[test["is_words"] == text_type].reset_index(drop=True)

    train_split, val_split = group_train_val_split(train_part)
    train_split, val_split, test_part = attach_user_baselines(
        train_split, train_split, val_split, test_part,
    )

    test_preds, val_preds = train_bert_model(train_split, val_split, test_part, NUMERIC)
    test_parts.append(prediction_frame(test_part, test_preds))
    val_parts.append(prediction_frame(val_split, val_preds, with_labels=True))

save_predictions(pd.concat(test_parts), "submission_m10", test_order=test)
save_predictions(pd.concat(val_parts), "val_m10")
