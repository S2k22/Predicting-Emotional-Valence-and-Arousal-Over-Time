"""M9: DistilBERT + structural/intensity + temporal + user baseline features."""

from bert import set_seed, train_bert_model
from common import (add_features, attach_user_baselines, get_numeric_features,
                    group_train_val_split, load_data, prediction_frame, save_predictions)

set_seed()
train, test = load_data()
train = add_features(train)
test = add_features(test)
NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

# User baselines come from the training split only, for validation and test alike
train_split, val_split = group_train_val_split(train)
train_split, val_split, test_feat = attach_user_baselines(
    train_split, train_split, val_split, test,
)

test_preds, val_preds = train_bert_model(train_split, val_split, test_feat, NUMERIC)
save_predictions(prediction_frame(test_feat, test_preds), "submission_m9", test_order=test)
save_predictions(prediction_frame(val_split, val_preds, with_labels=True), "val_m9")
