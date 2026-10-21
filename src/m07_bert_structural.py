"""M7: DistilBERT + structural/intensity features."""

from bert import set_seed, train_bert_model
from common import (add_features, get_numeric_features, group_train_val_split, load_data,
                    prediction_frame, save_predictions)

set_seed()
train, test = load_data()
train = add_features(train, structural=True, temporal=False)
test = add_features(test, structural=True, temporal=False)
NUMERIC = get_numeric_features(structural=True)

train_split, val_split = group_train_val_split(train)
test_preds, val_preds = train_bert_model(train_split, val_split, test, NUMERIC)
save_predictions(prediction_frame(test, test_preds), "submission_m7", test_order=test)
save_predictions(prediction_frame(val_split, val_preds, with_labels=True), "val_m7")
