"""M6: DistilBERT, text only."""

from bert import set_seed, train_bert_model
from common import group_train_val_split, load_data, prediction_frame, save_predictions

set_seed()
train, test = load_data()
train_split, val_split = group_train_val_split(train)

test_preds, val_preds = train_bert_model(train_split, val_split, test)
save_predictions(prediction_frame(test, test_preds), "submission_m6")
save_predictions(prediction_frame(val_split, val_preds, with_labels=True), "val_m6")
