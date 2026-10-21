"""M5: TF-IDF + all features + text-type split (essays vs word lists)."""

from utils import *

train, test = load_data()
train = add_features(train)
test = add_features(test)

NUMERIC = get_numeric_features(structural=True, temporal=True, user=True)

# ---- Full train → test predictions (split by text type) ----
all_test_preds = []
for is_words_val in [0, 1]:
    tr = train[train["is_words"] == is_words_val].copy()
    te = test[test["is_words"] == is_words_val].copy()
    tr, te = attach_user_baselines(tr, tr, te)

    model = build_tfidf_model(NUMERIC)
    model.fit(get_tfidf_input(tr, NUMERIC), tr[TARGET_COLS])
    preds = model.predict(get_tfidf_input(te, NUMERIC))
    all_test_preds.append(pd.DataFrame({
        "user_id": te["user_id"].values, "text_id": te["text_id"].values,
        "pred_valence": preds[:, 0], "pred_arousal": preds[:, 1],
    }))

submission = pd.concat(all_test_preds)
submission = test[["user_id", "text_id"]].merge(submission, on=["user_id", "text_id"], how="left")
OUTPUTS_DIR.mkdir(exist_ok=True)
submission.to_csv(OUTPUTS_DIR / "submission_m5.csv", index=False)
print(f"Saved submission to {OUTPUTS_DIR / 'submission_m5.csv'}")

# ---- Validation predictions (for M11 weight tuning) ----
all_val_preds = []
for is_words_val in [0, 1]:
    tr = train[train["is_words"] == is_words_val].copy()
    train_split, val_split = group_train_val_split(tr)
    train_split, val_split = attach_user_baselines(train_split, train_split, val_split)

    model = build_tfidf_model(NUMERIC)
    model.fit(get_tfidf_input(train_split, NUMERIC), train_split[TARGET_COLS])
    val_preds = model.predict(get_tfidf_input(val_split, NUMERIC))
    all_val_preds.append(pd.DataFrame({
        "user_id": val_split["user_id"].values, "text_id": val_split["text_id"].values,
        "pred_valence": val_preds[:, 0], "pred_arousal": val_preds[:, 1],
        "true_valence": val_split["valence"].values, "true_arousal": val_split["arousal"].values,
    }))

pd.concat(all_val_preds).to_csv(OUTPUTS_DIR / "val_m5.csv", index=False)
print(f"Saved validation predictions to {OUTPUTS_DIR / 'val_m5.csv'}")
