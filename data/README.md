# Data

The dataset is not included in this repository. It is the longitudinal
ecological momentary assessment (EMA) data from SemEval-2026 Task 2
(Predicting Variation in Emotional Valence and Arousal over Time from
Ecological Essays), which we received through the IN5550 course at the
University of Oslo. Get it from the task organisers and place the two files
here:

```
data/
├── train.csv
└── test.csv
```

Expected columns:

| Column | Description |
|---|---|
| `user_id` | Participant ID |
| `text_id` | Entry ID |
| `text` | Free-text essay or comma-separated list of feeling words |
| `timestamp` | Time of the entry, e.g. `2021-06-08 12:26:16` |
| `collection_phase` | Study phase (1 to 7) |
| `is_words` | `True` for feeling-word lists, `False` for essays |
| `valence` | Self-reported valence, -2 to 2 |
| `arousal` | Self-reported arousal, 0 to 2 |
| `is_seen_user` | Test set only: whether the user also appears in the training set |

To keep the data somewhere else, set `DATA_DIR`:

```bash
DATA_DIR=/path/to/data python src/m01_tfidf_text.py
```
