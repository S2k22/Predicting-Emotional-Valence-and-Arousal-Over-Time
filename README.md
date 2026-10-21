# Predicting Emotional Valence and Arousal Over Time

Predicting how people feel from short diary entries they write repeatedly over
several years.

Each entry is either a short essay or a list of feeling words, written as part of an
ecological momentary assessment (EMA) study. For every entry we predict the writer's
self-reported **valence** (how pleasant they feel, −2 to 2) and **arousal** (how
activated they feel, 0 to 2). The task asks for two things at once:

- **Within-person:** following how one person's mood changes from entry to entry.
- **Between-person:** telling apart people who are generally more positive or more
  energetic than others.

The data and evaluation come from SemEval-2026 Task 2 (Subtask 1), used in the
WNNLP 2026 workshop at the University of Oslo.

## Approach

We run a controlled ablation with 11 models. Feature groups are added one at a time
to two model families, so each group's contribution can be measured separately, and
the best model of each family is then combined in a weighted ensemble.

| ID | Model |
|---|---|
| M1 | TF-IDF + Ridge regression, text only |
| M2 | M1 + structural/intensity features (length, punctuation, capitals) |
| M3 | M2 + temporal features (collection phase, days since first and previous entry) |
| M4 | M3 + user baselines (the user's mean valence and arousal in the training data) |
| M5 | M4 with separate models for essays and feeling-word lists |
| M6–M10 | The same five steps with fine-tuned multilingual DistilBERT |
| M11 | Weighted average of M5 and M10 |

## Results

The main metric is the composite Pearson correlation, which combines within-person and
between-person correlation using Fisher's z (see `src/metrics.py`). Higher is better.

| Model | Valence | Arousal |
|---|---|---|
| M1 TF-IDF text only | 0.597 | 0.447 |
| M2 + structural | 0.581 | 0.458 |
| M3 + temporal | 0.596 | 0.454 |
| M4 + user baselines | 0.635 | **0.509** |
| M5 + text-type split | 0.608 | 0.454 |
| M6 DistilBERT text only | 0.609 | 0.379 |
| M7 + structural | 0.618 | 0.457 |
| M8 + temporal | 0.632 | 0.409 |
| M9 + user baselines | 0.584 | 0.392 |
| M10 + text-type split | 0.619 | 0.349 |
| **M11 hybrid (M5 + M10)** | **0.658** | 0.494 |

Test set: 1,737 entries from 91 users. Full tables with within-person, between-person
and MAE scores for both the validation and test sets are in `results/`.

**Main findings**

- **The hybrid ensemble is the best overall system:** 0.66 for valence and 0.49 for
  arousal. It is significantly better than both of its components on valence, based
  on a bootstrap over users. For arousal, it is not significantly different from the
  best single model (M4).
- **The two model families are complementary.** DistilBERT tracks changes within a
  person better, while TF-IDF better separates people with different baselines. The
  ensemble has the highest within-person correlation for both valence and arousal.
- **Whether personalisation helps depends on the evaluation.** User baselines improve
  the TF-IDF model on the test set, where 40% of entries come from users seen in
  training, but hurt on the validation split, where every user is new.
- **Arousal is harder to predict than valence** for every model.

## Repository structure

```
├── data/                  # put train.csv and test.csv here (see data/README.md)
├── results/
│   ├── predictions/       # predictions behind the results above
│   ├── test_metrics.csv
│   └── validation_metrics.csv
├── slurm/run_all.slurm    # example job script for a GPU cluster
├── src/
│   ├── common.py          # data loading, features, splits, TF-IDF pipeline
│   ├── bert.py            # DistilBERT regressor and training loop
│   ├── metrics.py         # within-person, between-person and composite metrics
│   ├── m01_…m05_*.py      # TF-IDF models
│   ├── m06_…m10_*.py      # DistilBERT models
│   ├── m11_hybrid.py      # ensemble
│   ├── evaluate.py        # validation and test tables
│   └── error_analysis.py  # error breakdowns for M10
├── requirements.txt
└── run_all.sh
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The TF-IDF models only need NumPy, pandas, SciPy and scikit-learn. PyTorch and
transformers are needed for M6–M10.

The dataset is not included. See `data/README.md` for where it comes from and the
expected format.

## Running

```bash
python src/m01_tfidf_text.py     # each model writes outputs/submission_mX.csv (test)
                                 # and outputs/val_mX.csv (validation)
python src/m11_hybrid.py         # needs M5 and M10 first
python src/evaluate.py           # writes the validation and test metric tables
```

Or run everything with `bash run_all.sh`. To score the stored predictions instead of
new ones:

```bash
python src/evaluate.py --pred-dir results/predictions --out-dir results
```

Each DistilBERT model takes a few minutes on one GPU. If the machine has no internet
access, download `distilbert-base-multilingual-cased` beforehand and set
`MODEL_NAME=/path/to/model`.

## Implementation details

- **Validation:** 15% of training users are held out with `GroupShuffleSplit`, so no
  user appears in both training and validation. M5 and M10 make this split separately
  for essays and word lists. User baselines for validation are computed from the
  training part only.
- **TF-IDF:** unigrams and bigrams, `min_df=2`, up to 20,000 features, Ridge with
  `alpha=1.0`. Numeric features are standardised inside the pipeline. For test
  predictions, these models are retrained on all training data.
- **DistilBERT:** `distilbert-base-multilingual-cased`. The [CLS] embedding is
  concatenated with the numeric features and passed to a two-layer head (128 units,
  dropout 0.25). Training uses AdamW with learning rate 1e-5, weight decay 0.01,
  batch size 16 and max length 128, for up to 8 epochs with linear warm-up over the
  first 10% of steps. The epoch with the lowest validation loss is kept. These models
  are trained on the 85% training part, since the validation part is needed for
  model selection.
- **Ensemble:** the weight is chosen from 0.0, 0.1, …, 1.0 by maximising the mean
  pooled Pearson r for valence and arousal on validation. This selects w = 0.4
  (40% TF-IDF, 60% DistilBERT).
- **Seed:** 42 for everything.
- **Temporal features:** in the run behind `results/`, timestamps were parsed with the
  wrong format, so the two time-gap features were always zero and only the collection
  phase carried temporal information. The code now parses timestamps correctly.
  Rerunning the TF-IDF models with the fix changes their scores by at most 0.025, so
  numbers from a fresh run will differ slightly from the tables above.

## Authors

Noor Fatima, Andre Maharaj Gregussen and Mykhailo Karpiuk, University of Oslo.

## Citation

```bibtex
@inproceedings{fatima2026valence,
  title     = {Predicting Emotional Valence and Arousal Over Time: A Hybrid Approach
               Combining Lexical and Contextual Representations},
  author    = {Fatima, Noor and Gregussen, Andre Maharaj and Karpiuk, Mykhailo},
  booktitle = {WNNLP 2026, University of Oslo},
  year      = {2026}
}
```
