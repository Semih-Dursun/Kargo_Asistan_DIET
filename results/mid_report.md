# Mid-project report: Team 22 (Token Eight)

DRAFT. Numbers are copied from `results/mid_summary.md` (produced by `summarize.py`).
Check wording and edit before submitting.

## Question

On MASSIVE tr-TR (test: 2,974 utterances), does adding frozen BERTurk features to Rasa's DIET
raise intent macro-F1 over the identical sparse-only DIET pipeline (Config A)? Config B (with
BERTurk) is a final-project item. This report covers the baselines and Config A only.

## What was run

- Data: MASSIVE tr-TR, train / dev / test = 11,514 / 2,033 / 2,974 (matches the proposal).
- Majority class and TF-IDF (char 1-4) + logistic regression. `C` chosen on dev only.
- Config A: WhitespaceTokenizer, CountVectors (word), CountVectors (char_wb 1-4), DIETClassifier,
  100 epochs, seeds 42, 43, 44.
- Metric: intent macro-F1 over the 59 intents that occur in the evaluated split. Accuracy as secondary.
- Intervals: 95% bootstrap (1,000 resamples of the test set); paired bootstrap for differences.

## Results (MASSIVE tr-TR)

| Model | Dev macro-F1 | Test macro-F1 | Test accuracy |
|---|---|---|---|
| Majority class | 0.21 | 0.22 | 7.03 |
| TF-IDF char 1-4 + LR | 82.65 | 81.72 (95% CI 79.3-83.7) | 83.42 |
| DIET Config A, mean of 3 seeds | 78.27 ± 1.28 | 77.32 ± 0.76 | 80.37 ± 0.05 |

Config A per seed on test: 76.62 / 77.22 / 78.12 macro-F1. (± is the sample standard
deviation over the 3 seeds.)

Config A minus TF-IDF+LR on test (paired bootstrap, 95% interval):

| Seed | Difference (points) | 95% interval |
|---|---|---|
| 42 | -5.10 | [-6.74, -3.10] |
| 43 | -4.50 | [-6.39, -2.41] |
| 44 | -3.60 | [-5.48, -1.66] |

## DIET tuning on dev (Config A only)

TF-IDF's `C` was tuned on dev, so we gave DIET a small dev-only search. One parameter changed
at a time, seed 42, test never used. Decision rule fixed before looking at results: confirm the
best candidate on 3 seeds on dev, and adopt it only if its 3-seed dev mean beats the default
mean by about one seed standard deviation (about 1.3 points).

| Candidate (seed 42) | Dev macro-F1 | Dev accuracy |
|---|---|---|
| default | 79.71 | 81.21 |
| epochs 50 | 77.07 | 80.42 |
| drop_rate 0.4 | 78.74 | 81.31 |
| regularization 0.01 | 78.94 | 81.11 |
| hidden layers [256,128] | 79.13 | 80.91 |
| no transformer layers | 80.64 | 80.87 |
| weight_sparsity 0.9 | 80.82 | 82.05 |

The best candidate (weight_sparsity 0.9) was confirmed on 3 seeds: dev 79.45 ± 1.32 against
78.27 ± 1.28 for the default (+1.2 points). This is below our adoption threshold, so the
default stays the main Config A and the tuned variant is reported as a dev-only exploratory
result. We did not run it on test. Its gain is within seed noise. Only one parameter at a time
was varied; combinations and other values were not tried.

## What this shows and does not show

- The simple baseline is strong: TF-IDF+LR scores 81.7, above the 70-78 range the proposal
  expected. Config A (77.3) falls inside that range but is 3.6-5.1 points below TF-IDF+LR in
  every seed, and each interval excludes zero.
- The room left for BERTurk is small. A full BERTurk fine-tune is published at 85.0, only about
  3 points above TF-IDF+LR. The proposal already allows a "no" answer.
- This does not say whether BERTurk helps inside DIET (Config B is not run yet). It says
  nothing about the cargo domain. It does not say why Config A is below TF-IDF+LR; we have not
  tested a cause.

## Deviations from the proposal and caveats (state these in the talk)

1. Test has 59 intents, not 60 (`cooking_query` is absent); dev also has 59. Macro-F1 is over
   the intents present.
2. `FallbackClassifier` is off for MASSIVE (it has no out-of-scope class). It will be on for the
   cargo set.
3. DIET batch size is capped at `[64, 128]` instead of the default `[64, 256]`, because the 4 GB
   GPU ran out of memory. Applied to A, B and C alike. Not tuned on dev.
4. Rasa removes exact duplicate training examples (11,514 -> 11,181). Evaluation uses the full test
   CSV; the baseline trains on the same de-duplicated text.
5. The TF-IDF `C` search ended at its upper edge (`C=100`, grid 1-100). A larger `C` could make
   the baseline slightly stronger; it was not tried.
6. DIET tuning was a small one-at-a-time search on dev (see above), not a full grid. Entity F1
   (secondary) is not computed yet. The same tuning budget must be given to Config B later.
7. Entity spans with punctuation (for example "g. p. s.") are misaligned by the whitespace
   tokenizer; this may lower entity F1 later.
8. Seeds only change initialisation; GPU training may not be bit-for-bit repeatable.

## Cargo set status

- Schema draft: `docs/cargo_schema.md` (13 intents, 5 entity types, 2 boundary examples per
  intent, confusable pairs). Drafted with Claude; the team still has to review it.
- Training utterances: not collected yet. Target for mid: ~300 annotated utterances written by the team.
  Check progress with `cargo_stats.py`.

## Next (final)

Config B and the ablation (Config C), paired bootstrap B - A, cargo test set from 5 outside
writers with out-of-scope inputs, error analysis on the confusable intent pairs, MASSIVE
learning curve (low-resource condition), Rasa demo.

## Reproduce

```
python prepare_massive.py            # data/raw/*.parquet -> data/massive
PYTHONPATH=. python baselines.py
python make_configs.py
bash run_mid.sh                      # trains A for seeds 42-44, then summarize.py
```
