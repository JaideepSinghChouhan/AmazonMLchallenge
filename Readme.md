# Business Entity Resolution — MVP

End-to-end pipeline that reads TSV business records from three sources,
generates candidates, scores pairs with a tabular classifier, and writes
`output/matching_results.tsv` + `output/candidate_pairs.tsv`.

## 1. Problem
Given a Source-1 reference record, predict the set of Source-2 and Source-3
records referring to the same real-world business. Zero, one, or many matches
are allowed. Metric is **macro-averaged per-entity F0.5** (precision weighted 2×).

## 2. Dataset
- train: `dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv`
- test:  `dataset/test/{test_source1,test_source2,test_source3}.tsv`
- Override the root with `BER_DATA_DIR=/path/to/dataset`.

## 3. Architecture
DATA → PREPROCESS → BLOCK (multi-strategy) → FEATURES → MODEL → ENTITY DECISION → SUBMISSION

## 4. Preprocessing
Multiple representations per field (never one):
- name: `normalize_name` (NFKC, lowercase, punctuation-stripped, abbrev-expanded),
  `core_name` (generic tokens removed), `name_sorted`, token sets, char n-grams.
- address: `normalize_address` (abbrev expansion preserving numbers), token set,
  numeric-token set, postal-code set (4–6 digits — covers India PIN, US ZIP, France).
- country: light canonicalization only, **open-set** — no hard-coded list,
  unseen labels (e.g. France) flow through unchanged.
Missing values → `""` in normalization, and feature functions return **NaN**
whenever either side is empty (never two blanks = match).

## 5. Blocking
Six strategies unioned + deduped + capped (200/S1):
1. exact `core_name`  2. exact `name_sorted`  3. rare-token inverted index
4. numeric-address index  5. postal-code index  6. TF-IDF top-K (word 1–2 grams
on `name || addr`) via `linear_kernel`.
We log candidate recall on the validation split.

## 6. Feature Engineering
~35 features: name (exact, core-exact, sorted-exact, Jaccard, overlap, common
tokens, char-ngram Jaccard, Levenshtein, token-set ratio, TF-IDF cosine,
length diff), address (same family + numeric Jaccard, numeric conflict, postal
equality/conflict), country (equal, missing one/both, mismatch), missingness
flags for name / address, pair-source flag, and conflict interactions
(`name_high & address_bad`, `address_high & name_bad`, `country_mismatch & name_high`).
TF-IDF vectorizers are **fit only on the model-training corpus** to avoid leakage.

## 7. Training
Positive = candidate pairs present in ground truth. Negatives = candidate pairs
not in ground truth **plus** one random negative per S1. Class imbalance handled
with per-sample weights. Default classifier: `HistGradientBoostingClassifier`
(scikit-learn, BSD-3). Alternatives via `BER_MODEL={hgb,rf,logreg}`.

## 8. Validation
Entity-level split on Source-1 (80/20) — no S1 appears in both sides. Candidate
generation is re-built for validation entities. Macro-F0.5 is computed with
`evaluation.macro_f05`, mirroring the official formula.

## 9. Decision Layer
Per-S1, we keep every candidate whose probability ≥ threshold. Empty list is a
valid decision (singleton). We do **not** force top-1. The threshold is chosen on
the validation split by **maximizing macro-F0.5** (never F1).

## 10. Evaluation
Macro F0.5 = mean over all S1 of per-entity F0.5, with singleton handling:
empty ∩ empty → 1.0, empty ∩ non-empty → 0.0.

## 11. Output Format
`output/matching_results.tsv`:Both TSV, ID lists comma-separated, one row per test S1, empty list allowed,
no duplicates, matches ⊆ candidates. `src/submission.py` re-checks these rules
after writing.

## 12. How to run
```bash
pip install -r requirements.txt
python run.py --mode eda       # data stats
python run.py --mode validate  # 80/20 entity split, macro-F0.5
python run.py --mode all       # validate + retrain on full train + write outputs