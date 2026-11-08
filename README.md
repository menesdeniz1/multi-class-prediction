# Multi-class prediction of cirrhosis outcomes

Course/practice project: predict the outcome of patients with cirrhosis as one of three classes — `C` (censored), `CL` (censored due to liver transplant) or `D` (death) — and output class probabilities in a Kaggle-style submission file (`Status_C`, `Status_CL`, `Status_D`).

## Data

`train.csv` (15,000 rows) and `test.csv` (10,000 rows) with clinical features such as `N_Days`, `Drug`, `Age`, `Sex`, `Bilirubin`, `Albumin`, `Copper`, `Prothrombin` and `Stage`; target column `Status`. `sample_submission.csv` shows the expected output format. The CSVs come from a public competition dataset — check its original terms before reusing them.

## Method

- **Preprocessing** (scikit-learn `ColumnTransformer`): numeric columns → median imputation + standard scaling; categorical columns → most-frequent imputation + one-hot encoding.
- **Model:** XGBoost (`multi:softprob`, 3 classes, learning rate 0.03, max depth 8, 4,687 trees, subsample 0.7).
- **Validation:** 5-fold `StratifiedKFold`, multi-class log loss averaged over folds.
- **Output:** `submission.csv` with per-class probabilities for the test set.

## Files

- `check_missing_values.py` — missing-value percentages and a heatmap.
- `data_project.py` — first version: the preprocessor is fitted on the full training set before cross-validation.
- `mezun_tayfa.py` — improved version: the preprocessor is fitted inside each fold (no validation leakage).

## Run

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
python mezun_tayfa.py
```

## Limitations

The test predictions come from the model of the last fold only (no fold averaging or refit on all data), and no leaderboard score is recorded in this repository.
