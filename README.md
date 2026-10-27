# Telecom Customer Churn Prediction

**Question:** Which customers of a telecom company are likely to churn, and which signals matter most?

**A default random forest classified about 95% of customers correctly in 5-fold cross-validation (baseline 84%) and caught about 81% of the customers who churned.

## Data

- UCI Machine Learning Repository, "Iranian Churn Dataset": 3,150 customers, 13 predictor columns plus a `Churn` label (1 = churned, 0 = stayed).
- About 15.7% of customers churned, so the classes are imbalanced. Always predicting "stayed" would score 84.3% accuracy on the test set, so accuracy alone is not the measure.
- Columns include call failures, complaints, subscription length, charge amount, seconds of use, frequency of use and SMS, distinct numbers called, age and age group, tariff plan, status and customer value.
- No missing values. Usage columns have long right tails (for example, SMS counts and seconds of use), which look like heavy users rather than errors, so no rows were removed.

## Method

1. Cleaned column names (several contained double spaces).
2. Checked missing values, column types and outliers.
3. Split the data 80/20 into train and test sets, stratified on `Churn` so both sets keep the same churn rate (630 test customers, 99 of them churners).
4. Trained a `RandomForestClassifier` with default settings (`random_state=42`).
5. Evaluated with 5-fold cross-validation (accuracy and churner recall, mean ± standard deviation) and on the held-out test set (precision, recall and F1 for each class), and compared against the “always predict stayed” baseline.
6. Checked feature importances, then retrained without `Complains` and `Status` to see whether the model depended on them.

## Results (single test split, 630 customers)

5-fold cross-validation (mean ± standard deviation):

| | All columns | Without `Complains` and `Status` |
|---|---|---|
| Accuracy | 0.955 ± 0.007 | 0.938 ± 0.012 |
| Churn recall | 0.81 ± 0.04 | 0.74 ± 0.05 |

| | Baseline (always "stayed") | All columns | Without `Complains` and `Status` |
|---|---|---|---|
| Accuracy | 0.843 | 0.96 | 0.95 |
| Churn precision | n/a | 0.89 | 0.87 |
| Churn recall | 0 | 0.86 | 0.77 |
| Churn F1 | n/a | 0.88 | 0.82 |

Top feature importances with all columns: `Complains` (0.19), `Seconds of Use` (0.13), `Status` (0.12), `Subscription Length` (0.12), `Frequency of use` (0.11). `Tariff Plan` and `Charge Amount` contributed almost nothing.

## What this suggests

- Importance is spread across complaints, usage and tenure rather than resting on one column.
- `Complains` and `Status` help the model catch more churners (recall 0.81 vs 0.74 in cross-validation), but usage and billing columns alone still identify about three in four.
- Complaints may be recorded shortly before a customer leaves, so they may be a late signal rather than an early warning. Feature importance shows what the model uses, not what causes churn.

## Limitations

- Single train/test split and default settings. The numbers would move a little with a different split, so treat them as approximate.
- No hyperparameter tuning.
- Accuracy is flattering when 84% of customers stay, so recall on churners is the number that matters for a retention campaign.
- The dataset is a public sample, so these results are not a claim about any real company.

## Next steps

- Compare against logistic regression as a simple baseline.
- Try a business framing: choose a probability cutoff that trades off missed churners against wasted retention offers.

## Run it

- Python with `pandas`, `numpy` and `scikit-learn`.
- Download the dataset from the UCI repository, set the file path at the top of `Iranian_Churn.py`, and run the script top to bottom.
