
# Which customers of this telecom are likely to churn, and which usage and billing patterns signal it? 
# The target is Churn (1 = churned), which is 15.7% of the 3,150 customers. A useful model catches most 
# churners (recall) without flagging too many loyal customers (precision).

# Result (5-fold cross-validation, mean ± standard deviation):
# All features: accuracy 0.955 ± 0.007, churner recall 0.81 ± 0.04
# Without Complains and Status: accuracy 0.938 ± 0.012, churner recall 0.74 ± 0.05
# Baseline (predict nobody churns): 0.843 accuracy

import pandas as pd
my_df = pd.read_csv("Customer_Churn.csv")
my_df.columns = my_df.columns.str.replace(r"\s+", " ", regex=True)

print(my_df.shape)
print(my_df.columns)
my_df["Churn"].mean()
my_df.isna().sum()
my_df.info()
pd.set_option("display.max_columns", None)
my_df.describe()

X = my_df.drop(["Churn"], axis=1)
y = my_df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify= y)

print(y_train.mean())
print(y_test.mean())

baseline_accuracy = 1 - y_test.mean()

# Churn is a yes/no outcome, so I used a random forest classifier and judged it on
# recall and precision for churners, not accuracy alone (baseline 0.843 by predicting nobody churns).

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

importances = pd.Series(model.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False))

# Complains                  0.187890
# Seconds of Use             0.126241
# Status                     0.121759
# Subscription Length        0.119365
# Frequency of use           0.109693
# Customer Value             0.077212
# Distinct Called Numbers    0.077142
# Call Failure               0.059473
# Frequency of SMS           0.051846
# Age Group                  0.028458
# Age                        0.026520
# Charge Amount              0.013186
# Tariff Plan                0.001215

X_train2 = X_train.drop(columns=["Complains", "Status"])
X_test2 = X_test.drop(columns=["Complains", "Status"])
model2 = RandomForestClassifier(random_state=42)
model2.fit(X_train2, y_train)
print(classification_report(y_test, model2.predict(X_test2)))

#              precision    recall  f1-score   support
#
#           0       0.96      0.98      0.97       531
#           1       0.87      0.77      0.82        99
#
#     accuracy                           0.95       630
#    macro avg       0.92      0.87      0.89       630
# weighted avg       0.94      0.95      0.94       630


from sklearn.model_selection import cross_validate

scores = cross_validate(model, X, y, cv=5, scoring=["accuracy", "recall"])
scores["test_accuracy"].mean(), scores["test_accuracy"].std()

print("with all features")
print("accuracy", scores["test_accuracy"].mean(), scores["test_accuracy"].std())
print("recall  ", scores["test_recall"].mean(), scores["test_recall"].std())

X2 = X.drop(columns=["Complains", "Status"])
scores2 = cross_validate(model2, X2, y, cv=5, scoring=["accuracy", "recall"])
print("without Complains and Status")
print("accuracy", scores2["test_accuracy"].mean(), scores2["test_accuracy"].std())
print("recall  ", scores2["test_recall"].mean(), scores2["test_recall"].std())