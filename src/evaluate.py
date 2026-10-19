from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models/random_forest_model.joblib")
test = pd.read_csv(ROOT / "data/processed/test.csv")

X_test = test[model.feature_names_in_]
y_test = test["label"]

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Test samples:", len(test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Precision:", round(precision_score(y_test, predictions), 3))
print("Recall:", round(recall_score(y_test, predictions), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
