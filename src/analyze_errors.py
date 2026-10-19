from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models/random_forest_model.joblib")
test = pd.read_csv(ROOT / "data/processed/test.csv")

X_test = test[model.feature_names_in_]

test["predicted"] = model.predict(X_test)
test["spring_probability"] = model.predict_proba(X_test)[:, 1]

errors = test[test["label"] != test["predicted"]].copy()

output = ROOT / "results/prediction_errors.csv"
errors.to_csv(output, index=False)

print("Total test samples:", len(test))
print("Correct predictions:", len(test) - len(errors))
print("Wrong predictions:", len(errors))
print("Missed springs:", ((errors["label"] == 1)).sum())
print("Background predicted as springs:", ((errors["label"] == 0)).sum())
print("Saved to:", output)
