from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

# فقط داده‌های آموزش؛ مجموعه آزمون نهایی دست‌نخورده می‌ماند
data = pd.read_csv(ROOT / "data/processed/train.csv")

exclude = [
    "x", "y", "label",
    "block_x", "block_y", "spatial_block"
]

features = [c for c in data.columns if c not in exclude]

X = data[features]
y = data["label"]
groups = data["spatial_block"]

cv = GroupKFold(n_splits=5)

results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    predictions = model.predict(X.iloc[val_idx])
    probabilities = model.predict_proba(X.iloc[val_idx])[:, 1]

    accuracy = accuracy_score(y.iloc[val_idx], predictions)

    if y.iloc[val_idx].nunique() == 2:
        auc = roc_auc_score(y.iloc[val_idx], probabilities)
    else:
        auc = float("nan")

    results.append({
        "fold": fold,
        "accuracy": accuracy,
        "roc_auc": auc
    })

results = pd.DataFrame(results)

print(results.to_string(index=False))
print("\nMean Accuracy:", round(results["accuracy"].mean(), 3))
print("Mean ROC-AUC:", round(results["roc_auc"].mean(), 3))

output = ROOT / "results/spatial_cv.csv"
results.to_csv(output, index=False)

print("Saved to:", output)
