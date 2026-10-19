from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "data/processed/train.csv")

exclude = [
    "x", "y", "label",
    "block_x", "block_y", "spatial_block"
]

features = [c for c in train.columns if c not in exclude]

X_train = train[features]
y_train = train["label"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

output = ROOT / "models/random_forest_model.joblib"
joblib.dump(model, output)

print("Training samples:", len(X_train))
print("Number of features:", len(features))
print("Number of trees:", model.n_estimators)
print("Model trained successfully!")
print("Saved to:", output)
