from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models/random_forest_model.joblib")

importance = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
).sort_values(ascending=True)

plt.figure(figsize=(9, 7))
importance.plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

output = ROOT / "results/feature_importance.png"
plt.savefig(output, dpi=200)
plt.show()

print("Top 5 important features:")
print(importance.tail(5).sort_values(ascending=False))
print("Saved to:", output)
