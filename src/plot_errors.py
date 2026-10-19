from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

data = pd.read_csv(ROOT / "results/prediction_errors.csv")

missed = data[data["label"] == 1]
background_errors = data[data["label"] == 0]

plt.figure(figsize=(9, 7))

plt.scatter(
    missed["x"], missed["y"],
    color="red", s=45,
    label="Missed Springs"
)

plt.scatter(
    background_errors["x"], background_errors["y"],
    color="blue", s=45,
    label="Background Predicted as Spring"
)

plt.xlabel("Easting (meters)")
plt.ylabel("Northing (meters)")
plt.title("Spatial Distribution of Prediction Errors")
plt.legend()
plt.axis("equal")
plt.tight_layout()

output = ROOT / "results/prediction_errors_map.png"
plt.savefig(output, dpi=200)

print("Missed springs:", len(missed))
print("Background errors:", len(background_errors))
print("Saved to:", output)

plt.show()
