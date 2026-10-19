from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

test = pd.read_csv(ROOT / "data/processed/test.csv")
errors = pd.read_csv(ROOT / "results/prediction_errors.csv")

summary = test.groupby("spatial_block").agg(
    total=("label", "size"),
    springs=("label", "sum")
)

error_counts = errors.groupby("spatial_block").size()

summary["errors"] = error_counts.reindex(
    summary.index, fill_value=0
)

summary["error_rate"] = (
    summary["errors"] / summary["total"]
)

summary = summary.sort_values(
    ["errors", "error_rate"],
    ascending=False
)

output = ROOT / "results/block_errors.csv"
summary.to_csv(output)

print("Test blocks:", len(summary))
print("Blocks with errors:", (summary["errors"] > 0).sum())
print("\nTop 10 blocks with most errors:")
print(summary.head(10).to_string())
print("\nSaved to:", output)
