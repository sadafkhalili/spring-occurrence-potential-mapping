from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent

source = ROOT / "results/spring_potential.tif"
output = ROOT / "results/spring_potential_classes.tif"

with rasterio.open(source) as src:
    scores = src.read(1, masked=True)
    profile = src.profile.copy()

    classes = np.zeros(src.shape, dtype=np.uint8)

    valid = ~np.ma.getmaskarray(scores)
    values = scores.data[valid]

    classes[valid] = (
        np.digitize(values, [0.2, 0.4, 0.6, 0.8]) + 1
    ).astype(np.uint8)

    profile.update(
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(classes, 1)

print("Class 1: Very Low")
print("Class 2: Low")
print("Class 3: Moderate")
print("Class 4: High")
print("Class 5: Very High")
print("Saved to:", output)
