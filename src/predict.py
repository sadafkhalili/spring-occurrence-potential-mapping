from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models/random_forest_model.joblib")

mask_path = ROOT / "data/processed/valid_area_mask.tif"
output = ROOT / "results/spring_potential.tif"

with rasterio.open(mask_path) as src:
    valid = src.read(1) == 1
    profile = src.profile.copy()
    height, width = src.shape

# خواندن نقشه‌ها به ترتیب ویژگی‌های مدل
layers = []

for feature in model.feature_names_in_:
    if feature == "Land cover":
        path = ROOT / "data/processed/Land_cover_aligned.tif"
    else:
        path = ROOT / "data/raw" / f"{feature}.tif"

    with rasterio.open(path) as src:
        layers.append(src.read(1))

# نقشه خروجی
prediction = np.full(
    (height, width), -9999, dtype=np.float32
)

# پردازش بخش‌بخش برای کاهش مصرف حافظه
rows, cols = np.where(valid)
batch_size = 50000

for start in range(0, len(rows), batch_size):
    end = start + batch_size

    r = rows[start:end]
    c = cols[start:end]

    X = pd.DataFrame({
        feature: layer[r, c]
        for feature, layer in zip(
            model.feature_names_in_, layers
        )
    })

    scores = model.predict_proba(X)[:, 1]
    prediction[r, c] = scores

    print(
        f"Processed {min(end, len(rows)):,} / {len(rows):,}"
    )

profile.update(
    dtype="float32",
    count=1,
    nodata=-9999,
    compress="lzw"
)

with rasterio.open(output, "w", **profile) as dst:
    dst.write(prediction, 1)

print("\nGIS prediction map created!")
print("Saved to:", output)
