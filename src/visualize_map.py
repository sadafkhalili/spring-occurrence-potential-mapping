from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

with rasterio.open(ROOT / "results/spring_potential.tif") as src:
    data = src.read(1, masked=True)
    bounds = src.bounds

values = np.ma.masked_where(
    (data < 0) | (data > 1),
    data
)

plt.figure(figsize=(10, 9))

plt.imshow(
    values,
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    extent=[
        bounds.left, bounds.right,
        bounds.bottom, bounds.top
    ],
    origin="upper"
)

plt.colorbar(label="Relative Spring Occurrence Score")

plt.title("Spring Occurrence Potential Map")
plt.xlabel("Easting (meters)")
plt.ylabel("Northing (meters)")
plt.tight_layout()

output = ROOT / "results/spring_potential_map.png"
plt.savefig(output, dpi=200)

print("Map saved to:", output)

plt.show()
