from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent

raster_path = ROOT / "results/spring_potential_classes.tif"
springs_path = ROOT / "data/raw/Shapefile/Springer_points.shp"

colors = [
    "#c62828",
    "#f57c00",
    "#fdd835",
    "#8bc34a",
    "#1b7837"
]

labels = [
    "Very Low",
    "Low",
    "Moderate",
    "High",
    "Very High"
]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap.N)

with rasterio.open(raster_path) as src:
    data = src.read(1, masked=True)
    bounds = src.bounds
    springs = gpd.read_file(springs_path).to_crs(src.crs)

fig, ax = plt.subplots(figsize=(9, 10))

ax.imshow(
    data,
    cmap=cmap,
    norm=norm,
    extent=[
        bounds.left, bounds.right,
        bounds.bottom, bounds.top
    ],
    origin="upper",
    interpolation="nearest"
)

ax.scatter(
    springs.geometry.x,
    springs.geometry.y,
    s=8,
    c="black",
    alpha=0.75,
    label="Recorded Springs"
)

legend_items = [
    Patch(facecolor=color, label=label)
    for color, label in zip(colors, labels)
]

legend_items.append(
    plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=6,
        label="Recorded Springs"
    )
)

ax.legend(
    handles=legend_items,
    loc="upper right",
    fontsize=9
)

ax.set_title(
    "Relative Spring Occurrence Potential Map",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

plt.tight_layout()

output = ROOT / "results/final_spring_potential_map.png"
plt.savefig(output, dpi=300, bbox_inches="tight")

print("Final map saved to:", output)

plt.show()
