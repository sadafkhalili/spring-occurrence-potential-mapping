from pathlib import Path
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

springs = gpd.read_file(
    ROOT / "data/raw/Shapefile/Springer_points.shp"
)

with rasterio.open(ROOT / "results/spring_potential.tif") as src:
    data = src.read(1, masked=True)
    bounds = src.bounds
    springs = springs.to_crs(src.crs)

fig, ax = plt.subplots(figsize=(10, 9))

image = ax.imshow(
    data,
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    extent=[
        bounds.left, bounds.right,
        bounds.bottom, bounds.top
    ],
    origin="upper"
)

ax.scatter(
    springs.geometry.x,
    springs.geometry.y,
    s=12,
    c="black",
    marker="o",
    label="Recorded Springs",
    alpha=0.8
)

fig.colorbar(
    image,
    ax=ax,
    label="Relative Spring Occurrence Score"
)

ax.set_title("Spring Potential Map with Recorded Springs")
ax.set_xlabel("Easting (meters)")
ax.set_ylabel("Northing (meters)")
ax.legend()
ax.set_aspect("equal")

plt.tight_layout()

output = ROOT / "results/map_with_springs.png"
plt.savefig(output, dpi=200)

print("Number of springs:", len(springs))
print("Saved to:", output)

plt.show()
