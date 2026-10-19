from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Aspect",
    "Distance to fault",
    "Distance to river",
    "Elevation",
    "LS factor",
    "Lithology",
    "NDVI",
    "Plan curvature",
    "Profile curvature",
    "Rainfall",
    "SPI",
    "Slope",
    "Soil",
    "TWI",
    "Land cover",
]

SPRINGS_FILE = RAW_DIR / "Shapefile" / "Springer_points.shp"
STUDY_AREA_FILE = RAW_DIR / "Shapefile" / "Study_area.shp"

print("Data preparation script initialized.")
print(f"Environmental features: {len(FEATURES)}")
print(f"Spring points file exists: {SPRINGS_FILE.exists()}")
print(f"Study area file exists: {STUDY_AREA_FILE.exists()}")

from rasterio.warp import reproject, Resampling

def align_land_cover():
    source_path = RAW_DIR / "Land cover.tif"
    reference_path = RAW_DIR / "Elevation.tif"
    output_path = PROCESSED_DIR / "Land_cover_aligned.tif"

    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()

        with rasterio.open(source_path) as source:
            profile.update(
                dtype=source.dtypes[0],
                nodata=source.nodata,
                count=1
            )

            with rasterio.open(output_path, "w", **profile) as destination:
                reproject(
                    source=rasterio.band(source, 1),
                    destination=rasterio.band(destination, 1),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=reference.transform,
                    dst_crs=reference.crs,
                    dst_nodata=source.nodata,
                    resampling=Resampling.nearest
                )

    print("Land cover alignment completed.")


def extract_spring_features():
    springs = gpd.read_file(SPRINGS_FILE)
    coordinates = [(p.x, p.y) for p in springs.geometry]

    data = {
        "spring_id": np.arange(1, len(springs) + 1),
        "x": [p[0] for p in coordinates],
        "y": [p[1] for p in coordinates],
        "label": np.ones(len(springs), dtype=int),
    }

    for feature in FEATURES:
        if feature == "Land cover":
            raster_path = Path("data/processed/Land_cover_aligned.tif")
        else:
            raster_path = RAW_DIR / f"{feature}.tif"

        with rasterio.open(raster_path) as raster:
            values = list(raster.sample(coordinates, masked=True))
            data[feature] = [
                float(v[0]) if not np.ma.is_masked(v[0]) else np.nan
                for v in values
            ]

    df = pd.DataFrame(data)
    output_path = PROCESSED_DIR / "spring_features.csv"
    df.to_csv(output_path, index=False)

    print("Spring features extracted:", df.shape)
    print("Missing values:", df.isna().sum().sum())


from scipy.spatial import cKDTree

def calculate_distance_to_springs(x_coordinates, y_coordinates):
    springs = gpd.read_file(SPRINGS_FILE)
    spring_xy = np.column_stack([
        springs.geometry.x.to_numpy(),
        springs.geometry.y.to_numpy()
    ])

    tree = cKDTree(spring_xy)
    candidate_xy = np.column_stack([x_coordinates, y_coordinates])

    distances, _ = tree.query(candidate_xy)
    return distances


def sample_background_points(n_points=236, min_distance=500, seed=42):
    mask_path = PROCESSED_DIR / "valid_area_mask.tif"

    with rasterio.open(mask_path) as raster:
        mask = raster.read(1) == 1
        rows, cols = np.where(mask)

        xs, ys = rasterio.transform.xy(
            raster.transform, rows, cols, offset="center"
        )

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    distances = calculate_distance_to_springs(xs, ys)
    eligible = np.where(distances >= min_distance)[0]

    if len(eligible) < n_points:
        raise ValueError("Not enough eligible background pixels.")

    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=n_points, replace=False)

    background = pd.DataFrame({
        "x": xs[selected],
        "y": ys[selected],
        "label": np.zeros(n_points, dtype=int)
    })

    print("Eligible pixels:", len(eligible))
    print("Selected background points:", len(background))
    print("Minimum distance to springs:", distances[selected].min())

    return background


def extract_background_features(background):
    coordinates = list(zip(background["x"], background["y"]))
    result = background.copy()

    for feature in FEATURES:
        if feature == "Land cover":
            raster_path = PROCESSED_DIR / "Land_cover_aligned.tif"
        else:
            raster_path = RAW_DIR / f"{feature}.tif"

        with rasterio.open(raster_path) as raster:
            values = list(raster.sample(coordinates, masked=True))
            result[feature] = [
                float(v[0]) if not np.ma.is_masked(v[0]) else np.nan
                for v in values
            ]

    print("Background features extracted:", result.shape)
    print("Missing values:", result.isna().sum().sum())

    return result


def combine_datasets(springs, background):
    springs = springs.rename(columns={"Land_cover_aligned": "Land cover"})
    columns = ["x", "y", "label"] + FEATURES

    combined = pd.concat(
        [
            springs[columns],
            background[columns]
        ],
        ignore_index=True
    )

    print("Combined dataset shape:", combined.shape)
    print("Spring samples:", (combined["label"] == 1).sum())
    print("Background samples:", (combined["label"] == 0).sum())
    print("Missing values:", combined.isna().sum().sum())

    return combined


def save_dataset(dataset, filename="groundwater_dataset.csv"):
    output_path = PROCESSED_DIR / filename

    if output_path.exists():
        print("File already exists. No changes made:", output_path)
        return

    dataset.to_csv(output_path, index=False)
    print("Dataset saved:", output_path)
    print("Dataset shape:", dataset.shape)


def main():
    print("Starting data preparation...")

    if not (PROCESSED_DIR / "Land_cover_aligned.tif").exists():
        align_land_cover()

    if not (PROCESSED_DIR / "spring_features.csv").exists():
        extract_spring_features()

    save_valid_area_mask()

    springs = pd.read_csv(PROCESSED_DIR / "spring_features.csv")

    background = sample_background_points()
    background = extract_background_features(background)

    save_dataset(background, "background_features.csv")

    dataset = combine_datasets(springs, background)
    save_dataset(dataset)

    print("Data preparation completed.")



from rasterio.features import geometry_mask

def create_study_area_mask():
    study_area = gpd.read_file(STUDY_AREA_FILE)

    with rasterio.open(RAW_DIR / "Elevation.tif") as raster:
        study_area = study_area.to_crs(raster.crs)

        mask = geometry_mask(
            study_area.geometry,
            out_shape=(raster.height, raster.width),
            transform=raster.transform,
            invert=True
        )

    print("Study area pixels:", mask.sum())

    return mask


def create_valid_area_mask():
    valid = create_study_area_mask()

    for feature in FEATURES:
        if feature == "Land cover":
            path = PROCESSED_DIR / "Land_cover_aligned.tif"
        else:
            path = RAW_DIR / f"{feature}.tif"

        with rasterio.open(path) as raster:
            data = raster.read(1, masked=True)
            valid &= ~np.ma.getmaskarray(data)
            valid &= np.isfinite(data.data)

    print("Valid pixels:", int(valid.sum()))
    return valid


def save_valid_area_mask():
    output = PROCESSED_DIR / "valid_area_mask.tif"

    if output.exists():
        print("Valid area mask already exists.")
        return

    valid = create_valid_area_mask()

    with rasterio.open(RAW_DIR / "Elevation.tif") as reference:
        profile = reference.profile.copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(valid.astype("uint8"), 1)

    print("Valid area mask saved:", output)


if __name__ == "__main__":
    main()
