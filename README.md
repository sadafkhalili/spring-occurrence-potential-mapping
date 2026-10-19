# Relative Spring Occurrence Potential Mapping Using Machine Learning

## Project Overview
This project uses Random Forest and GIS environmental data to map the relative potential of recorded spring occurrence.

## Dataset
- 236 recorded spring locations
- 236 sampled background locations
- 15 environmental features
- 30-meter raster resolution

## Machine Learning Model
Algorithm: Random Forest

Evaluation:
- Spatial Test Accuracy: 70.0%
- Spatial Test ROC-AUC: 0.793
- Mean Spatial Cross-Validation Accuracy: 76.4%
- Mean Spatial Cross-Validation ROC-AUC: 0.855

## Outputs
- Continuous spring occurrence score map (GeoTIFF)
- Five-class relative potential map (GeoTIFF)
- Final visualization (PNG)

## Installation
pip install -r requirements.txt

## Important Limitation
Background locations are not confirmed groundwater absence points.
Model scores represent relative spring occurrence potential,
not verified groundwater presence probabilities.

## Data Source
The environmental raster layers and recorded spring locations
were obtained from the public GitHub repository:

https://github.com/Geomatics-GIS/Groundwater-Potential-Maps

The original dataset is not redistributed in this repository.
Users should obtain the source data from the original repository.

## Reproducibility
The project includes Python scripts for data preparation,
model training, evaluation, and map generation.

The dataset contains 236 recorded springs and 236 sampled
background points. Background points are selected at least
500 meters away from recorded springs using random seed 42.

A spatial train/test split is used to reduce spatial leakage.

## How to Run

Run commands from the project root directory.
1. Install: pip install -r requirements.txt
2. Download source data into data/raw/
3. Prepare data: python src/prepare_data.py
4. Train model: python src/train.py
5. Evaluate: python src/evaluate.py
6. Spatial CV: python src/spatial_cv.py
7. Predict: python src/predict.py
8. Classify: python src/classify_map.py
9. Final map: python src/final_map.py
