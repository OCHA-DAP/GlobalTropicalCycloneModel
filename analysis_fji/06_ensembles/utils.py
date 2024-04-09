import os
from pathlib import Path
import pandas as pd
import numpy as np

# Load Fiji + Synthetic tracks
def get_training_dataset_fji():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_fji_with_synthetic_ensambles.csv"
    return pd.read_csv(filename)

# Load PHL
def get_training_dataset_phl():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_phl/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_phl_new.csv"
    return pd.read_csv(filename)

# Load VNM
def get_training_dataset_viet():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_vnm/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_viet_complete_interpolated_wind.csv"
    return pd.read_csv(filename)

# Load grids by mun
def get_municipality_grids():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/02_model_features/02_housing_damage/input"
    )
    filename = input_dir / "grid_municipality_info.csv"
    return pd.read_csv(filename)

# Combined dataset
def get_combined_dataset(rwi_features=False):
    # Load datasets
    df_fji = get_training_dataset_fji()
    df_phl = get_training_dataset_phl()
    df_viet = get_training_dataset_viet()
    # Standardized features names
    df_fji = df_fji.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses',
         'mean_altitude':'mean_elev'},
         axis=1)
    df_viet = df_viet.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses'},
         axis=1)
    df_phl = df_phl.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses'},
         axis=1)

    all_features = [
        "grid_point_id",
        "wind_speed",
        "track_distance",
        "total_houses",
        "rainfall_max_6h",
        "rainfall_max_24h",
        "coast_length",
        "with_coast",
        "mean_elev",
        "mean_slope",
        #"mean_rug",
        "IWI",
        #"rwi",
        #"scaled_distance",
        "light_index",
        "country",
        "percent_houses_damaged",
        "typhoon_name"]
    if rwi_features:
        all_features = all_features + ['scaled_distance', 'rwi']
    # New feature 'country'
    df_phl['country'] = 'phl'
    df_fji['country'] = 'fji'
    df_viet['country'] = 'viet'

    # All together
    df_combined = pd.concat([df_phl[all_features], df_fji[all_features], df_viet[all_features]], axis=0)

    # Set any values of damage houses >100% to 100%
    df_combined.loc[df_combined["percent_houses_damaged"] > 100, "percent_houses_damaged"] = 100

    return df_combined
