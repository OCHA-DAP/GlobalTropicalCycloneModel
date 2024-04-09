# +
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor

RS_BASE = 12345


# -

def get_training_dataset_hti():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_hti.csv"
    return pd.read_csv(filename)


def get_stationary_data_hti():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/03_model_input_dataset"
    )
    filename = input_dir / "hti_stationary_data.csv"
    return pd.read_csv(filename)

def get_municipality_grids():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features/02_housing_damage/input"
    )
    filename = input_dir / "grid_municipality_info.csv"
    return pd.read_csv(filename)
