# +
import os
from pathlib import Path

import pandas as pd

RS_BASE = 12345


# -


def get_training_dataset():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/03_new_model_training"
    )
    filename = input_dir / "new_model_training_dataset_fji.csv"
    return pd.read_csv(filename)


def get_training_dataset_complete():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/03_new_model_training"
    )
    filename = input_dir / "new_model_training_dataset_fji_complete.csv"
    return pd.read_csv(filename)


# not done yet
def weight_file(x):
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/02_new_model_input/02_housing_damage/input/Google Footprint Data"
    )
    weight_filename = input_dir / "ggl_grid_to_mun_weights.csv"
    return pd.read_csv(weight_filename)
