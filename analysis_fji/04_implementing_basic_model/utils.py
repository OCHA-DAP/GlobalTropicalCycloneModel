# +
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor

RS_BASE = 12345


# -

def get_training_dataset():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_new_model_training"
    )
    filename = input_dir / "new_model_training_dataset_fji.csv"
    return pd.read_csv(filename)

def get_training_dataset_complete():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_new_model_training"
    )
    filename = input_dir / "new_model_training_dataset_fji_complete.csv"
    return pd.read_csv(filename)

def get_training_dataset_phl():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis/03_new_model_training"
    )
    filename = input_dir / "new_model_training_dataset.csv"
    return pd.read_csv(filename)

def get_stationary_data_fiji():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_new_model_training"
    )
    filename = input_dir / "fiji_stationary_data.csv"
    return pd.read_csv(filename)


def get_combined_dataset():
    # Load both datasets
    df_fji = get_training_dataset_complete()
    df_phl = get_training_dataset_phl()
    # Standardized features names
    df_fji = df_fji.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses',
         'mean_altitude':'mean_elev'},
         axis=1)
    all_features = [
        "wind_speed",
        "track_distance",
        "total_houses",
        "rainfall_max_6h",
        "rainfall_max_24h",
        "coast_length",
        "with_coast",
        "mean_elev",
        "mean_slope",
        #"IWI",
        "country",
        "percent_houses_damaged",
        "typhoon_name"]
    # New feature 'country'
    df_phl['country'] = 'phl'
    df_fji['country'] = 'fji'

    # All together
    df_combined = pd.concat([df_phl[all_features], df_fji[all_features]], axis=0)

    # Set any values of damage houses >100% to 100%
    df_combined.loc[df_combined["percent_houses_damaged"] > 100, "percent_houses_damaged"] = 100

    return df_combined

def get_municipality_grids():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/02_new_model_input/02_housing_damage/input"
    )
    filename = input_dir / "grid_municipality_info.csv"
    return pd.read_csv(filename)

def xgb_model_combined_data_LOOCV(df_combined, df_fji, features, bins, fji_weight):
    # Dataframe Fiji

    fji_typhoons = df_fji.typhoon_name.unique()

    # Bins
    num_bins = len(bins)

    # The model
    rmse_total_fji = []
    rmse_bin_fji = []
    avg_error_bin_fji = []

    y_test_typhoon_fji  = []
    y_pred_typhoon_fji  = []

    for typhoon in fji_typhoons:

        """ PART 1: Train/Test """

        # LOOCV
        df_test = df_fji[df_fji["typhoon_name"] == typhoon] # Test set: Fiji
        df_train = df_combined[df_combined["typhoon_name"] != typhoon] # Train set: everything

        # Class weight
        weights = np.where(df_train['country'] == 'phl', 1, fji_weight) # Let's give more weight to Fiji

        # Split X and y from dataframe features
        X_test = df_test[features]
        X_train = df_train[features]

        y_train = df_train["percent_houses_damaged"]
        y_test = df_test["percent_houses_damaged"]

        # Stratify data
        bin_index_test = np.digitize(y_test, bins=bins[:-1])

        """ PART 2: XGB regressor """
        # create an XGBoost Regressor
        xgb = XGBRegressor(
            base_score=0.5,
            booster="gbtree",
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            colsample_bytree=0.8,
            gamma=3,
            eta=0.01,
            importance_type="gain",
            learning_rate=0.1,
            max_delta_step=0,
            max_depth=4,
            min_child_weight=1,
            missing=1,
            n_estimators=100,
            early_stopping_rounds=10,
            n_jobs=1,
            nthread=None,
            objective="reg:squarederror",
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=None,
            silent=None,
            subsample=0.8,
            verbosity=0,
            eval_metric=["rmse", "logloss"],
            random_state=0,
        )


        # fit it on the training set
        eval_set = [(X_train, y_train)]
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False, sample_weight=weights) #xgb_model

        # make predictions on Fiji
        y_pred_fji = xgb.predict(X_test)

        # Save y_test y_pred
        y_test_typhoon_fji.append(y_test)
        y_pred_typhoon_fji.append(y_pred_fji)

        # Calculate root mean squared error in total
        mse_test = mean_squared_error(y_test, y_pred_fji)
        rmse_test = np.sqrt(mse_test)
        rmse_total_fji.append(rmse_test)

        # Per bin (Stratification)
        rmse_test_bin = []
        avg_error_bin = []
        for bin_num in range(num_bins)[1:]:
            if (len(y_test[bin_index_test == bin_num]) != 0 and len(y_pred_fji[bin_index_test == bin_num]) != 0):
                # Estimation of RMSE for test data per each bin
                mse_test = mean_squared_error(y_test[bin_index_test == bin_num], y_pred_fji[bin_index_test == bin_num])
                rmse_test = np.sqrt(mse_test)
                rmse_test_bin.append(rmse_test)
                # Avg error
                mean_difference = np.mean(y_test[bin_index_test == bin_num] - y_pred_fji[bin_index_test == bin_num])
                avg_error_bin.append(mean_difference)
            else:
                rmse_test_bin.append(np.nan)
                avg_error_bin.append(np.nan)

        rmse_bin_fji.append(rmse_test_bin)
        avg_error_bin_fji.append(avg_error_bin)

    # RMSE & Avg error per bin
    rmse_strat_fji = []
    avg_error_strat_fji = []
    for i in range(num_bins - 1):
        #RMSE
        test_rmse_bin = np.nanmean(np.array(rmse_bin_fji)[:,i])
        rmse_strat_fji.append(test_rmse_bin)
        #AVG error
        test_avg_bin = np.nanmean(np.array(avg_error_bin_fji)[:,i])
        avg_error_strat_fji.append(test_avg_bin)

    return y_test_typhoon_fji, y_pred_typhoon_fji, rmse_strat_fji, avg_error_strat_fji
