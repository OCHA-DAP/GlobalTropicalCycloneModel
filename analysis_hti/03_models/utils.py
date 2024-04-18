# +
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor


RS_BASE = 12345


# -
def get_training_dataset_hti(balanced=False):
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/03_model_input_dataset"
    )
    if balanced:
        filename = input_dir / "new_model_training_dataset_hti_with_nodmg.csv"
    else:
        filename = input_dir / "new_model_training_dataset_hti.csv"
    return pd.read_csv(filename)

def get_training_dataset_fji():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_fji_interpolated_wind_new_bld_count_using_pop.csv"
    return pd.read_csv(filename)

def get_training_dataset_phl():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_phl/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_phl_new.csv"
    return pd.read_csv(filename)

def get_training_dataset_vnm():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_vnm/03_model_input_dataset"
    )
    filename = input_dir / "new_model_training_dataset_viet_complete_interpolated_wind.csv"
    return pd.read_csv(filename)

def get_stationary_data_hti():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/03_model_input_dataset"
    )
    filename = input_dir / "hti_stationary_data.csv"
    return pd.read_csv(filename)

def get_stationary_data_vnm():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_viet/03_model_input_dataset"
    )
    filename = input_dir / "viet_stationary_data_interpolated_wind.csv"
    return pd.read_csv(filename)

def get_combined_dataset(from_phl=False):
    # Load datasets
    df_hti = get_training_dataset_hti()
    df_fji = get_training_dataset_fji()
    df_phl = get_training_dataset_phl()
    df_vnm = get_training_dataset_vnm()

    # Standardized features names
    df_fji = df_fji.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses',
         'mean_altitude':'mean_elev'},
         axis=1)
    df_vnm = df_vnm.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses'},
         axis=1)
    df_phl = df_phl.rename(
        {'perc_dmg_grid':'percent_houses_damaged',
         'total_buildings':'total_houses'},
         axis=1)

    if from_phl:
        df_hti = df_hti.rename(
        {'perc_dmg_grid_from_phl':'percent_houses_damaged',
         'total_buildings':'total_houses'},
         axis=1)
    else:
        df_hti = df_hti.rename(
            {'perc_dmg_grid':'percent_houses_damaged',
            'total_buildings':'total_houses'},
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
        "IWI",
        "country",
        "percent_houses_damaged",
        "typhoon_name"]

    # New feature 'country'
    df_hti['country'] = 'hti'
    df_phl['country'] = 'phl'
    df_fji['country'] = 'fji'
    df_vnm['country'] = 'viet'

    # All together
    df_combined = pd.concat([df_hti[all_features], df_phl[all_features], df_fji[all_features], df_vnm[all_features]], axis=0)

    # Set any values of damage houses >100% to 100%
    df_combined.loc[df_combined["percent_houses_damaged"] > 100, "percent_houses_damaged"] = 100

    return df_combined

def get_municipality_grids():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features/02_housing_damage/input"
    )
    filename = input_dir / "grid_municipality_info.csv"
    return pd.read_csv(filename)

def xgb_model_combined_data_LOOCV(df_combined, df_hti, features, bins, hti_weight, fji_weight=1, phl_weight=1, viet_weight=1):
    # Dataframe foir testing: HAITI
    hti_aux = df_hti[['typhoon_name', 'typhoon_year']].drop_duplicates()

    # Bins
    num_bins = len(bins)

    # The model
    rmse_total = []
    rmse_bin = []
    avg_error_bin = []

    y_test_typhoon  = []
    y_pred_typhoon  = []

    for typhoon, year in zip(hti_aux['typhoon_name'], hti_aux['typhoon_year']):

        """ PART 1: Train/Test """

        # LOOCV
        df_test = df_hti[
            (df_hti["typhoon_name"] == typhoon) &
            (df_hti["typhoon_year"] == year)] # Test set: HTI
        df_train = df_combined[
            (df_combined["typhoon_name"] != typhoon)
            #& (df_combined["typhoon_year"] != year) #I'm loosing information here... but its what it is for now
            ] # Train set: everything

        # Class weight
        weights = np.select(
            [
                (df_train['country'] == 'phl'),
                (df_train['country'] == 'viet'),
                (df_train['country'] == 'fji'),
                (df_train['country'] == 'hti')
            ],
            [
                phl_weight,
                viet_weight,
                fji_weight,
                hti_weight
            ],
            default=1
        )

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

        # Fit the model
        eval_set = [(X_train, y_train)]
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False, sample_weight=weights) #xgb_model

        # make predictions on Fiji
        y_pred = xgb.predict(X_test)

        # Save y_test y_pred
        y_test_typhoon.append(y_test)
        y_pred_typhoon.append(y_pred)

        # Calculate root mean squared error in total
        mse_test = mean_squared_error(y_test, y_pred)
        rmse_test = np.sqrt(mse_test)
        rmse_total.append(rmse_test)

        # Per bin (Stratification)
        rmse_test_bin = []
        avg_error_bin = []
        for bin_num in range(num_bins)[1:]:
            if (len(y_test[bin_index_test == bin_num]) != 0 and len(y_pred[bin_index_test == bin_num]) != 0):
                # Estimation of RMSE for test data per each bin
                mse_test = mean_squared_error(y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num])
                rmse_test = np.sqrt(mse_test)
                rmse_test_bin.append(rmse_test)
                # Avg error
                mean_difference = np.mean(y_test[bin_index_test == bin_num] - y_pred[bin_index_test == bin_num])
                avg_error_bin.append(mean_difference)
            else:
                rmse_test_bin.append(np.nan)
                avg_error_bin.append(np.nan)

        rmse_bin.append(rmse_test_bin)
        avg_error_bin.append(avg_error_bin)

        # RMSE & Avg error per bin
        rmse_strat = []
        avg_error_strat = []
        for i in range(num_bins - 1):
            #RMSE
            test_rmse_bin = np.nanmean(np.array(rmse_bin)[:,i])
            rmse_strat.append(test_rmse_bin)
            # #AVG error
            # test_avg_bin = np.nanmean(np.array(avg_error_bin)[:,i])
            # avg_error_strat.append(test_avg_bin)

    return y_test_typhoon, y_pred_typhoon, rmse_strat, rmse_total

def xgb_model_pop_data_LOOCV(df_hti, features, bins):
    # Dataframe foir testing: HAITI
    hti_aux = df_hti[['typhoon_name', 'typhoon_year']].drop_duplicates()

    # Bins
    num_bins = len(bins)

    # The model
    rmse_total = []
    rmse_bin = []
    avg_error_bin = []

    y_test_typhoon  = []
    y_pred_typhoon  = []

    for typhoon, year in zip(hti_aux['typhoon_name'], hti_aux['typhoon_year']):

        """ PART 1: Train/Test """

        # LOOCV
        df_test = df_hti[
            (df_hti["typhoon_name"] == typhoon) &
            (df_hti["typhoon_year"] == year)] # Test set: HTI event
        df_train = df_hti[
            (df_hti["typhoon_name"] != typhoon) &
            (df_hti["typhoon_year"] != year)] # Train set: everything else


        # Split X and y from dataframe features
        X_test = df_test[features]
        X_train = df_train[features]

        y_train = df_train["perc_aff_pop_grid"]
        y_test = df_test["perc_aff_pop_grid"]

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

        # Fit the model
        eval_set = [(X_train, y_train)]
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False) #xgb_model

        # make predictions on Fiji
        y_pred = xgb.predict(X_test)

        # Save y_test y_pred
        y_test_typhoon.append(y_test)
        y_pred_typhoon.append(y_pred)

        # Calculate root mean squared error in total
        mse_test = mean_squared_error(y_test, y_pred)
        rmse_test = np.sqrt(mse_test)
        rmse_total.append(rmse_test)

        # Per bin (Stratification)
        rmse_test_bin = []
        avg_error_bin = []
        for bin_num in range(num_bins)[1:]:
            if (len(y_test[bin_index_test == bin_num]) != 0 and len(y_pred[bin_index_test == bin_num]) != 0):
                # Estimation of RMSE for test data per each bin
                mse_test = mean_squared_error(y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num])
                rmse_test = np.sqrt(mse_test)
                rmse_test_bin.append(rmse_test)
                # Avg error
                mean_difference = np.mean(y_test[bin_index_test == bin_num] - y_pred[bin_index_test == bin_num])
                avg_error_bin.append(mean_difference)
            else:
                rmse_test_bin.append(np.nan)
                avg_error_bin.append(np.nan)

        rmse_bin.append(rmse_test_bin)
        avg_error_bin.append(avg_error_bin)

        # RMSE & Avg error per bin
        rmse_strat = []
        avg_error_strat = []
        for i in range(num_bins - 1):
            #RMSE
            test_rmse_bin = np.nanmean(np.array(rmse_bin)[:,i])
            rmse_strat.append(test_rmse_bin)
            # #AVG error
            # test_avg_bin = np.nanmean(np.array(avg_error_bin)[:,i])
            # avg_error_strat.append(test_avg_bin)

    return y_test_typhoon, y_pred_typhoon, rmse_strat, rmse_total
