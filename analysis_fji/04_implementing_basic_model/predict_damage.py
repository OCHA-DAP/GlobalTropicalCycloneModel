import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor

from utils import get_combined_dataset, get_municipality_grids
from input_dataset import create_input_dataset, create_windfield_dataset

def load_datasets():
    # Load grids by municipality
    grid_mun = get_municipality_grids()[['id','NAME_2']]

    # Load dataset
    df_combined = get_combined_dataset()
    df_combined = df_combined.rename({
        'mean_elev':'mean_altitude',
        'total_houses':'total_buildings'
        }, axis=1)
    df_fji = df_combined[df_combined.country == 'fji']

    # Features NO Rainfall
    features_drop = [
        "wind_speed",
        "track_distance",
        "total_buildings",
        #"rainfall_max_6h",
        #"rainfall_max_24h",
        "coast_length",
        "with_coast",
        "mean_altitude",
        "mean_slope",
    ]
    return grid_mun, df_combined, features_drop

def apply_model(list_forecast):
    # Load datasets from Fiji and Philippines
    grid_mun, df_combined, features_drop = load_datasets()

    #Fiji weight
    fji_weight = 2
    list_df_out = []
    for forecast in list_forecast:
        # Definfe train/test
        df_test = forecast
        df_train = df_combined.copy()

        # Class weight
        weights = np.where(df_train['country'] == 'phl', 1, fji_weight) # Let's give more weight to Fiji

        # Split X and y from dataframe features
        X_test = df_test[features_drop]
        X_train = df_train[features_drop]
        y_train = df_train["percent_houses_damaged"]

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

        # Fit it on the training set
        eval_set = [(X_train, y_train)]
        xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False, sample_weight=weights) #xgb_model

        # Make predictions on new data
        y_pred = xgb.predict(X_test)

        # Join with forecast
        df_test['perc_dmg_pred'] = y_pred

        # Set damage predicted < 0 to 0 -- Just if you want -- In some extreme cases (typhoons), the damage is always > 0
        #df_test.loc[df_test['perc_dmg_pred'] < 0, 'perc_dmg_pred'] = 0

        # Agreggate by municipality
        dmg_by_mun = grid_mun.merge(df_test, left_on='id', right_on='grid_point_id')[
            ['NAME_2','perc_dmg_pred']
            ].groupby('NAME_2').mean().reset_index().rename({
                'NAME_2':'municipality'
            }, axis=1)

        list_df_out.append(dmg_by_mun)
    return list_df_out
