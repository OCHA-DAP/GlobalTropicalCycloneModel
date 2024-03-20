```python
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from utils import get_combined_dataset_interpolated_with_viet_new_bld_count_using_pop, get_training_dataset_phl
```

## Load data


```python
df_combined = get_combined_dataset_interpolated_with_viet_new_bld_count_using_pop()
df_phl = df_combined[df_combined.country=='phl']

df_phl_training = get_training_dataset_phl()
df_phl_training = df_phl_training.rename(
    {'perc_dmg_grid':'percent_houses_damaged',
        'total_buildings':'total_houses'},
        axis=1)
```


```python
# Load RWI index
base_dir_phl = Path(os.getenv("STORM_DATA_DIR")) / "analysis_phl/02_model_features/"
vul_dir_phl = base_dir_phl / "05_vulnerability/output/"
rwi_phl = pd.read_csv(vul_dir_phl / "phl_rwi_bygrid.csv")

# base_dir_viet = Path(os.getenv("STORM_DATA_DIR")) / "analysis_viet/02_new_model_input/"
# vul_dir_viet = base_dir_viet / "05_vulnerability/output/"
# rwi_viet = pd.read_csv(vul_dir_viet / "viet_rwi_bygrid.csv")
```

## Stratification


```python
plt.hist(df_phl.percent_houses_damaged)
plt.yscale('log')
plt.show()
```



![png](xgboost_model_files/xgboost_model_5_0.png)




```python
# Paper stratification: [0, 0.00009], (0.00009, 1], (1, 10], (10, 50], and (50,100]
bins_def = np.array([0, 0.00009, 1, 10, 50, 100])
num_bins = len(bins_def)-1

# For future plots
str_bin = []
for i in range(len(bins_def[:-1])):
    a = str(np.round(bins_def[i+1],6))
    b = str(np.round(bins_def[i],6))
    str_bin.append('{} - {}'.format(b,a))
print(str_bin)
```

    ['0.0 - 9e-05', '9e-05 - 1.0', '1.0 - 10.0', '10.0 - 50.0', '50.0 - 100.0']



```python
set(df_combined[df_combined.country=='viet'].typhoon_name.unique()) & set(df_phl.typhoon_name.unique())
```




    {'CONSON',
     'DURIAN',
     'HAIMA',
     'KALMAEGI',
     'KAMMURI',
     'KETSANA',
     'MANGKHUT',
     'NARI',
     'NESAT',
     'RAMMASUN'}



## Define model


```python
import shap
def xgb_combined_model_LOOCV(df_combined, df_phl, features, bins, fji_weight=1, phl_weight=1, viet_weight=1, combined=False, fi=False):
    # Dataframe Fiji
    typhoons = df_phl.typhoon_name.unique()

    # Bins
    num_bins = len(bins)

    # The model
    rmse_total = []
    rmse_bin = []
    avg_error_bin = []

    y_test_typhoon  = []
    y_pred_typhoon  = []
    shap_values_list = []

    for typhoon in typhoons:

        """ PART 1: Train/Test """

        # LOOCV
        df_test = df_phl[df_phl["typhoon_name"] == typhoon] # Test set: Fiji
        if combined == True:
            df_train = df_combined[~((df_combined["typhoon_name"] == typhoon) & (df_combined["country"] == 'phl'))] # Train set: everything
            # Class weight
            weights = np.select(
                [
                    (df_train['country'] == 'phl'),
                    (df_train['country'] == 'viet'),
                    (df_train['country'] == 'fji')
                ],
                [
                    phl_weight,
                    viet_weight,
                    fji_weight
                ],
                default=1
            )
        else:
            df_train = df_phl[df_phl["typhoon_name"] != typhoon] # Train set: Just phl typhoons


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
        if combined:
            xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False, sample_weight=weights) #xgb_model
        else:
            xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False) #xgb_model

        # Shap values
        if fi:
            # Initialize an explainer for the typhoon
            explainer = shap.Explainer(xgb, X_train)
            shap_values = explainer(X_test)
            shap_values_list.append(shap_values)

        # make predictions
        y_pred_fji = xgb.predict(X_test)

        # Save y_test y_pred
        y_test_typhoon.append(y_test)
        y_pred_typhoon.append(y_pred_fji)

        # Calculate root mean squared error in total
        mse_test = mean_squared_error(y_test, y_pred_fji)
        rmse_test = np.sqrt(mse_test)
        rmse_total.append(rmse_test)

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

        rmse_bin.append(rmse_test_bin)
        avg_error_bin.append(avg_error_bin)

    # RMSE & Avg error per bin
    rmse_strat = []
    for i in range(num_bins - 1):
        #RMSE
        test_rmse_bin = np.nanmean(np.array(rmse_bin)[:,i])
        rmse_strat.append(test_rmse_bin)

    if fi:
        return rmse_strat, shap_values_list
    else:
        return rmse_strat
```

    IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html


## Run model


```python
# Features
features = [
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
]

features_rwi = features + ["rwi"]
features_scaled_rwi = features + ["scaled_distance"]
features_noiwi = [feature for feature in features if feature != 'IWI']
```


```python
rmse_strat_combined= xgb_combined_model_LOOCV(
    df_combined=df_combined,
    df_phl=df_phl,
    features=features,
    bins=bins_def,
    fji_weight=1,
    phl_weight=3,
    viet_weight=1,
    combined=True,
    fi=False)
```


```python
rmse_strat, shap_list = xgb_combined_model_LOOCV(
    df_combined=df_combined,
    df_phl=df_phl,
    features=features,
    bins=bins_def,
    fji_weight=1,
    phl_weight=1,
    viet_weight=1,
    combined=False,
    fi=True)
```


```python
# Add rwi index (scaled distance)
rmse_strat_rwi_scaled, shap_list_rwi_scaled = xgb_combined_model_LOOCV(
    df_combined=df_combined,
    df_phl=df_phl_training,
    features=features_scaled_rwi,
    bins=bins_def,
    fji_weight=1,
    phl_weight=3,
    viet_weight=1,
    combined=False,
    fi=True)
```


```python
# Add rwi index (just RWI)
rmse_strat_rwi, shap_list_rwi = xgb_combined_model_LOOCV(
    df_combined=df_combined,
    df_phl=df_phl_training,
    features=features_rwi,
    bins=bins_def,
    fji_weight=1,
    phl_weight=3,
    viet_weight=1,
    combined=False,
    fi=True)
```


```python
# Romove IWI
rmse_strat_noiwi, shap_list_noiwi = xgb_combined_model_LOOCV(
    df_combined=df_combined,
    df_phl=df_phl_training,
    features=features_noiwi,
    bins=bins_def,
    fji_weight=1,
    phl_weight=3,
    viet_weight=1,
    combined=False,
    fi=True)
```

## Model comparison


```python
fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.plot(range(num_bins)[1:], rmse_strat[1:], 'ro', alpha=0.5, label='Trained on PHL typhoons')
#ax.plot(range(num_bins)[1:], rmse_strat_noiwi[1:], 'yo', alpha=0.5, label='Trained on PHL typhoons (no IWI)')
#ax.plot(range(num_bins)[1:], rmse_strat_rwi[1:], 'go', alpha=0.5, label='Trained on PHL typhoons + RWI')
ax.plot(range(num_bins)[1:], rmse_strat_rwi_scaled[1:], 'ko', alpha=0.5, label='Trained on PHL typhoons + Scaled Distance')
ax.plot(range(num_bins)[1:], rmse_strat_combined[1:], 'bo', alpha=0.5, label='Trained on PHL+FJI+VIET typhoons')
ax.set_yscale('log')

ax.set_xticks(range(num_bins)[1:], str_bin[1:], rotation=45)
ax.set_xlabel('Damage [%]')
ax.set_ylabel('RMSE')
ax.grid()
ax.set_title('XGBoost Regression model \n using LOOCV for Philippines typhoons')
ax.legend()

plt.tight_layout()
plt.show()
```



![png](xgboost_model_files/xgboost_model_18_0.png)



## Feature importance


```python
# Compute mean SHAP values
#OBS: [shap_old[0].values[:,i].mean() for i in range(len(features))] is just basically shap_old[0].mean(0)
mean_shap_values = shap_list_rwi[0].mean(0)
for shap_values in shap_list_rwi[1:]:
    mean_shap_values += shap_values.mean(0)
mean_shap_values /= len(shap_list_rwi)

fig, ax = plt.subplots(1,1)
# Extract feature names and SHAP values from the explanation object
shap_features = mean_shap_values.feature_names
shap_features_values = mean_shap_values.values

# Sort features based on their absolute mean SHAP values
sorted_indices = np.argsort(np.abs(shap_features_values))#[::-1]
sorted_features_names = [shap_features[i] for i in sorted_indices]
sorted_shap_values_abs = np.abs(shap_features_values[sorted_indices])
sorted_shap_values = shap_features_values[sorted_indices]

ax.barh(sorted_features_names, sorted_shap_values, color='skyblue')
ax.set_xlabel('Mean SHAP Value')
ax.set_title('SHAP Values (with RWI)')
#ax.set_xscale('log')

plt.show()
```



![png](xgboost_model_files/xgboost_model_20_0.png)




```python
# Compute mean SHAP values
#OBS: [shap_old[0].values[:,i].mean() for i in range(len(features))] is just basically shap_old[0].mean(0)
mean_shap_values = shap_list_rwi_scaled[0].mean(0)
for shap_values in shap_list_rwi_scaled[1:]:
    mean_shap_values += shap_values.mean(0)
mean_shap_values /= len(shap_list_rwi)

fig, ax = plt.subplots(1,1)
# Extract feature names and SHAP values from the explanation object
shap_features = mean_shap_values.feature_names
shap_features_values = mean_shap_values.values

# Sort features based on their absolute mean SHAP values
sorted_indices = np.argsort(np.abs(shap_features_values))#[::-1]
sorted_features_names = [shap_features[i] for i in sorted_indices]
sorted_shap_values_abs = np.abs(shap_features_values[sorted_indices])
sorted_shap_values = shap_features_values[sorted_indices]

ax.barh(sorted_features_names, sorted_shap_values_abs, color='skyblue')
ax.set_xlabel('Mean SHAP Value')
ax.set_title('SHAP Values (with Scaled RWI Distance)')
#ax.set_xscale('log')

plt.show()
```



![png](xgboost_model_files/xgboost_model_21_0.png)




```python
# Compute mean SHAP values
#OBS: [shap_old[0].values[:,i].mean() for i in range(len(features))] is just basically shap_old[0].mean(0)
mean_shap_values = shap_list[0].mean(0)
for shap_values in shap_list[1:]:
    mean_shap_values += shap_values.mean(0)
mean_shap_values /= len(shap_list_rwi)

fig, ax = plt.subplots(1,1)
# Extract feature names and SHAP values from the explanation object
shap_features = mean_shap_values.feature_names
shap_features_values = mean_shap_values.values

# Sort features based on their absolute mean SHAP values
sorted_indices = np.argsort(np.abs(shap_features_values))#[::-1]
sorted_features_names = [shap_features[i] for i in sorted_indices]
sorted_shap_values_abs = np.abs(shap_features_values[sorted_indices])
sorted_shap_values = shap_features_values[sorted_indices]

ax.barh(sorted_features_names, sorted_shap_values_abs, color='skyblue')
ax.set_xlabel('Mean SHAP Value')
ax.set_title('SHAP Values (Classic features)')
#ax.set_xscale('log')

plt.show()
```



![png](xgboost_model_files/xgboost_model_22_0.png)



Positive values means that
- The higher the feature, the more damage it predicts.

Negative values means that
- The higher the feature, the less damage it predicts.

This makes sense:
- High IWI means rich people --> Richer people equals less damage.
- High windspeed equals more damage.


```python
# Also How much important is wind speed than track_distance?
sorted_shap_values[-1] / sorted_shap_values[-2] #45% more important
```




    1.4521878455311747
