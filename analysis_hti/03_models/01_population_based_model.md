```python
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import numpy as np
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
import geopandas as gpd
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
from xgboost import plot_importance
import shap
import ast

from utils import get_training_dataset_hti, get_municipality_grids
```

    IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html


## Loading data


```python
# For Checking (Number of buildings destroyed per mun)
input_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/02_housing_damage/input/"
)
# Grid dir
grid_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/02_housing_damage/output/"
)

actual_mun_dmg = pd.read_csv(input_dir / "impact_data_bld_hti.csv")

# Load Fiji Shapefile
shp = gpd.read_file(
    input_dir / "shapefile_hti_fixed.gpkg"
)
shp = shp.to_crs('EPSG:4326')

# Load grid
grid = gpd.read_file(grid_dir / "hti_0.1_degree_grid_land_overlap.gpkg")


# Load typhoon track
intersection = actual_mun_dmg[['typhoon_name', 'Year', 'sid']].drop_duplicates().reset_index(drop=True)
```


```python
# Features
features= [
    "wind_speed",
    "track_distance",
    "total_houses",
    "total_pop",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "coast_length",
    "with_coast",
    "mean_elev",
    "mean_slope",
    "IWI"
]
```


```python
# Load HTI + id
df_hti= get_training_dataset_hti()
df_hti = df_hti.rename({
    "mean_altitude": "mean_elev",
    "total_buildings": "total_houses",
    "perc_dmg_grid": "percent_houses_damaged"
}, axis=1)
```

## Stratification


```python
plt.hist(df_hti.perc_aff_pop_grid)
plt.yscale('log')
plt.xlabel('% of affected pop')
plt.title('HTI population affected by grid')
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_6_0.png)




```python
# Stratification
dmg = np.array(df_hti.perc_aff_pop_grid.to_list())
zero_dmg = np.round((np.count_nonzero(dmg == 0) / len(dmg)) , 2 )

# Define ranges for each group
x0 = list(np.linspace(0, zero_dmg, 1))   # zero damage
x1 = list(np.linspace(zero_dmg, 0.9, 2))  # almost no damage
x2 = list(np.linspace(0.935, 1, 5))  # all the damage
x3=x0+x1+x2

bins = []
for i in x3:
    bins.append(np.quantile(dmg, i))

# Histogram after stratification
samples_per_bin, bins_def = np.histogram(dmg, bins=bins)

# Define number of bins
num_bins = len(samples_per_bin)

# For future plots
str_bin = []
for i in range(len(bins_def[:-1])):
    a = str(np.round(bins_def[i+1],3))
    b = str(np.round(bins_def[i],3))
    str_bin.append('{} - {}'.format(b,a))
print(str_bin)
```

    ['0.0 - 0.0', '0.0 - 0.002', '0.002 - 0.007', '0.007 - 0.013', '0.013 - 0.04', '0.04 - 0.092', '0.092 - 29.698']


## Define the model


```python
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

```

## The model


```python
# Fji + Phl + Viet + Hti
y_test_typhoon_hti, y_pred_typhoon_hti, rmse_strat, rmse_total = xgb_model_pop_data_LOOCV(
    df_hti=df_hti,
    bins=bins_def,
    features=features
)
```

    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice
    Mean of empty slice


## Results per bin


```python
fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.plot(range(num_bins), rmse_strat, 'rs', alpha=0.5, label='XGBoost model')


ax.set_xticks(range(num_bins), str_bin, rotation=45)
ax.set_xlabel('Damage [%]')
ax.set_ylabel('RMSE')
ax.grid()
ax.set_title('XGBoost Regression model (just HTI TCs) \n using LOOCV')
ax.legend()

plt.tight_layout()
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_13_0.png)



## Results at municipality level


```python
df_hti.columns
```




    Index(['index', 'typhoon_name', 'typhoon_year', 'event_level', 'grid_point_id',
           'total_buildings_damaged', 'total_pop_affected', 'total_houses',
           'total_pop', 'percent_houses_damaged', 'perc_aff_pop_grid', 'track_id',
           'wind_speed', 'track_distance', 'rainfall_max_6h', 'rainfall_max_24h',
           'with_coast', 'coast_length', 'mean_elev', 'mean_slope', 'IWI'],
          dtype='object')




```python
# typhoons
hti_typhoons = df_hti.typhoon_name.unique()

# Calculate buildings destroyed by municipality and % of buildings destroyed by mun
mun_id = get_municipality_grids()[['id','ADM1_PCODE']].drop_duplicates()

def num_people_aff_mun(mun, typhoon, year, y_pred_typhoon, df_hti, real=False):
    k = hti_typhoons.tolist().index(typhoon)
    df_typhoon = df_hti[(df_hti.typhoon_name==typhoon) & (df_hti.typhoon_year==year)]
    # Add feature "predictive_damage"
    df_typhoon['predicted_damage'] = y_pred_typhoon[k]

    mun_ids = mun_id[mun_id.ADM1_PCODE == mun].id.to_list()
    cells_in_mun = df_typhoon[df_typhoon.typhoon_name == typhoon].set_index('grid_point_id').loc[mun_ids]

    if real:
        damage_grid = np.array(cells_in_mun.perc_aff_pop_grid.to_list()) # Real dmg
    else:
        damage_grid = np.array(cells_in_mun.predicted_damage.to_list()) # Dmg predicted by cell

    # Number of people
    N_people_grid = np.array(cells_in_mun.total_pop.to_list()) # People by cell
    N_people_mun = np.sum(N_people_grid) # Total bld in mun

    # Calculate % of people (and N of people) affected by mun
    N_people_aff_pred_mun = np.sum(damage_grid) * (N_people_mun / 100)
    perc_aff_mun = np.sum(damage_grid)

    return N_people_aff_pred_mun, perc_aff_mun

# def calculate_actual_perc_dmg(x, i, y_pred_typhoon, df_fji, typhoon, year):
#     try:
#         return num_bld_destroyed_mun(mun=x['ADM1_PCODE'], typhoon=typhoon, year=year, y_pred_typhoon=y_pred_typhoon, df_fji=df_fji, real=True)[i]
#     except:
#         return 0
# def calculate_pred_perc_dmg(x, i, y_pred_typhoon, df_fji, typhoon, year):
#     try:
#         return num_bld_destroyed_mun(mun=x['ADM1_PCODE'], typhoon=typhoon, year=year, y_pred_typhoon=y_pred_typhoon, df_fji=df_fji, real=False)[i]
#     except:
#         return 0
```


```python
# Good example
typhoon='JEANNE'
year=2004
mun = 'HT05'

num_people_aff_mun(mun=mun, typhoon=typhoon,
                      year=year, y_pred_typhoon=y_pred_typhoon_hti, df_hti=df_hti, real=False)
```




    (11814.64221745748, 0.6810203795321286)



### General results


```python
from sklearn.metrics import mean_squared_error

typhoons_aux = df_hti_complete[['typhoon_name', 'typhoon_year']].drop_duplicates()
rmse_tot = []
avg_error_tot = []
for typhoon, year in zip(typhoons_aux.typhoon_name, typhoons_aux.typhoon_year):
    # Classic approach PHL+FJI+VIET
    shp_old = shp.copy().drop_duplicates(subset='ADM1_PCODE')
    shp_old['actual_perc_dmg'] = shp_old.apply(calculate_actual_perc_dmg, df_fji=df_hti_complete,
                                                y_pred_typhoon=y_pred_typhoon_hti, i=1, typhoon=typhoon, year=year, axis=1)
    shp_old['pred_perc_dmg'] = shp_old.apply(calculate_pred_perc_dmg, df_fji=df_hti_complete,
                                            y_pred_typhoon=y_pred_typhoon_hti, i=1, typhoon=typhoon, year=year, axis=1)

    # Modify values
    shp_old['actual_perc_dmg'] = shp_old['actual_perc_dmg'].apply(lambda x: 0 if x < 0 else (100 if x > 100 else x))
    shp_old['pred_perc_dmg'] = shp_old['pred_perc_dmg'].apply(lambda x: 0 if x < 0 else (100 if x > 100 else x))
    # Now calculate prediciton error
    shp_old['prediction_error'] = shp_old['actual_perc_dmg'] - shp_old['pred_perc_dmg'] # in percentual points

    actual_dmg_old = shp_old['actual_perc_dmg']
    pred_dmg_old = shp_old['pred_perc_dmg']

    muns = shp_old['ADM1_PCODE']
    rmse = []
    avg_error = []
    for i in range(len(muns)):
        rmse.append(np.sqrt(mean_squared_error([actual_dmg_old[i]], [pred_dmg_old[i]])))
        avg_error.append(np.mean(actual_dmg_old[i] - pred_dmg_old[i]))

    rmse_tot.append(rmse)
    avg_error_tot.append(avg_error)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/pandas/core/indexes/base.py:3790, in Index.get_loc(self, key)
       3789 try:
    -> 3790     return self._engine.get_loc(casted_key)
       3791 except KeyError as err:


    File index.pyx:152, in pandas._libs.index.IndexEngine.get_loc()


    File index.pyx:181, in pandas._libs.index.IndexEngine.get_loc()


    File pandas/_libs/hashtable_class_helper.pxi:2606, in pandas._libs.hashtable.Int64HashTable.get_item()


    File pandas/_libs/hashtable_class_helper.pxi:2630, in pandas._libs.hashtable.Int64HashTable.get_item()


    KeyError: 6


    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    Cell In[110], line 27
         25 avg_error = []
         26 for i in range(len(muns)):
    ---> 27     rmse.append(np.sqrt(mean_squared_error([actual_dmg_old[i]], [pred_dmg_old[i]])))
         28     avg_error.append(np.mean(actual_dmg_old[i] - pred_dmg_old[i]))
         30 rmse_tot.append(rmse)


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/pandas/core/series.py:1040, in Series.__getitem__(self, key)
       1037     return self._values[key]
       1039 elif key_is_scalar:
    -> 1040     return self._get_value(key)
       1042 # Convert generator to list before going through hashable part
       1043 # (We will iterate through the generator there to check for slices)
       1044 if is_iterator(key):


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/pandas/core/series.py:1156, in Series._get_value(self, label, takeable)
       1153     return self._values[label]
       1155 # Similar to Index.get_value, but we do not fall back to positional
    -> 1156 loc = self.index.get_loc(label)
       1158 if is_integer(loc):
       1159     return self._values[loc]


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/pandas/core/indexes/base.py:3797, in Index.get_loc(self, key)
       3792     if isinstance(casted_key, slice) or (
       3793         isinstance(casted_key, abc.Iterable)
       3794         and any(isinstance(x, slice) for x in casted_key)
       3795     ):
       3796         raise InvalidIndexError(key)
    -> 3797     raise KeyError(key) from err
       3798 except TypeError:
       3799     # If we have a listlike key, _check_indexing_error will raise
       3800     #  InvalidIndexError. Otherwise we fall through and re-raise
       3801     #  the TypeError.
       3802     self._check_indexing_error(key)


    KeyError: 6



```python
# RMSE per mun
rmse_mun_fji = []
avg_error_mun_fji = []
for i in range(len(muns)):
    #RMSE
    test_rmse_mun = np.nanmean(np.array(rmse_tot)[:,i])
    rmse_mun_fji.append(test_rmse_mun)
    #AVG ERROR
    test_avg_error_mun = np.nanmean(np.array(avg_error_tot)[:,i])
    avg_error_mun_fji.append(test_avg_error_mun)
```


```python
fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].plot(rmse_mun_fji, 'o-', label='HTI weight: 4', alpha=0.8)
ax[0].set_xticks(range(len(muns)), muns, rotation='vertical')  # Set x-axis ticks and labels
ax[0].set_title('RMSE')
ax[0].set_ylabel('RMSE')
ax[0].legend()

ax[1].plot(avg_error_mun_fji, 'o-', label='HTI weight: 4', alpha=0.8)
ax[1].axhline(y=0, color='k', linestyle='--')
ax[1].set_xticks(range(len(muns)), muns, rotation='vertical')  # Set x-axis ticks and labels
ax[1].set_title('Average Error (Real - Predicted)')
ax[1].set_ylabel('Avg Error')
ax[1].legend()
plt.grid()

plt.suptitle('Tropical Cyclones, Haiti \nHTI+PHL+FJI+VNM model')
plt.tight_layout()
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_21_0.png)



### Specific cases


```python
# Good examples
i=5
df_hti_aux = df_hti[['typhoon_name', 'typhoon_year']].drop_duplicates()
df_hti_aux_subset = df_hti_aux.iloc[[2,12,13,14,19,21]]
typhoon = df_hti_aux_subset.iloc[i].typhoon_name
year = df_hti_aux_subset.iloc[i].typhoon_year
```


```python
shp_reduced = shp[['ADM1_PCODE', 'geometry']].copy()
# Apply the function for real percentage damage
result_actual = shp_reduced.apply(lambda row: num_people_aff_mun(mun=row['ADM1_PCODE'],
                                                            typhoon=typhoon,
                                                            year=year,
                                                            y_pred_typhoon=y_pred_typhoon_hti,
                                                            df_hti=df_hti,
                                                            real=True), axis=1)

# Apply the function for predicted percentage damage
result_pred = shp_reduced.apply(lambda row: num_people_aff_mun(mun=row['ADM1_PCODE'],
                                                          typhoon=typhoon,
                                                          year=year,
                                                          y_pred_typhoon=y_pred_typhoon_hti,
                                                          df_hti=df_hti,
                                                          real=False), axis=1)

# Unpack the resulting tuples into separate columns for actual and predicted percentage damage
shp_reduced['actual_N_bld_dest_mun'], shp_reduced['actual_perc_dmg'] = zip(*result_actual)
shp_reduced['pred_N_bld_dest_mun'], shp_reduced['pred_perc_dmg'] = zip(*result_pred)

# Modify values
shp_reduced['actual_perc_dmg'] = shp_reduced['actual_perc_dmg'].apply(lambda x: 0 if x < 0 else (100 if x > 100 else x))
shp_reduced['pred_perc_dmg'] = shp_reduced['pred_perc_dmg'].apply(lambda x: 0 if x < 0 else (100 if x > 100 else x))
# Now calculate prediciton error
shp_reduced['prediction_error'] = shp_reduced['actual_perc_dmg'] - shp_reduced['pred_perc_dmg'] # in percentual points

```


```python
# Load track
id = intersection[intersection['typhoon_name'] == typhoon].sid.to_list()
track = TCTracks.from_ibtracs_netcdf(storm_id=id)
tc_track = track.get_track()

points_ib = gpd.points_from_xy(tc_track.lon, tc_track.lat)
tc_track_line_ib = LineString(points_ib)

geometries_ib = gpd.GeoSeries([tc_track_line_ib])
line_gdf_ib = gpd.GeoDataFrame(geometry=geometries_ib)

# Plots
cmap='Reds'
cmap_blue = 'Blues'
cmap_red = 'Reds_r'
```

    2024-04-04 09:27:23,524 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.



```python
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax = ax.flatten()
# Check prediction_error column for values > 0 and < 0
positive_error = shp_reduced[shp_reduced['prediction_error'] >= 0]
negative_error = shp_reduced[shp_reduced['prediction_error'] < 0]


# Plotting the maps
fiji_plot_1 = shp_reduced.plot(column='actual_perc_dmg', cmap=cmap, linewidth=0.2, ax=ax[0], edgecolor='0.3', legend=True)
fiji_plot_2 = shp_reduced.plot(column='pred_perc_dmg', cmap=cmap, linewidth=0.2, ax=ax[1], edgecolor='0.3', legend=True)
fiji_plot_3_blue = positive_error.plot(column='prediction_error', cmap=cmap_blue, linewidth=0.2, ax=ax[2], edgecolor='0.3', vmin=0, legend=True)
fiji_plot_3_red = negative_error.plot(column='prediction_error', cmap=cmap_red, linewidth=0.2, ax=ax[2], edgecolor='0.3', vmax=0, legend=True)


line_gdf_ib.plot(ax=ax[0], color='k', linewidth=1, label='Typhoon track')  # Plot the LineString in black
line_gdf_ib.plot(ax=ax[1], color='k', linewidth=1, label='Typhoon track')  # Plot the LineString in black
line_gdf_ib.plot(ax=ax[2], color='k', linewidth=1, label='Typhoon track')  # Plot the LineString in black


# Create custom legends
blue_patch = mpatches.Patch(color='#6495ED', label='Underestimated damage')
red_patch = mpatches.Patch(color='#800000', label='Overestimated damage')
ax[2].legend(handles=[blue_patch, red_patch], loc='lower left', bbox_to_anchor=(-0.05, -0.2))

ax[0].set_title('Actual % of Affected population \nby department (ADM1)')
ax[1].set_title('Predicted % of Affected population \nby department (ADM1)')
ax[2].set_title('Prediction Error \n $actual_{dmg} - predicted_{dmg}$ \n(in percentage points)')

ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')


# All Fiji map
ax[0].set_xlim([-76, -70])
ax[0].set_ylim([17, 21.5])
ax[1].set_xlim([-76, -70])
ax[1].set_ylim([17, 21.5])
ax[2].set_xlim([-76, -70])
ax[2].set_ylim([17, 21.5])

plt.suptitle("Typhoon {}({})\nXGBoost model with LOOCV train-test split".format(typhoon, year))
plt.tight_layout()
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_26_0.png)




```python
actual_dmg = shp_reduced.drop_duplicates(subset='ADM1_PCODE')['actual_perc_dmg'].reset_index(drop=True)
pred_dmg = shp_reduced.drop_duplicates(subset='ADM1_PCODE')['pred_perc_dmg'].reset_index(drop=True)
error_baseline = actual_dmg - actual_dmg
error_old = actual_dmg - pred_dmg
muns = shp_reduced.drop_duplicates(subset='ADM1_PCODE')['ADM1_PCODE'].reset_index(drop=True)
```


```python
plt.plot(error_baseline, 'o-',label='Perfect model', alpha=1)
plt.plot(error_old, 'o-', label='XGBoost model', alpha=0.4)


plt.xticks(range(len(muns)), muns, rotation='vertical')  # Set x-axis ticks and labels
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.title('Tropical Cyclone: {} ({}) \n Actual % of affected pop - Predicted % of affected pop \nby department (ADM1)'.format(typhoon, year))
plt.ylabel('% Points')

plt.grid()
plt.legend(loc='upper left')
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_28_0.png)



## ADM0 aggregation


```python
# typhoons
hti_typhoons = df_hti.typhoon_name.unique()

# Calculate buildings destroyed by municipality and % of buildings destroyed by mun
mun_id_adm0 = get_municipality_grids()[['id','ADM1_PCODE']].drop_duplicates()
mun_id_adm0['ADM0_PCODE'] = 'HT'

def num_people_aff_adm0(typhoon, year, y_pred_typhoon, df_hti, real=False):
    k = hti_typhoons.tolist().index(typhoon)
    df_typhoon = df_hti[(df_hti.typhoon_name==typhoon) & (df_hti.typhoon_year==year)]
    # Add feature "predictive_damage"
    df_typhoon['predicted_damage'] = y_pred_typhoon[k]

    mun_ids = mun_id_adm0[mun_id_adm0.ADM0_PCODE == 'HT'].id.to_list() # All
    cells_in_mun = df_typhoon[df_typhoon.typhoon_name == typhoon].set_index('grid_point_id').loc[mun_ids]

    if real:
        damage_grid = np.array(cells_in_mun.perc_aff_pop_grid.to_list()) # Real dmg
    else:
        damage_grid = np.array(cells_in_mun.predicted_damage.to_list()) # Dmg predicted by cell

    # Number of people
    N_people_grid = np.array(cells_in_mun.total_pop.to_list()) # People by cell
    N_people_mun = np.sum(N_people_grid) # Total bld in mun

    # Calculate % of people (and N of people) affected by mun
    N_people_aff_pred_mun = np.sum(damage_grid) * (N_people_mun / 100)
    perc_aff_mun = np.sum(damage_grid)

    return N_people_aff_pred_mun, perc_aff_mun
```


```python
df_hti_aux = df_hti[['typhoon_name', 'typhoon_year']].drop_duplicates()
actual_damage_typhoons = []
pred_damage_typhoons = []
for typhoon, year in zip(df_hti_aux.typhoon_name, df_hti_aux.typhoon_year):
    actual_N_dmg, actual_dmg = num_people_aff_adm0(typhoon=typhoon,
                                    year=year,
                                    y_pred_typhoon=y_pred_typhoon_hti,
                                    df_hti=df_hti,
                                    real=True)
    pred_N_dmg, pred_dmg = num_people_aff_adm0(typhoon=typhoon,
                                    year=year,
                                    y_pred_typhoon=y_pred_typhoon_hti,
                                    df_hti=df_hti,
                                    real=False)

    actual_damage_typhoons.append(actual_dmg)
    pred_damage_typhoons.append(pred_dmg)
```


```python
typhoons= df_hti_aux.typhoon_name.to_list()
years = df_hti_aux.typhoon_year.to_list()
```


```python
plt.plot(actual_damage_typhoons, pred_damage_typhoons, 'o', label='Haiti events')
plt.xlabel('Actual % of affected pop')
plt.ylabel('Predicted % of affected pop')

# # Add text annotations for each point
# for i, (x, y, typhoon, year) in enumerate(zip(actual_damage_typhoons, pred_damage_typhoons, typhoons, years)):
#     plt.text(x, y, f'{typhoon}', fontsize=6, ha='left', va='center')

# Calculate the range for the 45-degree line
min_val = min(min(actual_damage_typhoons), min(pred_damage_typhoons))
max_val = max(max(actual_damage_typhoons), max(pred_damage_typhoons))
line_range = np.linspace(min_val, max_val, 100)

# Plot the 45-degree line within the range
plt.plot(line_range, line_range, 'r-', label='Perfect model')
plt.title('% of affected population at ADM0 level for every TC')
plt.legend()
plt.grid()
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_33_0.png)




```python
df_hti_aux_subset = df_hti_aux.iloc[[2,12,13,14,19,21]]
actual_damage_typhoons_subset = np.array(actual_damage_typhoons)[[2,12,13,14,19,21]]
pred_damage_typhoons_subset = np.array(pred_damage_typhoons)[[2,12,13,14,19,21]]
typhoons_subset = df_hti_aux_subset.typhoon_name.to_list()
years_subset = df_hti_aux_subset.typhoon_year.to_list()
```


```python
plt.plot(actual_damage_typhoons_subset, pred_damage_typhoons_subset, 'o')
plt.xlabel('Actual % of affected pop')
plt.ylabel('Predicted % of affected pop')
plt.title('% of affected population at ADM0 level for every TC')

# Add text annotations for each point
for i, (x, y, typhoon, year) in enumerate(zip(actual_damage_typhoons_subset, pred_damage_typhoons_subset, typhoons_subset, years_subset)):
    plt.text(x, y, f'{typhoon} ({year})', fontsize=6, ha='left', va='center')

# Calculate the range for the 45-degree line
min_val = min(min(actual_damage_typhoons_subset), min(pred_damage_typhoons_subset))
max_val = max(max(actual_damage_typhoons_subset), max(pred_damage_typhoons_subset))
line_range = np.linspace(min_val, max_val, 100)

# Plot the 45-degree line within the range
plt.plot(line_range, line_range, 'r-', label='Perfect model')

plt.legend()
plt.grid()
plt.show()
```



![png](01_population_based_model_files/01_population_based_model_35_0.png)
