# Apply Model

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from pathlib import Path
import os
from datetime import datetime

from utils import get_combined_dataset, get_municipality_grids
from input_dataset import (
    create_input_dataset,
    create_windfield_dataset,
    create_rainfall_dataset,
)
from predict_damage import apply_model
```

## Manual proccess (rainfall data not contemplated)


```python
# Load grids by municipality
grid_mun = get_municipality_grids()[['id','NAME_2']]

# Load dataset
df_combined = get_combined_dataset()
df_combined = df_combined.rename({
    'mean_elev':'mean_altitude',
    'total_houses':'total_buildings'
    }, axis=1)
df_fji = df_combined[df_combined.country == 'fji']

# Typhoons
fji_typhoons = df_fji.typhoon_name.unique()

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

# Stratification
dmg = np.array(df_fji.percent_houses_damaged.to_list())
zero_dmg = np.round((np.count_nonzero(dmg == 0) / len(dmg)) , 2 )

# Define ranges for each group
x0 = list(np.linspace(0, zero_dmg, 1))   # zero damage
x1 = list(np.linspace(zero_dmg, 0.93, 2))  # almost no damage
x2 = list(np.linspace(0.935, 1, 5))  # all the damage
x3=x0+x1+x2

bins = []
for i in x3:
    bins.append(np.quantile(dmg, i))

# Histogram after stratification
samples_per_bin_fji, bins_def_fji = np.histogram(dmg, bins=bins)

# Define number of bins
num_bins_fji = len(samples_per_bin_fji)

# For future plots
str_bin_fji = []
for i in range(len(bins_def_fji[:-1])):
    a = str(np.round(bins_def_fji[i+1],3))
    b = str(np.round(bins_def_fji[i],3))
    str_bin_fji.append('{} - {}'.format(b,a))

# Rename
bins = bins_def_fji.copy()
```


```python
# Getting data from ECMWF
df_windfield = create_windfield_dataset()
```

```python
# Merging windspeed data with stationary data
df_input = create_input_dataset(df_windfield)
df_input
```



```python
# Group the DataFrame by the 'unique_id' column
grouped = df_input.groupby('unique_id')

# Create a list of Forecasts DataFrames, one for each unique_id
list_forecast = [group for name, group in grouped]
```


```python
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

    # Set damage predicted < 0 to 0
    df_test.loc[df_test['perc_dmg_pred'] < 0, 'perc_dmg_pred'] = 0

    # Agreggate by municipality
    dmg_by_mun = grid_mun.merge(df_test, left_on='id', right_on='grid_point_id')[
        ['NAME_2','perc_dmg_pred']
        ].groupby('NAME_2').mean().reset_index().rename({
            'NAME_2':'municipality'
        }, axis=1)

    list_df_out.append(dmg_by_mun)
```

## Automated proccess (includes rainfall data)


```python
# Getting data from ECMWF (takes 40s to run aprox)
df_windfield = create_windfield_dataset(thres=120)
```

```python
# Check if some forecasts take place on Fiji
df_windfield[df_windfield.in_fiji]
```
```python
# Get rainfall data (might take a couple of minutes- more like ~4 minutes)
df_rainfall = create_rainfall_dataset(df_windfield)
```

```python
# Load rainfall data (only if there are some wind forecasts on Fiji)
rain_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/03_rainfall/output"
)
today = datetime.now().strftime("%Y%m%d")
filename = rain_dir / "NOMADS"/ today /"rainfall_data_rw_mean.csv"
df_rainfall = pd.read_csv(filename)
```


```python
# Merging windspeed data with stationary data
df_input = create_input_dataset(df_windfield, df_rainfall)

# Group the DataFrame by the 'unique_id' column
grouped = df_input.groupby('unique_id')

# Create a list of Forecasts DataFrames, one for each unique_id
list_forecast = [group for name, group in grouped]
```


```python
# Apply model
list_df_out = apply_model(list_forecast)

# If you want to add more information to the output dataset (like the time period the windspeed was measured)
list_output = []
for i, df_out in enumerate(list_df_out):
    df_aux = df_out.copy()
    time_init, time_end = list_forecast[i][['time_init', 'time_end']].iloc[0]
    npoints = len(df_aux)

    df_aux['forecast_time_init'] = [time_init] * npoints
    df_aux['forecast_time_end'] = [time_end] * npoints
    list_output.append(df_aux)
```


```python
list_output[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>municipality</th>
      <th>perc_dmg_pred</th>
      <th>forecast_time_init</th>
      <th>forecast_time_end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ba</td>
      <td>0.014184</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bua</td>
      <td>-0.010197</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cakaudrove</td>
      <td>-0.010138</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kadavu</td>
      <td>0.029455</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lau</td>
      <td>-0.010547</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lomaiviti</td>
      <td>-0.006072</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Macuata</td>
      <td>-0.012121</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nadroga/Navosa</td>
      <td>0.104006</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Naitasiri</td>
      <td>0.012073</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Namosi</td>
      <td>0.011827</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ra</td>
      <td>0.010899</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rewa</td>
      <td>-0.003083</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Rotuma</td>
      <td>-0.012672</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Serua</td>
      <td>0.041130</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tailevu</td>
      <td>0.008783</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
    </tr>
  </tbody>
</table>
</div>
