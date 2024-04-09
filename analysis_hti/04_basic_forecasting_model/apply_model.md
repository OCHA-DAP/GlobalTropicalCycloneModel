```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from pathlib import Path
import os
from datetime import datetime

from utils import get_training_dataset_hti, get_municipality_grids
from input_dataset import create_input_dataset, create_windfield_dataset, create_rainfall_dataset, trigger_hti
from predict_damage import apply_model
```

## Automated proccess (includes rainfall data)


```python
# Getting data from ECMWF (takes 40s to run aprox)
df_windfield = create_windfield_dataset(thres=120, deg=3)
```

    /Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/HAITI clean/04_basic_forecasting_model/input_dataset.py:34: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grids.geometry = grids.geometry.to_crs(grids.crs).centroid
    Download: 100%|██████████| 33/33 [00:03<00:00, 10.58 files/s]
    Processing: 100%|██████████| 33/33 [00:02<00:00, 12.66 files/s]

    2024-04-09 17:50:30,373 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.






```python
# Check if some forecasts take place on HTI
trigger_hti(df_windfield=df_windfield)
```




    False




```python
# Get rainfall data (might take a couple of minutes- more like ~4 minutes)
df_rainfall = create_rainfall_dataset(df_windfield)
```

    prcp_gb2 not found for today.
    prcp_gb2 not found for today.
    Downloading  https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20240409/06/prcp_bc_gb2/



```python
# Load rainfall data (only if there are some wind forecasts on Fiji)
rain_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_hti/02_model_features/03_rainfall/output"
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
      <td>-0.025132</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bua</td>
      <td>-0.298872</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cakaudrove</td>
      <td>-0.667101</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kadavu</td>
      <td>-0.094245</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lau</td>
      <td>-0.580376</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lomaiviti</td>
      <td>-0.118332</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Macuata</td>
      <td>-0.202598</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nadroga/Navosa</td>
      <td>-0.119503</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Naitasiri</td>
      <td>0.115723</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Namosi</td>
      <td>0.022670</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ra</td>
      <td>0.053872</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rewa</td>
      <td>-0.032840</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Rotuma</td>
      <td>-0.045703</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Serua</td>
      <td>-0.018696</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tailevu</td>
      <td>0.054701</td>
      <td>2023-12-07</td>
      <td>2023-12-09 12:00:00</td>
    </tr>
  </tbody>
</table>
</div>
