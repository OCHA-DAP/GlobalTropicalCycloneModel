```python
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from shapely.geometry import LineString, Point
from climada.hazard import Centroids, TCTracks, TropCyclone, Hazard
import xarray as xr
import datetime
#from pykrige.ok import OrdinaryKriging
from utils import get_stationary_data_fiji
```


```python
output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_model_features/02_housing_damage/output/"
)

# Load grid and stationary data
df = get_stationary_data_fiji()
grids = gpd.read_file(output_dir / "fji_0.1_degree_grid_land_overlap_new.gpkg")
grids.geometry = grids.geometry.to_crs(grids.crs).centroid
df_stationary = df.merge(grids, right_on='id', left_on='grid_point_id').drop(['index', 'id'], axis=1)
```

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_58568/386341765.py:9: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grids.geometry = grids.geometry.to_crs(grids.crs).centroid


### Load forecasts


```python
# Folder path
folder_path = '/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/Forecasts_Fiji/forecasts/'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Get the full file paths
csv_file_paths = [os.path.join(folder_path, filename) for filename in csv_files]
```


```python
csv_files
```




    ['20121213T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201217T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121215T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201216T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121214T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201214T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121216T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200405T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200404T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201215T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121217T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200408T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200407T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121215T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20121215T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200407T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201211T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121214T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200406T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200406T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121214T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20121216T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200406T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201212T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121216T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200407T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121217T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201214T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201214T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201215T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201215T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201217T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121213T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200406T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121213T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201217T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201216T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200407T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201216T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121216T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201214T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200408T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200404T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200408T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200405T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200405T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121217T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201215T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200409T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200405T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121215T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201217T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201219T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121213T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20121214T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201216T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200408T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201218T000000Z_Official_Forecast_Track_2021_02F_YASA.csv']




```python
# Evan typhoon 1
df_forecast = pd.read_csv(csv_file_paths[1], header=6, engine='python').iloc[1:][
    ["Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']",'Latitude', 'Longitude', 'MeanWind', 'Pressure', 'PressureOCI', 'Category','RadiusOCI', 'RadiusMaxWinds']].rename(
        {"Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']":'forecast_time'},
        axis=1
    )

# Create a GeoDataFrame from the DataFrame with latitude and longitude
geometry = [Point(xy) for xy in zip(df_forecast['Longitude'], df_forecast['Latitude'])]
df_forecast = gpd.GeoDataFrame(df_forecast, geometry=geometry)

# Convert 'forecast_time' to datetime format
df_forecast['forecast_time'] = pd.to_datetime(df_forecast['forecast_time'])

# Sort the DataFrame by 'forecast_time' in ascending order
df_forecast = df_forecast.sort_values(by='forecast_time')

# Calculate the time step by subtracting consecutive 'forecast_time' values
df_forecast['time_step'] = df_forecast['forecast_time'].diff().dt.total_seconds() / 3600
df_forecast['time_step'].fillna(0, inplace=True)

# Atmospheric pressure ---THIS IS IMPORTANT!!!!---
df_forecast['AtmPressure'] = 1013 #millibars

"""
Wind moves from areas of high pressure to areas of low pressure.
The greater the difference in pressure between two points,
the stronger the force driving the wind.
We need to be precise when defining enviromental pressure.
Since here we dont have access to that, and, since enviromental pressure
is almost constant, I took the value of Wikipedia: https://en.wikipedia.org/wiki/Atmospheric_pressure
which is 1,013 hPa or 1013 mb.

UPDATE: We have a feature caled PressureOCI. Looking into detail, it's really similar to the values from the IbTracks datasets,
I assume this is the enviromental pressure.
"""

# Basin
df_forecast['basin'] = 'SP'

# geometry2 = [Point(xy) for xy in zip(df_forecast2['Longitude'], df_forecast2['Latitude'])]
# df_forecast2 = gpd.GeoDataFrame(df_forecast2, geometry=geometry2)

df_forecast.head()
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
      <th>forecast_time</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MeanWind</th>
      <th>Pressure</th>
      <th>PressureOCI</th>
      <th>Category</th>
      <th>RadiusOCI</th>
      <th>RadiusMaxWinds</th>
      <th>geometry</th>
      <th>time_step</th>
      <th>AtmPressure</th>
      <th>basin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-12-17 06:00:00+00:00</td>
      <td>-16.60000</td>
      <td>178.80000</td>
      <td>130.0</td>
      <td>907.0</td>
      <td>1004.0</td>
      <td>5.0</td>
      <td>240.0</td>
      <td>30.0</td>
      <td>POINT (178.800 -16.600)</td>
      <td>0.0</td>
      <td>1013</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-12-17 08:00:00+00:00</td>
      <td>-16.80238</td>
      <td>179.07420</td>
      <td>128.3</td>
      <td>909.3</td>
      <td>1004.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>POINT (179.074 -16.802)</td>
      <td>2.0</td>
      <td>1013</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-12-17 12:00:00+00:00</td>
      <td>-17.20714</td>
      <td>179.62262</td>
      <td>125.0</td>
      <td>914.0</td>
      <td>1004.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>POINT (179.623 -17.207)</td>
      <td>4.0</td>
      <td>1013</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-12-17 14:00:00+00:00</td>
      <td>-17.44048</td>
      <td>179.84406</td>
      <td>123.3</td>
      <td>916.0</td>
      <td>1004.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>POINT (179.844 -17.440)</td>
      <td>2.0</td>
      <td>1013</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-12-17 18:00:00+00:00</td>
      <td>-17.90714</td>
      <td>180.28690</td>
      <td>120.0</td>
      <td>920.0</td>
      <td>1004.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>POINT (180.287 -17.907)</td>
      <td>4.0</td>
      <td>1013</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(sorted(pd.read_csv(csv_file_paths[0], header=6, engine='python').iloc[1:].columns))
```

    ['Category', 'CurrentIntensity', 'CycloneStatus', 'DataSource', 'DataTNoDvorak', 'EyeRadius', 'FinalT', 'GaleRadius', 'HowEyeRadius', 'HowGaleRadius', 'HowGust', 'HowHurricaneRadius', 'HowLocation', 'HowMaxWindRadius', 'HowMaxWindSpeed', 'HowPressure', 'HowStormRadius', 'HurricaneRadius', 'Land/Water', 'Latitude', 'Longitude', 'MeanWind', 'ModelTNoDvorak', 'NEGaleRadius', 'NEHurricaneRadius', 'NEStormRadius', 'NEStrongGaleRadius', 'NWGaleRadius', 'NWHurricaneRadius', 'NWStormRadius', 'NWStrongGaleRadius', 'P/W_Method', 'P5Wind', 'PatternTNoDvorak', 'Pressure', 'PressureOCI', 'Radius1000hPa', 'RadiusMaxWinds', 'RadiusOCI', 'SEGaleRadius', 'SEHurricaneRadius', 'SEStormRadius', 'SEStrongGaleRadius', 'SWGaleRadius', 'SWHurricaneRadius', 'SWStormRadius', 'SWStrongGaleRadius', 'StormRadius', 'StrongGaleRadius', 'Symbol', "Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']", 'UncEyeRadius', 'UncMaxWindSpeed', 'UncPressure', 'Uncertainty', 'VerticalExtent', 'WindGust']


### Calculate complete windfield


```python
# The idea is to compute the windfield in every grid cell. Not just in the locations where we have points.
fig, ax = plt.subplots(1,1)
grids.plot(ax=ax)
df_forecast.plot(column='MeanWind', cmap='coolwarm', markersize=20, legend=True, ax=ax)

plt.show()
```



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_8_0.png)



Almost sure the units are Knots


```python
# From ibtracs
yasa_track = TCTracks.from_ibtracs_netcdf(storm_id='2020346S13168')
```

Let's load a custom track in CLIMADA


```python
# Define a custom id
custom_idno = 123
custom_sid = str(custom_idno)

# Create xarray
def adjust_tracks(forecast_df):
    track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', np.array(forecast_df.MeanWind.values, dtype='float32')), #0.514444 --> kn to m/s
            #'environmental_pressure': ('time', forecast_df.AtmPressure.values), # Atmospheric pressure
            'environmental_pressure': ('time', forecast_df.PressureOCI.values), # I assume its enviromental pressure
            'central_pressure': ('time',forecast_df.Pressure.values),
            'lat': ('time',forecast_df.Latitude.values),
            'lon': ('time', forecast_df.Longitude.values),
            'radius_max_wind': ('time', forecast_df.RadiusMaxWinds.values),
            'radius_oci': ('time',forecast_df.RadiusOCI.values), # Works even if there is a bunch of nans. Doesnt change the windspeed values
            'time_step': ('time', forecast_df.time_step),
            'basin': ('time', np.array(forecast_df.basin, dtype='<U2'))
        },
        coords={
            'time': forecast_df.forecast_time.values,
        },
        attrs={
            'max_sustained_wind_unit': 'kn',
            'central_pressure_unit': 'mb',
            'name': 'Custom',
            #'sid': yasa_track.data[0].sid,
            'sid' : custom_sid,
            'orig_event_flag': True,
            'data_provider': 'Custom',
            #'id_no': yasa_track.data[0].id_no,
            'id_no' : custom_idno,
            'category': int(max(forecast_df.Category.iloc)),
        }
    )
    track = track.set_coords(['lat', 'lon'])
    return track
```


```python
# Load custom track
tracks = TCTracks()
tracks.data = [adjust_tracks(df_forecast)]
```

Observations:

-  Max sustained wind isn't even important.
-  The most important features are CentralPressure and AtmosphericPressure.
-  Everything else is irrelevant and I can put whatever I want in any other feature BUT I have to respect the type of variable.


```python
tracks.data[0]
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:                 (time: 16)
Coordinates:
    lat                     (time) float64 -16.6 -16.8 -17.21 ... -23.46 -23.89
    lon                     (time) float64 178.8 179.1 179.6 ... 181.2 180.9
  * time                    (time) datetime64[ns] 2020-12-17T06:00:00 ... 202...
Data variables:
    max_sustained_wind      (time) float32 130.0 128.3 125.0 ... 85.0 80.0 70.0
    environmental_pressure  (time) float64 1.004e+03 1.004e+03 ... 1.004e+03
    central_pressure        (time) float64 907.0 909.3 914.0 ... 955.0 963.0
    radius_max_wind         (time) float64 30.0 30.0 30.0 ... 30.0 30.0 30.0
    radius_oci              (time) float64 240.0 nan nan nan ... nan nan nan nan
    time_step               (time) float64 0.0 2.0 4.0 2.0 ... 6.0 6.0 6.0 6.0
    basin                   (time) &lt;U2 &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27; ... &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27;
Attributes:
    max_sustained_wind_unit:  kn
    central_pressure_unit:    mb
    name:                     Custom
    sid:                      123
    orig_event_flag:          True
    data_provider:            Custom
    id_no:                    123
    category:                 5</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-16a1b973-fff7-4298-bec8-e9d0dc32c130' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-16a1b973-fff7-4298-bec8-e9d0dc32c130' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 16</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-06ec47a8-707e-49b1-92ae-672ef89345d2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-06ec47a8-707e-49b1-92ae-672ef89345d2' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-16.6 -16.8 ... -23.46 -23.89</div><input id='attrs-b8772644-78cd-4f0b-b48f-74dbac653ed6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b8772644-78cd-4f0b-b48f-74dbac653ed6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-49197d51-d0bd-4e3f-aae8-9d895ff9b0a8' class='xr-var-data-in' type='checkbox'><label for='data-49197d51-d0bd-4e3f-aae8-9d895ff9b0a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-16.6    , -16.80238, -17.20714, -17.44048, -17.90714, -18.14048,
       -18.60714, -19.325  , -20.04286, -20.68929, -21.33571, -21.89821,
       -22.46071, -22.95893, -23.45714, -23.89176])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>178.8 179.1 179.6 ... 181.2 180.9</div><input id='attrs-874e8fb3-b469-4c90-bbd6-f529dccf24b1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-874e8fb3-b469-4c90-bbd6-f529dccf24b1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c3747ecc-27ba-4cd1-88e7-b1577859698f' class='xr-var-data-in' type='checkbox'><label for='data-c3747ecc-27ba-4cd1-88e7-b1577859698f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([178.8    , 179.0742 , 179.62262, 179.84406, 180.2869 , 180.50833,
       180.95119, 181.3369 , 181.72262, 181.84404, 181.96548, 181.82976,
       181.69405, 181.42976, 181.16548, 180.85202])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2020-12-17T06:00:00 ... 2020-12-...</div><input id='attrs-9257209f-8fa4-4b96-9f1f-c19161533557' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9257209f-8fa4-4b96-9f1f-c19161533557' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-63531012-3fe2-4fce-9197-65723e43411a' class='xr-var-data-in' type='checkbox'><label for='data-63531012-3fe2-4fce-9197-65723e43411a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2020-12-17T06:00:00.000000000&#x27;, &#x27;2020-12-17T08:00:00.000000000&#x27;,
       &#x27;2020-12-17T12:00:00.000000000&#x27;, &#x27;2020-12-17T14:00:00.000000000&#x27;,
       &#x27;2020-12-17T18:00:00.000000000&#x27;, &#x27;2020-12-17T20:00:00.000000000&#x27;,
       &#x27;2020-12-18T00:00:00.000000000&#x27;, &#x27;2020-12-18T06:00:00.000000000&#x27;,
       &#x27;2020-12-18T12:00:00.000000000&#x27;, &#x27;2020-12-18T18:00:00.000000000&#x27;,
       &#x27;2020-12-19T00:00:00.000000000&#x27;, &#x27;2020-12-19T06:00:00.000000000&#x27;,
       &#x27;2020-12-19T12:00:00.000000000&#x27;, &#x27;2020-12-19T18:00:00.000000000&#x27;,
       &#x27;2020-12-20T00:00:00.000000000&#x27;, &#x27;2020-12-20T06:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d772683e-8756-4e04-b550-2867d579dab9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d772683e-8756-4e04-b550-2867d579dab9' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>max_sustained_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>130.0 128.3 125.0 ... 80.0 70.0</div><input id='attrs-4cf5c654-3d79-4808-af81-aa035fe2c3f3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4cf5c654-3d79-4808-af81-aa035fe2c3f3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-73c48c18-12b3-422d-bf6b-7f6c4c3fddb5' class='xr-var-data-in' type='checkbox'><label for='data-73c48c18-12b3-422d-bf6b-7f6c4c3fddb5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([130. , 128.3, 125. , 123.3, 120. , 118.3, 115. , 110. , 110. ,
       105. , 105. , 100. ,  95. ,  85. ,  80. ,  70. ], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>environmental_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.004e+03 1.004e+03 ... 1.004e+03</div><input id='attrs-d354fb7a-139e-43a1-81df-88a12d019b7f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d354fb7a-139e-43a1-81df-88a12d019b7f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9b85b63b-8867-4c9c-b7d2-e3986a4c4fb5' class='xr-var-data-in' type='checkbox'><label for='data-9b85b63b-8867-4c9c-b7d2-e3986a4c4fb5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1004., 1004., 1004., 1004., 1004., 1004., 1004., 1004., 1004.,
       1004., 1004., 1004., 1004., 1004., 1004., 1004.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>central_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>907.0 909.3 914.0 ... 955.0 963.0</div><input id='attrs-3fe3cd55-45eb-4253-bcda-86e5e17e4bd4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3fe3cd55-45eb-4253-bcda-86e5e17e4bd4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-40feaa4b-7b3f-4152-bb50-9d00f2f164a7' class='xr-var-data-in' type='checkbox'><label for='data-40feaa4b-7b3f-4152-bb50-9d00f2f164a7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([907. , 909.3, 914. , 916. , 920. , 921. , 923. , 928. , 927. ,
       932. , 932. , 936. , 941. , 951. , 955. , 963. ])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>radius_max_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>30.0 30.0 30.0 ... 30.0 30.0 30.0</div><input id='attrs-18ce5d84-4333-4262-8167-1a87002f808c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-18ce5d84-4333-4262-8167-1a87002f808c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dcd015c4-6606-45f9-ab71-efacec50fa30' class='xr-var-data-in' type='checkbox'><label for='data-dcd015c4-6606-45f9-ab71-efacec50fa30' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
       30., 30., 30.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>radius_oci</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>240.0 nan nan nan ... nan nan nan</div><input id='attrs-cfce0b57-569e-4ba2-94f1-ca5f526d0b4c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cfce0b57-569e-4ba2-94f1-ca5f526d0b4c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1f94c588-a287-45f2-8733-751b8dbac57a' class='xr-var-data-in' type='checkbox'><label for='data-1f94c588-a287-45f2-8733-751b8dbac57a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([240.,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
        nan,  nan,  nan,  nan,  nan])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>time_step</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 2.0 4.0 2.0 ... 6.0 6.0 6.0 6.0</div><input id='attrs-c02ae0b8-ce08-4228-8eea-a206dbabba2d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c02ae0b8-ce08-4228-8eea-a206dbabba2d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-533a2fcd-d3f1-4913-b4be-246125e674bf' class='xr-var-data-in' type='checkbox'><label for='data-533a2fcd-d3f1-4913-b4be-246125e674bf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0., 2., 4., 2., 4., 2., 4., 6., 6., 6., 6., 6., 6., 6., 6., 6.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>basin</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U2</div><div class='xr-var-preview xr-preview'>&#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27; ... &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27;</div><input id='attrs-55653d1b-0d7e-4375-9d02-4fdd25535199' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-55653d1b-0d7e-4375-9d02-4fdd25535199' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3ace591b-d7fb-47c8-b1af-2b3f6618bb75' class='xr-var-data-in' type='checkbox'><label for='data-3ace591b-d7fb-47c8-b1af-2b3f6618bb75' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;,
       &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;], dtype=&#x27;&lt;U2&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-54578f90-6a42-4c2f-be1f-ae013f5cfa77' class='xr-section-summary-in' type='checkbox'  ><label for='section-54578f90-6a42-4c2f-be1f-ae013f5cfa77' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-bc678a0f-9a86-444f-99a1-659d55edfb1a' class='xr-index-data-in' type='checkbox'/><label for='index-bc678a0f-9a86-444f-99a1-659d55edfb1a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2020-12-17 06:00:00&#x27;, &#x27;2020-12-17 08:00:00&#x27;,
               &#x27;2020-12-17 12:00:00&#x27;, &#x27;2020-12-17 14:00:00&#x27;,
               &#x27;2020-12-17 18:00:00&#x27;, &#x27;2020-12-17 20:00:00&#x27;,
               &#x27;2020-12-18 00:00:00&#x27;, &#x27;2020-12-18 06:00:00&#x27;,
               &#x27;2020-12-18 12:00:00&#x27;, &#x27;2020-12-18 18:00:00&#x27;,
               &#x27;2020-12-19 00:00:00&#x27;, &#x27;2020-12-19 06:00:00&#x27;,
               &#x27;2020-12-19 12:00:00&#x27;, &#x27;2020-12-19 18:00:00&#x27;,
               &#x27;2020-12-20 00:00:00&#x27;, &#x27;2020-12-20 06:00:00&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;time&#x27;, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d1879d96-3916-4a2b-84f5-b4d761f64b0f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d1879d96-3916-4a2b-84f5-b4d761f64b0f' class='xr-section-summary' >Attributes: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>max_sustained_wind_unit :</span></dt><dd>kn</dd><dt><span>central_pressure_unit :</span></dt><dd>mb</dd><dt><span>name :</span></dt><dd>Custom</dd><dt><span>sid :</span></dt><dd>123</dd><dt><span>orig_event_flag :</span></dt><dd>True</dd><dt><span>data_provider :</span></dt><dd>Custom</dd><dt><span>id_no :</span></dt><dd>123</dd><dt><span>category :</span></dt><dd>5</dd></dl></div></li></ul></div></div>




```python
yasa_track.data[0]
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:                 (time: 64)
Coordinates:
  * time                    (time) datetime64[ns] 2020-12-13 ... 2020-12-20T1...
    lat                     (time) float32 -15.2 -15.17 -15.2 ... -24.21 -24.6
    lon                     (time) float32 173.1 172.7 172.5 ... 181.2 181.3
Data variables:
    radius_max_wind         (time) float32 50.0 50.0 50.0 ... 50.0 50.0 50.0
    radius_oci              (time) float32 230.0 190.0 150.0 ... 200.0 200.0
    max_sustained_wind      (time) float32 34.09 36.36 39.77 ... 36.36 34.09
    central_pressure        (time) float32 997.0 996.0 995.0 ... 995.0 997.0
    environmental_pressure  (time) float64 1.005e+03 1.004e+03 ... 1.006e+03
    time_step               (time) float64 3.0 3.0 3.0 3.0 ... 3.0 3.0 3.0 3.0
    basin                   (time) &lt;U2 &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27; ... &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27;
Attributes:
    max_sustained_wind_unit:  kn
    central_pressure_unit:    mb
    orig_event_flag:          True
    data_provider:            ibtracs_mixed:lat(official_3h),lon(official_3h)...
    category:                 5
    name:                     YASA
    sid:                      2020346S13168
    id_no:                    2020346113168.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-33fd8a0e-6a02-4ba7-8146-bbb9ac1c48d0' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-33fd8a0e-6a02-4ba7-8146-bbb9ac1c48d0' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 64</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-f1908939-fb75-4426-886f-c49612478a68' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f1908939-fb75-4426-886f-c49612478a68' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2020-12-13 ... 2020-12-20T18:00:00</div><input id='attrs-a0befe50-5032-4d7f-bacf-9cdee8716716' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a0befe50-5032-4d7f-bacf-9cdee8716716' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a98c5f96-52ae-49d0-8705-f22817161312' class='xr-var-data-in' type='checkbox'><label for='data-a98c5f96-52ae-49d0-8705-f22817161312' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2020-12-13T00:00:00.000000000&#x27;, &#x27;2020-12-13T03:00:00.000000000&#x27;,
       &#x27;2020-12-13T06:00:00.000000000&#x27;, &#x27;2020-12-13T09:00:00.000000000&#x27;,
       &#x27;2020-12-13T12:00:00.000000000&#x27;, &#x27;2020-12-13T15:00:00.000000000&#x27;,
       &#x27;2020-12-13T18:00:00.000000000&#x27;, &#x27;2020-12-13T21:00:00.000000000&#x27;,
       &#x27;2020-12-14T00:00:00.000000000&#x27;, &#x27;2020-12-14T03:00:00.000000000&#x27;,
       &#x27;2020-12-14T06:00:00.000000000&#x27;, &#x27;2020-12-14T09:00:00.000000000&#x27;,
       &#x27;2020-12-14T12:00:00.000000000&#x27;, &#x27;2020-12-14T15:00:00.000000000&#x27;,
       &#x27;2020-12-14T18:00:00.000000000&#x27;, &#x27;2020-12-14T21:00:00.000000000&#x27;,
       &#x27;2020-12-15T00:00:00.000000000&#x27;, &#x27;2020-12-15T03:00:00.000000000&#x27;,
       &#x27;2020-12-15T06:00:00.000000000&#x27;, &#x27;2020-12-15T09:00:00.000000000&#x27;,
       &#x27;2020-12-15T12:00:00.000000000&#x27;, &#x27;2020-12-15T15:00:00.000000000&#x27;,
       &#x27;2020-12-15T18:00:00.000000000&#x27;, &#x27;2020-12-15T21:00:00.000000000&#x27;,
       &#x27;2020-12-16T00:00:00.000000000&#x27;, &#x27;2020-12-16T03:00:00.000000000&#x27;,
       &#x27;2020-12-16T06:00:00.000000000&#x27;, &#x27;2020-12-16T09:00:00.000000000&#x27;,
       &#x27;2020-12-16T12:00:00.000000000&#x27;, &#x27;2020-12-16T15:00:00.000000000&#x27;,
       &#x27;2020-12-16T18:00:00.000000000&#x27;, &#x27;2020-12-16T21:00:00.000000000&#x27;,
       &#x27;2020-12-17T00:00:00.000000000&#x27;, &#x27;2020-12-17T03:00:00.000000000&#x27;,
       &#x27;2020-12-17T06:00:00.000000000&#x27;, &#x27;2020-12-17T07:00:00.000000000&#x27;,
       &#x27;2020-12-17T09:00:00.000000000&#x27;, &#x27;2020-12-17T12:00:00.000000000&#x27;,
       &#x27;2020-12-17T15:00:00.000000000&#x27;, &#x27;2020-12-17T18:00:00.000000000&#x27;,
       &#x27;2020-12-17T21:00:00.000000000&#x27;, &#x27;2020-12-18T00:00:00.000000000&#x27;,
       &#x27;2020-12-18T03:00:00.000000000&#x27;, &#x27;2020-12-18T06:00:00.000000000&#x27;,
       &#x27;2020-12-18T09:00:00.000000000&#x27;, &#x27;2020-12-18T12:00:00.000000000&#x27;,
       &#x27;2020-12-18T15:00:00.000000000&#x27;, &#x27;2020-12-18T18:00:00.000000000&#x27;,
       &#x27;2020-12-18T21:00:00.000000000&#x27;, &#x27;2020-12-19T00:00:00.000000000&#x27;,
       &#x27;2020-12-19T03:00:00.000000000&#x27;, &#x27;2020-12-19T06:00:00.000000000&#x27;,
       &#x27;2020-12-19T09:00:00.000000000&#x27;, &#x27;2020-12-19T12:00:00.000000000&#x27;,
       &#x27;2020-12-19T15:00:00.000000000&#x27;, &#x27;2020-12-19T18:00:00.000000000&#x27;,
       &#x27;2020-12-19T21:00:00.000000000&#x27;, &#x27;2020-12-20T00:00:00.000000000&#x27;,
       &#x27;2020-12-20T03:00:00.000000000&#x27;, &#x27;2020-12-20T06:00:00.000000000&#x27;,
       &#x27;2020-12-20T09:00:00.000000000&#x27;, &#x27;2020-12-20T12:00:00.000000000&#x27;,
       &#x27;2020-12-20T15:00:00.000000000&#x27;, &#x27;2020-12-20T18:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-15.2 -15.17 -15.2 ... -24.21 -24.6</div><input id='attrs-e20412d5-a499-4248-bc1a-4cb59f1eb3a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e20412d5-a499-4248-bc1a-4cb59f1eb3a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-136d5f92-46fb-4726-86f0-771725a1c8cf' class='xr-var-data-in' type='checkbox'><label for='data-136d5f92-46fb-4726-86f0-771725a1c8cf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-15.2     , -15.172693, -15.2     , -15.319935, -15.5     ,
       -15.71497 , -15.9     , -16.010073, -16.      , -15.81497 ,
       -15.6     , -15.542496, -15.5     , -15.357452, -15.2     ,
       -15.077543, -15.      , -14.992531, -15.      , -14.957386,
       -14.9     , -14.857623, -14.8     , -14.669821, -14.6     ,
       -14.71262 , -14.9     , -15.049695, -15.2     , -15.335211,
       -15.5     , -15.734777, -16.      , -16.236345, -16.5     ,
       -16.6     , -16.9     , -17.2     , -17.552794, -17.9     ,
       -18.15316 , -18.4     , -18.7     , -19.2     , -19.414701,
       -19.5     , -19.652195, -19.9     , -20.3     , -21.      ,
       -21.368101, -21.6     , -22.000177, -22.4     , -22.72992 ,
       -23.      , -23.22991 , -23.4     , -23.507683, -23.6     ,
       -23.712439, -23.9     , -24.212538, -24.6     ], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>173.1 172.7 172.5 ... 181.2 181.3</div><input id='attrs-84bc3294-c3c7-442c-afca-f8ec66e81b56' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-84bc3294-c3c7-442c-afca-f8ec66e81b56' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78a071f2-841c-4738-b116-7e02a0e96fba' class='xr-var-data-in' type='checkbox'><label for='data-78a071f2-841c-4738-b116-7e02a0e96fba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([173.1    , 172.73749, 172.5    , 172.46246, 172.5    , 172.48   ,
       172.4    , 172.20004, 172.     , 171.91997, 171.9    , 171.88501,
       171.9    , 171.91248, 172.     , 172.22006, 172.5    , 172.77243,
       173.     , 173.09248, 173.2    , 173.49251, 173.8    , 173.9351 ,
       174.1    , 174.46986, 174.9    , 175.22015, 175.6    , 176.17761,
       176.8    , 177.31506, 177.8    , 178.27318, 178.8    , 179.     ,
       179.4    , 180.3    , 180.59036, 180.6    , 180.76508, 181.1    ,
       181.7    , 182.     , 182.16158, 182.2    , 182.2214 , 182.1    ,
       181.8    , 181.7    , 181.64555, 181.7    , 181.942  , 182.2    ,
       182.30264, 182.3    , 182.25252, 182.1    , 181.80748, 181.5    ,
       181.29753, 181.2    , 181.2173 , 181.3    ], dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-16f215aa-00ff-443a-9993-5c48a4807f29' class='xr-section-summary-in' type='checkbox'  checked><label for='section-16f215aa-00ff-443a-9993-5c48a4807f29' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>radius_max_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>50.0 50.0 50.0 ... 50.0 50.0 50.0</div><input id='attrs-6abded9d-ffea-4757-8e60-e8d776f03c8b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6abded9d-ffea-4757-8e60-e8d776f03c8b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-816b2554-3ecb-4222-bb02-9aa240e95782' class='xr-var-data-in' type='checkbox'><label for='data-816b2554-3ecb-4222-bb02-9aa240e95782' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([50., 50., 50., 37., 25., 27., 30., 30., 30., 25., 20., 20., 20.,
       20., 20., 16., 12., 12., 12., 12., 12., 13., 15., 17., 20., 17.,
       15., 15., 15., 15., 15., 13., 12.,  8.,  5.,  6., 10., 15., 15.,
       15., 15., 15., 15., 15., 20., 25., 27., 30., 30., 30., 30., 30.,
       30., 30., 30., 30., 40., 50., 50., 50., 50., 50., 50., 50.],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>radius_oci</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>230.0 190.0 150.0 ... 200.0 200.0</div><input id='attrs-cecd8fef-7a90-489a-a2d1-ad0e857ee5fb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cecd8fef-7a90-489a-a2d1-ad0e857ee5fb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-922459ca-459f-46b2-a283-6fbc7cdd2dfc' class='xr-var-data-in' type='checkbox'><label for='data-922459ca-459f-46b2-a283-6fbc7cdd2dfc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([230., 190., 150., 152., 155., 165., 175., 180., 185., 187., 190.,
       195., 200., 200., 200., 200., 200., 200., 200., 207., 215., 215.,
       215., 215., 215., 210., 205., 195., 185., 185., 185., 180., 175.,
       185., 195., 196., 200., 205., 205., 205., 205., 205., 200., 195.,
       190., 185., 185., 185., 197., 210., 212., 215., 207., 200., 200.,
       200., 207., 215., 207., 200., 200., 200., 200., 200.],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>max_sustained_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>34.09 36.36 39.77 ... 36.36 34.09</div><input id='attrs-010460e2-f935-4936-92d5-f732291ba23f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-010460e2-f935-4936-92d5-f732291ba23f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca0d3298-ef98-4bc5-9977-981b47c2ceaa' class='xr-var-data-in' type='checkbox'><label for='data-ca0d3298-ef98-4bc5-9977-981b47c2ceaa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 34.090908,  36.363636,  39.772728,  45.454544,  51.136364,
        53.409092,  56.818184,  62.5     ,  68.181816,  73.86364 ,
        79.545456,  79.545456,  79.545456,  81.818184,  85.22727 ,
        87.5     ,  90.90909 ,  96.59091 , 102.27273 , 104.545456,
       107.954544, 115.90909 , 125.      , 130.68182 , 136.36363 ,
       136.36363 , 136.36363 , 136.36363 , 136.36363 , 138.63637 ,
       142.04546 , 138.63637 , 136.36363 , 132.95454 , 130.68182 ,
       130.68182 , 119.318184, 107.954544,  96.59091 ,  85.22727 ,
        81.818184,  79.545456,  79.545456,  85.22727 ,  85.22727 ,
        85.22727 ,  79.545456,  73.86364 ,  73.86364 ,  68.181816,
        64.77273 ,  62.5     ,  59.090908,  56.818184,  56.818184,
        56.818184,  53.409092,  51.136364,  47.727272,  45.454544,
        42.045456,  39.772728,  36.363636,  34.090908], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>central_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>997.0 996.0 995.0 ... 995.0 997.0</div><input id='attrs-c726da7b-0518-4105-865b-89e6638a1ac6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c726da7b-0518-4105-865b-89e6638a1ac6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0f0e2a2e-2aaa-4a98-881e-21057044d5a8' class='xr-var-data-in' type='checkbox'><label for='data-0f0e2a2e-2aaa-4a98-881e-21057044d5a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([997., 996., 995., 993., 991., 989., 987., 984., 982., 977., 972.,
       972., 972., 969., 967., 966., 966., 959., 953., 949., 946., 939.,
       932., 925., 919., 920., 921., 920., 920., 918., 917., 920., 923.,
       925., 928., 928., 940., 954., 961., 969., 969., 970., 971., 967.,
       965., 964., 968., 972., 973., 978., 979., 980., 982., 985., 984.,
       984., 985., 986., 988., 990., 991., 993., 995., 997.],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>environmental_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.005e+03 1.004e+03 ... 1.006e+03</div><input id='attrs-a3fdc7c6-1680-4d8e-ae46-b5f88589a2a1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a3fdc7c6-1680-4d8e-ae46-b5f88589a2a1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3cb8041d-bbba-4acf-97fb-e2dfef5be15e' class='xr-var-data-in' type='checkbox'><label for='data-3cb8041d-bbba-4acf-97fb-e2dfef5be15e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1005., 1004., 1003., 1004., 1005., 1005., 1006., 1006., 1006.,
       1005., 1004., 1004., 1005., 1005., 1006., 1006., 1006., 1004.,
       1003., 1004., 1005., 1004., 1003., 1003., 1003., 1002., 1002.,
       1002., 1003., 1002., 1001., 1001., 1002., 1001., 1001., 1001.,
       1002., 1004., 1004., 1004., 1004., 1004., 1004., 1004., 1004.,
       1005., 1005., 1005., 1005., 1005., 1004., 1004., 1004., 1005.,
       1005., 1005., 1005., 1005., 1004., 1004., 1005., 1006., 1006.,
       1006.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>time_step</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.0 3.0 3.0 3.0 ... 3.0 3.0 3.0 3.0</div><input id='attrs-56e79d23-d3ac-46b4-95c1-a7da33c143a7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-56e79d23-d3ac-46b4-95c1-a7da33c143a7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8fe34c24-53dd-4105-a1cb-bb0d9fd69820' class='xr-var-data-in' type='checkbox'><label for='data-8fe34c24-53dd-4105-a1cb-bb0d9fd69820' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       1.00000001, 1.99999999, 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        , 3.        ,
       3.        , 3.        , 3.        , 3.        ])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>basin</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U2</div><div class='xr-var-preview xr-preview'>&#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27; ... &#x27;SP&#x27; &#x27;SP&#x27; &#x27;SP&#x27;</div><input id='attrs-bd2af334-cc17-4115-88f0-37a751de1f1d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bd2af334-cc17-4115-88f0-37a751de1f1d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f016ec42-6d29-41ca-a1c0-22f34d835714' class='xr-var-data-in' type='checkbox'><label for='data-f016ec42-6d29-41ca-a1c0-22f34d835714' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;,
       &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;,
       &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;,
       &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;,
       &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;,
       &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;, &#x27;SP&#x27;], dtype=&#x27;&lt;U2&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2d98ca41-a8e6-4040-b077-a84cecb79b3a' class='xr-section-summary-in' type='checkbox'  ><label for='section-2d98ca41-a8e6-4040-b077-a84cecb79b3a' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-05c785c1-42fe-436e-b841-34a38b073a95' class='xr-index-data-in' type='checkbox'/><label for='index-05c785c1-42fe-436e-b841-34a38b073a95' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2020-12-13 00:00:00&#x27;, &#x27;2020-12-13 03:00:00&#x27;,
               &#x27;2020-12-13 06:00:00&#x27;, &#x27;2020-12-13 09:00:00&#x27;,
               &#x27;2020-12-13 12:00:00&#x27;, &#x27;2020-12-13 15:00:00&#x27;,
               &#x27;2020-12-13 18:00:00&#x27;, &#x27;2020-12-13 21:00:00&#x27;,
               &#x27;2020-12-14 00:00:00&#x27;, &#x27;2020-12-14 03:00:00&#x27;,
               &#x27;2020-12-14 06:00:00&#x27;, &#x27;2020-12-14 09:00:00&#x27;,
               &#x27;2020-12-14 12:00:00&#x27;, &#x27;2020-12-14 15:00:00&#x27;,
               &#x27;2020-12-14 18:00:00&#x27;, &#x27;2020-12-14 21:00:00&#x27;,
               &#x27;2020-12-15 00:00:00&#x27;, &#x27;2020-12-15 03:00:00&#x27;,
               &#x27;2020-12-15 06:00:00&#x27;, &#x27;2020-12-15 09:00:00&#x27;,
               &#x27;2020-12-15 12:00:00&#x27;, &#x27;2020-12-15 15:00:00&#x27;,
               &#x27;2020-12-15 18:00:00&#x27;, &#x27;2020-12-15 21:00:00&#x27;,
               &#x27;2020-12-16 00:00:00&#x27;, &#x27;2020-12-16 03:00:00&#x27;,
               &#x27;2020-12-16 06:00:00&#x27;, &#x27;2020-12-16 09:00:00&#x27;,
               &#x27;2020-12-16 12:00:00&#x27;, &#x27;2020-12-16 15:00:00&#x27;,
               &#x27;2020-12-16 18:00:00&#x27;, &#x27;2020-12-16 21:00:00&#x27;,
               &#x27;2020-12-17 00:00:00&#x27;, &#x27;2020-12-17 03:00:00&#x27;,
               &#x27;2020-12-17 06:00:00&#x27;, &#x27;2020-12-17 07:00:00&#x27;,
               &#x27;2020-12-17 09:00:00&#x27;, &#x27;2020-12-17 12:00:00&#x27;,
               &#x27;2020-12-17 15:00:00&#x27;, &#x27;2020-12-17 18:00:00&#x27;,
               &#x27;2020-12-17 21:00:00&#x27;, &#x27;2020-12-18 00:00:00&#x27;,
               &#x27;2020-12-18 03:00:00&#x27;, &#x27;2020-12-18 06:00:00&#x27;,
               &#x27;2020-12-18 09:00:00&#x27;, &#x27;2020-12-18 12:00:00&#x27;,
               &#x27;2020-12-18 15:00:00&#x27;, &#x27;2020-12-18 18:00:00&#x27;,
               &#x27;2020-12-18 21:00:00&#x27;, &#x27;2020-12-19 00:00:00&#x27;,
               &#x27;2020-12-19 03:00:00&#x27;, &#x27;2020-12-19 06:00:00&#x27;,
               &#x27;2020-12-19 09:00:00&#x27;, &#x27;2020-12-19 12:00:00&#x27;,
               &#x27;2020-12-19 15:00:00&#x27;, &#x27;2020-12-19 18:00:00&#x27;,
               &#x27;2020-12-19 21:00:00&#x27;, &#x27;2020-12-20 00:00:00&#x27;,
               &#x27;2020-12-20 03:00:00&#x27;, &#x27;2020-12-20 06:00:00&#x27;,
               &#x27;2020-12-20 09:00:00&#x27;, &#x27;2020-12-20 12:00:00&#x27;,
               &#x27;2020-12-20 15:00:00&#x27;, &#x27;2020-12-20 18:00:00&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;time&#x27;, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-969c2378-c165-4674-aeee-328ed5ed6558' class='xr-section-summary-in' type='checkbox'  checked><label for='section-969c2378-c165-4674-aeee-328ed5ed6558' class='xr-section-summary' >Attributes: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>max_sustained_wind_unit :</span></dt><dd>kn</dd><dt><span>central_pressure_unit :</span></dt><dd>mb</dd><dt><span>orig_event_flag :</span></dt><dd>True</dd><dt><span>data_provider :</span></dt><dd>ibtracs_mixed:lat(official_3h),lon(official_3h),wind(official_3h),pres(official_3h),rmw(usa),poci(usa),roci(usa)</dd><dt><span>category :</span></dt><dd>5</dd><dt><span>name :</span></dt><dd>YASA</dd><dt><span>sid :</span></dt><dd>2020346S13168</dd><dt><span>id_no :</span></dt><dd>2020346113168.0</dd></dl></div></li></ul></div></div>




```python
yasa_track.data[0].environmental_pressure
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;environmental_pressure&#x27; (time: 64)&gt;
array([1005., 1004., 1003., 1004., 1005., 1005., 1006., 1006., 1006.,
       1005., 1004., 1004., 1005., 1005., 1006., 1006., 1006., 1004.,
       1003., 1004., 1005., 1004., 1003., 1003., 1003., 1002., 1002.,
       1002., 1003., 1002., 1001., 1001., 1002., 1001., 1001., 1001.,
       1002., 1004., 1004., 1004., 1004., 1004., 1004., 1004., 1004.,
       1005., 1005., 1005., 1005., 1005., 1004., 1004., 1004., 1005.,
       1005., 1005., 1005., 1005., 1004., 1004., 1005., 1006., 1006.,
       1006.])
Coordinates:
  * time     (time) datetime64[ns] 2020-12-13 ... 2020-12-20T18:00:00
    lat      (time) float32 -15.2 -15.17 -15.2 -15.32 ... -23.9 -24.21 -24.6
    lon      (time) float32 173.1 172.7 172.5 172.5 ... 181.3 181.2 181.2 181.3</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'environmental_pressure'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 64</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-22838f70-7463-4424-b234-d72a9498b88e' class='xr-array-in' type='checkbox' checked><label for='section-22838f70-7463-4424-b234-d72a9498b88e' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>1.005e+03 1.004e+03 1.003e+03 ... 1.006e+03 1.006e+03 1.006e+03</span></div><div class='xr-array-data'><pre>array([1005., 1004., 1003., 1004., 1005., 1005., 1006., 1006., 1006.,
       1005., 1004., 1004., 1005., 1005., 1006., 1006., 1006., 1004.,
       1003., 1004., 1005., 1004., 1003., 1003., 1003., 1002., 1002.,
       1002., 1003., 1002., 1001., 1001., 1002., 1001., 1001., 1001.,
       1002., 1004., 1004., 1004., 1004., 1004., 1004., 1004., 1004.,
       1005., 1005., 1005., 1005., 1005., 1004., 1004., 1004., 1005.,
       1005., 1005., 1005., 1005., 1004., 1004., 1005., 1006., 1006.,
       1006.])</pre></div></div></li><li class='xr-section-item'><input id='section-bc0c046a-e51a-43a6-a7cd-5cb18f411f84' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bc0c046a-e51a-43a6-a7cd-5cb18f411f84' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2020-12-13 ... 2020-12-20T18:00:00</div><input id='attrs-3e1dc8e8-435e-4bc7-90bb-fc09bd6e42d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3e1dc8e8-435e-4bc7-90bb-fc09bd6e42d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b78d6488-0fa8-4a91-b376-f2fc8bf863e9' class='xr-var-data-in' type='checkbox'><label for='data-b78d6488-0fa8-4a91-b376-f2fc8bf863e9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2020-12-13T00:00:00.000000000&#x27;, &#x27;2020-12-13T03:00:00.000000000&#x27;,
       &#x27;2020-12-13T06:00:00.000000000&#x27;, &#x27;2020-12-13T09:00:00.000000000&#x27;,
       &#x27;2020-12-13T12:00:00.000000000&#x27;, &#x27;2020-12-13T15:00:00.000000000&#x27;,
       &#x27;2020-12-13T18:00:00.000000000&#x27;, &#x27;2020-12-13T21:00:00.000000000&#x27;,
       &#x27;2020-12-14T00:00:00.000000000&#x27;, &#x27;2020-12-14T03:00:00.000000000&#x27;,
       &#x27;2020-12-14T06:00:00.000000000&#x27;, &#x27;2020-12-14T09:00:00.000000000&#x27;,
       &#x27;2020-12-14T12:00:00.000000000&#x27;, &#x27;2020-12-14T15:00:00.000000000&#x27;,
       &#x27;2020-12-14T18:00:00.000000000&#x27;, &#x27;2020-12-14T21:00:00.000000000&#x27;,
       &#x27;2020-12-15T00:00:00.000000000&#x27;, &#x27;2020-12-15T03:00:00.000000000&#x27;,
       &#x27;2020-12-15T06:00:00.000000000&#x27;, &#x27;2020-12-15T09:00:00.000000000&#x27;,
       &#x27;2020-12-15T12:00:00.000000000&#x27;, &#x27;2020-12-15T15:00:00.000000000&#x27;,
       &#x27;2020-12-15T18:00:00.000000000&#x27;, &#x27;2020-12-15T21:00:00.000000000&#x27;,
       &#x27;2020-12-16T00:00:00.000000000&#x27;, &#x27;2020-12-16T03:00:00.000000000&#x27;,
       &#x27;2020-12-16T06:00:00.000000000&#x27;, &#x27;2020-12-16T09:00:00.000000000&#x27;,
       &#x27;2020-12-16T12:00:00.000000000&#x27;, &#x27;2020-12-16T15:00:00.000000000&#x27;,
       &#x27;2020-12-16T18:00:00.000000000&#x27;, &#x27;2020-12-16T21:00:00.000000000&#x27;,
       &#x27;2020-12-17T00:00:00.000000000&#x27;, &#x27;2020-12-17T03:00:00.000000000&#x27;,
       &#x27;2020-12-17T06:00:00.000000000&#x27;, &#x27;2020-12-17T07:00:00.000000000&#x27;,
       &#x27;2020-12-17T09:00:00.000000000&#x27;, &#x27;2020-12-17T12:00:00.000000000&#x27;,
       &#x27;2020-12-17T15:00:00.000000000&#x27;, &#x27;2020-12-17T18:00:00.000000000&#x27;,
       &#x27;2020-12-17T21:00:00.000000000&#x27;, &#x27;2020-12-18T00:00:00.000000000&#x27;,
       &#x27;2020-12-18T03:00:00.000000000&#x27;, &#x27;2020-12-18T06:00:00.000000000&#x27;,
       &#x27;2020-12-18T09:00:00.000000000&#x27;, &#x27;2020-12-18T12:00:00.000000000&#x27;,
       &#x27;2020-12-18T15:00:00.000000000&#x27;, &#x27;2020-12-18T18:00:00.000000000&#x27;,
       &#x27;2020-12-18T21:00:00.000000000&#x27;, &#x27;2020-12-19T00:00:00.000000000&#x27;,
       &#x27;2020-12-19T03:00:00.000000000&#x27;, &#x27;2020-12-19T06:00:00.000000000&#x27;,
       &#x27;2020-12-19T09:00:00.000000000&#x27;, &#x27;2020-12-19T12:00:00.000000000&#x27;,
       &#x27;2020-12-19T15:00:00.000000000&#x27;, &#x27;2020-12-19T18:00:00.000000000&#x27;,
       &#x27;2020-12-19T21:00:00.000000000&#x27;, &#x27;2020-12-20T00:00:00.000000000&#x27;,
       &#x27;2020-12-20T03:00:00.000000000&#x27;, &#x27;2020-12-20T06:00:00.000000000&#x27;,
       &#x27;2020-12-20T09:00:00.000000000&#x27;, &#x27;2020-12-20T12:00:00.000000000&#x27;,
       &#x27;2020-12-20T15:00:00.000000000&#x27;, &#x27;2020-12-20T18:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-15.2 -15.17 -15.2 ... -24.21 -24.6</div><input id='attrs-1c85ee9d-1c2f-4988-87e9-75ffef109909' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c85ee9d-1c2f-4988-87e9-75ffef109909' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4e436fe3-857d-4050-8986-4ebd7cd1cc97' class='xr-var-data-in' type='checkbox'><label for='data-4e436fe3-857d-4050-8986-4ebd7cd1cc97' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-15.2     , -15.172693, -15.2     , -15.319935, -15.5     ,
       -15.71497 , -15.9     , -16.010073, -16.      , -15.81497 ,
       -15.6     , -15.542496, -15.5     , -15.357452, -15.2     ,
       -15.077543, -15.      , -14.992531, -15.      , -14.957386,
       -14.9     , -14.857623, -14.8     , -14.669821, -14.6     ,
       -14.71262 , -14.9     , -15.049695, -15.2     , -15.335211,
       -15.5     , -15.734777, -16.      , -16.236345, -16.5     ,
       -16.6     , -16.9     , -17.2     , -17.552794, -17.9     ,
       -18.15316 , -18.4     , -18.7     , -19.2     , -19.414701,
       -19.5     , -19.652195, -19.9     , -20.3     , -21.      ,
       -21.368101, -21.6     , -22.000177, -22.4     , -22.72992 ,
       -23.      , -23.22991 , -23.4     , -23.507683, -23.6     ,
       -23.712439, -23.9     , -24.212538, -24.6     ], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>173.1 172.7 172.5 ... 181.2 181.3</div><input id='attrs-8c62d02b-0150-4964-be59-c60dd2f4a62d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8c62d02b-0150-4964-be59-c60dd2f4a62d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-480aa82f-ae25-48db-ae05-67bdf0980bac' class='xr-var-data-in' type='checkbox'><label for='data-480aa82f-ae25-48db-ae05-67bdf0980bac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([173.1    , 172.73749, 172.5    , 172.46246, 172.5    , 172.48   ,
       172.4    , 172.20004, 172.     , 171.91997, 171.9    , 171.88501,
       171.9    , 171.91248, 172.     , 172.22006, 172.5    , 172.77243,
       173.     , 173.09248, 173.2    , 173.49251, 173.8    , 173.9351 ,
       174.1    , 174.46986, 174.9    , 175.22015, 175.6    , 176.17761,
       176.8    , 177.31506, 177.8    , 178.27318, 178.8    , 179.     ,
       179.4    , 180.3    , 180.59036, 180.6    , 180.76508, 181.1    ,
       181.7    , 182.     , 182.16158, 182.2    , 182.2214 , 182.1    ,
       181.8    , 181.7    , 181.64555, 181.7    , 181.942  , 182.2    ,
       182.30264, 182.3    , 182.25252, 182.1    , 181.80748, 181.5    ,
       181.29753, 181.2    , 181.2173 , 181.3    ], dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-021593a0-99a9-4d0e-aca6-8e3a2a5733db' class='xr-section-summary-in' type='checkbox'  ><label for='section-021593a0-99a9-4d0e-aca6-8e3a2a5733db' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-78c5dc2d-dee7-4e9e-8137-c397276e330a' class='xr-index-data-in' type='checkbox'/><label for='index-78c5dc2d-dee7-4e9e-8137-c397276e330a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2020-12-13 00:00:00&#x27;, &#x27;2020-12-13 03:00:00&#x27;,
               &#x27;2020-12-13 06:00:00&#x27;, &#x27;2020-12-13 09:00:00&#x27;,
               &#x27;2020-12-13 12:00:00&#x27;, &#x27;2020-12-13 15:00:00&#x27;,
               &#x27;2020-12-13 18:00:00&#x27;, &#x27;2020-12-13 21:00:00&#x27;,
               &#x27;2020-12-14 00:00:00&#x27;, &#x27;2020-12-14 03:00:00&#x27;,
               &#x27;2020-12-14 06:00:00&#x27;, &#x27;2020-12-14 09:00:00&#x27;,
               &#x27;2020-12-14 12:00:00&#x27;, &#x27;2020-12-14 15:00:00&#x27;,
               &#x27;2020-12-14 18:00:00&#x27;, &#x27;2020-12-14 21:00:00&#x27;,
               &#x27;2020-12-15 00:00:00&#x27;, &#x27;2020-12-15 03:00:00&#x27;,
               &#x27;2020-12-15 06:00:00&#x27;, &#x27;2020-12-15 09:00:00&#x27;,
               &#x27;2020-12-15 12:00:00&#x27;, &#x27;2020-12-15 15:00:00&#x27;,
               &#x27;2020-12-15 18:00:00&#x27;, &#x27;2020-12-15 21:00:00&#x27;,
               &#x27;2020-12-16 00:00:00&#x27;, &#x27;2020-12-16 03:00:00&#x27;,
               &#x27;2020-12-16 06:00:00&#x27;, &#x27;2020-12-16 09:00:00&#x27;,
               &#x27;2020-12-16 12:00:00&#x27;, &#x27;2020-12-16 15:00:00&#x27;,
               &#x27;2020-12-16 18:00:00&#x27;, &#x27;2020-12-16 21:00:00&#x27;,
               &#x27;2020-12-17 00:00:00&#x27;, &#x27;2020-12-17 03:00:00&#x27;,
               &#x27;2020-12-17 06:00:00&#x27;, &#x27;2020-12-17 07:00:00&#x27;,
               &#x27;2020-12-17 09:00:00&#x27;, &#x27;2020-12-17 12:00:00&#x27;,
               &#x27;2020-12-17 15:00:00&#x27;, &#x27;2020-12-17 18:00:00&#x27;,
               &#x27;2020-12-17 21:00:00&#x27;, &#x27;2020-12-18 00:00:00&#x27;,
               &#x27;2020-12-18 03:00:00&#x27;, &#x27;2020-12-18 06:00:00&#x27;,
               &#x27;2020-12-18 09:00:00&#x27;, &#x27;2020-12-18 12:00:00&#x27;,
               &#x27;2020-12-18 15:00:00&#x27;, &#x27;2020-12-18 18:00:00&#x27;,
               &#x27;2020-12-18 21:00:00&#x27;, &#x27;2020-12-19 00:00:00&#x27;,
               &#x27;2020-12-19 03:00:00&#x27;, &#x27;2020-12-19 06:00:00&#x27;,
               &#x27;2020-12-19 09:00:00&#x27;, &#x27;2020-12-19 12:00:00&#x27;,
               &#x27;2020-12-19 15:00:00&#x27;, &#x27;2020-12-19 18:00:00&#x27;,
               &#x27;2020-12-19 21:00:00&#x27;, &#x27;2020-12-20 00:00:00&#x27;,
               &#x27;2020-12-20 03:00:00&#x27;, &#x27;2020-12-20 06:00:00&#x27;,
               &#x27;2020-12-20 09:00:00&#x27;, &#x27;2020-12-20 12:00:00&#x27;,
               &#x27;2020-12-20 15:00:00&#x27;, &#x27;2020-12-20 18:00:00&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;time&#x27;, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-189fe8b1-8839-48b6-b033-9e2d598bf134' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-189fe8b1-8839-48b6-b033-9e2d598bf134' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
cent = Centroids.from_geodataframe(grids)

tc = TropCyclone.from_tracks(
    tracks, centroids=cent, store_windfields=True, intensity_thres=0
)
```

    2024-01-05 13:22:46,943 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



```python
# Just to check
np.sort(tc.intensity.toarray().flatten())[-10:]
```




    array([72.10223834, 72.31296562, 72.93854395, 72.97697109, 72.97792903,
           73.11851391, 73.17804901, 73.40158883, 73.59083684, 73.72871348])




```python
def windfield_to_grid(tc, tracks, grids):
    df_windfield = pd.DataFrame()

    for intensity_sparse, event_id in zip(tc.intensity, tc.event_name):
        # Get the windfield
        windfield = intensity_sparse.toarray().flatten()
        npoints = len(windfield)
        # Get the track distance
        tc_track = tracks.get_track(track_name=event_id)
        points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
        tc_track_line = LineString(points)
        DEG_TO_KM = 111.1
        tc_track_distance = grids["geometry"].apply(
            lambda point: point.distance(tc_track_line) * DEG_TO_KM
        )
        # Add to DF
        df_to_add = pd.DataFrame(
            dict(
                track_id=[event_id] * npoints,
                grid_point_id=grids["id"],
                wind_speed=windfield,
                track_distance=tc_track_distance,
                geometry = grids.geometry
            )
        )
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
    return df_windfield

df_windfield = windfield_to_grid(tc=tc, tracks=tracks, grids=grids)
```


```python
# Track
tc_track = tracks.get_track(track_name=event_id)
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])
```


```python
# Plot intensity
name='YASA custom track'
fig, ax = plt.subplots(1,1)
geo_windfield = gpd.GeoDataFrame(df_windfield)
geo_windfield.plot(column='wind_speed', cmap='Reds', linewidth=0.2, edgecolor='0.3', ax=ax, legend=True)
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')
track_points.plot(ax=ax, color='k', linewidth=1, label='Typhoon track datapoints')

# ax.axis('off')
ax.set_xlim(176, 182)
ax.set_ylim(-20, -12)
ax.set_title(name)
plt.show()
```



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_22_0.png)



## Automation

I have NDMO tracks from:


```python
unique_tracks = []
for i in range(len(csv_file_paths)):
    unique_tracks.append(csv_file_paths[i].split('_')[-1].split('.')[0])

set(unique_tracks)
```




    {'EVAN', 'Harold', 'YASA'}



Lets work with YASA


```python
YASA_files = []
for file in csv_file_paths:
    if file.endswith('YASA.csv'):
        YASA_files.append(file)
YASA_files = sorted(YASA_files) # This is the same track but on different dates
```


```python
# Loading custom tracks
custom_idno = 456
custom_sid = str(custom_idno)

df_forecast_all = pd.DataFrame()
list_forecasts = []
for i in range(len(YASA_files)):
    df_forecast = pd.read_csv(YASA_files[i], header=6, engine='python').iloc[1:][
        ["Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']",'Latitude', 'Longitude', 'MeanWind', 'Pressure', 'PressureOCI', 'Category','RadiusOCI', 'RadiusMaxWinds']].rename(
            {"Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']":'forecast_time'},
            axis=1
        )

    # Create a GeoDataFrame from the DataFrame with latitude and longitude
    geometry = [Point(xy) for xy in zip(df_forecast['Longitude'], df_forecast['Latitude'])]
    df_forecast = gpd.GeoDataFrame(df_forecast, geometry=geometry)

    # Convert 'forecast_time' to datetime format
    df_forecast['forecast_time'] = pd.to_datetime(df_forecast['forecast_time'])

    # Sort the DataFrame by 'forecast_time' in ascending order
    df_forecast = df_forecast.sort_values(by='forecast_time')

    # Calculate the time step by subtracting consecutive 'forecast_time' values
    df_forecast['time_step'] = df_forecast['forecast_time'].diff().dt.total_seconds() / 3600
    df_forecast['time_step'].fillna(0, inplace=True)

    # Atmospheric pressure ---THIS IS IMPORTANT!!!!---
    df_forecast['AtmPressure'] = 1013 #millibars

    # Basin
    df_forecast['basin'] = 'SP'

    df_forecast_all = pd.concat([df_forecast_all, df_forecast])
    list_forecasts.append(df_forecast)
```

### All tracks append


```python
# Create xarray object
tracks_all = TCTracks()
tracks_all.data = [adjust_tracks(df_forecast_all)]

# Compute windfield
cent = Centroids.from_geodataframe(grids)
tc_all = TropCyclone.from_tracks(
    tracks_all, centroids=cent, store_windfields=True, intensity_thres=0
)
df_windfield_all = windfield_to_grid(tc=tc_all, tracks=tracks_all, grids=grids)
```

    2024-01-05 12:57:35,510 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.


    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/climada/hazard/trop_cyclone.py:1222: RuntimeWarning: divide by zero encountered in divide
      si_track["vtrans"].values[1:, :] = vec[:, 0, 0] / si_track["tstep"].values[1:, None]
    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/climada/hazard/trop_cyclone.py:1223: RuntimeWarning: divide by zero encountered in divide
      si_track["vtrans_norm"].values[1:] = norm[:, 0, 0] / si_track["tstep"].values[1:]
    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/climada/hazard/trop_cyclone.py:1228: RuntimeWarning: invalid value encountered in multiply
      si_track["vtrans"].values[msk, :] *= fact[:, None]
    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/climada/hazard/trop_cyclone.py:1229: RuntimeWarning: invalid value encountered in multiply
      si_track["vtrans_norm"].values[msk] *= fact



```python
# Track lines
tc_track = tracks_all.get_track(track_name='456')
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

# Plot intensity
name='YASA custom track'
fig, ax = plt.subplots(1,1)
geo_windfield = gpd.GeoDataFrame(df_windfield_all)
geo_windfield.plot(column='wind_speed', cmap='Reds', linewidth=0.2, edgecolor='0.3', ax=ax, legend=True)
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')
track_points.plot(ax=ax, color='k', linewidth=1, label='Typhoon track datapoints')

# ax.axis('off')
# ax.set_xlim(176, 182)
# ax.set_ylim(-20, -12)
ax.set_title(name)
plt.show()

```



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_31_0.png)



### One by one


```python
# Create xarray object
track = TCTracks()
tracks_sep = TCTracks()
for i in range(len(list_forecasts)):
    custom_idno = i
    custom_sid = str(custom_idno)
    track.data = [adjust_tracks(list_forecasts[i])]
    tracks_sep.append(track.get_track())

# Compute windfield
cent = Centroids.from_geodataframe(grids)
tc_sep = TropCyclone.from_tracks(
    tracks_sep, centroids=cent, store_windfields=True, intensity_thres=0
)
df_windfield_sep = windfield_to_grid(tc=tc_sep, tracks=tracks_sep, grids=grids)
```

    2024-01-05 13:11:52,570 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



```python
track.data = [adjust_tracks(list_forecasts[-5])]
tc_aux = TropCyclone.from_tracks(
    track, centroids=cent, store_windfields=True, intensity_thres=0
)

#tc_aux.intensity.toarray().flatten()
```


```python
# Track lines

fig, ax = plt.subplots(4,5, figsize=(15,15))
ax = ax.flatten()
name = 'YASA tracks'
for i in range(len(YASA_files[:17])):
    # Create track
    tc_track = tracks_sep.get_track(track_name=str(i))
    date_range = str(np.array(tc_track.time[0]))[:10] + ' to ' +str(np.array(tc_track.time[-1]))[:10]
    points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
    track_points = gpd.GeoDataFrame(geometry=points)
    tc_track_line = LineString(points)
    track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

    # Plot intensity
    geo_windfield = gpd.GeoDataFrame(df_windfield_sep[df_windfield_sep.track_id == str(i)])
    geo_windfield.plot(column='wind_speed', cmap='Reds', linewidth=0.2, edgecolor='0.3', ax=ax[i], legend=True)
    track_line.plot(ax=ax[i], color='k', linewidth=1, label='Typhoon track')
    track_points.plot(ax=ax[i], color='k', linewidth=1, label='Typhoon track datapoints')
    ax[i].set_title(date_range, size=8)

# ax.axis('off')
# ax.set_xlim(176, 182)
# ax.set_ylim(-20, -12)

fig.delaxes(ax[17])
fig.delaxes(ax[18])
fig.delaxes(ax[19])

plt.suptitle(name)
plt.tight_layout()
plt.show()
```



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_35_0.png)
