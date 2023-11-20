# Gather the data needed to train the model

In this notebook we combine all of the data from
step 2. The contents of this notebook is mirrored
in `utils.py` so that it can be used in other notebooks.


```python
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/02_new_model_input"
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_new_model_training"
)
```

## Read in buliding damage


```python
# Read in the building damage data
filename = input_dir / "02_housing_damage/output/building_damage_bygrid_new.csv"
df_damage = pd.read_csv(filename)

# Read in buildings per grid
filename = input_dir / "02_housing_damage/output/num_building_bygrid.csv"
df_buildings_raw = pd.read_csv(filename)
```


```python
df_damage.sort_values('id')
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
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>ADM1_NAME</th>
      <th>ADM2_NAME</th>
      <th>typhoon</th>
      <th>Year</th>
      <th>Totally</th>
      <th>Partially</th>
      <th>total</th>
      <th>id</th>
      <th>Municipality</th>
      <th>numbuildings_x</th>
      <th>numbuildings_y</th>
      <th>frac_bld</th>
      <th>perc_dmg_grid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1696</th>
      <td>1696</td>
      <td>49</td>
      <td>Western Division</td>
      <td>Ba</td>
      <td>Ana</td>
      <td>2021</td>
      <td>7.0</td>
      <td>241.0</td>
      <td>248.000000</td>
      <td>355</td>
      <td>Ba</td>
      <td>0</td>
      <td>57364</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1876</th>
      <td>1876</td>
      <td>54</td>
      <td>Western Division</td>
      <td>Ba</td>
      <td>Evan</td>
      <td>2012</td>
      <td>699.0</td>
      <td>614.0</td>
      <td>437.666667</td>
      <td>355</td>
      <td>Ba</td>
      <td>0</td>
      <td>57364</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>1576</td>
      <td>29</td>
      <td>Western Division</td>
      <td>Ba</td>
      <td>Harold</td>
      <td>2020</td>
      <td>33.0</td>
      <td>162.0</td>
      <td>195.000000</td>
      <td>355</td>
      <td>Ba</td>
      <td>0</td>
      <td>57364</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1756</th>
      <td>1756</td>
      <td>15</td>
      <td>Western Division</td>
      <td>Ba</td>
      <td>Sarai</td>
      <td>2019</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>11.333333</td>
      <td>355</td>
      <td>Ba</td>
      <td>0</td>
      <td>57364</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1816</th>
      <td>1816</td>
      <td>19</td>
      <td>Western Division</td>
      <td>Ba</td>
      <td>Tino</td>
      <td>2019</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>355</td>
      <td>Ba</td>
      <td>0</td>
      <td>57364</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2326</th>
      <td>2326</td>
      <td>56</td>
      <td>Eastern Division</td>
      <td>Lau</td>
      <td>Gita</td>
      <td>2018</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>7.000000</td>
      <td>5331</td>
      <td>Lau</td>
      <td>0</td>
      <td>6835</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2401</th>
      <td>2401</td>
      <td>17</td>
      <td>Eastern Division</td>
      <td>Lau</td>
      <td>Sarai</td>
      <td>2019</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>6.500000</td>
      <td>5331</td>
      <td>Lau</td>
      <td>0</td>
      <td>6835</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2551</th>
      <td>2551</td>
      <td>58</td>
      <td>Eastern Division</td>
      <td>Lau</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>47.0</td>
      <td>174.0</td>
      <td>55.250000</td>
      <td>5331</td>
      <td>Lau</td>
      <td>0</td>
      <td>6835</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2176</th>
      <td>2176</td>
      <td>31</td>
      <td>Eastern Division</td>
      <td>Lau</td>
      <td>Harold</td>
      <td>2020</td>
      <td>108.0</td>
      <td>77.0</td>
      <td>185.000000</td>
      <td>5331</td>
      <td>Lau</td>
      <td>0</td>
      <td>6835</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2476</th>
      <td>2476</td>
      <td>21</td>
      <td>Eastern Division</td>
      <td>Lau</td>
      <td>Tino</td>
      <td>2019</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>Lau</td>
      <td>0</td>
      <td>6835</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2819 rows × 15 columns</p>
</div>




```python
# Select and rename columns in the damage dataset
# drop any rows that don't have a typhoon name
columns_to_keep = {
    "id": "grid_point_id",
    "numbuildings_x": "total_buildings",
    "typhoon": "typhoon_name",
    "total": "total_buildings_damaged",
    "perc_dmg_grid": "perc_dmg_grid"
}

df_damage = (
    df_damage.dropna(subset="typhoon")
    .loc[:, list(columns_to_keep.keys())]
    .rename(columns=columns_to_keep)
)
df_damage["typhoon_name"] = df_damage["typhoon_name"].str.upper()
column_name = "grid_point_id"
df_damage[column_name] = df_damage[column_name].astype(int)

df_damage.sort_values('grid_point_id')
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
      <th>grid_point_id</th>
      <th>total_buildings</th>
      <th>typhoon_name</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1696</th>
      <td>355</td>
      <td>0</td>
      <td>ANA</td>
      <td>248.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1876</th>
      <td>355</td>
      <td>0</td>
      <td>EVAN</td>
      <td>437.666667</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>355</td>
      <td>0</td>
      <td>HAROLD</td>
      <td>195.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1756</th>
      <td>355</td>
      <td>0</td>
      <td>SARAI</td>
      <td>11.333333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1816</th>
      <td>355</td>
      <td>0</td>
      <td>TINO</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2326</th>
      <td>5331</td>
      <td>0</td>
      <td>GITA</td>
      <td>7.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2401</th>
      <td>5331</td>
      <td>0</td>
      <td>SARAI</td>
      <td>6.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2551</th>
      <td>5331</td>
      <td>0</td>
      <td>TOMAS</td>
      <td>55.250000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2176</th>
      <td>5331</td>
      <td>0</td>
      <td>HAROLD</td>
      <td>185.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2476</th>
      <td>5331</td>
      <td>0</td>
      <td>TINO</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2819 rows × 5 columns</p>
</div>




```python
len(df_damage.grid_point_id.unique())
```




    421




```python
# Rename colums in the buildings dataset
df_buildings = df_buildings_raw[['id','numbuildings']].rename({'id':'grid_point_id', 'numbuildings':'total_buildings'},axis=1)
```


```python
df_buildings
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
      <th>grid_point_id</th>
      <th>total_buildings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>355</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>409</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>456</td>
      <td>213</td>
    </tr>
    <tr>
      <th>3</th>
      <td>510</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>511</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>416</th>
      <td>5122</td>
      <td>76</td>
    </tr>
    <tr>
      <th>417</th>
      <td>5123</td>
      <td>0</td>
    </tr>
    <tr>
      <th>418</th>
      <td>5221</td>
      <td>0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>5223</td>
      <td>0</td>
    </tr>
    <tr>
      <th>420</th>
      <td>5331</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 2 columns</p>
</div>



## Read in windfield


```python
# Read in the data file

filename = input_dir / "01_windfield/windfield_data_fji_new.csv"

df_windfield = pd.read_csv(filename)
df_windfield.columns
```




    Index(['Unnamed: 0', 'typhoon_id', 'typhoon_name', 'typhoon_year',
           'grid_point_id', 'wind_speed', 'track_distance', 'geometry'],
          dtype='object')




```python
# Select columns
columns_to_keep = [
    "typhoon_name",
    "typhoon_year",
    "grid_point_id",
    "wind_speed",
    "track_distance",
]
df_windfield = df_windfield.loc[:, columns_to_keep]
df_windfield["typhoon_name"] = df_windfield["typhoon_name"].str.upper()
df_windfield
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
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>355</td>
      <td>11.489542</td>
      <td>297.755905</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>409</td>
      <td>16.290230</td>
      <td>150.425391</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>456</td>
      <td>12.035214</td>
      <td>286.692573</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>510</td>
      <td>16.976736</td>
      <td>148.319961</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>511</td>
      <td>16.959047</td>
      <td>159.228640</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5122</td>
      <td>0.000000</td>
      <td>340.122623</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5123</td>
      <td>0.000000</td>
      <td>335.597712</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5221</td>
      <td>0.000000</td>
      <td>350.361565</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5223</td>
      <td>0.000000</td>
      <td>350.269410</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5331</td>
      <td>0.000000</td>
      <td>335.268400</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 5 columns</p>
</div>




```python
len(df_windfield.grid_point_id.unique())
```




    421



## Read in rainfall


```python
filename = input_dir / "03_rainfall/output/rainfall_data_rw_mean.csv"
df_rainfall = pd.read_csv(filename)
df_rainfall[["typhoon_name", "typhoon_year"]] = df_rainfall[
    "typhoon"
].str.split("(\d+)", expand=True)[[0, 1]]
df_rainfall["typhoon_name"] = df_rainfall["typhoon_name"].str.upper()
df_rainfall["typhoon_year"] = df_rainfall["typhoon_year"].astype(int)
df_rainfall = df_rainfall.rename(columns={"id": "grid_point_id"}).loc[
    :,
    [
        "typhoon_name",
        "typhoon_year",
        "grid_point_id",
        "rainfall_max_6h",
        "rainfall_max_24h",
        "Centroid"
    ],
]
df_rainfall
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
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>355</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>409</td>
      <td>1.866667</td>
      <td>0.612500</td>
      <td>176.95E_-12.45N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>456</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.95E_-17.15N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>510</td>
      <td>3.241667</td>
      <td>1.010417</td>
      <td>177.05E_-12.45N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TOMAS</td>
      <td>2010</td>
      <td>511</td>
      <td>2.366667</td>
      <td>0.806250</td>
      <td>177.05E_-12.55N</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5122</td>
      <td>0.666667</td>
      <td>0.362500</td>
      <td>181.55E_-19.15N</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5123</td>
      <td>0.766667</td>
      <td>0.422917</td>
      <td>181.55E_-19.25N</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5221</td>
      <td>0.675000</td>
      <td>0.360417</td>
      <td>181.65E_-18.95N</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5223</td>
      <td>0.900000</td>
      <td>0.437500</td>
      <td>181.65E_-19.15N</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>ANA</td>
      <td>2021</td>
      <td>5331</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 6 columns</p>
</div>




```python
len(df_rainfall.grid_point_id.unique())
```




    421



## Check grid points ids matches


```python
wind_ids = df_windfield.grid_point_id.unique()
house_ids = df_damage.grid_point_id.unique()
rain_ids = df_rainfall.grid_point_id.unique()
```


```python
len(set(wind_ids) & set(house_ids))
```




    421




```python
len(set(rain_ids) & set(house_ids))
```




    421



Everything is ok!!!!!

## Read in IWI and topography


```python
filename_iwi = input_dir / "05_vulnerablility/output/fji_rwi_bygrid_new.csv"
df_iwi = pd.read_csv(filename_iwi)
```


```python
filename_topo = input_dir / "04_topography/output/topography_variables_bygrid_new.csv"
df_topo = pd.read_csv(filename_topo)
```

## Merge the datasets


```python
# Windfield and Rainfall are COMPLETE datasets. Damage has just damage info in the cells where are in fact damage.
index = ["typhoon_name", "grid_point_id"]
object_list = [df_damage, df_rainfall]

# First merge rainfall and damage
df_all = pd.concat(
    objs=[aux.set_index(index) for aux in object_list], axis=1, join="outer"
)

# Now windfield
df_all = df_windfield.set_index(index).merge(
    df_all, left_index=True, right_index=True, how="left"
)

# Now merge the IWI, buildings and Topological data
df_all.reset_index(inplace=True)
merge1 = df_all.merge(df_iwi, on='grid_point_id', how='left')
merge2 = merge1.merge(df_buildings, on='grid_point_id', how='left')
df_all = merge2.merge(df_topo, left_on='grid_point_id', right_on='id').set_index(index)
df_all = df_all.rename({'total_buildings_y':'total_buildings'}, axis=1)
```


```python
df_all
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
      <th></th>
      <th>typhoon_year_x</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year_y</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
    </tr>
    <tr>
      <th>typhoon_name</th>
      <th>grid_point_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOMAS</th>
      <th>355</th>
      <td>2010</td>
      <td>11.489542</td>
      <td>297.755905</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2010</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>EVAN</th>
      <th>355</th>
      <td>2012</td>
      <td>39.054266</td>
      <td>60.865460</td>
      <td>0.0</td>
      <td>437.666667</td>
      <td>0.0</td>
      <td>2012</td>
      <td>0.141667</td>
      <td>0.070833</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>WINSTON</th>
      <th>355</th>
      <td>2016</td>
      <td>51.661172</td>
      <td>40.384660</td>
      <td>0.0</td>
      <td>7735.000000</td>
      <td>0.0</td>
      <td>2016</td>
      <td>0.508333</td>
      <td>0.189583</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>GITA</th>
      <th>355</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>444.116254</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">SARAI</th>
      <th>355</th>
      <td>2019</td>
      <td>20.284731</td>
      <td>122.202046</td>
      <td>0.0</td>
      <td>11.333333</td>
      <td>0.0</td>
      <td>2019</td>
      <td>15.700000</td>
      <td>4.856250</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5331</th>
      <td>2019</td>
      <td>28.861931</td>
      <td>97.172675</td>
      <td>0.0</td>
      <td>6.500000</td>
      <td>0.0</td>
      <td>2019</td>
      <td>2.741667</td>
      <td>1.158333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TINO</th>
      <th>5331</th>
      <td>2020</td>
      <td>9.664967</td>
      <td>252.982657</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2020</td>
      <td>0.233333</td>
      <td>0.058333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>HAROLD</th>
      <th>5331</th>
      <td>2020</td>
      <td>32.093306</td>
      <td>62.764842</td>
      <td>0.0</td>
      <td>185.000000</td>
      <td>0.0</td>
      <td>2020</td>
      <td>0.508333</td>
      <td>0.166667</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>YASA</th>
      <th>5331</th>
      <td>2020</td>
      <td>36.540718</td>
      <td>34.441416</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020</td>
      <td>0.491667</td>
      <td>0.131250</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ANA</th>
      <th>5331</th>
      <td>2021</td>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2021</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 17 columns</p>
</div>



## Complete df


```python
# Assume all zeros
df = df_all.fillna(0)
```


```python
df_complete = df.copy()
df_complete.reset_index(inplace=True)
df_complete
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
      <th>typhoon_name</th>
      <th>grid_point_id</th>
      <th>typhoon_year_x</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year_y</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TOMAS</td>
      <td>355</td>
      <td>2010</td>
      <td>11.489542</td>
      <td>297.755905</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2010</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EVAN</td>
      <td>355</td>
      <td>2012</td>
      <td>39.054266</td>
      <td>60.865460</td>
      <td>0.0</td>
      <td>437.666667</td>
      <td>0.0</td>
      <td>2012</td>
      <td>0.141667</td>
      <td>0.070833</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WINSTON</td>
      <td>355</td>
      <td>2016</td>
      <td>51.661172</td>
      <td>40.384660</td>
      <td>0.0</td>
      <td>7735.000000</td>
      <td>0.0</td>
      <td>2016</td>
      <td>0.508333</td>
      <td>0.189583</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GITA</td>
      <td>355</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>444.116254</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SARAI</td>
      <td>355</td>
      <td>2019</td>
      <td>20.284731</td>
      <td>122.202046</td>
      <td>0.0</td>
      <td>11.333333</td>
      <td>0.0</td>
      <td>2019</td>
      <td>15.700000</td>
      <td>4.856250</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>SARAI</td>
      <td>5331</td>
      <td>2019</td>
      <td>28.861931</td>
      <td>97.172675</td>
      <td>0.0</td>
      <td>6.500000</td>
      <td>0.0</td>
      <td>2019</td>
      <td>2.741667</td>
      <td>1.158333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>TINO</td>
      <td>5331</td>
      <td>2020</td>
      <td>9.664967</td>
      <td>252.982657</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2020</td>
      <td>0.233333</td>
      <td>0.058333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>HAROLD</td>
      <td>5331</td>
      <td>2020</td>
      <td>32.093306</td>
      <td>62.764842</td>
      <td>0.0</td>
      <td>185.000000</td>
      <td>0.0</td>
      <td>2020</td>
      <td>0.508333</td>
      <td>0.166667</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>YASA</td>
      <td>5331</td>
      <td>2020</td>
      <td>36.540718</td>
      <td>34.441416</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2020</td>
      <td>0.491667</td>
      <td>0.131250</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>ANA</td>
      <td>5331</td>
      <td>2021</td>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2021</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 19 columns</p>
</div>




```python
# How many points do we have for each typhoon?
df_complete.groupby('typhoon_name').count()['grid_point_id']
```




    typhoon_name
    ANA        421
    EVAN       421
    GITA       421
    HAROLD     421
    SARAI      421
    TINO       421
    TOMAS      421
    WINSTON    421
    YASA       421
    Name: grid_point_id, dtype: int64



## Incomplete df


```python
# Assume all zeros
df = df_all.fillna(0)
```


```python
# Drop rows with 0 buildings
df = df[df["total_buildings"] != 0]
df_incomplete = df.copy()
```


```python
df.sort_values('perc_dmg_grid', ascending=False)
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
      <th></th>
      <th>typhoon_year_x</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year_y</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
    </tr>
    <tr>
      <th>typhoon_name</th>
      <th>grid_point_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>YASA</th>
      <th>2873</th>
      <td>2020</td>
      <td>38.140583</td>
      <td>36.663163</td>
      <td>9963.0</td>
      <td>4021.0</td>
      <td>9.805340</td>
      <td>2020</td>
      <td>1.825000</td>
      <td>0.595833</td>
      <td>179.35E_-16.45N</td>
      <td>85.4</td>
      <td>9963</td>
      <td>2873</td>
      <td>1</td>
      <td>12306.788561</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WINSTON</th>
      <th>1670</th>
      <td>2016</td>
      <td>67.175189</td>
      <td>10.014419</td>
      <td>5041.0</td>
      <td>3570.0</td>
      <td>8.994541</td>
      <td>2016</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>178.15E_-17.35N</td>
      <td>73.0</td>
      <td>5041</td>
      <td>1670</td>
      <td>1</td>
      <td>49042.568466</td>
      <td>146.172840</td>
      <td>74.163409</td>
    </tr>
    <tr>
      <th>HAROLD</th>
      <th>1687</th>
      <td>2020</td>
      <td>48.011328</td>
      <td>5.389440</td>
      <td>932.0</td>
      <td>1578.0</td>
      <td>5.225792</td>
      <td>2020</td>
      <td>2.633333</td>
      <td>1.279167</td>
      <td>178.15E_-19.05N</td>
      <td>71.4</td>
      <td>932</td>
      <td>1687</td>
      <td>1</td>
      <td>62351.916992</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">WINSTON</th>
      <th>2380</th>
      <td>2016</td>
      <td>74.273432</td>
      <td>30.136886</td>
      <td>1286.0</td>
      <td>1487.0</td>
      <td>4.245975</td>
      <td>2016</td>
      <td>0.083333</td>
      <td>0.022917</td>
      <td>178.85E_-17.65N</td>
      <td>71.4</td>
      <td>1286</td>
      <td>2380</td>
      <td>1</td>
      <td>19932.444313</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>966</th>
      <td>2016</td>
      <td>54.922953</td>
      <td>18.896682</td>
      <td>15721.0</td>
      <td>7735.0</td>
      <td>3.695402</td>
      <td>2016</td>
      <td>0.066667</td>
      <td>0.018750</td>
      <td>177.45E_-17.65N</td>
      <td>86.0</td>
      <td>15721</td>
      <td>966</td>
      <td>1</td>
      <td>24414.711690</td>
      <td>110.060957</td>
      <td>70.804054</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>EVAN</th>
      <th>1979</th>
      <td>2012</td>
      <td>24.711041</td>
      <td>134.059126</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2012</td>
      <td>0.316667</td>
      <td>0.089583</td>
      <td>178.45E_-17.95N</td>
      <td>85.4</td>
      <td>1733</td>
      <td>1979</td>
      <td>0</td>
      <td>0.000000</td>
      <td>84.953704</td>
      <td>29.965563</td>
    </tr>
    <tr>
      <th>TOMAS</th>
      <th>1979</th>
      <td>2010</td>
      <td>24.383370</td>
      <td>117.780218</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2010</td>
      <td>0.125000</td>
      <td>0.041667</td>
      <td>178.45E_-17.95N</td>
      <td>85.4</td>
      <td>1733</td>
      <td>1979</td>
      <td>0</td>
      <td>0.000000</td>
      <td>84.953704</td>
      <td>29.965563</td>
    </tr>
    <tr>
      <th>YASA</th>
      <th>1978</th>
      <td>2020</td>
      <td>12.440303</td>
      <td>147.762756</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.975000</td>
      <td>0.368750</td>
      <td>178.45E_-17.85N</td>
      <td>85.4</td>
      <td>396</td>
      <td>1978</td>
      <td>0</td>
      <td>0.000000</td>
      <td>287.563272</td>
      <td>80.876073</td>
    </tr>
    <tr>
      <th>TINO</th>
      <th>1978</th>
      <td>2020</td>
      <td>0.000000</td>
      <td>314.896066</td>
      <td>396.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>3.208333</td>
      <td>0.952083</td>
      <td>178.45E_-17.85N</td>
      <td>85.4</td>
      <td>396</td>
      <td>1978</td>
      <td>0</td>
      <td>0.000000</td>
      <td>287.563272</td>
      <td>80.876073</td>
    </tr>
    <tr>
      <th>ANA</th>
      <th>5122</th>
      <td>2021</td>
      <td>0.000000</td>
      <td>340.122623</td>
      <td>76.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2021</td>
      <td>0.666667</td>
      <td>0.362500</td>
      <td>181.55E_-19.15N</td>
      <td>71.4</td>
      <td>76</td>
      <td>5122</td>
      <td>1</td>
      <td>49388.931854</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>2979 rows × 17 columns</p>
</div>



OBS: total_buildings_damage is a fractonary number because we are splitting the number of houses destroyed per grid cell.

OBS: Is not a ptoblem that sometimes total_buildings_damaged > total_buildings because total_buildings_damaged is at municipality level and total_buildings is by grid.


```python
# Example
df[df.index.get_level_values('typhoon_name') == 'GITA']
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
      <th></th>
      <th>typhoon_year_x</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year_y</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
    </tr>
    <tr>
      <th>typhoon_name</th>
      <th>grid_point_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">GITA</th>
      <th>456</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>440.974516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.95E_-17.15N</td>
      <td>86.0</td>
      <td>213</td>
      <td>456</td>
      <td>1</td>
      <td>30905.368530</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>561</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>395.189828</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>177.05E_-17.55N</td>
      <td>76.1</td>
      <td>332</td>
      <td>561</td>
      <td>1</td>
      <td>21225.900224</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>562</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>384.519933</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>177.05E_-17.65N</td>
      <td>76.1</td>
      <td>233</td>
      <td>562</td>
      <td>1</td>
      <td>21374.012317</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>658</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>435.466405</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>177.15E_-17.15N</td>
      <td>86.0</td>
      <td>76</td>
      <td>658</td>
      <td>1</td>
      <td>29712.081119</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>659</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>424.645867</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>177.15E_-17.25N</td>
      <td>86.0</td>
      <td>334</td>
      <td>659</td>
      <td>1</td>
      <td>34539.391946</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5016</th>
      <td>2018</td>
      <td>16.239738</td>
      <td>283.300002</td>
      <td>180.0</td>
      <td>7.0</td>
      <td>0.002697</td>
      <td>2018</td>
      <td>0.050000</td>
      <td>0.012500</td>
      <td>181.45E_-18.65N</td>
      <td>71.4</td>
      <td>180</td>
      <td>5016</td>
      <td>1</td>
      <td>15177.090805</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5021</th>
      <td>2018</td>
      <td>22.572718</td>
      <td>227.750024</td>
      <td>183.0</td>
      <td>7.0</td>
      <td>0.002742</td>
      <td>2018</td>
      <td>0.008333</td>
      <td>0.002083</td>
      <td>181.45E_-19.15N</td>
      <td>71.4</td>
      <td>183</td>
      <td>5021</td>
      <td>1</td>
      <td>107203.695162</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5115</th>
      <td>2018</td>
      <td>0.000000</td>
      <td>305.575581</td>
      <td>162.0</td>
      <td>7.0</td>
      <td>0.002427</td>
      <td>2018</td>
      <td>0.016667</td>
      <td>0.004167</td>
      <td>181.55E_-18.45N</td>
      <td>71.4</td>
      <td>162</td>
      <td>5115</td>
      <td>1</td>
      <td>22629.289248</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5117</th>
      <td>2018</td>
      <td>16.095278</td>
      <td>283.359540</td>
      <td>100.0</td>
      <td>7.0</td>
      <td>0.001498</td>
      <td>2018</td>
      <td>0.025000</td>
      <td>0.006250</td>
      <td>181.55E_-18.65N</td>
      <td>71.4</td>
      <td>100</td>
      <td>5117</td>
      <td>1</td>
      <td>11610.403830</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5122</th>
      <td>2018</td>
      <td>22.171682</td>
      <td>227.822819</td>
      <td>76.0</td>
      <td>7.0</td>
      <td>0.001139</td>
      <td>2018</td>
      <td>0.008333</td>
      <td>0.002083</td>
      <td>181.55E_-19.15N</td>
      <td>71.4</td>
      <td>76</td>
      <td>5122</td>
      <td>1</td>
      <td>49388.931854</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>331 rows × 17 columns</p>
</div>




```python
# Spoiler
plt.hist(df.perc_dmg_grid)
plt.title('Distribution of housing damage at grid level')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```



![png](01_collate_data_files/01_collate_data_38_0.png)



## Create stationary dataset of Fiji


```python
df_fji = df_complete[['grid_point_id',
            'IWI',
            'total_buildings',
            'with_coast',
            'coast_length',
            'mean_altitude',
            'mean_slope']].drop_duplicates()
df_fji
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
      <th>grid_point_id</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>355</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>409</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>9835.966285</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>456</td>
      <td>86.0</td>
      <td>213</td>
      <td>1</td>
      <td>30905.368530</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>510</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>19870.176731</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>511</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>32765.248313</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3744</th>
      <td>5122</td>
      <td>71.4</td>
      <td>76</td>
      <td>1</td>
      <td>49388.931854</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3753</th>
      <td>5123</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>10886.461556</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3762</th>
      <td>5221</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>1516.399318</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3771</th>
      <td>5223</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>10173.166160</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3780</th>
      <td>5331</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 7 columns</p>
</div>



## Write out dataset


```python
df_complete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_fji_complete.csv", index=False
)

df_incomplete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_fji.csv", index=False
)

df_fji.reset_index().to_csv(
    output_dir / "fiji_stationary_data.csv", index=False
)
```
