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
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/02_model_features"
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/03_model_input_dataset"
)
```

## Read in buliding damage


```python
# Read in the building damage data
#filename = input_dir / "02_housing_damage/output/building_damage_bygrid_new.csv"
#filename = input_dir / "02_housing_damage/output/building_damage_bygrid_new_rural_urban_bld.csv"
filename = input_dir / "02_housing_damage/output/building_damage_bygrid_new_rural_urban_bld_using_pop.csv"
df_damage = pd.read_csv(filename)

# Read in buildings per grid
#filename = input_dir / "02_housing_damage/output/num_building_bygrid.csv"
#filename = input_dir / "02_housing_damage/output/num_building_bygrid_from_urban_rural.csv"
filename = input_dir / "02_housing_damage/output/num_building_bygrid_from_urban_rural_using_pop.csv"
df_buildings_raw = pd.read_csv(filename)
```


```python
df_damage.sort_values('perc_dmg_grid')
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
      <th>Year</th>
      <th>Totally</th>
      <th>Partially</th>
      <th>total</th>
      <th>id</th>
      <th>numbuildings_x</th>
      <th>numbuildings_y</th>
      <th>numbuildings_z</th>
      <th>frac_bld</th>
      <th>perc_dmg_grid</th>
      <th>numbuildings</th>
      <th>typhoon_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1894</th>
      <td>2018.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2976</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>68.0</td>
      <td>Gita</td>
    </tr>
    <tr>
      <th>2373</th>
      <td>2019.0</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>34.0</td>
      <td>1177</td>
      <td>0.0</td>
      <td>64821.0</td>
      <td>64821.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>2374</th>
      <td>2019.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3680</td>
      <td>0.0</td>
      <td>27368.0</td>
      <td>27368.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>2375</th>
      <td>2019.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3683</td>
      <td>0.0</td>
      <td>27368.0</td>
      <td>27368.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2019.0</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>34.0</td>
      <td>1059</td>
      <td>0.0</td>
      <td>64821.0</td>
      <td>64821.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Sarai</td>
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
    </tr>
    <tr>
      <th>563</th>
      <td>2020.0</td>
      <td>572.0</td>
      <td>1006.0</td>
      <td>1578.0</td>
      <td>1989</td>
      <td>185.0</td>
      <td>1570.0</td>
      <td>1570.0</td>
      <td>0.117834</td>
      <td>11.843482</td>
      <td>185.0</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>554</th>
      <td>2020.0</td>
      <td>572.0</td>
      <td>1006.0</td>
      <td>1578.0</td>
      <td>1687</td>
      <td>206.0</td>
      <td>1570.0</td>
      <td>1570.0</td>
      <td>0.131210</td>
      <td>13.187878</td>
      <td>206.0</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>2020.0</td>
      <td>565.0</td>
      <td>3456.0</td>
      <td>4021.0</td>
      <td>2873</td>
      <td>3715.0</td>
      <td>9841.0</td>
      <td>9841.0</td>
      <td>0.377502</td>
      <td>15.424618</td>
      <td>3715.0</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2016.0</td>
      <td>2813.0</td>
      <td>757.0</td>
      <td>3570.0</td>
      <td>1670</td>
      <td>1693.0</td>
      <td>6148.0</td>
      <td>6148.0</td>
      <td>0.275374</td>
      <td>15.990331</td>
      <td>1693.0</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>501</th>
      <td>2020.0</td>
      <td>572.0</td>
      <td>1006.0</td>
      <td>1578.0</td>
      <td>1587</td>
      <td>348.0</td>
      <td>1570.0</td>
      <td>1570.0</td>
      <td>0.221656</td>
      <td>22.278551</td>
      <td>348.0</td>
      <td>Harold</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 12 columns</p>
</div>




```python
# Select and rename columns in the damage dataset
# drop any rows that don't have a typhoon name
columns_to_keep = {
    "id": "grid_point_id",
    "numbuildings_x": "total_buildings",
    "typhoon_name": "typhoon_name",
    "total": "total_buildings_damaged",
    "perc_dmg_grid": "perc_dmg_grid"
}

df_damage = (
    df_damage.dropna(subset="typhoon_name")
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
      <th>2752</th>
      <td>355</td>
      <td>59.0</td>
      <td>TINO</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>355</td>
      <td>59.0</td>
      <td>YASA</td>
      <td>47.0</td>
      <td>0.000121</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>355</td>
      <td>0.0</td>
      <td>GITA</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>355</td>
      <td>59.0</td>
      <td>ANA</td>
      <td>248.0</td>
      <td>0.000638</td>
    </tr>
    <tr>
      <th>3594</th>
      <td>355</td>
      <td>0.0</td>
      <td>TOMAS</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <th>3185</th>
      <td>5331</td>
      <td>0.0</td>
      <td>EVAN</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>5331</td>
      <td>46.0</td>
      <td>ANA</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2764</th>
      <td>5331</td>
      <td>46.0</td>
      <td>TINO</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3606</th>
      <td>5331</td>
      <td>46.0</td>
      <td>TOMAS</td>
      <td>221.0</td>
      <td>0.015204</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>5331</td>
      <td>46.0</td>
      <td>GITA</td>
      <td>7.0</td>
      <td>0.003748</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 5 columns</p>
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
      <td>1981</td>
      <td>17899.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2081</td>
      <td>14357.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>11833.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>966</td>
      <td>12504.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>967</td>
      <td>7823.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1676</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1675</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>418</th>
      <td>1669</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>3375</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>420</th>
      <td>2369</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 2 columns</p>
</div>



## Read in windfield


```python
# Read in the data file
filename = input_dir / "01_windfield/windfield_data_fji_new_fixed_interpolated_overlap.csv"

df_windfield = pd.read_csv(filename)
df_windfield.columns
```




    Index(['typhoon_name', 'track_id', 'grid_point_id', 'wind_speed',
           'track_distance', 'geometry'],
          dtype='object')




```python
# Select columns
columns_to_keep = [
    "typhoon_name",
#    "typhoon_year",
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TOMAS</td>
      <td>355</td>
      <td>12.577735</td>
      <td>297.755905</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TOMAS</td>
      <td>409</td>
      <td>16.904912</td>
      <td>150.425391</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TOMAS</td>
      <td>456</td>
      <td>13.135196</td>
      <td>286.692573</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TOMAS</td>
      <td>510</td>
      <td>17.564677</td>
      <td>148.319961</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TOMAS</td>
      <td>511</td>
      <td>17.558868</td>
      <td>159.228640</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>ANA</td>
      <td>5122</td>
      <td>0.000000</td>
      <td>340.122623</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>ANA</td>
      <td>5123</td>
      <td>0.000000</td>
      <td>335.597712</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>ANA</td>
      <td>5221</td>
      <td>0.000000</td>
      <td>350.361565</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>ANA</td>
      <td>5223</td>
      <td>0.000000</td>
      <td>350.269410</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>ANA</td>
      <td>5331</td>
      <td>0.000000</td>
      <td>335.268400</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 4 columns</p>
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
filename_iwi = input_dir / "05_vulnerablility/output/fji_iwi_bygrid_new.csv"
df_iwi = pd.read_csv(filename_iwi)
```


```python
filename_topo = input_dir / "04_topography/output/topography_variables_bygrid_new.csv"
df_topo = pd.read_csv(filename_topo).rename({'mean_elev': 'mean_altitude'}, axis=1)
```

## Read in Light index


```python
filename_light = input_dir / "05_vulnerablility/output/light_index.csv"
df_light = pd.read_csv(filename_light).rename({'sum': 'light_index'}, axis=1)
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
merge3 = merge2.merge(df_topo, left_on='grid_point_id', right_on='id')
df_all = merge3.merge(df_light, left_on='grid_point_id', right_on='id').set_index(index)
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id_x</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
      <th>id_y</th>
      <th>light_index</th>
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
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOMAS</th>
      <th>355</th>
      <td>12.577735</td>
      <td>297.755905</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2010</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>EVAN</th>
      <th>355</th>
      <td>40.223381</td>
      <td>60.865460</td>
      <td>59.0</td>
      <td>1313.0</td>
      <td>0.001844</td>
      <td>2012</td>
      <td>0.141667</td>
      <td>0.070833</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>WINSTON</th>
      <th>355</th>
      <td>53.500992</td>
      <td>40.384660</td>
      <td>59.0</td>
      <td>7735.0</td>
      <td>0.019905</td>
      <td>2016</td>
      <td>0.508333</td>
      <td>0.189583</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>GITA</th>
      <th>355</th>
      <td>0.000000</td>
      <td>444.116254</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">SARAI</th>
      <th>355</th>
      <td>19.408615</td>
      <td>122.202046</td>
      <td>59.0</td>
      <td>34.0</td>
      <td>0.000048</td>
      <td>2019</td>
      <td>15.700000</td>
      <td>4.856250</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
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
    </tr>
    <tr>
      <th>5331</th>
      <td>26.329689</td>
      <td>97.172675</td>
      <td>46.0</td>
      <td>26.0</td>
      <td>0.001789</td>
      <td>2019</td>
      <td>2.741667</td>
      <td>1.158333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>TINO</th>
      <th>5331</th>
      <td>11.814649</td>
      <td>252.982657</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.233333</td>
      <td>0.058333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>HAROLD</th>
      <th>5331</th>
      <td>37.333507</td>
      <td>62.764842</td>
      <td>46.0</td>
      <td>185.0</td>
      <td>0.099060</td>
      <td>2020</td>
      <td>0.508333</td>
      <td>0.166667</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>YASA</th>
      <th>5331</th>
      <td>36.352948</td>
      <td>34.441416</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.491667</td>
      <td>0.131250</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>ANA</th>
      <th>5331</th>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2021</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 18 columns</p>
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id_x</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
      <th>id_y</th>
      <th>light_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TOMAS</td>
      <td>355</td>
      <td>12.577735</td>
      <td>297.755905</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2010</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EVAN</td>
      <td>355</td>
      <td>40.223381</td>
      <td>60.865460</td>
      <td>59.0</td>
      <td>1313.0</td>
      <td>0.001844</td>
      <td>2012</td>
      <td>0.141667</td>
      <td>0.070833</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WINSTON</td>
      <td>355</td>
      <td>53.500992</td>
      <td>40.384660</td>
      <td>59.0</td>
      <td>7735.0</td>
      <td>0.019905</td>
      <td>2016</td>
      <td>0.508333</td>
      <td>0.189583</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GITA</td>
      <td>355</td>
      <td>0.000000</td>
      <td>444.116254</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SARAI</td>
      <td>355</td>
      <td>19.408615</td>
      <td>122.202046</td>
      <td>59.0</td>
      <td>34.0</td>
      <td>0.000048</td>
      <td>2019</td>
      <td>15.700000</td>
      <td>4.856250</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>SARAI</td>
      <td>5331</td>
      <td>26.329689</td>
      <td>97.172675</td>
      <td>46.0</td>
      <td>26.0</td>
      <td>0.001789</td>
      <td>2019</td>
      <td>2.741667</td>
      <td>1.158333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>TINO</td>
      <td>5331</td>
      <td>11.814649</td>
      <td>252.982657</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.233333</td>
      <td>0.058333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>HAROLD</td>
      <td>5331</td>
      <td>37.333507</td>
      <td>62.764842</td>
      <td>46.0</td>
      <td>185.0</td>
      <td>0.099060</td>
      <td>2020</td>
      <td>0.508333</td>
      <td>0.166667</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>YASA</td>
      <td>5331</td>
      <td>36.352948</td>
      <td>34.441416</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.491667</td>
      <td>0.131250</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>ANA</td>
      <td>5331</td>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2021</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 20 columns</p>
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id_x</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
      <th>id_y</th>
      <th>light_index</th>
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
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HAROLD</th>
      <th>1587</th>
      <td>43.046120</td>
      <td>19.785665</td>
      <td>348.0</td>
      <td>1578.0</td>
      <td>22.278551</td>
      <td>2020</td>
      <td>2.541667</td>
      <td>1.177083</td>
      <td>178.05E_-19.15N</td>
      <td>71.4</td>
      <td>348.0</td>
      <td>1587</td>
      <td>1</td>
      <td>43033.688427</td>
      <td>99.072651</td>
      <td>8.655745</td>
      <td>1587</td>
      <td>626.567505</td>
    </tr>
    <tr>
      <th>WINSTON</th>
      <th>1670</th>
      <td>69.374838</td>
      <td>10.014419</td>
      <td>1693.0</td>
      <td>3570.0</td>
      <td>15.990331</td>
      <td>2016</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>178.15E_-17.35N</td>
      <td>73.0</td>
      <td>1693.0</td>
      <td>1670</td>
      <td>1</td>
      <td>49042.568466</td>
      <td>37.162737</td>
      <td>5.577366</td>
      <td>1670</td>
      <td>1043.006104</td>
    </tr>
    <tr>
      <th>YASA</th>
      <th>2873</th>
      <td>41.398814</td>
      <td>36.663163</td>
      <td>3715.0</td>
      <td>4021.0</td>
      <td>15.424618</td>
      <td>2020</td>
      <td>1.825000</td>
      <td>0.595833</td>
      <td>179.35E_-16.45N</td>
      <td>85.4</td>
      <td>3715.0</td>
      <td>2873</td>
      <td>1</td>
      <td>12306.788561</td>
      <td>29.891967</td>
      <td>4.919905</td>
      <td>2873</td>
      <td>1007.199585</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">HAROLD</th>
      <th>1687</th>
      <td>44.796041</td>
      <td>5.389440</td>
      <td>206.0</td>
      <td>1578.0</td>
      <td>13.187878</td>
      <td>2020</td>
      <td>2.633333</td>
      <td>1.279167</td>
      <td>178.15E_-19.05N</td>
      <td>71.4</td>
      <td>206.0</td>
      <td>1687</td>
      <td>1</td>
      <td>62351.916992</td>
      <td>76.218944</td>
      <td>7.520840</td>
      <td>1687</td>
      <td>590.369080</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>48.225127</td>
      <td>17.109128</td>
      <td>185.0</td>
      <td>1578.0</td>
      <td>11.843482</td>
      <td>2020</td>
      <td>3.783333</td>
      <td>1.329167</td>
      <td>178.45E_-18.95N</td>
      <td>71.4</td>
      <td>185.0</td>
      <td>1989</td>
      <td>1</td>
      <td>60343.175054</td>
      <td>-11.114740</td>
      <td>6.383458</td>
      <td>1989</td>
      <td>522.173706</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>TINO</th>
      <th>1878</th>
      <td>0.000000</td>
      <td>329.088673</td>
      <td>414.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>3.266667</td>
      <td>1.383333</td>
      <td>178.35E_-17.95N</td>
      <td>85.4</td>
      <td>414.0</td>
      <td>1878</td>
      <td>0</td>
      <td>0.000000</td>
      <td>70.774618</td>
      <td>8.569301</td>
      <td>1878</td>
      <td>733.683228</td>
    </tr>
    <tr>
      <th>GITA</th>
      <th>1878</th>
      <td>0.000000</td>
      <td>328.168371</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.008333</td>
      <td>0.002083</td>
      <td>178.35E_-17.95N</td>
      <td>85.4</td>
      <td>414.0</td>
      <td>1878</td>
      <td>0</td>
      <td>0.000000</td>
      <td>70.774618</td>
      <td>8.569301</td>
      <td>1878</td>
      <td>733.683228</td>
    </tr>
    <tr>
      <th>EVAN</th>
      <th>1878</th>
      <td>26.500437</td>
      <td>125.601902</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2012</td>
      <td>0.475000</td>
      <td>0.206250</td>
      <td>178.35E_-17.95N</td>
      <td>85.4</td>
      <td>414.0</td>
      <td>1878</td>
      <td>0</td>
      <td>0.000000</td>
      <td>70.774618</td>
      <td>8.569301</td>
      <td>1878</td>
      <td>733.683228</td>
    </tr>
    <tr>
      <th>TOMAS</th>
      <th>1878</th>
      <td>25.055829</td>
      <td>128.885112</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2010</td>
      <td>0.283333</td>
      <td>0.072917</td>
      <td>178.35E_-17.95N</td>
      <td>85.4</td>
      <td>414.0</td>
      <td>1878</td>
      <td>0</td>
      <td>0.000000</td>
      <td>70.774618</td>
      <td>8.569301</td>
      <td>1878</td>
      <td>733.683228</td>
    </tr>
    <tr>
      <th>ANA</th>
      <th>5331</th>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2021</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
  </tbody>
</table>
<p>2232 rows × 18 columns</p>
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id_x</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
      <th>id_y</th>
      <th>light_index</th>
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
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">GITA</th>
      <th>355</th>
      <td>0.000000</td>
      <td>444.116254</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>510</th>
      <td>0.000000</td>
      <td>831.285250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>1.650000</td>
      <td>0.412500</td>
      <td>177.05E_-12.45N</td>
      <td>71.4</td>
      <td>73.0</td>
      <td>510</td>
      <td>1</td>
      <td>19870.176731</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>510</td>
      <td>2067.388184</td>
    </tr>
    <tr>
      <th>511</th>
      <td>0.000000</td>
      <td>828.608237</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.983333</td>
      <td>0.245833</td>
      <td>177.05E_-12.55N</td>
      <td>71.4</td>
      <td>245.0</td>
      <td>511</td>
      <td>1</td>
      <td>32765.248313</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>511</td>
      <td>1773.442383</td>
    </tr>
    <tr>
      <th>558</th>
      <td>0.000000</td>
      <td>427.334117</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>177.05E_-17.25N</td>
      <td>86.0</td>
      <td>78.0</td>
      <td>558</td>
      <td>1</td>
      <td>1127.174596</td>
      <td>0.011073</td>
      <td>0.010481</td>
      <td>558</td>
      <td>649.564270</td>
    </tr>
    <tr>
      <th>562</th>
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
      <td>131.0</td>
      <td>562</td>
      <td>1</td>
      <td>21374.012317</td>
      <td>1.014349</td>
      <td>0.431089</td>
      <td>562</td>
      <td>1318.218140</td>
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
    </tr>
    <tr>
      <th>5016</th>
      <td>19.569000</td>
      <td>283.300002</td>
      <td>102.0</td>
      <td>7.0</td>
      <td>0.008311</td>
      <td>2018</td>
      <td>0.050000</td>
      <td>0.012500</td>
      <td>181.45E_-18.65N</td>
      <td>71.4</td>
      <td>102.0</td>
      <td>5016</td>
      <td>1</td>
      <td>15177.090805</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5016</td>
      <td>840.377014</td>
    </tr>
    <tr>
      <th>5020</th>
      <td>23.265902</td>
      <td>238.860019</td>
      <td>58.0</td>
      <td>7.0</td>
      <td>0.004726</td>
      <td>2018</td>
      <td>0.008333</td>
      <td>0.002083</td>
      <td>181.45E_-19.05N</td>
      <td>71.4</td>
      <td>58.0</td>
      <td>5020</td>
      <td>1</td>
      <td>1782.807607</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5020</td>
      <td>778.695312</td>
    </tr>
    <tr>
      <th>5115</th>
      <td>0.000000</td>
      <td>305.575581</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>0.002689</td>
      <td>2018</td>
      <td>0.016667</td>
      <td>0.004167</td>
      <td>181.55E_-18.45N</td>
      <td>71.4</td>
      <td>33.0</td>
      <td>5115</td>
      <td>1</td>
      <td>22629.289248</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5115</td>
      <td>799.209534</td>
    </tr>
    <tr>
      <th>5122</th>
      <td>23.746221</td>
      <td>227.822819</td>
      <td>20.0</td>
      <td>7.0</td>
      <td>0.001630</td>
      <td>2018</td>
      <td>0.008333</td>
      <td>0.002083</td>
      <td>181.55E_-19.15N</td>
      <td>71.4</td>
      <td>20.0</td>
      <td>5122</td>
      <td>1</td>
      <td>49388.931854</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5122</td>
      <td>837.122009</td>
    </tr>
    <tr>
      <th>5331</th>
      <td>29.414019</td>
      <td>151.285088</td>
      <td>46.0</td>
      <td>7.0</td>
      <td>0.003748</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 18 columns</p>
</div>




```python
# Spoiler
plt.hist(df.perc_dmg_grid)
plt.title('Distribution of housing damage at grid level')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```



![png](01_collate_data_files/01_collate_data_40_0.png)



## Create stationary dataset of Fiji


```python
df_fji = df_complete[[
            'grid_point_id',
            'IWI',
            'light_index',
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
      <th>light_index</th>
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
      <td>789.968262</td>
      <td>59.0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
    </tr>
    <tr>
      <th>9</th>
      <td>409</td>
      <td>71.4</td>
      <td>1721.661255</td>
      <td>0.0</td>
      <td>1</td>
      <td>9835.966285</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>456</td>
      <td>86.0</td>
      <td>574.940063</td>
      <td>0.0</td>
      <td>1</td>
      <td>30905.368530</td>
      <td>0.465151</td>
      <td>0.128264</td>
    </tr>
    <tr>
      <th>27</th>
      <td>510</td>
      <td>71.4</td>
      <td>2067.388184</td>
      <td>73.0</td>
      <td>1</td>
      <td>19870.176731</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>511</td>
      <td>71.4</td>
      <td>1773.442383</td>
      <td>245.0</td>
      <td>1</td>
      <td>32765.248313</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>3744</th>
      <td>5122</td>
      <td>71.4</td>
      <td>837.122009</td>
      <td>20.0</td>
      <td>1</td>
      <td>49388.931854</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3753</th>
      <td>5123</td>
      <td>71.4</td>
      <td>1286.776855</td>
      <td>0.0</td>
      <td>1</td>
      <td>10886.461556</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3762</th>
      <td>5221</td>
      <td>71.4</td>
      <td>578.233887</td>
      <td>0.0</td>
      <td>1</td>
      <td>1516.399318</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3771</th>
      <td>5223</td>
      <td>71.4</td>
      <td>823.425171</td>
      <td>0.0</td>
      <td>1</td>
      <td>10173.166160</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3780</th>
      <td>5331</td>
      <td>71.4</td>
      <td>1243.329590</td>
      <td>46.0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 8 columns</p>
</div>




```python
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_buildings_x</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>typhoon_year</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>Centroid</th>
      <th>IWI</th>
      <th>total_buildings</th>
      <th>id_x</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
      <th>id_y</th>
      <th>light_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TOMAS</td>
      <td>355</td>
      <td>12.577735</td>
      <td>297.755905</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2010</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EVAN</td>
      <td>355</td>
      <td>40.223381</td>
      <td>60.865460</td>
      <td>59.0</td>
      <td>1313.0</td>
      <td>0.001844</td>
      <td>2012</td>
      <td>0.141667</td>
      <td>0.070833</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WINSTON</td>
      <td>355</td>
      <td>53.500992</td>
      <td>40.384660</td>
      <td>59.0</td>
      <td>7735.0</td>
      <td>0.019905</td>
      <td>2016</td>
      <td>0.508333</td>
      <td>0.189583</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GITA</td>
      <td>355</td>
      <td>0.000000</td>
      <td>444.116254</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2018</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SARAI</td>
      <td>355</td>
      <td>19.408615</td>
      <td>122.202046</td>
      <td>59.0</td>
      <td>34.0</td>
      <td>0.000048</td>
      <td>2019</td>
      <td>15.700000</td>
      <td>4.856250</td>
      <td>176.85E_-17.15N</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>355</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.000084</td>
      <td>0.000188</td>
      <td>355</td>
      <td>789.968262</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>3784</th>
      <td>SARAI</td>
      <td>5331</td>
      <td>26.329689</td>
      <td>97.172675</td>
      <td>46.0</td>
      <td>26.0</td>
      <td>0.001789</td>
      <td>2019</td>
      <td>2.741667</td>
      <td>1.158333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>TINO</td>
      <td>5331</td>
      <td>11.814649</td>
      <td>252.982657</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.233333</td>
      <td>0.058333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>HAROLD</td>
      <td>5331</td>
      <td>37.333507</td>
      <td>62.764842</td>
      <td>46.0</td>
      <td>185.0</td>
      <td>0.099060</td>
      <td>2020</td>
      <td>0.508333</td>
      <td>0.166667</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>YASA</td>
      <td>5331</td>
      <td>36.352948</td>
      <td>34.441416</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2020</td>
      <td>0.491667</td>
      <td>0.131250</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>ANA</td>
      <td>5331</td>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2021</td>
      <td>1.483333</td>
      <td>0.783333</td>
      <td>181.75E_-19.85N</td>
      <td>71.4</td>
      <td>46.0</td>
      <td>5331</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5331</td>
      <td>1243.329590</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 20 columns</p>
</div>



## Write out dataset


```python
df_complete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_fji_complete_interpolated_wind_new_bld_count_using_pop.csv", index=False
)

df_incomplete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_fji_interpolated_wind_new_bld_count_using_pop.csv", index=False
)

df_fji.reset_index().to_csv(
    output_dir / "fiji_stationary_data_interpolated_wind_new_bld_count_using_pop.csv", index=False
)
```
