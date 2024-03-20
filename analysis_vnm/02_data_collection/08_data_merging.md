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
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_vnm/02_model_features"
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_vnm/03_model_input_dataset"
)
```

## Read in typhoons ids


```python
typhoons_ids = pd.read_csv(input_dir / "01_windfield/typhoons.csv")
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
# Rename some columns
df_damage = df_damage.rename({
    "id": "grid_point_id",
    "numbuildings": "total_buildings",
    "total_bld_dmg": "total_buildings_damaged",
}, axis=1)
df_damage['typhoon_name'] = df_damage['typhoon_name'].str.upper()
df_damage.Year = df_damage.Year.astype('int64')
df_damage

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
      <th>Year</th>
      <th>total_buildings_damaged</th>
      <th>region_affected</th>
      <th>grid_point_id</th>
      <th>Region_x</th>
      <th>numbuildings_x</th>
      <th>numbuildings_y</th>
      <th>numbuildings_z</th>
      <th>frac_bld</th>
      <th>perc_dmg_grid</th>
      <th>Region</th>
      <th>total_buildings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LINDA</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>2007</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NW</td>
      <td>333.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LINDA</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>2008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NW</td>
      <td>294.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NW</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LINDA</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>2187</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NW</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LINDA</td>
      <td>1997</td>
      <td>0.0</td>
      <td>0</td>
      <td>2188</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NW</td>
      <td>313.0</td>
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
    </tr>
    <tr>
      <th>128839</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>3321.0</td>
      <td>SC</td>
      <td>30180</td>
      <td>SC</td>
      <td>1.0</td>
      <td>7991119.0</td>
      <td>12306771.0</td>
      <td>8.125608e-08</td>
      <td>2.192707e-09</td>
      <td>SC</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>128840</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>3321.0</td>
      <td>SC</td>
      <td>30359</td>
      <td>SC</td>
      <td>0.0</td>
      <td>7991119.0</td>
      <td>12306771.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>SC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>128841</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>3321.0</td>
      <td>SC</td>
      <td>30360</td>
      <td>SC</td>
      <td>0.0</td>
      <td>7991119.0</td>
      <td>12306771.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>SC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>128842</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>3321.0</td>
      <td>SC</td>
      <td>30540</td>
      <td>SC</td>
      <td>0.0</td>
      <td>7991119.0</td>
      <td>12306771.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>SC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>128843</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>3321.0</td>
      <td>SC</td>
      <td>30541</td>
      <td>SC</td>
      <td>1.0</td>
      <td>7991119.0</td>
      <td>12306771.0</td>
      <td>8.125608e-08</td>
      <td>2.192707e-09</td>
      <td>SC</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>128844 rows × 13 columns</p>
</div>




```python
# Rename colums in the buildings dataset
df_buildings = df_buildings_raw[['id','numbuildings']].rename({'id':'grid_point_id', 'numbuildings':'total_buildings'},axis=1)
```

## Read in windfield


```python
# Read in the data file
filename = input_dir / "01_windfield/windfield_data_viet_new_fixed_interpolated_overlap.csv"

df_windfield = pd.read_csv(filename)
df_windfield = df_windfield.drop('geometry', axis=1)
df_windfield = df_windfield.drop_duplicates()
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
      <th>track_id</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LINDA</td>
      <td>1997298N06140</td>
      <td>2007</td>
      <td>0.0</td>
      <td>1161.028073</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LINDA</td>
      <td>1997298N06140</td>
      <td>2008</td>
      <td>0.0</td>
      <td>1151.966170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>1997298N06140</td>
      <td>2009</td>
      <td>0.0</td>
      <td>1142.904268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LINDA</td>
      <td>1997298N06140</td>
      <td>2187</td>
      <td>0.0</td>
      <td>1176.517574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LINDA</td>
      <td>1997298N06140</td>
      <td>2188</td>
      <td>0.0</td>
      <td>1167.455671</td>
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
      <th>135997</th>
      <td>BULBUL:MATMO</td>
      <td>2019302N11118</td>
      <td>30180</td>
      <td>0.0</td>
      <td>633.611165</td>
    </tr>
    <tr>
      <th>135998</th>
      <td>BULBUL:MATMO</td>
      <td>2019302N11118</td>
      <td>30359</td>
      <td>0.0</td>
      <td>633.611167</td>
    </tr>
    <tr>
      <th>135999</th>
      <td>BULBUL:MATMO</td>
      <td>2019302N11118</td>
      <td>30360</td>
      <td>0.0</td>
      <td>638.462776</td>
    </tr>
    <tr>
      <th>136000</th>
      <td>BULBUL:MATMO</td>
      <td>2019302N11118</td>
      <td>30540</td>
      <td>0.0</td>
      <td>643.661443</td>
    </tr>
    <tr>
      <th>136001</th>
      <td>BULBUL:MATMO</td>
      <td>2019302N11118</td>
      <td>30541</td>
      <td>0.0</td>
      <td>648.437860</td>
    </tr>
  </tbody>
</table>
<p>128844 rows × 5 columns</p>
</div>




```python
len(df_windfield.grid_point_id.unique())
```




    3579



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
df_rainfall = df_rainfall.drop('Centroid', axis=1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2007</td>
      <td>1.833333</td>
      <td>0.741667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2008</td>
      <td>2.050000</td>
      <td>0.870833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2009</td>
      <td>3.441667</td>
      <td>1.152083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2187</td>
      <td>2.100000</td>
      <td>0.856250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2188</td>
      <td>2.566667</td>
      <td>0.935417</td>
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
      <th>125260</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30180</td>
      <td>5.958333</td>
      <td>3.097917</td>
    </tr>
    <tr>
      <th>125261</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30359</td>
      <td>8.891667</td>
      <td>3.091667</td>
    </tr>
    <tr>
      <th>125262</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30360</td>
      <td>5.550000</td>
      <td>2.656250</td>
    </tr>
    <tr>
      <th>125263</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30540</td>
      <td>9.341667</td>
      <td>3.193750</td>
    </tr>
    <tr>
      <th>125264</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30541</td>
      <td>6.641667</td>
      <td>2.818750</td>
    </tr>
  </tbody>
</table>
<p>125265 rows × 5 columns</p>
</div>



Rainfall does not include LINDA1997


```python
len(df_rainfall.grid_point_id.unique())
```




    3579




```python
set(df_windfield.typhoon_name.unique()) - set(df_rainfall.typhoon_name.unique())
```




    {'LINDA'}



## Check grid points ids matches


```python
wind_ids = df_windfield.grid_point_id.unique()
house_ids = df_damage.grid_point_id.unique()
rain_ids = df_rainfall.grid_point_id.unique()
```


```python
len(set(wind_ids) & set(house_ids))
```




    3579




```python
len(set(rain_ids) & set(house_ids))
```




    3579



Everything is ok!!!!!

## Read in IWI


```python
filename_iwi = input_dir / "05_vulnerability/output/viet_iwi_bygrid_new.csv"
df_iwi = pd.read_csv(filename_iwi)
```

## Read in RWI


```python
filename_RWI = input_dir / "05_vulnerability/output/viet_rwi_bygrid.csv"
df_rwi = pd.read_csv(filename_RWI)
```

## Read in topography


```python
filename_topo = input_dir / "04_topography/output/topography_variables_bygrid_new.csv"
df_topo = pd.read_csv(filename_topo)
```

## Read in Light index


```python
filename_light = input_dir / "05_vulnerability/output/light_index.csv"
df_light = pd.read_csv(filename_light).rename({'sum': 'light_index'}, axis=1)
```

## Merge the datasets


```python
# Merge windfield and ids
df_windfield_total = typhoons_ids.merge(df_windfield, left_on='id', right_on='track_id')[[
    'typhoon_name_y', 'Year', 'grid_point_id', 'wind_speed', 'track_distance'
]]
df_windfield_total


# Merge rainfall and windfield
df_rainfall_total = df_rainfall.rename({
    'typhoon_name': 'typhoon_name_y',
    'typhoon_year': 'Year'
}, axis=1)

df_merge_wind_rain = df_windfield_total.merge(df_rainfall_total, on=['typhoon_name_y', 'Year', 'grid_point_id'])
df_merge_wind_rain = df_merge_wind_rain.rename({'typhoon_name_y': 'typhoon_name'}, axis=1)

# Merge with damage
df_damage_total = df_damage[['typhoon_name', 'Year', 'grid_point_id', 'total_buildings_damaged', 'perc_dmg_grid', 'total_buildings' ]]
df_merge_typhoons = df_merge_wind_rain.merge(df_damage_total, on=['typhoon_name', 'Year', 'grid_point_id']).drop_duplicates()

# Merge with topography data
df_topo_total = df_topo.rename({'id':'grid_point_id'}, axis=1)
df_merge_topo = df_merge_typhoons.merge(df_topo_total, on='grid_point_id')

# Merge with vulnerability indexes
df_vul = df_iwi.merge(df_light,left_on='grid_point_id', right_on='id')
df_vul_rwi = df_rwi[['id', 'scaled_distance', 'rwi']].merge(df_vul, left_on='id', right_on='grid_point_id')
df_all = df_merge_topo.merge(df_vul_rwi, on='grid_point_id')
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
      <th>typhoon_name</th>
      <th>Year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>total_buildings</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
      <th>id_x</th>
      <th>scaled_distance</th>
      <th>rwi</th>
      <th>IWI</th>
      <th>id_y</th>
      <th>light_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2007</td>
      <td>0.0</td>
      <td>395.069708</td>
      <td>1.833333</td>
      <td>0.741667</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>XANGSANE</td>
      <td>2006</td>
      <td>2007</td>
      <td>0.0</td>
      <td>837.203208</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>2007</td>
      <td>0.0</td>
      <td>1484.443184</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LEKIMA</td>
      <td>2007</td>
      <td>2007</td>
      <td>0.0</td>
      <td>576.704212</td>
      <td>4.158333</td>
      <td>1.710417</td>
      <td>113623.0</td>
      <td>1.181028e-05</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KAMMURI</td>
      <td>2008</td>
      <td>2007</td>
      <td>0.0</td>
      <td>540.464849</td>
      <td>6.183333</td>
      <td>2.456250</td>
      <td>26635.0</td>
      <td>1.312183e-05</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
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
      <th>125260</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>30541</td>
      <td>0.0</td>
      <td>1115.573644</td>
      <td>4.741667</td>
      <td>2.204167</td>
      <td>3321.0</td>
      <td>2.192707e-09</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125261</th>
      <td>SON-TINH</td>
      <td>2012</td>
      <td>30541</td>
      <td>0.0</td>
      <td>399.816218</td>
      <td>0.708333</td>
      <td>0.408333</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125262</th>
      <td>KAI-TAK</td>
      <td>2012</td>
      <td>30541</td>
      <td>0.0</td>
      <td>895.199828</td>
      <td>1.225000</td>
      <td>0.527083</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125263</th>
      <td>PODUL</td>
      <td>2013</td>
      <td>30541</td>
      <td>0.0</td>
      <td>695.108199</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3067.0</td>
      <td>2.025002e-09</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125264</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30541</td>
      <td>0.0</td>
      <td>648.437860</td>
      <td>6.641667</td>
      <td>2.818750</td>
      <td>1486.0</td>
      <td>9.811391e-10</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
  </tbody>
</table>
<p>125265 rows × 20 columns</p>
</div>



## Complete df


```python
# Assume all zeros
df = df_all.fillna(0)
```


```python
df_complete = df.copy()
df_complete.reset_index(drop=True)
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
      <th>Year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>total_buildings</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
      <th>id_x</th>
      <th>scaled_distance</th>
      <th>rwi</th>
      <th>IWI</th>
      <th>id_y</th>
      <th>light_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAMREY</td>
      <td>2005</td>
      <td>2007</td>
      <td>0.0</td>
      <td>395.069708</td>
      <td>1.833333</td>
      <td>0.741667</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>XANGSANE</td>
      <td>2006</td>
      <td>2007</td>
      <td>0.0</td>
      <td>837.203208</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>2007</td>
      <td>0.0</td>
      <td>1484.443184</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LEKIMA</td>
      <td>2007</td>
      <td>2007</td>
      <td>0.0</td>
      <td>576.704212</td>
      <td>4.158333</td>
      <td>1.710417</td>
      <td>113623.0</td>
      <td>1.181028e-05</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KAMMURI</td>
      <td>2008</td>
      <td>2007</td>
      <td>0.0</td>
      <td>540.464849</td>
      <td>6.183333</td>
      <td>2.456250</td>
      <td>26635.0</td>
      <td>1.312183e-05</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
      <td>2007</td>
      <td>0.444038</td>
      <td>0.371000</td>
      <td>74.2</td>
      <td>2007</td>
      <td>589.62085</td>
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
      <th>125260</th>
      <td>SINLAKU</td>
      <td>2020</td>
      <td>30541</td>
      <td>0.0</td>
      <td>1115.573644</td>
      <td>4.741667</td>
      <td>2.204167</td>
      <td>3321.0</td>
      <td>2.192707e-09</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125261</th>
      <td>SON-TINH</td>
      <td>2012</td>
      <td>30541</td>
      <td>0.0</td>
      <td>399.816218</td>
      <td>0.708333</td>
      <td>0.408333</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125262</th>
      <td>KAI-TAK</td>
      <td>2012</td>
      <td>30541</td>
      <td>0.0</td>
      <td>895.199828</td>
      <td>1.225000</td>
      <td>0.527083</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125263</th>
      <td>PODUL</td>
      <td>2013</td>
      <td>30541</td>
      <td>0.0</td>
      <td>695.108199</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3067.0</td>
      <td>2.025002e-09</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
    <tr>
      <th>125264</th>
      <td>BULBUL:MATMO</td>
      <td>2019</td>
      <td>30541</td>
      <td>0.0</td>
      <td>648.437860</td>
      <td>6.641667</td>
      <td>2.818750</td>
      <td>1486.0</td>
      <td>9.811391e-10</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30541</td>
      <td>0.607827</td>
      <td>0.003629</td>
      <td>80.9</td>
      <td>30541</td>
      <td>578.44104</td>
    </tr>
  </tbody>
</table>
<p>125265 rows × 20 columns</p>
</div>




```python
# How many points do we have for each typhoon?
df_complete.groupby('typhoon_name').count()['grid_point_id']
```




    typhoon_name
    BEBINCA         3579
    BULBUL:MATMO    3579
    CONSON          3579
    DAMREY          7158
    DIANMU          3579
    DOKSURI         3579
    DURIAN          3579
    GAEMI           3579
    HAIMA           3579
    JEBI            3579
    KAI-TAK         3579
    KALMAEGI        3579
    KAMMURI         3579
    KETSANA         3579
    KUJIRA          3579
    LEKIMA          3579
    MANGKHUT        3579
    MINDULLE        3579
    MIRINAE         3579
    NARI            3579
    NESAT           3579
    PAKHAR          3579
    PODUL           7158
    RAI             3579
    RAMMASUN        3579
    SINLAKU         3579
    SON-TINH        3579
    SONCA           3579
    TALAS           3579
    VICENTE         3579
    WIPHA           3579
    WUTIP           3579
    XANGSANE        3579
    Name: grid_point_id, dtype: int64



OBS: total_buildings_damage is a fractonary number because we are splitting the number of houses destroyed per grid cell.

OBS: Is not a ptoblem that sometimes total_buildings_damaged > total_buildings because total_buildings_damaged is at municipality level and total_buildings is by grid.


```python
# Spoiler
plt.hist(df.perc_dmg_grid)
plt.yscale('log')
plt.title('Distribution of housing damage at grid level')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```



![png](08_data_merging_files/08_data_merging_38_0.png)



## Create stationary dataset


```python
df_stat = df_complete[[
            'grid_point_id',
            'IWI',
            'rwi',
            'scaled_distance',
            'light_index',
            'total_buildings',
            'with_coast',
            'coast_length',
            'mean_elev',
            'mean_slope']].drop_duplicates()
df_stat
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
      <th>rwi</th>
      <th>scaled_distance</th>
      <th>light_index</th>
      <th>total_buildings</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007</td>
      <td>74.2</td>
      <td>0.371000</td>
      <td>0.444038</td>
      <td>589.620850</td>
      <td>333.0</td>
      <td>1</td>
      <td>9240.659348</td>
      <td>985.104826</td>
      <td>20.474829</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2008</td>
      <td>74.2</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>641.250366</td>
      <td>294.0</td>
      <td>1</td>
      <td>18120.985665</td>
      <td>1276.734225</td>
      <td>21.390162</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2009</td>
      <td>74.2</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>608.715332</td>
      <td>21.0</td>
      <td>1</td>
      <td>965.114251</td>
      <td>1162.446697</td>
      <td>21.541014</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2187</td>
      <td>74.2</td>
      <td>-0.465000</td>
      <td>0.816760</td>
      <td>1075.123291</td>
      <td>0.0</td>
      <td>1</td>
      <td>5424.446827</td>
      <td>801.625901</td>
      <td>23.054955</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2188</td>
      <td>74.2</td>
      <td>-0.543000</td>
      <td>0.851535</td>
      <td>644.235657</td>
      <td>313.0</td>
      <td>1</td>
      <td>21703.068644</td>
      <td>1047.179465</td>
      <td>21.058342</td>
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
    </tr>
    <tr>
      <th>125090</th>
      <td>30180</td>
      <td>80.9</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>555.685303</td>
      <td>1.0</td>
      <td>1</td>
      <td>13130.384968</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>125125</th>
      <td>30359</td>
      <td>80.9</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>585.181396</td>
      <td>0.0</td>
      <td>1</td>
      <td>5616.457618</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>125160</th>
      <td>30360</td>
      <td>80.9</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>569.950684</td>
      <td>0.0</td>
      <td>1</td>
      <td>8188.877461</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>125195</th>
      <td>30540</td>
      <td>80.9</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>700.367615</td>
      <td>0.0</td>
      <td>1</td>
      <td>8813.010041</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>125230</th>
      <td>30541</td>
      <td>80.9</td>
      <td>0.003629</td>
      <td>0.607827</td>
      <td>578.441040</td>
      <td>1.0</td>
      <td>1</td>
      <td>6328.547070</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>3579 rows × 10 columns</p>
</div>



## Write out dataset


```python
df_complete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_viet_complete_interpolated_wind.csv", index=False
)

df_stat.reset_index().to_csv(
    output_dir / "viet_stationary_data_interpolated_wind.csv", index=False
)
```
