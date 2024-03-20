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
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_phl/02_model_features"
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_phl/03_model_input_dataset"
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
      <th>typhoon_name</th>
      <th>Year</th>
      <th>grid_point_id</th>
      <th>total_buildings_damaged</th>
      <th>total_buildings</th>
      <th>Region</th>
      <th>perc_dmg_grid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74519</th>
      <td>NAKRI</td>
      <td>2019</td>
      <td>20681</td>
      <td>0.0</td>
      <td>468</td>
      <td>PH112508000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>99160</th>
      <td>HAGUPIT</td>
      <td>2014</td>
      <td>11005</td>
      <td>0.0</td>
      <td>0</td>
      <td>PH156606000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>99159</th>
      <td>HAGUPIT</td>
      <td>2014</td>
      <td>11004</td>
      <td>0.0</td>
      <td>39</td>
      <td>PH156606000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>99158</th>
      <td>HAGUPIT</td>
      <td>2014</td>
      <td>10974</td>
      <td>0.0</td>
      <td>32</td>
      <td>PH175308000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>99157</th>
      <td>HAGUPIT</td>
      <td>2014</td>
      <td>10960</td>
      <td>0.0</td>
      <td>0</td>
      <td>PH175310000</td>
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
    </tr>
    <tr>
      <th>3726</th>
      <td>HAIYAN</td>
      <td>2013</td>
      <td>13119</td>
      <td>3481.0</td>
      <td>26</td>
      <td>PH060414000</td>
      <td>13388.461538</td>
    </tr>
    <tr>
      <th>18630</th>
      <td>MANGKHUT</td>
      <td>2018</td>
      <td>11051</td>
      <td>1187.0</td>
      <td>8</td>
      <td>PH012823000</td>
      <td>14837.500000</td>
    </tr>
    <tr>
      <th>100602</th>
      <td>PHANFONE</td>
      <td>2019</td>
      <td>13119</td>
      <td>8783.0</td>
      <td>26</td>
      <td>PH060414000</td>
      <td>33780.769231</td>
    </tr>
    <tr>
      <th>59616</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>16943</td>
      <td>4392.0</td>
      <td>10</td>
      <td>PH052003000</td>
      <td>43920.000000</td>
    </tr>
    <tr>
      <th>67068</th>
      <td>NOCK-TEN</td>
      <td>2016</td>
      <td>16943</td>
      <td>4868.0</td>
      <td>10</td>
      <td>PH052003000</td>
      <td>48680.000000</td>
    </tr>
  </tbody>
</table>
<p>149040 rows × 7 columns</p>
</div>




```python
# Rename colums in the buildings dataset
df_buildings = df_buildings_raw[['id','numbuildings']].rename({'id':'grid_point_id', 'numbuildings':'total_buildings'},axis=1)
```

## Read in windfield


```python
# Read in the data file
filename = input_dir / "01_windfield/windfield_data_phl.csv"

df_windfield = pd.read_csv(filename)
df_windfield = df_windfield.drop('geometry', axis=1)
df_windfield = df_windfield.drop_duplicates()
df_windfield = df_windfield.rename({
    'typhoon_id':'track_id',
    'typhoon_year':'Year'}, axis=1)
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
      <th>Unnamed: 0</th>
      <th>track_id</th>
      <th>typhoon_name</th>
      <th>Year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.0</td>
      <td>308.690020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4475</td>
      <td>0.0</td>
      <td>623.151133</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4639</td>
      <td>0.0</td>
      <td>588.305668</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4640</td>
      <td>0.0</td>
      <td>599.219433</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4641</td>
      <td>0.0</td>
      <td>610.140281</td>
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
      <th>145309</th>
      <td>145309</td>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20677</td>
      <td>0.0</td>
      <td>644.615067</td>
    </tr>
    <tr>
      <th>145310</th>
      <td>145310</td>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20678</td>
      <td>0.0</td>
      <td>655.724121</td>
    </tr>
    <tr>
      <th>145311</th>
      <td>145311</td>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20679</td>
      <td>0.0</td>
      <td>666.833174</td>
    </tr>
    <tr>
      <th>145312</th>
      <td>145312</td>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20680</td>
      <td>0.0</td>
      <td>677.942228</td>
    </tr>
    <tr>
      <th>145313</th>
      <td>145313</td>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>689.051282</td>
    </tr>
  </tbody>
</table>
<p>145314 rows × 7 columns</p>
</div>




```python
len(df_windfield.grid_point_id.unique())
```




    3726



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
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.122917</td>
      <td>0.035937</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4475</td>
      <td>0.097917</td>
      <td>0.027083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4639</td>
      <td>0.535417</td>
      <td>0.146354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4640</td>
      <td>0.358333</td>
      <td>0.101562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4641</td>
      <td>0.216667</td>
      <td>0.058333</td>
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
      <th>219829</th>
      <td>NOUL</td>
      <td>2015</td>
      <td>20677</td>
      <td>0.812500</td>
      <td>0.375521</td>
    </tr>
    <tr>
      <th>219830</th>
      <td>NOUL</td>
      <td>2015</td>
      <td>20678</td>
      <td>0.808333</td>
      <td>0.354687</td>
    </tr>
    <tr>
      <th>219831</th>
      <td>NOUL</td>
      <td>2015</td>
      <td>20679</td>
      <td>1.447917</td>
      <td>0.369792</td>
    </tr>
    <tr>
      <th>219832</th>
      <td>NOUL</td>
      <td>2015</td>
      <td>20680</td>
      <td>2.535417</td>
      <td>0.640625</td>
    </tr>
    <tr>
      <th>219833</th>
      <td>NOUL</td>
      <td>2015</td>
      <td>20681</td>
      <td>1.797917</td>
      <td>0.459375</td>
    </tr>
  </tbody>
</table>
<p>219834 rows × 5 columns</p>
</div>



Rainfall does not include LINDA1997


```python
len(df_rainfall.grid_point_id.unique())
```




    3726




```python
set(df_windfield.typhoon_name.unique()) - set(df_rainfall.typhoon_name.unique())
```




    set()



## Check grid points ids matches


```python
wind_ids = df_windfield.grid_point_id.unique()
house_ids = df_damage.grid_point_id.unique()
rain_ids = df_rainfall.grid_point_id.unique()
```


```python
len(set(wind_ids) & set(house_ids))
```




    3726




```python
len(set(rain_ids) & set(house_ids))
```




    3726



Everything is ok!!!!!

## Read in IWI


```python
filename_iwi = input_dir / "05_vulnerability/output/phl_iwi_bygrid_new.csv"
df_iwi = pd.read_csv(filename_iwi)
```

## Read in RWI


```python
filename_RWI = input_dir / "05_vulnerability/output/phl_rwi_bygrid.csv"
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
df_windfield_total = typhoons_ids.merge(df_windfield, left_on='typhoon_id', right_on='track_id')[[
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
      <th>...</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
      <th>mean_rug</th>
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
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.0</td>
      <td>308.690020</td>
      <td>0.122917</td>
      <td>0.035937</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FENGSHEN</td>
      <td>2008</td>
      <td>101</td>
      <td>0.0</td>
      <td>753.271212</td>
      <td>6.937500</td>
      <td>4.539583</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>2009</td>
      <td>101</td>
      <td>0.0</td>
      <td>474.619388</td>
      <td>14.862500</td>
      <td>5.548958</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CONSON</td>
      <td>2010</td>
      <td>101</td>
      <td>0.0</td>
      <td>571.914912</td>
      <td>8.593750</td>
      <td>2.535938</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NESAT</td>
      <td>2011</td>
      <td>101</td>
      <td>0.0</td>
      <td>707.652211</td>
      <td>4.083333</td>
      <td>2.275521</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>145309</th>
      <td>SAUDEL</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>951.187307</td>
      <td>1.006250</td>
      <td>0.660937</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145310</th>
      <td>GONI</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>753.893882</td>
      <td>1.631250</td>
      <td>0.543750</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145311</th>
      <td>VAMCO</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>671.488529</td>
      <td>2.806250</td>
      <td>1.148958</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145312</th>
      <td>VONGFONG</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>546.200936</td>
      <td>3.789583</td>
      <td>1.076042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145313</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>689.051282</td>
      <td>3.268750</td>
      <td>1.164062</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
  </tbody>
</table>
<p>145314 rows × 21 columns</p>
</div>




```python
# How many points do we have for each typhoon?
df_complete.groupby('typhoon_name').count()['grid_point_id']
```




    typhoon_name
    BOPHA        3726
    CONSON       3726
    DURIAN       3726
    FENGSHEN     3726
    FUNG-WONG    3726
    GONI         7452
    HAGUPIT      3726
    HAIMA        3726
    HAIYAN       3726
    JANGMI       3726
    KALMAEGI     3726
    KAMMURI      3726
    KETSANA      3726
    KOPPU        3726
    KROSA        3726
    LINFA        3726
    LINGLING     3726
    MANGKHUT     3726
    MEKKHALA     3726
    MELOR        3726
    MERANTI      3726
    MOLAVE       3726
    MUJIGAE      3726
    NAKRI        3726
    NARI         3726
    NESAT        3726
    NOCK-TEN     3726
    NOUL         3726
    PHANFONE     3726
    RAMMASUN     3726
    SARIKA       3726
    SAUDEL       3726
    TOKAGE       3726
    USAGI        3726
    UTOR         3726
    VAMCO        3726
    VONGFONG     3726
    YUTU         3726
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



![png](08_data_merging_files/08_data_merging_37_0.png)



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
            'mean_slope',
            'mean_rug']].drop_duplicates()
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
      <th>mean_rug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101</td>
      <td>47.8</td>
      <td>-0.212496</td>
      <td>0.668811</td>
      <td>1496.591553</td>
      <td>0</td>
      <td>1</td>
      <td>3445.709753</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4475</td>
      <td>47.8</td>
      <td>-0.527000</td>
      <td>0.799419</td>
      <td>518.590576</td>
      <td>3</td>
      <td>1</td>
      <td>8602.645832</td>
      <td>0.422334</td>
      <td>0.127924</td>
      <td>0.217314</td>
    </tr>
    <tr>
      <th>78</th>
      <td>4639</td>
      <td>47.8</td>
      <td>-0.283000</td>
      <td>0.698090</td>
      <td>502.378876</td>
      <td>11</td>
      <td>1</td>
      <td>5084.012925</td>
      <td>0.050729</td>
      <td>0.027975</td>
      <td>0.053621</td>
    </tr>
    <tr>
      <th>117</th>
      <td>4640</td>
      <td>47.8</td>
      <td>-0.358889</td>
      <td>0.729605</td>
      <td>535.827576</td>
      <td>587</td>
      <td>1</td>
      <td>55607.865950</td>
      <td>5.896272</td>
      <td>1.365539</td>
      <td>2.291861</td>
    </tr>
    <tr>
      <th>156</th>
      <td>4641</td>
      <td>47.8</td>
      <td>-0.462800</td>
      <td>0.772757</td>
      <td>578.120667</td>
      <td>974</td>
      <td>1</td>
      <td>35529.342507</td>
      <td>26.036955</td>
      <td>4.594237</td>
      <td>7.283852</td>
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
    </tr>
    <tr>
      <th>145119</th>
      <td>20677</td>
      <td>63.9</td>
      <td>0.508167</td>
      <td>0.369532</td>
      <td>4119.430176</td>
      <td>7393</td>
      <td>1</td>
      <td>21559.003490</td>
      <td>5.634180</td>
      <td>1.059275</td>
      <td>1.752799</td>
    </tr>
    <tr>
      <th>145158</th>
      <td>20678</td>
      <td>63.9</td>
      <td>-0.174100</td>
      <td>0.652865</td>
      <td>4084.401123</td>
      <td>2528</td>
      <td>1</td>
      <td>12591.742022</td>
      <td>28.449444</td>
      <td>2.649756</td>
      <td>4.313097</td>
    </tr>
    <tr>
      <th>145197</th>
      <td>20679</td>
      <td>63.9</td>
      <td>-0.244286</td>
      <td>0.682012</td>
      <td>4021.677246</td>
      <td>1484</td>
      <td>1</td>
      <td>19740.596834</td>
      <td>3.123472</td>
      <td>0.614328</td>
      <td>0.988790</td>
    </tr>
    <tr>
      <th>145236</th>
      <td>20680</td>
      <td>63.9</td>
      <td>0.038000</td>
      <td>0.564784</td>
      <td>3987.096680</td>
      <td>2798</td>
      <td>1</td>
      <td>26363.303778</td>
      <td>26.898075</td>
      <td>2.066074</td>
      <td>3.275555</td>
    </tr>
    <tr>
      <th>145275</th>
      <td>20681</td>
      <td>63.9</td>
      <td>-0.175000</td>
      <td>0.653239</td>
      <td>3881.783691</td>
      <td>468</td>
      <td>1</td>
      <td>9359.492382</td>
      <td>8.298210</td>
      <td>0.764651</td>
      <td>1.179309</td>
    </tr>
  </tbody>
</table>
<p>3726 rows × 11 columns</p>
</div>



## Write out dataset


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
      <th>Year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_buildings_damaged</th>
      <th>perc_dmg_grid</th>
      <th>total_buildings</th>
      <th>...</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
      <th>mean_rug</th>
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
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.0</td>
      <td>308.690020</td>
      <td>0.122917</td>
      <td>0.035937</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FENGSHEN</td>
      <td>2008</td>
      <td>101</td>
      <td>0.0</td>
      <td>753.271212</td>
      <td>6.937500</td>
      <td>4.539583</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>2009</td>
      <td>101</td>
      <td>0.0</td>
      <td>474.619388</td>
      <td>14.862500</td>
      <td>5.548958</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CONSON</td>
      <td>2010</td>
      <td>101</td>
      <td>0.0</td>
      <td>571.914912</td>
      <td>8.593750</td>
      <td>2.535938</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NESAT</td>
      <td>2011</td>
      <td>101</td>
      <td>0.0</td>
      <td>707.652211</td>
      <td>4.083333</td>
      <td>2.275521</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>3445.709753</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>0.668811</td>
      <td>-0.212496</td>
      <td>47.8</td>
      <td>101</td>
      <td>1496.591553</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>145309</th>
      <td>SAUDEL</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>951.187307</td>
      <td>1.006250</td>
      <td>0.660937</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145310</th>
      <td>GONI</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>753.893882</td>
      <td>1.631250</td>
      <td>0.543750</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145311</th>
      <td>VAMCO</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>671.488529</td>
      <td>2.806250</td>
      <td>1.148958</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145312</th>
      <td>VONGFONG</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>546.200936</td>
      <td>3.789583</td>
      <td>1.076042</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
    <tr>
      <th>145313</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>689.051282</td>
      <td>3.268750</td>
      <td>1.164062</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>468</td>
      <td>...</td>
      <td>9359.492382</td>
      <td>8.29821</td>
      <td>0.764651</td>
      <td>1.179309</td>
      <td>20681</td>
      <td>0.653239</td>
      <td>-0.175000</td>
      <td>63.9</td>
      <td>20681</td>
      <td>3881.783691</td>
    </tr>
  </tbody>
</table>
<p>145314 rows × 21 columns</p>
</div>




```python
df_complete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_phl_new.csv", index=False
)

df_stat.reset_index().to_csv(
    output_dir / "phl_stationary_data_new.csv", index=False
)
```
