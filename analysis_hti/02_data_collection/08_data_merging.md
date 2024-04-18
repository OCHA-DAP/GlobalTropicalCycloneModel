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
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/02_model_features"
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis_hti/03_model_input_dataset"
)
output_dir.mkdir(exist_ok=True)
```

## Typhoons


```python
# Read in the data file
filename = input_dir / "01_windfield/windfield_data_hti_overlap.csv"

df_typhoons = pd.read_csv(filename)
df_typhoons = df_typhoons[['typhoon_name','typhoon_year', 'affected_pop']].drop_duplicates()
df_typhoons_nodmg = df_typhoons[df_typhoons.affected_pop == False].reset_index(drop=True)
df_typhoons_nodmg['event_level'] = 'ADM0'
```

## Read in buliding damage


```python
# Read in the building damage data
filename = input_dir / "02_housing_damage/output/building_damage_bygrid.csv"
df_damage_bld = pd.read_csv(filename)

# Read in the people affected data
filename = input_dir / "02_housing_damage/output/pop_affected_bygrid.csv"
df_damage_pop = pd.read_csv(filename)

df_damage = df_damage_bld.merge(df_damage_pop)

# Buildings per grid
df_buildings_raw = df_damage[
    ['id', 'numbuildings_x']
    ].drop_duplicates().rename({'numbuildings_x':'numbuildings'}, axis=1)
```


```python
df_damage.typhoon_name.unique()
```




    array(['LILI', 'IVAN', 'JEANNE', 'DENNIS', 'EMILY', 'STAN', 'ALPHA',
           'ERNESTO', 'DEAN', 'NOEL', 'OLGA', 'FAY', 'GUSTAV', 'HANNA', 'IKE',
           'TOMAS', 'IRENE', 'ISAAC', 'SANDY', 'ERIKA', 'MATTHEW', 'IRMA',
           'LAURA', 'ELSA'], dtype=object)




```python
# Rename some columns
df_damage = df_damage.rename({
    "id": "grid_point_id",
    "numbuildings_x": "total_buildings",
    "bld_affected": "total_buildings_damaged",
    "bld_affected_from_phl": "total_buildings_damaged_from_phl",
    "affected_population":"total_pop_affected"
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
      <th>event_level</th>
      <th>grid_point_id</th>
      <th>total_buildings_damaged</th>
      <th>total_buildings_damaged_from_phl</th>
      <th>total_buildings</th>
      <th>numbuildings_z</th>
      <th>frac_bld</th>
      <th>perc_dmg_grid</th>
      <th>perc_dmg_grid_from_phl</th>
      <th>total_pop_affected</th>
      <th>total_pop</th>
      <th>numpeople_z</th>
      <th>frac_people</th>
      <th>perc_aff_pop_grid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>899</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>0.0</td>
      <td>4034248.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>250.0</td>
      <td>0.000000e+00</td>
      <td>1.076409e+07</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>1333</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>0.0</td>
      <td>4034248.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>250.0</td>
      <td>0.000000e+00</td>
      <td>1.076409e+07</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>571</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>0.0</td>
      <td>4034248.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>250.0</td>
      <td>0.000000e+00</td>
      <td>1.076409e+07</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>694</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>0.0</td>
      <td>4034248.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>250.0</td>
      <td>0.000000e+00</td>
      <td>1.076409e+07</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>948</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>0.0</td>
      <td>4034248.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>250.0</td>
      <td>0.000000e+00</td>
      <td>1.076409e+07</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
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
    </tr>
    <tr>
      <th>8020</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1034</td>
      <td>1.130102</td>
      <td>322.950409</td>
      <td>52534.0</td>
      <td>2342743.0</td>
      <td>0.022424</td>
      <td>0.000001</td>
      <td>0.000104</td>
      <td>3.0</td>
      <td>4.704280e+04</td>
      <td>6.219108e+06</td>
      <td>0.007564</td>
      <td>3.648868e-07</td>
    </tr>
    <tr>
      <th>8021</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1076</td>
      <td>1.130102</td>
      <td>322.950409</td>
      <td>103463.0</td>
      <td>2342743.0</td>
      <td>0.044163</td>
      <td>0.000002</td>
      <td>0.000205</td>
      <td>3.0</td>
      <td>2.987461e+05</td>
      <td>6.219108e+06</td>
      <td>0.048037</td>
      <td>2.317220e-06</td>
    </tr>
    <tr>
      <th>8022</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1159</td>
      <td>1.130102</td>
      <td>322.950409</td>
      <td>170524.0</td>
      <td>2342743.0</td>
      <td>0.072788</td>
      <td>0.000004</td>
      <td>0.000338</td>
      <td>3.0</td>
      <td>1.451328e+05</td>
      <td>6.219108e+06</td>
      <td>0.023337</td>
      <td>1.125721e-06</td>
    </tr>
    <tr>
      <th>8023</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1118</td>
      <td>1.130102</td>
      <td>322.950409</td>
      <td>277305.0</td>
      <td>2342743.0</td>
      <td>0.118368</td>
      <td>0.000006</td>
      <td>0.000550</td>
      <td>3.0</td>
      <td>1.468212e+06</td>
      <td>6.219108e+06</td>
      <td>0.236081</td>
      <td>1.138816e-05</td>
    </tr>
    <tr>
      <th>8024</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>1.130102</td>
      <td>322.950409</td>
      <td>300289.0</td>
      <td>2342743.0</td>
      <td>0.128178</td>
      <td>0.000006</td>
      <td>0.000596</td>
      <td>3.0</td>
      <td>7.071491e+05</td>
      <td>6.219108e+06</td>
      <td>0.113706</td>
      <td>5.484992e-06</td>
    </tr>
  </tbody>
</table>
<p>8025 rows × 16 columns</p>
</div>




```python
import ast

# grids
df_aux = df_damage[['grid_point_id', 'total_buildings', 'total_pop']].drop_duplicates()
grid_cells = df_aux.grid_point_id.tolist()
total_bld = df_aux.total_buildings.tolist()
total_pop = df_aux.total_pop.tolist()
print(len(grid_cells))

# Add no dmg data
df_typhoons_nodmg['grid_point_id'] = str(grid_cells)
df_typhoons_nodmg['total_buildings'] = str(total_bld)
df_typhoons_nodmg['total_pop'] = str(total_pop)
df_typhoons_nodmg['grid_point_id'] = df_typhoons_nodmg['grid_point_id'].astype('str').apply(ast.literal_eval)
df_typhoons_nodmg['total_buildings'] = df_typhoons_nodmg['total_buildings'].astype('str').apply(ast.literal_eval)
df_typhoons_nodmg['total_pop'] = df_typhoons_nodmg['total_pop'].astype('str').apply(ast.literal_eval)
df_typhoons_nodmg_exploded = df_typhoons_nodmg.explode(['grid_point_id', 'total_buildings', 'total_pop'])
df_typhoons_nodmg_exploded = df_typhoons_nodmg_exploded.rename({'typhoon_year':'Year'}, axis=1)
```

    321



```python
df_damage_all = pd.concat([df_damage, df_typhoons_nodmg_exploded]).fillna(0)
```


```python
# Rename colums in the buildings dataset
df_buildings = df_buildings_raw[['id','numbuildings']].rename(
    {'id':'grid_point_id', 'numbuildings':'total_buildings'},axis=1)
```

## Read in windfield


```python
# Read in the data file
filename = input_dir / "01_windfield/windfield_data_hti_overlap.csv"

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
      <th>typhoon_year</th>
      <th>track_id</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>affected_pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>235</td>
      <td>13.856544</td>
      <td>121.524783</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>236</td>
      <td>14.311211</td>
      <td>116.787127</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>237</td>
      <td>14.766644</td>
      <td>112.480322</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>238</td>
      <td>15.217762</td>
      <td>109.138587</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>277</td>
      <td>13.663473</td>
      <td>131.573998</td>
      <td>True</td>
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
      <th>16045</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>2022311N21293</td>
      <td>1404</td>
      <td>0.000000</td>
      <td>556.443231</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16046</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>2022311N21293</td>
      <td>1405</td>
      <td>0.000000</td>
      <td>559.319502</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16047</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>2022311N21293</td>
      <td>1406</td>
      <td>0.000000</td>
      <td>562.400573</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16048</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>2022311N21293</td>
      <td>1407</td>
      <td>0.000000</td>
      <td>565.683099</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16049</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>2022311N21293</td>
      <td>1414</td>
      <td>0.000000</td>
      <td>593.995213</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>16050 rows × 7 columns</p>
</div>




```python
len(df_windfield.grid_point_id.unique())
```




    321



## Read in rainfall


```python
filename = input_dir / "03_rainfall/output/rainfall_data_rw_mean.csv"
filename_nodmg = input_dir / "03_rainfall/output/rainfall_data_nodmg_rw_mean.csv"
df_rainfall_nodmg = pd.read_csv(filename_nodmg)
df_rainfall_dmg = pd.read_csv(filename)
df_rainfall = pd.concat([df_rainfall_dmg, df_rainfall_nodmg])

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
      <td>LILI</td>
      <td>2002</td>
      <td>235</td>
      <td>5.083333</td>
      <td>3.375000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LILI</td>
      <td>2002</td>
      <td>236</td>
      <td>10.116667</td>
      <td>5.589583</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LILI</td>
      <td>2002</td>
      <td>237</td>
      <td>12.216667</td>
      <td>7.145833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LILI</td>
      <td>2002</td>
      <td>238</td>
      <td>18.041667</td>
      <td>9.760417</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LILI</td>
      <td>2002</td>
      <td>277</td>
      <td>5.450000</td>
      <td>3.462500</td>
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
      <th>8020</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>1404</td>
      <td>2.250000</td>
      <td>0.595833</td>
    </tr>
    <tr>
      <th>8021</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>1405</td>
      <td>3.950000</td>
      <td>1.025000</td>
    </tr>
    <tr>
      <th>8022</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>1406</td>
      <td>1.908333</td>
      <td>0.477083</td>
    </tr>
    <tr>
      <th>8023</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>1407</td>
      <td>3.133333</td>
      <td>0.783333</td>
    </tr>
    <tr>
      <th>8024</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>1414</td>
      <td>1.408333</td>
      <td>0.458333</td>
    </tr>
  </tbody>
</table>
<p>16050 rows × 5 columns</p>
</div>



Rainfall does not include LINDA1997


```python
len(df_rainfall.grid_point_id.unique())
```




    321




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




    321




```python
len(set(rain_ids) & set(house_ids))
```




    321



Everything is ok!!!!!

## Read in IWI


```python
filename_iwi = input_dir / "05_vulnerability/output/hti_iwi_bygrid_new.csv"
df_iwi = pd.read_csv(filename_iwi)
```

## Read in topography


```python
filename_topo = input_dir / "04_topography/output/topography_variables_bygrid.csv"
df_topo = pd.read_csv(filename_topo)
df_topo = df_topo.rename({'id':'grid_point_id'}, axis=1)
```

## Merge the datasets


```python
# Merge windfield and rainfall
df_merge_wind_rain = df_windfield.merge(df_rainfall)

# Merge with damage
df_merge_typhoons = df_damage_all[['typhoon_name', 'Year', 'event_level',
           'grid_point_id',
           'total_buildings_damaged', 'total_buildings_damaged_from_phl',
           'total_pop_affected',
           'total_buildings', 'total_pop',
           'perc_dmg_grid', 'perc_dmg_grid_from_phl',
           'perc_aff_pop_grid']
           ].rename({'Year':'typhoon_year'}, axis=1).merge(df_merge_wind_rain)

# Merge with topography data
df_merge_topo = df_merge_typhoons.merge(df_topo, on='grid_point_id')

# Merge with vulnerability indexes
df_all = df_merge_topo.merge(df_iwi, on='grid_point_id')
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
      <th>typhoon_year</th>
      <th>event_level</th>
      <th>grid_point_id</th>
      <th>total_buildings_damaged</th>
      <th>total_buildings_damaged_from_phl</th>
      <th>total_pop_affected</th>
      <th>total_buildings</th>
      <th>total_pop</th>
      <th>perc_dmg_grid</th>
      <th>...</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>affected_pop</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
      <th>IWI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>899</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>320.213466</td>
      <td>True</td>
      <td>0.366667</td>
      <td>0.091667</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IVAN</td>
      <td>2004</td>
      <td>ADM2</td>
      <td>899</td>
      <td>0.000000</td>
      <td>21.303592</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>415.471493</td>
      <td>True</td>
      <td>1.933333</td>
      <td>0.739583</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JEANNE</td>
      <td>2004</td>
      <td>ADM1</td>
      <td>899</td>
      <td>119832.972848</td>
      <td>14324.227304</td>
      <td>315594.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>14.298383</td>
      <td>110.539658</td>
      <td>True</td>
      <td>8.800000</td>
      <td>3.239583</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DENNIS</td>
      <td>2005</td>
      <td>ADM1</td>
      <td>899</td>
      <td>0.000000</td>
      <td>322.817273</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>9.467717</td>
      <td>306.785000</td>
      <td>True</td>
      <td>3.133333</td>
      <td>1.493750</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EMILY</td>
      <td>2005</td>
      <td>ADM2</td>
      <td>899</td>
      <td>0.000000</td>
      <td>15.478327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>501.742109</td>
      <td>True</td>
      <td>2.783333</td>
      <td>1.031250</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
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
      <th>16045</th>
      <td>IOTA</td>
      <td>2020</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>360.709621</td>
      <td>False</td>
      <td>4.825000</td>
      <td>1.327083</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16046</th>
      <td>NANA</td>
      <td>2020</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>440.564161</td>
      <td>False</td>
      <td>2.591667</td>
      <td>0.852083</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16047</th>
      <td>IDA</td>
      <td>2021</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>773.123661</td>
      <td>False</td>
      <td>2.658333</td>
      <td>0.708333</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16048</th>
      <td>LISA</td>
      <td>2022</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>337.440669</td>
      <td>False</td>
      <td>0.400000</td>
      <td>0.114583</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16049</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>646.912779</td>
      <td>False</td>
      <td>1.783333</td>
      <td>0.445833</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
  </tbody>
</table>
<p>16050 rows × 23 columns</p>
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
      <th>typhoon_year</th>
      <th>event_level</th>
      <th>grid_point_id</th>
      <th>total_buildings_damaged</th>
      <th>total_buildings_damaged_from_phl</th>
      <th>total_pop_affected</th>
      <th>total_buildings</th>
      <th>total_pop</th>
      <th>perc_dmg_grid</th>
      <th>...</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>affected_pop</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_elev</th>
      <th>mean_slope</th>
      <th>IWI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LILI</td>
      <td>2002</td>
      <td>ADM1</td>
      <td>899</td>
      <td>93.696877</td>
      <td>566.935811</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>320.213466</td>
      <td>True</td>
      <td>0.366667</td>
      <td>0.091667</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IVAN</td>
      <td>2004</td>
      <td>ADM2</td>
      <td>899</td>
      <td>0.000000</td>
      <td>21.303592</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>415.471493</td>
      <td>True</td>
      <td>1.933333</td>
      <td>0.739583</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JEANNE</td>
      <td>2004</td>
      <td>ADM1</td>
      <td>899</td>
      <td>119832.972848</td>
      <td>14324.227304</td>
      <td>315594.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>14.298383</td>
      <td>110.539658</td>
      <td>True</td>
      <td>8.800000</td>
      <td>3.239583</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DENNIS</td>
      <td>2005</td>
      <td>ADM1</td>
      <td>899</td>
      <td>0.000000</td>
      <td>322.817273</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>9.467717</td>
      <td>306.785000</td>
      <td>True</td>
      <td>3.133333</td>
      <td>1.493750</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EMILY</td>
      <td>2005</td>
      <td>ADM2</td>
      <td>899</td>
      <td>0.000000</td>
      <td>15.478327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>501.742109</td>
      <td>True</td>
      <td>2.783333</td>
      <td>1.031250</td>
      <td>1</td>
      <td>668.74247</td>
      <td>0.008149</td>
      <td>0.004711</td>
      <td>33.4</td>
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
      <th>16045</th>
      <td>IOTA</td>
      <td>2020</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>360.709621</td>
      <td>False</td>
      <td>4.825000</td>
      <td>1.327083</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16046</th>
      <td>NANA</td>
      <td>2020</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>440.564161</td>
      <td>False</td>
      <td>2.591667</td>
      <td>0.852083</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16047</th>
      <td>IDA</td>
      <td>2021</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>773.123661</td>
      <td>False</td>
      <td>2.658333</td>
      <td>0.708333</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16048</th>
      <td>LISA</td>
      <td>2022</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>337.440669</td>
      <td>False</td>
      <td>0.400000</td>
      <td>0.114583</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>16049</th>
      <td>NICOLE</td>
      <td>2022</td>
      <td>ADM0</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>646.912779</td>
      <td>False</td>
      <td>1.783333</td>
      <td>0.445833</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
  </tbody>
</table>
<p>16050 rows × 23 columns</p>
</div>




```python
df_complete.columns
```




    Index(['typhoon_name', 'typhoon_year', 'event_level', 'grid_point_id',
           'total_buildings_damaged', 'total_buildings_damaged_from_phl',
           'total_pop_affected', 'total_buildings', 'total_pop', 'perc_dmg_grid',
           'perc_dmg_grid_from_phl', 'perc_aff_pop_grid', 'track_id', 'wind_speed',
           'track_distance', 'affected_pop', 'rainfall_max_6h', 'rainfall_max_24h',
           'with_coast', 'coast_length', 'mean_elev', 'mean_slope', 'IWI'],
          dtype='object')




```python
# How many points do we have for each typhoon?
df_complete.groupby('typhoon_name').count()['grid_point_id']
```




    typhoon_name
    ALPHA        321
    BONNIE       321
    CRISTOBAL    321
    DANNY        321
    DEAN         321
    DELTA        321
    DENNIS       321
    EARL         321
    ELSA         321
    EMILY        642
    ERIKA        321
    ERNESTO      321
    ETA          321
    FAY          321
    FRANKLIN     321
    GERT         321
    GORDON       321
    GUSTAV       321
    HANNA        321
    HUMBERTO     321
    IDA          321
    IKE          321
    IOTA         321
    IRENE        321
    IRMA         321
    ISAAC        321
    IVAN         321
    JEANNE       321
    KARL         321
    KATE         321
    KATRINA      321
    KYLE         321
    LAURA        321
    LILI         321
    LISA         321
    MATTHEW      642
    MINDY        321
    NANA         321
    NICOLE       321
    NOEL         321
    OLGA         321
    OMAR         321
    OTTO         321
    RITA         321
    SANDY        321
    STAN         321
    TOMAS        321
    WILMA        321
    Name: grid_point_id, dtype: int64



OBS: total_buildings_damage is a fractonary number because we are splitting the number of houses destroyed per grid cell.

OBS: Is not a ptoblem that sometimes total_buildings_damaged > total_buildings because total_buildings_damaged is at municipality level and total_buildings is by grid.


```python
# Spoiler
plt.hist(df.perc_aff_pop_grid)
plt.yscale('log')
plt.title('Distribution of people affected at grid level')
plt.xlabel('% of people affected')
plt.ylabel('Count')
plt.show()
```



![png](08_data_merging_files/08_data_merging_38_0.png)




```python
# Spoiler
plt.hist(df.perc_dmg_grid)
plt.yscale('log')
plt.title('Distribution of housing damage at grid level')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```



![png](08_data_merging_files/08_data_merging_39_0.png)




```python
# Spoiler
plt.hist(df.perc_dmg_grid_from_phl)
plt.yscale('log')
plt.title('Distribution of housing damage at grid level \nFrom Philippines data')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```



![png](08_data_merging_files/08_data_merging_40_0.png)



## Create stationary dataset


```python
df_stat = df_complete[[
            'grid_point_id',
            'IWI',
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
      <td>899</td>
      <td>33.4</td>
      <td>0.0</td>
      <td>1</td>
      <td>668.742470</td>
      <td>0.008149</td>
      <td>0.004711</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1333</td>
      <td>36.7</td>
      <td>0.0</td>
      <td>1</td>
      <td>5457.737332</td>
      <td>4.050314</td>
      <td>0.528598</td>
    </tr>
    <tr>
      <th>100</th>
      <td>571</td>
      <td>28.7</td>
      <td>0.0</td>
      <td>1</td>
      <td>1363.143498</td>
      <td>0.015999</td>
      <td>0.009933</td>
    </tr>
    <tr>
      <th>150</th>
      <td>694</td>
      <td>53.1</td>
      <td>0.0</td>
      <td>1</td>
      <td>3892.528356</td>
      <td>0.020542</td>
      <td>0.015541</td>
    </tr>
    <tr>
      <th>200</th>
      <td>948</td>
      <td>53.1</td>
      <td>0.0</td>
      <td>1</td>
      <td>2577.940754</td>
      <td>0.308638</td>
      <td>0.083267</td>
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
      <th>15800</th>
      <td>983</td>
      <td>33.4</td>
      <td>97619.0</td>
      <td>1</td>
      <td>8130.354546</td>
      <td>41.891230</td>
      <td>4.504061</td>
    </tr>
    <tr>
      <th>15850</th>
      <td>1076</td>
      <td>53.1</td>
      <td>103463.0</td>
      <td>1</td>
      <td>12806.655888</td>
      <td>120.274683</td>
      <td>6.099569</td>
    </tr>
    <tr>
      <th>15900</th>
      <td>1159</td>
      <td>53.1</td>
      <td>170524.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>131.428020</td>
      <td>7.151623</td>
    </tr>
    <tr>
      <th>15950</th>
      <td>1118</td>
      <td>53.1</td>
      <td>277305.0</td>
      <td>1</td>
      <td>18853.739467</td>
      <td>124.936365</td>
      <td>6.479317</td>
    </tr>
    <tr>
      <th>16000</th>
      <td>1160</td>
      <td>53.1</td>
      <td>300289.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>150.243299</td>
      <td>4.928486</td>
    </tr>
  </tbody>
</table>
<p>321 rows × 7 columns</p>
</div>



## Write out dataset


```python
df_complete.reset_index().to_csv(
    output_dir / "new_model_training_dataset_hti_with_nodmg.csv", index=False
)

df_stat.reset_index().to_csv(
    output_dir / "hti_stationary_data.csv", index=False
)
```
