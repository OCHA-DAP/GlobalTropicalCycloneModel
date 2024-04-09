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

## Read in typhoons ids


```python
# typhoons_ids = pd.read_csv(input_dir / "01_windfield/windfield_data_hti_overlap.csv")
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
      <th>total_buildings</th>
      <th>numbuildings_z</th>
      <th>frac_bld</th>
      <th>perc_dmg_grid</th>
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
      <td>0.0</td>
      <td>4034248.0</td>
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
      <td>0.0</td>
      <td>4034248.0</td>
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
      <td>0.0</td>
      <td>4034248.0</td>
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
      <td>0.0</td>
      <td>4034248.0</td>
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
      <td>0.0</td>
      <td>4034248.0</td>
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
    </tr>
    <tr>
      <th>8020</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1034</td>
      <td>1.130102</td>
      <td>52534.0</td>
      <td>2342743.0</td>
      <td>0.022424</td>
      <td>0.000001</td>
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
      <td>103463.0</td>
      <td>2342743.0</td>
      <td>0.044163</td>
      <td>0.000002</td>
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
      <td>170524.0</td>
      <td>2342743.0</td>
      <td>0.072788</td>
      <td>0.000004</td>
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
      <td>277305.0</td>
      <td>2342743.0</td>
      <td>0.118368</td>
      <td>0.000006</td>
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
      <td>300289.0</td>
      <td>2342743.0</td>
      <td>0.128178</td>
      <td>0.000006</td>
      <td>3.0</td>
      <td>7.071491e+05</td>
      <td>6.219108e+06</td>
      <td>0.113706</td>
      <td>5.484992e-06</td>
    </tr>
  </tbody>
</table>
<p>8025 rows × 14 columns</p>
</div>




```python
# Rename colums in the buildings dataset
df_buildings = df_buildings_raw[['id','numbuildings']].rename({'id':'grid_point_id', 'numbuildings':'total_buildings'},axis=1)
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
    </tr>
    <tr>
      <th>1</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>236</td>
      <td>14.311211</td>
      <td>116.787127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>237</td>
      <td>14.766644</td>
      <td>112.480322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>238</td>
      <td>15.217762</td>
      <td>109.138587</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LILI</td>
      <td>2002</td>
      <td>2002265N10315</td>
      <td>277</td>
      <td>13.663473</td>
      <td>131.573998</td>
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
      <th>8020</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>2021182N09317</td>
      <td>1404</td>
      <td>5.833339</td>
      <td>254.561035</td>
    </tr>
    <tr>
      <th>8021</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>2021182N09317</td>
      <td>1405</td>
      <td>6.142006</td>
      <td>243.776764</td>
    </tr>
    <tr>
      <th>8022</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>2021182N09317</td>
      <td>1406</td>
      <td>6.468653</td>
      <td>232.992492</td>
    </tr>
    <tr>
      <th>8023</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>2021182N09317</td>
      <td>1407</td>
      <td>6.819306</td>
      <td>222.208221</td>
    </tr>
    <tr>
      <th>8024</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>2021182N09317</td>
      <td>1414</td>
      <td>10.251399</td>
      <td>146.718322</td>
    </tr>
  </tbody>
</table>
<p>8025 rows × 6 columns</p>
</div>




```python
len(df_windfield.grid_point_id.unique())
```




    321



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
      <td>ELSA</td>
      <td>2021</td>
      <td>1404</td>
      <td>1.775000</td>
      <td>0.881250</td>
    </tr>
    <tr>
      <th>8021</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>1405</td>
      <td>3.200000</td>
      <td>0.837500</td>
    </tr>
    <tr>
      <th>8022</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>1406</td>
      <td>3.633333</td>
      <td>0.929167</td>
    </tr>
    <tr>
      <th>8023</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>1407</td>
      <td>4.383333</td>
      <td>1.100000</td>
    </tr>
    <tr>
      <th>8024</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>1414</td>
      <td>1.525000</td>
      <td>0.595833</td>
    </tr>
  </tbody>
</table>
<p>8025 rows × 5 columns</p>
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
df_merge_typhoons = df_damage[['typhoon_name', 'Year', 'event_level',
           'grid_point_id',
           'total_buildings_damaged', 'total_pop_affected',
           'total_buildings', 'total_pop',
           'perc_dmg_grid', 'perc_aff_pop_grid']
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
      <th>total_pop_affected</th>
      <th>total_buildings</th>
      <th>total_pop</th>
      <th>perc_dmg_grid</th>
      <th>perc_aff_pop_grid</th>
      <th>track_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
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
      <td>250.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2002265N10315</td>
      <td>0.000000</td>
      <td>320.213466</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2004247N10332</td>
      <td>0.000000</td>
      <td>415.471493</td>
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
      <td>315594.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2004258N16300</td>
      <td>14.298383</td>
      <td>110.539658</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2005186N12299</td>
      <td>9.467717</td>
      <td>306.785000</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2005192N11318</td>
      <td>0.000000</td>
      <td>501.742109</td>
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
    </tr>
    <tr>
      <th>8020</th>
      <td>ERIKA</td>
      <td>2015</td>
      <td>ADM2</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015237N14315</td>
      <td>0.000000</td>
      <td>419.025406</td>
      <td>10.641667</td>
      <td>3.533333</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8021</th>
      <td>MATTHEW</td>
      <td>2016</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>772005.210029</td>
      <td>2100439.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>3.548527</td>
      <td>3.071354</td>
      <td>2016273N13300</td>
      <td>17.553459</td>
      <td>227.755339</td>
      <td>4.191667</td>
      <td>3.018750</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8022</th>
      <td>IRMA</td>
      <td>2017</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>15025.980806</td>
      <td>40092.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.027724</td>
      <td>0.024469</td>
      <td>2017242N16333</td>
      <td>12.991589</td>
      <td>286.188149</td>
      <td>4.408333</td>
      <td>1.833333</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8023</th>
      <td>LAURA</td>
      <td>2020</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>16556.238205</td>
      <td>44175.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.030548</td>
      <td>0.026961</td>
      <td>2020233N14313</td>
      <td>11.566673</td>
      <td>26.720308</td>
      <td>27.233333</td>
      <td>10.375000</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8024</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>1.130102</td>
      <td>3.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.000006</td>
      <td>0.000005</td>
      <td>2021182N09317</td>
      <td>10.066159</td>
      <td>152.497851</td>
      <td>4.783333</td>
      <td>1.235417</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
  </tbody>
</table>
<p>8025 rows × 20 columns</p>
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
      <th>total_pop_affected</th>
      <th>total_buildings</th>
      <th>total_pop</th>
      <th>perc_dmg_grid</th>
      <th>perc_aff_pop_grid</th>
      <th>track_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
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
      <td>250.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2002265N10315</td>
      <td>0.000000</td>
      <td>320.213466</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2004247N10332</td>
      <td>0.000000</td>
      <td>415.471493</td>
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
      <td>315594.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2004258N16300</td>
      <td>14.298383</td>
      <td>110.539658</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2005186N12299</td>
      <td>9.467717</td>
      <td>306.785000</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2005192N11318</td>
      <td>0.000000</td>
      <td>501.742109</td>
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
    </tr>
    <tr>
      <th>8020</th>
      <td>ERIKA</td>
      <td>2015</td>
      <td>ADM2</td>
      <td>1160</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015237N14315</td>
      <td>0.000000</td>
      <td>419.025406</td>
      <td>10.641667</td>
      <td>3.533333</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8021</th>
      <td>MATTHEW</td>
      <td>2016</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>772005.210029</td>
      <td>2100439.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>3.548527</td>
      <td>3.071354</td>
      <td>2016273N13300</td>
      <td>17.553459</td>
      <td>227.755339</td>
      <td>4.191667</td>
      <td>3.018750</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8022</th>
      <td>IRMA</td>
      <td>2017</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>15025.980806</td>
      <td>40092.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.027724</td>
      <td>0.024469</td>
      <td>2017242N16333</td>
      <td>12.991589</td>
      <td>286.188149</td>
      <td>4.408333</td>
      <td>1.833333</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8023</th>
      <td>LAURA</td>
      <td>2020</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>16556.238205</td>
      <td>44175.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.030548</td>
      <td>0.026961</td>
      <td>2020233N14313</td>
      <td>11.566673</td>
      <td>26.720308</td>
      <td>27.233333</td>
      <td>10.375000</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
    <tr>
      <th>8024</th>
      <td>ELSA</td>
      <td>2021</td>
      <td>ADM1</td>
      <td>1160</td>
      <td>1.130102</td>
      <td>3.0</td>
      <td>300289.0</td>
      <td>707149.104858</td>
      <td>0.000006</td>
      <td>0.000005</td>
      <td>2021182N09317</td>
      <td>10.066159</td>
      <td>152.497851</td>
      <td>4.783333</td>
      <td>1.235417</td>
      <td>0</td>
      <td>0.00000</td>
      <td>150.243299</td>
      <td>4.928486</td>
      <td>53.1</td>
    </tr>
  </tbody>
</table>
<p>8025 rows × 20 columns</p>
</div>




```python
# How many points do we have for each typhoon?
df_complete.groupby('typhoon_name').count()['grid_point_id']
```




    typhoon_name
    ALPHA      321
    DEAN       321
    DENNIS     321
    ELSA       321
    EMILY      642
    ERIKA      321
    ERNESTO    321
    FAY        321
    GUSTAV     321
    HANNA      321
    IKE        321
    IRENE      321
    IRMA       321
    ISAAC      321
    IVAN       321
    JEANNE     321
    LAURA      321
    LILI       321
    MATTHEW    321
    NOEL       321
    OLGA       321
    SANDY      321
    STAN       321
    TOMAS      321
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



![png](08_data_merging_files/08_data_merging_35_0.png)




```python
# Spoiler
plt.hist(df.perc_dmg_grid)
plt.yscale('log')
plt.title('Distribution of housing damage at grid level')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```



![png](08_data_merging_files/08_data_merging_36_0.png)



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
      <th>25</th>
      <td>1333</td>
      <td>36.7</td>
      <td>0.0</td>
      <td>1</td>
      <td>5457.737332</td>
      <td>4.050314</td>
      <td>0.528598</td>
    </tr>
    <tr>
      <th>50</th>
      <td>571</td>
      <td>28.7</td>
      <td>0.0</td>
      <td>1</td>
      <td>1363.143498</td>
      <td>0.015999</td>
      <td>0.009933</td>
    </tr>
    <tr>
      <th>75</th>
      <td>694</td>
      <td>53.1</td>
      <td>0.0</td>
      <td>1</td>
      <td>3892.528356</td>
      <td>0.020542</td>
      <td>0.015541</td>
    </tr>
    <tr>
      <th>100</th>
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
      <th>7900</th>
      <td>983</td>
      <td>33.4</td>
      <td>97619.0</td>
      <td>1</td>
      <td>8130.354546</td>
      <td>41.891230</td>
      <td>4.504061</td>
    </tr>
    <tr>
      <th>7925</th>
      <td>1076</td>
      <td>53.1</td>
      <td>103463.0</td>
      <td>1</td>
      <td>12806.655888</td>
      <td>120.274683</td>
      <td>6.099569</td>
    </tr>
    <tr>
      <th>7950</th>
      <td>1159</td>
      <td>53.1</td>
      <td>170524.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>131.428020</td>
      <td>7.151623</td>
    </tr>
    <tr>
      <th>7975</th>
      <td>1118</td>
      <td>53.1</td>
      <td>277305.0</td>
      <td>1</td>
      <td>18853.739467</td>
      <td>124.936365</td>
      <td>6.479317</td>
    </tr>
    <tr>
      <th>8000</th>
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
    output_dir / "new_model_training_dataset_hti.csv", index=False
)

df_stat.reset_index().to_csv(
    output_dir / "hti_stationary_data.csv", index=False
)
```
