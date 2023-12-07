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

```python
len(df_damage.grid_point_id.unique())
```

```python
# Rename colums in the buildings dataset
df_buildings = df_buildings_raw[['id','numbuildings']].rename({'id':'grid_point_id', 'numbuildings':'total_buildings'},axis=1)
```

```python
df_buildings
```

## Read in windfield

```python
# Read in the data file

filename = input_dir / "01_windfield/windfield_data_fji_new.csv"

df_windfield = pd.read_csv(filename)
df_windfield.columns
```

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

```python
len(df_windfield.grid_point_id.unique())
```

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

```python
len(df_rainfall.grid_point_id.unique())
```

## Check grid points ids matches

```python
wind_ids = df_windfield.grid_point_id.unique()
house_ids = df_damage.grid_point_id.unique()
rain_ids = df_rainfall.grid_point_id.unique()
```

```python
len(set(wind_ids) & set(house_ids))
```

```python
len(set(rain_ids) & set(house_ids))
```

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

```python
# How many points do we have for each typhoon?
df_complete.groupby('typhoon_name').count()['grid_point_id']
```

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

OBS: total_buildings_damage is a fractonary number because we are splitting the number of houses destroyed per grid cell.

OBS: Is not a ptoblem that sometimes total_buildings_damaged > total_buildings because total_buildings_damaged is at municipality level and total_buildings is by grid.

```python
# Example
df[df.index.get_level_values('typhoon_name') == 'GITA']
```

```python
# Spoiler
plt.hist(df.perc_dmg_grid)
plt.title('Distribution of housing damage at grid level')
plt.xlabel('% of buildings damaged')
plt.ylabel('Count')
plt.show()
```

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
