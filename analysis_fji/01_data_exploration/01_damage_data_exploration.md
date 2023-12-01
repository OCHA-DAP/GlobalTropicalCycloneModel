```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from pathlib import Path
```


```python
input_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/02_housing_damage/input/"
)
output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/02_housing_damage/output/"
)

# Load Fiji
fiji = gpd.read_file(
    input_dir / "adm2_shp_fixed.gpkg"
)
fiji = fiji.to_crs('EPSG:4326')

# Load hosuing damage
# df_housing = pd.read_csv(input_dir / "fji_impact_data/processed_house_impact_new.csv")
df_housing = pd.read_csv(input_dir / "fji_impact_data/processed_house_impact.csv")
```

## Municipality level housing damage analysis

Lets explore the housing dataset


```python
df_housing
```
```python
#How many provinces do we have?
len(df_housing.Province.unique())
```
```python
#How many typhoons do we have damage data for?
len(df_housing.nameyear.unique())
```
Lets create a new feature: the sum of major damage and destroyed

```python
df_housing['damage'] = df_housing['Destroyed'] + df_housing['Major Damage']
df_housing['damage'] = df_housing['damage'].astype(int)
```

What type of admin info do we have for each of the 9 typhoons?


```python
# Admin info for each typhoon
aux1 = df_housing[['Cyclone Name','ADM1_NAME']].drop_duplicates().groupby('Cyclone Name').count().rename({'ADM1_NAME':'ADM1_points'}, axis=1)
aux2 = df_housing[['Cyclone Name','ADM2_NAME']].dropna().groupby('Cyclone Name').count().rename({'ADM2_NAME':'ADM2_points'}, axis=1)
df_housing_info = aux1.merge(aux2,how='outer',left_index=True, right_index=True).fillna(0)
df_housing_info['ADM2_points'] = df_housing_info['ADM2_points'].astype('int')
df_housing_info
```
So

. We have adm1 info for every typhoon


. We have adm2 info for 5 out of 9 typhoon

The admin 1 typhoons. What regions do they affect?


```python
typhoons_adm1 = list(df_housing_info[df_housing_info.ADM2_points == 0].index)
aux3 = df_housing[['Cyclone Name','ADM1_NAME']].drop_duplicates()
aux3[aux3['Cyclone Name'].isin(typhoons_adm1)]
```
## Maps for the municipality level dataset


```python
typhoons_adm2 = list(df_housing_info[df_housing_info.ADM2_points != 0].index)
typhoons_adm1 = list(df_housing_info[df_housing_info.ADM2_points == 0].index)
```


```python
#how is the fiji dataset?
fiji.head(4)
```
### Amin 2 typhoons


```python
for t in typhoons_adm2:
    df_aux = df_housing[df_housing['Cyclone Name'] == t][['Cyclone Name','Major Damage','Destroyed','Province','Division','damage']]
    provinces = df_aux['Province'].to_list()

    fiji['colors'] = 'lightblue'  # Default color for all provinces
    fiji.loc[fiji['NAME_2'].isin(provinces), 'colors'] = 'red'  # Color for the specified provinces

    #to prevent problems
    fiji = fiji.to_crs('EPSG:4326')

    # Merge the Fiji shapefile with the numerical information DataFrame based on ADM2_CODE
    merged_df = fiji.merge(df_aux, left_on='NAME_2', right_on='Province')

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8))


    # Plot the Fiji provinces with the specified colors
    fiji.plot(ax=ax, color=fiji['colors'], edgecolor='gray', legend=True, label='Province Colors')


    # Annotate the map with numerical information
    for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontsize=11, fontweight='bold')

    ax.set_ylim(-19.5,-16)
    ax.set_xlim(176.5,182.5)
    ax.set_title('Typhoon {}, admin 2 level'.format(t))

    # Add a legend based on the colors of the provinces
    # Manually create an alternative legend at coordinates (177, -16.5)
    ax.add_patch(plt.Rectangle((176.6, -16.12), 0.05, 0.05, color='lightblue'))
    ax.annotate('Not affected areas', xy=(176.75, -16.1), fontsize=10, ha='left', va='center')

    ax.add_patch(plt.Rectangle((176.6, -16.22), 0.05, 0.05, color='red'))
    ax.annotate('Affected areas', xy=(176.75, -16.2), fontsize=10, ha='left', va='center')

    ax.plot(176.63, -16.30, 'ko', label='Major damage')
    ax.annotate('Damage (in numbers)', xy=(176.75, -16.3), fontsize=10, ha='left', va='center')

    #plt.savefig('typhhon_adm2_{}'.format(t))
    plt.show()

```

### Admin 1 typhoons


```python
for t in typhoons_adm1:
    df_aux = df_housing[df_housing['Cyclone Name'] == t][['Cyclone Name','Major Damage','Province','Division','damage']]
    divisions = df_aux['Division'].to_list()

    fiji['colors'] = 'lightblue'  # Default color for all provinces
    fiji.loc[fiji['NAME_1'].isin(divisions), 'colors'] = 'red'  # Color for the specified provinces

    #to prevent problems
    fiji = fiji.to_crs('EPSG:4326')

    # Merge the Fiji shapefile with the numerical information DataFrame based on ADM2_CODE
    merged_df = fiji.merge(df_aux, left_on='NAME_1', right_on='Division')

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8))


    # Plot the Fiji provinces with the specified colors
    fiji.plot(ax=ax, color=fiji['colors'], edgecolor='gray', legend=True, label='Region Colors')


    # Annotate the map with numerical information
    for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontsize=11, fontweight='bold')

    ax.set_ylim(-19.5,-16)
    ax.set_xlim(176.5,182.5)
    ax.set_title('Typhoon {}, admin 1 level'.format(t))

    # Add a legend based on the colors of the provinces
    # Manually create an alternative legend at coordinates (177, -16.5)
    ax.add_patch(plt.Rectangle((176.6, -16.12), 0.05, 0.05, color='lightblue'))
    ax.annotate('Not affected areas', xy=(176.75, -16.1), fontsize=10, ha='left', va='center')

    ax.add_patch(plt.Rectangle((176.6, -16.22), 0.05, 0.05, color='red'))
    ax.annotate('Affected areas', xy=(176.75, -16.2), fontsize=10, ha='left', va='center')

    ax.plot(176.63, -16.30, 'ko', label='Major damage')
    ax.annotate('Damage per region \n(in numbers)', xy=(176.75, -16.40), fontsize=10, ha='left', va='center')

    plt.savefig('typhhon_adm1_{}'.format(t))
    plt.show()

```
