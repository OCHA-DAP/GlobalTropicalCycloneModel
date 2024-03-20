# Windfields

This notebook is for downloading typhoon tracks from
IBTrACS and generating the windfields.


```python
from pathlib import Path
import os

from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
```


```python
DEG_TO_KM = 111.1  # Convert 1 degree to km
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_vnm/02_model_features"
```

## Get typhoon data

Typhoon IDs from IBTrACS can be manually taken from
[here](https://ncics.org/ibtracs/index.php?name=browse-name) but let's see what we have in the WP region using ibtracks


```python
# Import list of typhoons that affected vietnam
damage_df = pd.read_csv(input_dir / "02_housing_damage/input/viet_damage_adm1_level.csv")
damage_df['Year'] = damage_df.Year.astype('str')
damage_df['typhoon_name'] = damage_df.typhoon_name.str.upper()

# Select all typhoons that happened in the year range of our dataset and in the NorthWestPacific basin
sel_ibtracs_wp = TCTracks.from_ibtracs_netcdf(year_range=(1997, 2020), correct_pres=False, basin='WP')
sel_ibtracs_ni = TCTracks.from_ibtracs_netcdf(year_range=(2018, 2020), correct_pres=False, basin='NI') #For Matmo
```

    2024-02-20 20:07:57,985 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-02-20 20:09:03,344 - climada.hazard.tc_tracks - WARNING - 26 storm events are discarded because no valid wind/pressure values have been found: 1997225N09186, 1997236N10172, 1997273N13151, 1997296N07176, 1998204N17144, ...
    2024-02-20 20:09:03,353 - climada.hazard.tc_tracks - WARNING - 1 storm events are discarded because only one valid timestep has been found: 2004327N16125.
    2024-02-20 20:09:10,236 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.



```python
df_ibtracks = pd.DataFrame()
for i in range(len(sel_ibtracs_wp.data)):
    typhoon_name = sel_ibtracs_wp.data[i].name
    year = np.datetime64(np.array(sel_ibtracs_wp.data[i].time[0]), 'Y').astype('str')
    id = sel_ibtracs_wp.data[i].sid
    df_aux = pd.DataFrame({
        'typhoon_name':[typhoon_name],
        'Year':[year],
        'id':[id]
    })
    df_ibtracks = pd.concat([df_ibtracks, df_aux])

df_ibtracks_ni = pd.DataFrame()
for i in range(len(sel_ibtracs_ni.data)):
    typhoon_name = sel_ibtracs_ni.data[i].name
    year = np.datetime64(np.array(sel_ibtracs_ni.data[i].time[0]), 'Y').astype('str')
    id = sel_ibtracs_ni.data[i].sid
    df_aux = pd.DataFrame({
        'typhoon_name':[typhoon_name],
        'Year':[year],
        'id':[id]
    })
    df_ibtracks_ni = pd.concat([df_ibtracks_ni, df_aux])
```


```python
# Coincidences
df_in = df_ibtracks.merge(damage_df, on=['typhoon_name','Year']).drop_duplicates()
# Not matches
df_out = df_ibtracks.merge(damage_df, on=['typhoon_name','Year'], how='outer', indicator=True).loc[lambda x: x['_merge'] == 'right_only']

# Just duplicated ones
df_duplicated = df_in[df_in.duplicated(['typhoon_name', 'Year'], keep=False)]
```


```python
df_duplicated
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
      <th>id</th>
      <th>total_bld_dmg</th>
      <th>region_affected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>PODUL</td>
      <td>2019</td>
      <td>2019236N08143</td>
      <td>874.0</td>
      <td>['RRD', 'NC']</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PODUL</td>
      <td>2019</td>
      <td>2019237N08140</td>
      <td>874.0</td>
      <td>['RRD', 'NC']</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_out
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
      <th>id</th>
      <th>total_bld_dmg</th>
      <th>region_affected</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>725</th>
      <td>KAITAK</td>
      <td>2012</td>
      <td>NaN</td>
      <td>1297.0</td>
      <td>['NE']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>726</th>
      <td>SON</td>
      <td>2012</td>
      <td>NaN</td>
      <td>56109.0</td>
      <td>['RRD', 'NC']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>727</th>
      <td>SONTINH</td>
      <td>2012</td>
      <td>NaN</td>
      <td>60833.0</td>
      <td>['NE', 'RRD']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>728</th>
      <td>PUDOL</td>
      <td>2013</td>
      <td>NaN</td>
      <td>3067.0</td>
      <td>['SC', 'C']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>729</th>
      <td>MANGKUT</td>
      <td>2013</td>
      <td>NaN</td>
      <td>1633.0</td>
      <td>['RRD', 'NC']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>730</th>
      <td>KALMAGE</td>
      <td>2014</td>
      <td>NaN</td>
      <td>4117.0</td>
      <td>['NE', 'RRD']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>731</th>
      <td>MATMO</td>
      <td>2019</td>
      <td>NaN</td>
      <td>1486.0</td>
      <td>['SC', 'C']</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>



### Through Vietnam (duplicated analysis)


```python
ids_in = df_duplicated.id.to_list()
tracks_in = TCTracks()
for track in sel_ibtracs_wp.data:
    if str(int(track.id_no)) in ids_in:
        tracks_in.append(track)
```


```python
tracks_in.plot()
```




    <GeoAxes: >





![png](01_windfields_files/01_windfields_11_1.png)



THE DUPLICATED ONE


```python
#PODUL
podul1 = TCTracks.from_ibtracs_netcdf(storm_id='2019236N08143')
podul2 = TCTracks.from_ibtracs_netcdf(storm_id='2019237N08140')

podul1.plot()
plt.show()
podul2.plot()
plt.show()
```

    2024-01-23 22:29:52,260 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    2024-01-23 22:29:52,648 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.




![png](01_windfields_files/01_windfields_13_1.png)





![png](01_windfields_files/01_windfields_13_2.png)



The second typhoon didn't affected the area of interest


```python
df_in_fixed = df_in[df_in.id != '2019237N08140']
```

### df_out analysis


```python
df_ibtracks[df_ibtracks.typhoon_name.isin(df_out.typhoon_name)]
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
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SINLAKU</td>
      <td>2002</td>
      <td>2002240N16155</td>
    </tr>
    <tr>
      <th>0</th>
      <td>MATMO</td>
      <td>2008</td>
      <td>2008135N16125</td>
    </tr>
    <tr>
      <th>0</th>
      <td>SINLAKU</td>
      <td>2008</td>
      <td>2008252N16128</td>
    </tr>
    <tr>
      <th>0</th>
      <td>MATMO</td>
      <td>2014</td>
      <td>2014197N10137</td>
    </tr>
    <tr>
      <th>0</th>
      <td>SINLAKU</td>
      <td>2014</td>
      <td>2014329N08131</td>
    </tr>
  </tbody>
</table>
</div>



Which are the important ones?


```python
df_out
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
      <th>id</th>
      <th>total_bld_dmg</th>
      <th>region_affected</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>725</th>
      <td>KAITAK</td>
      <td>2012</td>
      <td>NaN</td>
      <td>1297.0</td>
      <td>['NE']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>726</th>
      <td>SON</td>
      <td>2012</td>
      <td>NaN</td>
      <td>56109.0</td>
      <td>['RRD', 'NC']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>727</th>
      <td>SONTINH</td>
      <td>2012</td>
      <td>NaN</td>
      <td>60833.0</td>
      <td>['NE', 'RRD']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>728</th>
      <td>PUDOL</td>
      <td>2013</td>
      <td>NaN</td>
      <td>3067.0</td>
      <td>['SC', 'C']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>729</th>
      <td>MANGKUT</td>
      <td>2013</td>
      <td>NaN</td>
      <td>1633.0</td>
      <td>['RRD', 'NC']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>730</th>
      <td>KALMAGE</td>
      <td>2014</td>
      <td>NaN</td>
      <td>4117.0</td>
      <td>['NE', 'RRD']</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>731</th>
      <td>MATMO</td>
      <td>2019</td>
      <td>NaN</td>
      <td>1486.0</td>
      <td>['SC', 'C']</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>



We need to look for them. At least SONTINH and SON


```python
# SONTINH is in reality SON-TINH
# SON is in fact SON-TINH. SON comes from the dataset https://docs.google.com/document/d/1GSclBZpiX3bXbtnVcifIsdeTylhuD8Yy/edit
# SONTINH comes from the IMHEM dataset (not so reliable).
# Lets work with SON and rename it SON-TINH to get the windfield data
# The same happens for some other events: kaitai is kai-tai, PUDOL is infact PODUL, MANGKUT is MANGKHUT, etc..
# MATMO affected 2 basins: WP and NI and its labeled in the NI basin, thats what happening. Also it has a different name.

df_aux_son = damage_df[damage_df.typhoon_name == 'SON']
df_aux_kai = damage_df[damage_df.typhoon_name == 'KAITAK']
df_aux_pod = damage_df[damage_df.typhoon_name == 'PUDOL']
df_aux_man = damage_df[damage_df.typhoon_name == 'MANGKUT']
df_aux_kal = damage_df[damage_df.typhoon_name == 'KALMAGE']
df_aux_mat = damage_df[damage_df.typhoon_name == 'MATMO']
id_son = df_ibtracks[df_ibtracks.typhoon_name =='SON-TINH'].iloc[0].id #2012
id_kai = df_ibtracks[df_ibtracks.typhoon_name =='KAI-TAK'].iloc[2].id #2012
id_pod = df_ibtracks[df_ibtracks.typhoon_name =='PODUL'].iloc[2].id #2013
id_man = df_ibtracks[df_ibtracks.typhoon_name =='MANGKHUT'].iloc[0].id #2013
id_kal = df_ibtracks[df_ibtracks.typhoon_name =='KALMAEGI'].iloc[2].id #2014
id_mat = df_ibtracks_ni[df_ibtracks_ni.typhoon_name =='BULBUL:MATMO'].iloc[0].id #2014

df_aux_son['id'] = id_son
df_aux_kai['id'] = id_kai
df_aux_pod['id'] = id_pod
df_aux_man['id'] = id_man
df_aux_kal['id'] = id_kal
df_aux_mat['id'] = id_mat

df_out_tot = pd.concat([df_aux_son, df_aux_kai, df_aux_pod, df_aux_man, df_aux_kal, df_aux_mat]).reset_index(drop=True)
```

#### Checks

Lets check if these events took place in Vietnam


```python
sontinh_track = TCTracks.from_ibtracs_netcdf(storm_id=id_son)
kaitak_track = TCTracks.from_ibtracs_netcdf(storm_id=id_kai)
podul_track = TCTracks.from_ibtracs_netcdf(storm_id=id_pod)
mangut_track = TCTracks.from_ibtracs_netcdf(storm_id=id_man)
kalmage_track = TCTracks.from_ibtracs_netcdf(storm_id=id_kal)
matmo_track = TCTracks.from_ibtracs_netcdf(storm_id=id_mat)
```


```python
ax1 = sontinh_track.plot()
ax1.set_title('SON-TINH')
ax2 = kaitak_track.plot()
ax2.set_title('KAI-TAK')
ax3 = podul_track.plot()
ax3.set_title('PODUL')
ax4 = mangut_track.plot()
ax4.set_title('MANGUL')
ax5 = kalmage_track.plot()
ax5.set_title('KALMAGE')
ax6 = matmo_track.plot()
ax6.set_title('MATMO')

plt.show()
```



![png](01_windfields_files/01_windfields_25_0.png)





![png](01_windfields_files/01_windfields_25_1.png)





![png](01_windfields_files/01_windfields_25_2.png)





![png](01_windfields_files/01_windfields_25_3.png)





![png](01_windfields_files/01_windfields_25_4.png)





![png](01_windfields_files/01_windfields_25_5.png)



Seems that all tyhoons affected the area of interest.

Podul is strange, but it might affected some small island. Looking at the coordinates, is in the Vietnam area at least.

#### Save it


```python
# Save all the events!
df_in_fixed_tot = pd.concat([df_in_fixed, df_out_tot]).reset_index(drop=True)
```


```python
# Save it
df_in_fixed_tot.to_csv(input_dir / '01_windfield/typhoons.csv', index=False)
```

IbTracks doesnt have information of windspeed about these specifuc typhoons.


```python
# So
intersection = df_in_fixed_tot.copy()
```

### The classic proccess of getting tracks


```python
# Download NECESARY tracks
sel_ibtracs = [];i=0
for track in intersection.id:
    sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=track))
    print('Track {} de {}'.format(i, len(intersection)-1))
    i+=1
```

    2024-02-20 20:09:48,751 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 0 de 37
    2024-02-20 20:09:52,302 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 1 de 37
    2024-02-20 20:09:55,711 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 2 de 37
    2024-02-20 20:09:59,122 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 3 de 37
    2024-02-20 20:10:02,431 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 4 de 37
    2024-02-20 20:10:05,841 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 5 de 37
    2024-02-20 20:10:09,463 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 6 de 37
    2024-02-20 20:10:12,832 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 7 de 37
    2024-02-20 20:10:16,580 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 8 de 37
    2024-02-20 20:10:19,855 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 9 de 37
    2024-02-20 20:10:22,906 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 10 de 37
    2024-02-20 20:10:25,988 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 11 de 37
    2024-02-20 20:10:29,275 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 12 de 37
    2024-02-20 20:10:32,772 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 13 de 37
    2024-02-20 20:10:36,236 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 14 de 37
    2024-02-20 20:10:39,853 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 15 de 37
    2024-02-20 20:10:43,326 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 16 de 37
    2024-02-20 20:10:46,409 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 17 de 37
    2024-02-20 20:10:49,493 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 18 de 37
    2024-02-20 20:10:52,665 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 19 de 37
    2024-02-20 20:10:55,720 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 20 de 37
    2024-02-20 20:10:58,875 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 21 de 37
    2024-02-20 20:11:01,949 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 22 de 37
    2024-02-20 20:11:05,268 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 23 de 37
    2024-02-20 20:11:08,430 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 24 de 37
    2024-02-20 20:11:11,590 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 25 de 37
    2024-02-20 20:11:14,620 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 26 de 37
    2024-02-20 20:11:17,832 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 27 de 37
    2024-02-20 20:11:20,849 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 28 de 37
    2024-02-20 20:11:24,018 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 29 de 37
    2024-02-20 20:11:27,057 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 30 de 37
    2024-02-20 20:11:30,261 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 31 de 37
    2024-02-20 20:11:33,334 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 32 de 37
    2024-02-20 20:11:36,506 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 33 de 37
    2024-02-20 20:11:39,569 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 34 de 37
    2024-02-20 20:11:42,766 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 35 de 37
    2024-02-20 20:11:45,935 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 36 de 37
    2024-02-20 20:11:49,112 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 37 de 37



```python
# Interpolation
#obs: .interp(x0,x,f(x)) gives the position of x0 in the fitting of (x,f(x))
#obs: daterange consider the track between certain intervals as discrete points instead of a continuous
tc_tracks = TCTracks()
for track in sel_ibtracs:
    tc_track = track.get_track()
    tc_track.interp(
        time = pd.date_range(tc_track.time.values[0], tc_track.time.values[-1], freq="30T")
    )
    tc_tracks.append(tc_track)
```


```python
# Plot the tracks
# Takes a while, especially after the interpolation.
ax = tc_tracks.plot()
ax.set_title('Vietnam and surroundings Typhoon Tracks', size=20)
#ax.legend(loc='upper left')
plt.show()
```



![png](01_windfields_files/01_windfields_35_0.png)



### Define some functions


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
                typhoon_name=[tc_track.name] * npoints,
                track_id=[event_id] * npoints,
                grid_point_id=grids["id"],
                wind_speed=windfield,
                track_distance=tc_track_distance,
                geometry = grids.geometry
            )
        )
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
    return df_windfield

# Define a function to calculate mean values for neighboring cells
def calculate_mean_for_neighbors(idx, gdf, buffer_size):
    row = gdf.iloc[idx]
    if row['wind_speed'] == 0:  # Check if wind_speed is 0
        buffered = row['geometry'].buffer(buffer_size)  # Adjust buffer size as needed

        # Find neighboring geometries that intersect with the buffer, excluding the current geometry
        neighbors = gdf[~gdf.geometry.equals(row['geometry']) & gdf.geometry.intersects(buffered)]

        if not neighbors.empty:
            # drop rows with 0 windspeed vals (we dont want to compute the mean while considering these cells)
            neighbors = neighbors[neighbors['wind_speed'] !=0]
            if len(neighbors) !=0:
                mean_val = neighbors['wind_speed'].mean()
            else:
                mean_val = 0
            return mean_val
    return row['wind_speed']  # Return the original value if no neighbors or wind_speed != 0

# Function to add interpolation points
def add_interpolation_points(data, num_points_between):
    new_x_list = []
    for i in range(len(data) - 1):
        start_point, end_point = data[i], data[i + 1]
        interp_x = list(np.linspace(start_point, end_point, num_points_between + 2))
        if i == 0:
            new_x_list.append(interp_x)
        elif i == (len(data) - 1):
            new_x_list.append(interp_x)
        else:
            new_x_list.append(interp_x[1:])

    new_x = np.concatenate(new_x_list)

    return new_x

# Create xarray
def adjust_tracks(forecast_df):
    track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', np.array(forecast_df.MeanWind.values, dtype='float32')), #0.514444 --> kn to m/s
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
            'name': name,
            'sid' : custom_sid,
            'orig_event_flag': True,
            'data_provider': 'Custom',
            'id_no' : custom_idno,
            'category': int(max(forecast_df.Category.iloc)),
        }
    )
    track = track.set_coords(['lat', 'lon'])
    return track
```

## Construct the windfield

The typhoon tracks will be used to construct the windfield.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.


```python
# Just grid-land overlap
filepath = (
    input_dir
    / "02_housing_damage/output/viet_0.1_degree_grid_centroids_land_overlap_new.gpkg"
)
gdf = gpd.read_file(filepath)
# Include oceans
filepath_complete = (
    input_dir
    / "02_housing_damage/output/viet_0.1_degree_grid_centroids_new.gpkg"
)
gdf_all = gpd.read_file(filepath_complete)

# Centroids
cent = Centroids.from_geodataframe(gdf) # grid-land overlap
cent_all = Centroids.from_geodataframe(gdf_all) # include oceans
```

    2024-02-20 20:12:29,215 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.
    2024-02-20 20:12:29,218 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



```python
cent.check()
cent.plot()
plt.show()
```



![png](01_windfields_files/01_windfields_40_0.png)



Add interpolation points


```python
tracks = TCTracks()
for i in range(len(tc_tracks.get_track())):
    # Define relevant features
    track_xarray = tc_tracks.get_track()[i]
    time_array = np.array(track_xarray.time)
    time_step_array = np.array(track_xarray.time_step)
    lat_array = np.array(track_xarray.lat)
    lon_array = np.array(track_xarray.lon)
    max_sustained_wind_array = np.array(track_xarray.max_sustained_wind)
    central_pressure_array = np.array(track_xarray.central_pressure)
    environmental_pressure_array = np.array(track_xarray.environmental_pressure)
    r_max_wind_array = np.array(track_xarray.radius_max_wind)
    r_oci_array = np.array(track_xarray.radius_oci)

    # Define new variables
    # Interpolate every important data
    w = max_sustained_wind_array.copy()
    t = time_array.copy()
    t_step = time_step_array.copy()
    lat = lat_array.copy()
    lon = lon_array.copy()
    cp = central_pressure_array.copy()
    ep = environmental_pressure_array.copy()
    rmax = r_max_wind_array.copy()
    roci = r_oci_array.copy()

    # Define the number of points to add between each pair of data points
    num_points_between = 2

    # Add interpolation points to regulat variables
    new_w = add_interpolation_points(w, num_points_between)
    new_t_step = add_interpolation_points(t_step, num_points_between)
    new_lat = add_interpolation_points(lat, num_points_between)
    new_lon = add_interpolation_points(lon, num_points_between)
    new_cp = add_interpolation_points(cp, num_points_between)
    new_ep = add_interpolation_points(ep, num_points_between)
    new_rmax = add_interpolation_points(rmax, num_points_between)
    new_roci = add_interpolation_points(roci, num_points_between)

    # Add interpolation points to time variables
    timestamps = np.array([date.astype('datetime64[s]').astype('int64') for date in t])# Convert to seconds
    new_t =  add_interpolation_points(timestamps, num_points_between)
    new_t = [np.datetime64(int(ts), 's') for ts in new_t]# Back to datetime format

    # Define dataframe
    df_t = pd.DataFrame({
        'MeanWind': new_w,
        'PressureOCI': new_ep,
        'Pressure': new_cp,
        'Latitude': new_lat,
        'Longitude': new_lon,
        'RadiusMaxWinds': new_rmax,
        'RadiusOCI': new_roci,
        'time_step': new_t_step,
        'basin': np.array([np.array(track_xarray.basin)[0]] * len(new_t)),
        'forecast_time': new_t,
        'Category': track_xarray.category
    })

    # Define a custom id
    custom_idno = track_xarray.id_no
    custom_sid = track_xarray.sid
    name = track_xarray.name# + ' interpolated'

    # Define track as climada likes it
    track = TCTracks()
    track.data = [adjust_tracks(df_t)]

    # Tracks modified
    tracks.append(track.get_track())
```

Create Tropcyclone class


```python
tc_all = TropCyclone.from_tracks(
    tracks, centroids=cent_all, store_windfields=True, intensity_thres=0
)

# Create grid-level windfield
df_windfield_interpolated = windfield_to_grid(tc=tc_all, tracks=tracks, grids=gdf_all)
```


```python
# Overlap
df_windfield_interpolated_overlap = df_windfield_interpolated[df_windfield_interpolated.grid_point_id.isin(gdf.id)]
```

## Examples


```python
typhoons = df_in_fixed_tot.typhoon_name.unique()
```


```python
# Let's look at a specific typhoon as an example.
name = typhoons[0]
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name, size=20)
plt.show()
```



![png](01_windfields_files/01_windfields_48_0.png)




```python
# Let's look at a specific typhoon as an example.
name = typhoons[1]
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name, size=20)
plt.show()
```



![png](01_windfields_files/01_windfields_49_0.png)



## Sanity checks


```python
# Check nans
df_windfield_interpolated.isna().sum()
```




    typhoon_name      0
    track_id          0
    grid_point_id     0
    wind_speed        0
    track_distance    0
    geometry          0
    dtype: int64




```python
# Check number of cells
print(len(df_windfield_interpolated[df_windfield_interpolated.typhoon_name=='LINDA']))
```

    30951



```python
grid_input = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_viet/02_new_model_input/02_housing_damage/output/"
)

# Load grid
grid_land_overlap = gpd.read_file(grid_input / "viet_0.1_degree_grid_land_overlap_new.gpkg")
grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)
grid = grid_land_overlap.copy()
```


```python
# Track path
tc_track = tracks.get_track()[2]
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

fig, ax = plt.subplots(1,1, figsize=(5,5))
# Plot intensity
name = tc_track.name
geo_windfield = gpd.GeoDataFrame(df_windfield_interpolated_overlap[df_windfield_interpolated_overlap.typhoon_name == name])
geo_grid_wind = grid.merge(geo_windfield[['grid_point_id', 'wind_speed']], left_on='id', right_on='grid_point_id')

geo_grid_wind.plot(column='wind_speed', cmap='Reds', linewidth=0.2, edgecolor='0.3', ax=ax, legend=True)
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

#ax.axis('off')
ax.set_xlim([101, 118.5])
ax.set_ylim([6.5, 24])
ax.set_title('{} (2006) \nWindspeed [m/s]'.format(name))
plt.show()
```



![png](01_windfields_files/01_windfields_54_0.png)




```python
# Some grid plots
fig, ax = plt.subplots(1,3, figsize=(10,4))
ax = ax.flatten()

for i in range(3):
    # Track path
    tc_track = tracks.get_track()[-3:][i]
    points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
    track_points = gpd.GeoDataFrame(geometry=points)
    tc_track_line = LineString(points)
    track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

    # Plot intensity
    name = tc_track.name
    geo_windfield = gpd.GeoDataFrame(df_windfield_interpolated_overlap[df_windfield_interpolated_overlap.typhoon_name == name])
    geo_windfield.plot(column='wind_speed', cmap='Reds', linewidth=0.2, edgecolor='0.3', ax=ax[i], legend=True)
    track_line.plot(ax=ax[i], color='k', linewidth=1, label='Typhoon track')

    ax[i].axis('off')
    ax[i].set_title(name)

plt.tight_layout()
plt.show()
```



![png](01_windfields_files/01_windfields_55_0.png)




```python
# Plot wind speed against track distance
df_windfield_interpolated_overlap.plot.scatter("track_distance", "wind_speed")
plt.xlabel('Track Distance [Km]')
plt.ylabel('Wind Speed [m/s]')

plt.show()
```



![png](01_windfields_files/01_windfields_56_0.png)



## Save everything


```python
# Save df as a csv file
df_windfield_interpolated_overlap.to_csv(
    input_dir / "01_windfield/windfield_data_viet_new_fixed_interpolated_overlap.csv",
    index=False
)
```

## Creating metadata

We need to create a dataframe with information about

-  Start date
-  landfalldate
-  landfall_time
-  End date


```python
from datetime import datetime
# Load Vietnam
viet_path = input_dir / '02_housing_damage/input/'
vietnam = gpd.read_file(
    viet_path / "adm2_shp_fixed.gpkg"
)
vietnam = vietnam.to_crs('EPSG:4326')
```


```python
df_metadata = pd.DataFrame()
for i in range(len(tracks.data)):
    # Basics
    startdate = np.datetime64(np.array(tracks.data[i].time[0]), 'D')
    enddate = np.datetime64(np.array(tracks.data[i].time[-1]), 'D')
    name = tracks.data[i].name
    year = tracks.data[i].sid[:4]
    nameyear = name + year

    # For the landfall
    # Track path
    tc_track = tracks.get_track()[i]
    points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
    track_points = gpd.GeoDataFrame(geometry=points)

    # Set crs
    track_points.crs = vietnam.crs

    try:
        # intersection --> Look for first intersection == landfall
        min_index = vietnam.sjoin(track_points)['index_right'].min()

        landfalldate = np.datetime64(np.array(tracks.data[i].time[min_index]), 'D')
        landfall_time = str(np.datetime64(np.array(tracks.data[i].time[min_index]), 's')).split('T')[1]
    except:
        # No landfall situation
        landfalldate = np.nan
        landfall_time = np.nan

    # Create df
    df_aux = pd.DataFrame({
        'typhoon': [nameyear],
        'startdate': [startdate],
        'enddate': [enddate],
        'landfalldate': [landfalldate],
        'landfall_time': [landfall_time]
    }
    )
    df_metadata = pd.concat([df_metadata, df_aux])
df_metadata = df_metadata.reset_index(drop=True)
```

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_2437/1386227293.py:39: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      df_metadata = pd.concat([df_metadata, df_aux])
    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_2437/1386227293.py:39: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      df_metadata = pd.concat([df_metadata, df_aux])
    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_2437/1386227293.py:39: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      df_metadata = pd.concat([df_metadata, df_aux])
    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_2437/1386227293.py:39: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      df_metadata = pd.concat([df_metadata, df_aux])
    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_2437/1386227293.py:39: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      df_metadata = pd.concat([df_metadata, df_aux])



```python
# Merge data idea
fig, ax = plt.subplots(1,1)
vietnam.plot(ax=ax)
track_points.plot(ax=ax, color='r')
plt.show()
```



![png](01_windfields_files/01_windfields_63_0.png)




```python
df_metadata.head()
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
      <th>typhoon</th>
      <th>startdate</th>
      <th>enddate</th>
      <th>landfalldate</th>
      <th>landfall_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LINDA1997</td>
      <td>1997-11-01</td>
      <td>1997-11-09</td>
      <td>1997-11-02</td>
      <td>10:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DAMREY2005</td>
      <td>2005-09-21</td>
      <td>2005-09-27</td>
      <td>2005-09-27</td>
      <td>03:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XANGSANE2006</td>
      <td>2006-09-26</td>
      <td>2006-10-01</td>
      <td>2006-10-01</td>
      <td>02:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN2006</td>
      <td>2006-11-26</td>
      <td>2006-12-05</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LEKIMA2007</td>
      <td>2007-09-30</td>
      <td>2007-10-04</td>
      <td>2007-10-03</td>
      <td>13:00:00</td>
    </tr>
  </tbody>
</table>
</div>



Look that for some typhoons, there's not even a landfall. Let's check them out


```python
nans_meta = df_metadata[df_metadata.landfall_time.isna()]
nans_meta
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
      <th>typhoon</th>
      <th>startdate</th>
      <th>enddate</th>
      <th>landfalldate</th>
      <th>landfall_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>DURIAN2006</td>
      <td>2006-11-26</td>
      <td>2006-12-05</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>VICENTE2012</td>
      <td>2012-07-21</td>
      <td>2012-07-24</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GAEMI2012</td>
      <td>2012-10-01</td>
      <td>2012-10-06</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SONCA2017</td>
      <td>2017-07-23</td>
      <td>2017-07-25</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PODUL2013</td>
      <td>2013-11-14</td>
      <td>2013-11-14</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1,5, figsize = (15,8))
for i in range(5):
    j = nans_meta.index[i]
    tc_track = tracks.get_track()[j]
    points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
    track_points = gpd.GeoDataFrame(geometry=points)
    #Plot
    vietnam.plot(ax=ax[i])
    track_points.plot(ax=ax[i], color='k', label='track')
    track_points.iloc[0:2].plot(ax=ax[i], color = 'r', label='beginning')

    ax[i].legend()
    ax[i].set_title(nans_meta.iloc[i].typhoon)
    ax[i].set_xlim([100,120])
    ax[i].set_ylim([5, 25])

plt.show()
```



![png](01_windfields_files/01_windfields_67_0.png)



We should treat all these cases separated from the rest.

We should consider for the rainfall adquisition:
-  VICENTE, GAEMI, SONCA, PODUL: from start to end
-  DURIAN: from 120lon onwards.


```python
# Durian
j = nans_meta.index[0]
durian_lon = tracks.get_track()[j].lon
durian_time = tracks.get_track()[j].time
df_durian = pd.DataFrame({'lon': durian_lon, 'time': durian_time})#.sort_values('lon')
durian_landfall_date = np.datetime64(df_durian[df_durian.lon <= 120].iloc[0].time, 'D')
durian_landfall_time = str(df_durian[df_durian.lon <= 120].iloc[0].time).split(' ')[1]
```


```python
df_metadata_fixed = pd.DataFrame()
for i in range(len(tracks.data)):
    # Basics
    startdate = np.datetime64(np.array(tracks.data[i].time[0]), 'D')
    enddate = np.datetime64(np.array(tracks.data[i].time[-1]), 'D')
    name = tracks.data[i].name
    year = tracks.data[i].sid[:4]
    nameyear = name + year

    # For the landfall
    # Track path
    tc_track = tracks.get_track()[i]
    points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
    track_points = gpd.GeoDataFrame(geometry=points)

    # Set crs
    track_points.crs = vietnam.crs

    try:
        # intersection --> Look for first intersection == landfall
        min_index = vietnam.sjoin(track_points)['index_right'].min()

        landfalldate = np.datetime64(np.array(tracks.data[i].time[min_index]), 'D')
        landfall_time = str(np.datetime64(np.array(tracks.data[i].time[min_index]), 's')).split('T')[1]
    except:
        if name == 'DURIAN':
            landfalldate = durian_landfall_date
            landfall_time = durian_landfall_time
        else:
            # No landfall situation --> Use last point
            landfalldate = enddate
            landfall_time = str(np.datetime64(np.array(tracks.data[i].time[-1]), 's')).split('T')[1]

    # Create df
    df_aux = pd.DataFrame({
        'typhoon': [nameyear],
        'startdate': [startdate],
        'enddate': [enddate],
        'landfalldate': [landfalldate],
        'landfall_time': [landfall_time]
    }
    )
    df_metadata_fixed = pd.concat([df_metadata_fixed, df_aux])
df_metadata_fixed = df_metadata_fixed.reset_index(drop=True)
```


```python
# Save it
df_metadata_fixed.to_csv(input_dir / '03_rainfall/input/metadata_typhoons.csv', index=False)
```


```python
df_metadata_fixed
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
      <th>typhoon</th>
      <th>startdate</th>
      <th>enddate</th>
      <th>landfalldate</th>
      <th>landfall_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LINDA1997</td>
      <td>1997-11-01</td>
      <td>1997-11-09</td>
      <td>1997-11-02</td>
      <td>10:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DAMREY2005</td>
      <td>2005-09-21</td>
      <td>2005-09-27</td>
      <td>2005-09-27</td>
      <td>03:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XANGSANE2006</td>
      <td>2006-09-26</td>
      <td>2006-10-01</td>
      <td>2006-10-01</td>
      <td>02:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN2006</td>
      <td>2006-11-26</td>
      <td>2006-12-05</td>
      <td>2006-12-01</td>
      <td>03:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LEKIMA2007</td>
      <td>2007-09-30</td>
      <td>2007-10-04</td>
      <td>2007-10-03</td>
      <td>13:00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KAMMURI2008</td>
      <td>2008-08-05</td>
      <td>2008-08-07</td>
      <td>2008-08-07</td>
      <td>09:00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KETSANA2009</td>
      <td>2009-09-26</td>
      <td>2009-09-30</td>
      <td>2009-09-29</td>
      <td>07:00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CONSON2010</td>
      <td>2010-07-12</td>
      <td>2010-07-17</td>
      <td>2010-07-17</td>
      <td>14:00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MINDULLE2010</td>
      <td>2010-08-23</td>
      <td>2010-08-24</td>
      <td>2010-08-24</td>
      <td>11:00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HAIMA2011</td>
      <td>2011-06-21</td>
      <td>2011-06-24</td>
      <td>2011-06-24</td>
      <td>11:00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NESAT2011</td>
      <td>2011-09-24</td>
      <td>2011-09-30</td>
      <td>2011-09-30</td>
      <td>06:00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PAKHAR2012</td>
      <td>2012-03-29</td>
      <td>2012-04-01</td>
      <td>2012-04-01</td>
      <td>12:00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>VICENTE2012</td>
      <td>2012-07-21</td>
      <td>2012-07-24</td>
      <td>2012-07-24</td>
      <td>12:00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GAEMI2012</td>
      <td>2012-10-01</td>
      <td>2012-10-06</td>
      <td>2012-10-06</td>
      <td>06:00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BEBINCA2013</td>
      <td>2013-06-20</td>
      <td>2013-06-24</td>
      <td>2013-06-23</td>
      <td>12:00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>JEBI2013</td>
      <td>2013-07-31</td>
      <td>2013-08-03</td>
      <td>2013-08-03</td>
      <td>02:00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MANGKHUT2013</td>
      <td>2013-08-06</td>
      <td>2013-08-07</td>
      <td>2013-08-07</td>
      <td>16:00:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>WUTIP2013</td>
      <td>2013-09-27</td>
      <td>2013-09-30</td>
      <td>2013-09-29</td>
      <td>02:00:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NARI2013</td>
      <td>2013-10-09</td>
      <td>2013-10-15</td>
      <td>2013-10-14</td>
      <td>23:00:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>RAMMASUN2014</td>
      <td>2014-07-12</td>
      <td>2014-07-19</td>
      <td>2014-07-19</td>
      <td>10:00:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KALMAEGI2014</td>
      <td>2014-09-12</td>
      <td>2014-09-17</td>
      <td>2014-09-16</td>
      <td>14:00:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KUJIRA2015</td>
      <td>2015-06-21</td>
      <td>2015-06-24</td>
      <td>2015-06-24</td>
      <td>07:00:00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MIRINAE2016</td>
      <td>2016-07-26</td>
      <td>2016-07-28</td>
      <td>2016-07-27</td>
      <td>17:00:00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>DIANMU2016</td>
      <td>2016-08-17</td>
      <td>2016-08-19</td>
      <td>2016-08-19</td>
      <td>05:00:00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>RAI2016</td>
      <td>2016-09-12</td>
      <td>2016-09-13</td>
      <td>2016-09-12</td>
      <td>20:00:00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>TALAS2017</td>
      <td>2017-07-15</td>
      <td>2017-07-17</td>
      <td>2017-07-16</td>
      <td>17:00:00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SONCA2017</td>
      <td>2017-07-23</td>
      <td>2017-07-25</td>
      <td>2017-07-25</td>
      <td>06:00:00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DOKSURI2017</td>
      <td>2017-09-12</td>
      <td>2017-09-15</td>
      <td>2017-09-14</td>
      <td>04:00:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>DAMREY2017</td>
      <td>2017-11-02</td>
      <td>2017-11-04</td>
      <td>2017-11-04</td>
      <td>00:00:00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WIPHA2019</td>
      <td>2019-07-30</td>
      <td>2019-08-03</td>
      <td>2019-08-02</td>
      <td>15:00:00</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PODUL2019</td>
      <td>2019-08-28</td>
      <td>2019-08-29</td>
      <td>2019-08-29</td>
      <td>17:00:00</td>
    </tr>
    <tr>
      <th>31</th>
      <td>SINLAKU2020</td>
      <td>2020-08-01</td>
      <td>2020-08-02</td>
      <td>2020-08-02</td>
      <td>08:00:00</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SON-TINH2012</td>
      <td>2012-10-23</td>
      <td>2012-10-29</td>
      <td>2012-10-26</td>
      <td>20:00:00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>KAI-TAK2012</td>
      <td>2012-08-13</td>
      <td>2012-08-18</td>
      <td>2012-08-17</td>
      <td>14:00:00</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PODUL2013</td>
      <td>2013-11-14</td>
      <td>2013-11-14</td>
      <td>2013-11-14</td>
      <td>18:00:00</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MANGKHUT2013</td>
      <td>2013-08-06</td>
      <td>2013-08-07</td>
      <td>2013-08-07</td>
      <td>16:00:00</td>
    </tr>
    <tr>
      <th>36</th>
      <td>KALMAEGI2014</td>
      <td>2014-09-12</td>
      <td>2014-09-17</td>
      <td>2014-09-16</td>
      <td>14:00:00</td>
    </tr>
    <tr>
      <th>37</th>
      <td>BULBUL:MATMO2019</td>
      <td>2019-10-29</td>
      <td>2019-11-11</td>
      <td>2019-10-30</td>
      <td>17:00:00</td>
    </tr>
  </tbody>
</table>
</div>
