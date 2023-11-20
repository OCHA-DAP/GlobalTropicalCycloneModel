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
df_housing = pd.read_csv(input_dir / "fji_impact_data/processed_house_impact_new.csv")
```

## Municipality level housing damage analysis

Lets explore the housing dataset


```python
df_housing
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
      <th>Division</th>
      <th>Province</th>
      <th>Destroyed</th>
      <th>Major Damage</th>
      <th>nameseason</th>
      <th>Tikina</th>
      <th>ADM1_NAME</th>
      <th>ADM1_PCODE</th>
      <th>ADM2_PCODE</th>
      <th>ADM2_NAME</th>
      <th>nameyear</th>
      <th>Name Season</th>
      <th>Cyclone Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016.0</td>
      <td>Central</td>
      <td>Naitasiri</td>
      <td>542.0</td>
      <td>365.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ409</td>
      <td>Naitasiri</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016.0</td>
      <td>Central</td>
      <td>Namosi</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ410</td>
      <td>Namosi</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016.0</td>
      <td>Central</td>
      <td>Rewa</td>
      <td>66.0</td>
      <td>78.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ412</td>
      <td>Rewa</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016.0</td>
      <td>Central</td>
      <td>Serua</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ413</td>
      <td>Serua</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016.0</td>
      <td>Central</td>
      <td>Tailevu</td>
      <td>395.0</td>
      <td>287.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ414</td>
      <td>Tailevu</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016.0</td>
      <td>Northern</td>
      <td>Cakaudrove</td>
      <td>1513.0</td>
      <td>2199.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ103</td>
      <td>Cakaudrove</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016.0</td>
      <td>Northern</td>
      <td>Bua</td>
      <td>524.0</td>
      <td>605.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ102</td>
      <td>Bua</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016.0</td>
      <td>Northern</td>
      <td>Macuata</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ107</td>
      <td>Macuata</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016.0</td>
      <td>Western</td>
      <td>Nadroga_Navosa</td>
      <td>113.0</td>
      <td>259.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ208</td>
      <td>Nadroga_Navosa</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016.0</td>
      <td>Western</td>
      <td>Ba</td>
      <td>3494.0</td>
      <td>4241.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ201</td>
      <td>Ba</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2016.0</td>
      <td>Western</td>
      <td>Ra</td>
      <td>2813.0</td>
      <td>757.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ211</td>
      <td>Ra</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016.0</td>
      <td>Eastern</td>
      <td>Lau</td>
      <td>328.0</td>
      <td>63.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ305</td>
      <td>Lau</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2016.0</td>
      <td>Eastern</td>
      <td>Kadavu</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ304</td>
      <td>Kadavu</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2016.0</td>
      <td>Eastern</td>
      <td>Lomaiviti</td>
      <td>1191.0</td>
      <td>296.0</td>
      <td>Winston 2015/2016</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ306</td>
      <td>Lomaiviti</td>
      <td>winston2016</td>
      <td>Winston 2015/2016</td>
      <td>Winston</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019.0</td>
      <td>Northern</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Sarai 2019/2020</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sarai2019</td>
      <td>Sarai 2019/2020</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019.0</td>
      <td>Western</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>Sarai 2019/2020</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sarai2019</td>
      <td>Sarai 2019/2020</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019.0</td>
      <td>Central</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>Sarai 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sarai2019</td>
      <td>Sarai 2019/2020</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019.0</td>
      <td>Eastern</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>Sarai 2019/2020</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sarai2019</td>
      <td>Sarai 2019/2020</td>
      <td>Sarai</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019.0</td>
      <td>Northern</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>Tino 2019/2020</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tino2020</td>
      <td>Tino 2019/2020</td>
      <td>Tino</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019.0</td>
      <td>Western</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Tino 2019/2020</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tino2020</td>
      <td>Tino 2019/2020</td>
      <td>Tino</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2019.0</td>
      <td>Central</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Tino 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tino2020</td>
      <td>Tino 2019/2020</td>
      <td>Tino</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2019.0</td>
      <td>Eastern</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Tino 2019/2020</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tino2020</td>
      <td>Tino 2019/2020</td>
      <td>Tino</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020.0</td>
      <td>Central</td>
      <td>Naitasiri</td>
      <td>14.0</td>
      <td>109.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ409</td>
      <td>Naitasiri</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020.0</td>
      <td>Central</td>
      <td>Namosi</td>
      <td>17.0</td>
      <td>38.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ410</td>
      <td>Namosi</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2020.0</td>
      <td>Central</td>
      <td>Rewa</td>
      <td>8.0</td>
      <td>33.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ412</td>
      <td>Rewa</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2020.0</td>
      <td>Central</td>
      <td>Serua</td>
      <td>8.0</td>
      <td>63.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ413</td>
      <td>Serua</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2020.0</td>
      <td>Central</td>
      <td>Tailevu</td>
      <td>58.0</td>
      <td>255.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ414</td>
      <td>Tailevu</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2020.0</td>
      <td>Northern</td>
      <td>Cakaudrove</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ103</td>
      <td>Cakaudrove</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2020.0</td>
      <td>Western</td>
      <td>Nadroga_Navosa</td>
      <td>125.0</td>
      <td>394.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ208</td>
      <td>Nadroga_Navosa</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020.0</td>
      <td>Western</td>
      <td>Ba</td>
      <td>33.0</td>
      <td>162.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ201</td>
      <td>Ba</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2020.0</td>
      <td>Western</td>
      <td>Ra</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ211</td>
      <td>Ra</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2020.0</td>
      <td>Eastern</td>
      <td>Lau</td>
      <td>108.0</td>
      <td>77.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ305</td>
      <td>Lau</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020.0</td>
      <td>Eastern</td>
      <td>Kadavu</td>
      <td>572.0</td>
      <td>1006.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ304</td>
      <td>Kadavu</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2020.0</td>
      <td>Eastern</td>
      <td>Lomaiviti</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Harold 2019/2020</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ306</td>
      <td>Lomaiviti</td>
      <td>harold2020</td>
      <td>Harold 2019/2020</td>
      <td>Harold</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2020.0</td>
      <td>Northern</td>
      <td>Bua</td>
      <td>513.0</td>
      <td>1034.0</td>
      <td>Yasa 2020/2021</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ102</td>
      <td>Bua</td>
      <td>yasa2020</td>
      <td>Yasa 2020/2021</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2020.0</td>
      <td>Northern</td>
      <td>Macuata</td>
      <td>565.0</td>
      <td>3456.0</td>
      <td>Yasa 2020/2021</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ107</td>
      <td>Macuata</td>
      <td>yasa2020</td>
      <td>Yasa 2020/2021</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2020.0</td>
      <td>Northern</td>
      <td>Cakaudrove</td>
      <td>455.0</td>
      <td>1594.0</td>
      <td>Yasa 2020/2021</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ103</td>
      <td>Cakaudrove</td>
      <td>yasa2020</td>
      <td>Yasa 2020/2021</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2020.0</td>
      <td>Western</td>
      <td>Ba</td>
      <td>6.0</td>
      <td>41.0</td>
      <td>Yasa 2020/2021</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ201</td>
      <td>Ba</td>
      <td>yasa2020</td>
      <td>Yasa 2020/2021</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020.0</td>
      <td>Western</td>
      <td>Ra</td>
      <td>2.0</td>
      <td>23.0</td>
      <td>Yasa 2020/2021</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ211</td>
      <td>Ra</td>
      <td>yasa2020</td>
      <td>Yasa 2020/2021</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2020.0</td>
      <td>Central</td>
      <td>Tailevu</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>Yasa 2020/2021</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ414</td>
      <td>Tailevu</td>
      <td>yasa2020</td>
      <td>Yasa 2020/2021</td>
      <td>Yasa</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2021.0</td>
      <td>Central</td>
      <td>Naitasiri</td>
      <td>3.0</td>
      <td>116.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ409</td>
      <td>Naitasiri</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2021.0</td>
      <td>Central</td>
      <td>Namosi</td>
      <td>4.0</td>
      <td>36.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ410</td>
      <td>Namosi</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2021.0</td>
      <td>Central</td>
      <td>Rewa</td>
      <td>7.0</td>
      <td>56.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ412</td>
      <td>Rewa</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2021.0</td>
      <td>Central</td>
      <td>Serua</td>
      <td>9.0</td>
      <td>54.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ413</td>
      <td>Serua</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2021.0</td>
      <td>Central</td>
      <td>Tailevu</td>
      <td>2.0</td>
      <td>11.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Central Division</td>
      <td>FJ4</td>
      <td>FJ414</td>
      <td>Tailevu</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2021.0</td>
      <td>Northern</td>
      <td>Cakaudrove</td>
      <td>4.0</td>
      <td>72.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ103</td>
      <td>Cakaudrove</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2021.0</td>
      <td>Northern</td>
      <td>Bua</td>
      <td>7.0</td>
      <td>55.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ102</td>
      <td>Bua</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2021.0</td>
      <td>Northern</td>
      <td>Macuata</td>
      <td>20.0</td>
      <td>321.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>FJ107</td>
      <td>Macuata</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2021.0</td>
      <td>Western</td>
      <td>Nadroga_Navosa</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ208</td>
      <td>Nadroga_Navosa</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2021.0</td>
      <td>Western</td>
      <td>Ba</td>
      <td>7.0</td>
      <td>241.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ201</td>
      <td>Ba</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2021.0</td>
      <td>Western</td>
      <td>Ra</td>
      <td>7.0</td>
      <td>42.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>FJ211</td>
      <td>Ra</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2021.0</td>
      <td>Eastern</td>
      <td>Lau</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ305</td>
      <td>Lau</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2021.0</td>
      <td>Eastern</td>
      <td>Kadavu</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ304</td>
      <td>Kadavu</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2021.0</td>
      <td>Eastern</td>
      <td>Lomaiviti</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>Ana 2020/2021</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ306</td>
      <td>Lomaiviti</td>
      <td>ana2021</td>
      <td>Ana 2020/2021</td>
      <td>Ana</td>
    </tr>
    <tr>
      <th>54</th>
      <td>NaN</td>
      <td>Western</td>
      <td>NaN</td>
      <td>699.0</td>
      <td>614.0</td>
      <td>Evan 2012/2013</td>
      <td>NaN</td>
      <td>Western Division</td>
      <td>FJ2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>evan2012</td>
      <td>Evan 2012/2013</td>
      <td>Evan</td>
    </tr>
    <tr>
      <th>55</th>
      <td>NaN</td>
      <td>Northern</td>
      <td>NaN</td>
      <td>124.0</td>
      <td>118.0</td>
      <td>Evan 2012/2013</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>evan2012</td>
      <td>Evan 2012/2013</td>
      <td>Evan</td>
    </tr>
    <tr>
      <th>56</th>
      <td>NaN</td>
      <td>Eastern</td>
      <td>Lau</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>Gita 2017/2018</td>
      <td>Ono</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>FJ305</td>
      <td>Lau</td>
      <td>gita2018</td>
      <td>Gita 2017/2018</td>
      <td>Gita</td>
    </tr>
    <tr>
      <th>57</th>
      <td>NaN</td>
      <td>Northern</td>
      <td>NaN</td>
      <td>97.0</td>
      <td>0.0</td>
      <td>Tomas 2009/2010</td>
      <td>NaN</td>
      <td>Northern Division</td>
      <td>FJ1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tomas2010</td>
      <td>Tomas 2009/2010</td>
      <td>Tomas</td>
    </tr>
    <tr>
      <th>58</th>
      <td>NaN</td>
      <td>Eastern</td>
      <td>NaN</td>
      <td>47.0</td>
      <td>174.0</td>
      <td>Tomas 2009/2010</td>
      <td>NaN</td>
      <td>Eastern Division</td>
      <td>FJ3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>tomas2010</td>
      <td>Tomas 2009/2010</td>
      <td>Tomas</td>
    </tr>
  </tbody>
</table>
</div>




```python
#How many provinces do we have?
len(df_housing.Province.unique())
```




    15




```python
#How many typhoons do we have damage data for?
len(df_housing.nameyear.unique())
```




    9



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
      <th>ADM1_points</th>
      <th>ADM2_points</th>
    </tr>
    <tr>
      <th>Cyclone Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ana</th>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Evan</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Gita</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Harold</th>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Sarai</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Tino</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Tomas</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Winston</th>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Yasa</th>
      <td>3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



So

. We have adm1 info for every typhoon


. We have adm2 info for 5 out of 9 typhoon

The admin 1 typhoons. What regions do they affect?


```python
typhoons_adm1 = list(df_housing_info[df_housing_info.ADM2_points == 0].index)
aux3 = df_housing[['Cyclone Name','ADM1_NAME']].drop_duplicates()
aux3[aux3['Cyclone Name'].isin(typhoons_adm1)]
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
      <th>Cyclone Name</th>
      <th>ADM1_NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Sarai</td>
      <td>Northern Division</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sarai</td>
      <td>Western Division</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sarai</td>
      <td>Central Division</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sarai</td>
      <td>Eastern Division</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Tino</td>
      <td>Northern Division</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Tino</td>
      <td>Western Division</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Tino</td>
      <td>Central Division</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Tino</td>
      <td>Eastern Division</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Evan</td>
      <td>Western Division</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Evan</td>
      <td>Northern Division</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Tomas</td>
      <td>Northern Division</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Tomas</td>
      <td>Eastern Division</td>
    </tr>
  </tbody>
</table>
</div>



## Maps for the municipality level dataset


```python
typhoons_adm2 = list(df_housing_info[df_housing_info.ADM2_points != 0].index)
typhoons_adm1 = list(df_housing_info[df_housing_info.ADM2_points == 0].index)
```


```python
#how is the fiji dataset?
fiji.head(4)
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
      <th>GID_2</th>
      <th>GID_0</th>
      <th>COUNTRY</th>
      <th>GID_1</th>
      <th>NAME_1</th>
      <th>NL_NAME_1</th>
      <th>NAME_2</th>
      <th>VARNAME_2</th>
      <th>NL_NAME_2</th>
      <th>TYPE_2</th>
      <th>ENGTYPE_2</th>
      <th>CC_2</th>
      <th>HASC_2</th>
      <th>str_geom</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FJI.1.1_1</td>
      <td>FJI</td>
      <td>Fiji</td>
      <td>FJI.1_1</td>
      <td>Central</td>
      <td>NA</td>
      <td>Naitasiri</td>
      <td>NA</td>
      <td>NA</td>
      <td>Province</td>
      <td>Province</td>
      <td>NA</td>
      <td>FJ.CENT</td>
      <td>POLYGON ((178.027924 -17.579861, 178.028732 -1...</td>
      <td>POLYGON ((178.02792 -17.57986, 178.02873 -17.5...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FJI.1.2_1</td>
      <td>FJI</td>
      <td>Fiji</td>
      <td>FJI.1_1</td>
      <td>Central</td>
      <td>NA</td>
      <td>Namosi</td>
      <td>NA</td>
      <td>NA</td>
      <td>Province</td>
      <td>Province</td>
      <td>NA</td>
      <td>FJ.CE.NM</td>
      <td>MULTIPOLYGON (((178.270523 -18.173244, 178.270...</td>
      <td>MULTIPOLYGON (((178.27052 -18.17324, 178.27080...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FJI.1.3_1</td>
      <td>FJI</td>
      <td>Fiji</td>
      <td>FJI.1_1</td>
      <td>Central</td>
      <td>NA</td>
      <td>Rewa</td>
      <td>NA</td>
      <td>NA</td>
      <td>Province</td>
      <td>Province</td>
      <td>NA</td>
      <td>FJ.CE.RW</td>
      <td>MULTIPOLYGON (((178.151672 -18.410278, 178.151...</td>
      <td>MULTIPOLYGON (((178.15167 -18.41028, 178.15111...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FJI.1.4_1</td>
      <td>FJI</td>
      <td>Fiji</td>
      <td>FJI.1_1</td>
      <td>Central</td>
      <td>NA</td>
      <td>Serua</td>
      <td>NA</td>
      <td>NA</td>
      <td>Province</td>
      <td>Province</td>
      <td>NA</td>
      <td>FJ.CE.SR</td>
      <td>MULTIPOLYGON (((178.073608 -18.433889, 178.073...</td>
      <td>MULTIPOLYGON (((178.07361 -18.43389, 178.07333...</td>
    </tr>
  </tbody>
</table>
</div>



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

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/3823529573.py:28: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_18_1.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/3823529573.py:28: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_18_3.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/3823529573.py:28: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_18_5.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/3823529573.py:28: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_18_7.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/3823529573.py:28: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_18_9.png)



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

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/871413463.py:23: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_20_1.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/871413463.py:23: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_20_3.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/871413463.py:23: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_20_5.png)



    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_96079/871413463.py:23: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      for x, y, label in zip(merged_df.geometry.centroid.x, merged_df.geometry.centroid.y, merged_df['damage']):




![png](01_damage_data_exploration_files/01_damage_data_exploration_20_7.png)
