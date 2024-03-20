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
    / "analysis_phl/02_model_features/02_housing_damage/input/"
)
output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_phl/02_model_features/02_housing_damage/output/"
)
wind_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_phl/02_model_features/01_windfield/"
)
iwi_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_phl/02_model_features/05_vulnerability/output/"
)
# Load IWI
iwi_grid = pd.read_csv(iwi_dir / "phl_iwi_bygrid_new.csv")

# Load RWI
rwi_grid = pd.read_csv(iwi_dir / "phl_rwi_bygrid.csv")

# Load PHL
phl = gpd.read_file(
    input_dir / "phl_adminboundaries_candidate_adm3.zip"
)
phl = phl.to_crs('EPSG:4326')

# Load impact data
impact_data = pd.read_csv(input_dir / "impact_data_clean_phl.csv")

# Ids by municipality
ids_mun_exploded = pd.read_csv(input_dir / "phl_grid_municipality_info.csv")
ids_mun = ids_mun_exploded[['id', 'ADM3_PCODE']].groupby('ADM3_PCODE')['id'].agg(list).reset_index()

# Num of buildings by grid
bld_grid = pd.read_csv(output_dir / "num_building_bygrid.csv")[['id', 'numbuildings']]
```

We need to aggregate mean windspeed and mean track distance to adm3 level. I picked these 3 features because these are the most important ones (in terms of shap values) in the philippines model trained with philippines data.

## Windspeed per amd3 mun


```python
# Windspeed by grid
wind_grid = pd.read_csv(wind_dir / "windfield_data_phl.csv")
```


```python
df_merged = pd.merge(ids_mun_exploded, wind_grid, left_on='id', right_on='grid_point_id', how='inner')
typhoons = df_merged.typhoon_id.unique()

wind_mun = pd.DataFrame()
for typhoon in typhoons:
    df_aux = df_merged[df_merged.typhoon_id==typhoon].groupby('ADM3_PCODE')['wind_speed'].mean().reset_index()

    df_aux['typhoon_year'] = df_merged[df_merged.typhoon_id==typhoon].typhoon_year.iloc[0]
    df_aux['typhoon_name'] = df_merged[df_merged.typhoon_id==typhoon].typhoon_name.iloc[0]
    wind_mun = pd.concat([wind_mun, df_aux])


# Add number of buildings
df_merged2 = df_merged.merge(bld_grid, on='id')
bld_mun = df_merged2.drop_duplicates('id').groupby('ADM3_PCODE')['numbuildings'].sum().reset_index()
wind_mun = wind_mun.merge(bld_mun, on='ADM3_PCODE')

# Standardized
wind_mun = wind_mun.rename({
    'typhoon_year':'Year',
    'ADM3_PCODE':'pcode',
        }, axis=1)
```


```python
wind_impact = impact_data.merge(wind_mun, on=['typhoon_name', 'Year', 'pcode'], how='right')
wind_impact = wind_impact.fillna(0)

# Create % of damage
wind_impact['perc_dmg'] = wind_impact['total_bld_dmg'] / wind_impact['numbuildings']

```


```python
wind_impact.sort_values('perc_dmg', ascending=False)
#bld_mun.sort_values('numbuildings')
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
      <th>total_bld_dmg</th>
      <th>pcode</th>
      <th>wind_speed</th>
      <th>numbuildings</th>
      <th>perc_dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28249</th>
      <td>HAGUPIT</td>
      <td>2014</td>
      <td>795.0</td>
      <td>PH084819000</td>
      <td>40.830132</td>
      <td>0</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>28248</th>
      <td>RAMMASUN</td>
      <td>2014</td>
      <td>3140.0</td>
      <td>PH084819000</td>
      <td>44.856617</td>
      <td>0</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>32439</th>
      <td>KAMMURI</td>
      <td>2019</td>
      <td>18963.0</td>
      <td>PH098309000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>32443</th>
      <td>GONI</td>
      <td>2020</td>
      <td>13695.0</td>
      <td>PH098309000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>32446</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>766.0</td>
      <td>PH098309000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>inf</td>
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
      <th>49618</th>
      <td>SAUDEL</td>
      <td>2020</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49619</th>
      <td>GONI</td>
      <td>2020</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49620</th>
      <td>VAMCO</td>
      <td>2020</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49621</th>
      <td>VONGFONG</td>
      <td>2020</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49622</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>10.228524</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>50325 rows × 7 columns</p>
</div>



We have acouple of rows with bad damage data (lot of bld destroyed but no bld in the adm3 mun). Since these cases add noise:

- more than 100% dmg
- Not representative
- Cannot distingue between households and buildings

We dont consider them!!


```python
wind_impact_fixed = wind_impact[wind_impact.numbuildings != 0].sort_values('perc_dmg', ascending=False)

# Also, consider >100% muns to 100%
wind_impact_fixed.loc[wind_impact_fixed['perc_dmg'] > 100, 'perc_dmg'] = 100
```


```python
wind_aux = wind_impact_fixed[(wind_impact.perc_dmg > 0) & (wind_impact.perc_dmg < 100)]
plt.plot(wind_aux.wind_speed, wind_aux.perc_dmg, 'o')

# Perform linear regression
coefficients = np.polyfit(wind_aux.wind_speed, np.log(wind_aux.perc_dmg), 1)
poly_function = np.poly1d(coefficients)

# Plot the regression line
plt.plot(wind_aux.wind_speed, np.exp(poly_function(wind_aux.wind_speed)), label=f'Linear Fit: {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

plt.yscale('log')
plt.xlabel('Mean Wind Speed [m/s]')
plt.ylabel('% of bld damaged')
plt.title('Mean windspeed by ADM3 municipality, Philippines')
plt.legend()
plt.show()
```



![png](model_correlations_files/model_correlations_10_0.png)



## Track distance by adm3 mun


```python
df_merged = pd.merge(ids_mun_exploded, wind_grid, left_on='id', right_on='grid_point_id', how='inner')
typhoons = df_merged.typhoon_id.unique()

track_distance_mun = pd.DataFrame()
for typhoon in typhoons:
    df_aux = df_merged[df_merged.typhoon_id==typhoon].groupby('ADM3_PCODE')['track_distance'].mean().reset_index()

    df_aux['typhoon_year'] = df_merged[df_merged.typhoon_id==typhoon].typhoon_year.iloc[0]
    df_aux['typhoon_name'] = df_merged[df_merged.typhoon_id==typhoon].typhoon_name.iloc[0]
    track_distance_mun = pd.concat([track_distance_mun, df_aux])

# Standardized
track_distance_mun = track_distance_mun.rename({
    'typhoon_year':'Year',
    'ADM3_PCODE':'pcode',
        }, axis=1)
```


```python
track_impact_fixed = wind_impact_fixed.merge(track_distance_mun, on=['typhoon_name', 'Year', 'pcode'])
track_impact_fixed = track_impact_fixed.fillna(0)
```


```python
track_aux = track_impact_fixed[(track_impact_fixed.perc_dmg < 100) & ((track_impact_fixed.perc_dmg > 0))]
plt.plot(track_aux.track_distance, track_aux.perc_dmg, 'o')


# Perform linear regression
coefficients = np.polyfit(track_aux.track_distance, np.log(track_aux.perc_dmg), 1)
poly_function = np.poly1d(coefficients)

# Plot the regression line
plt.plot(track_aux.track_distance, np.exp(poly_function(track_aux.track_distance)), label=f'Linear Fit: {coefficients[0]:.2f}x + {coefficients[1]:.2f}')


plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Mean Track distance [km]')
plt.ylabel('% of bld damaged')
plt.title('Mean Track Distance by ADM3 municipality, Philippines')
plt.legend()
plt.show()
```



![png](model_correlations_files/model_correlations_14_0.png)



## IWI by adm3 typhoons


```python
iwi_mun = pd.merge(ids_mun_exploded, iwi_grid,
         left_on='id', right_on='grid_point_id',
         how='inner')[
             ['ADM3_PCODE', 'IWI']
             ].drop_duplicates()

iwi_impact_fixed = track_impact_fixed.merge(iwi_mun, left_on='pcode', right_on='ADM3_PCODE')
```


```python
iwi_impact_fixed.IWI.hist(density=True)
plt.xlabel('IWI')
plt.ylabel('Density')
plt.title('IWI of each AMD3 mun')
plt.show()
```



![png](model_correlations_files/model_correlations_17_0.png)




```python
iwi_impact_aux = iwi_impact_fixed[(iwi_impact_fixed.perc_dmg < 100)  & (iwi_impact_fixed.track_distance < 300)]
plt.plot(iwi_impact_aux.IWI, iwi_impact_aux.perc_dmg, 'o')
plt.yscale('log')
# plt.xscale('log')
plt.xlabel('IWI')
plt.ylabel('% of bld damaged')
plt.title('IWI by ADM3 municipality, Philippines')
plt.show()
```



![png](model_correlations_files/model_correlations_18_0.png)



I dont like this plot: should be the opposite.

--> high IWI should = less damage.

- I dont like that this is at adm1.
- I dont like that this is a categorical variable.

## RWI by adm3 typhoons


```python
rwi_merged = pd.merge(ids_mun_exploded, rwi_grid,
            left_on='id', right_on='id',
            how='inner')

rwi_mun = rwi_merged.groupby('ADM3_PCODE')['scaled_distance'].mean().reset_index()
#rwi_mun = rwi_merged.groupby('ADM3_PCODE')['scaled_distance'].max().reset_index()
#rwi_mun = rwi_merged.groupby('ADM3_PCODE')['scaled_distance'].min().reset_index()

# Standardized
rwi_mun = rwi_mun.rename({
    'ADM3_PCODE':'pcode',
        }, axis=1)

rwi_impact_fixed = track_impact_fixed.merge(rwi_mun, left_on='pcode', right_on='pcode')
```


```python
rwi_impact_aux = rwi_impact_fixed[(rwi_impact_fixed.perc_dmg < 100)  & (rwi_impact_fixed.perc_dmg > 0)]
plt.plot(rwi_impact_aux.scaled_distance, rwi_impact_aux.perc_dmg, 'o')

# Perform linear regression
coefficients = np.polyfit(rwi_impact_aux.scaled_distance, np.log(rwi_impact_aux.perc_dmg), 1)
poly_function = np.poly1d(coefficients)

# Plot the regression line
plt.plot(rwi_impact_aux.scaled_distance, np.exp(poly_function(rwi_impact_aux.scaled_distance)), label=f'Linear Fit: {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Scaled distance (RWI)')
plt.ylabel('% of bld damaged')
plt.title('RWI distance by ADM3 municipality, Philippines')
plt.legend()
plt.show()
```



![png](model_correlations_files/model_correlations_22_0.png)



## Impactful typhoons subset (for better visualization)


```python
# Mosts impactful typhoons:
impactful = ['HAIYAN',
             'RAI',
             'BOPHA',
             'RAMMASUN',
             'MANGKHUT',
             'PARMA',
             'VAMCO',
             'GONI',
             'PAENG',
             'PEDRING']
```


```python
wind_impact_fixed[wind_impact_fixed.typhoon_name.isin(impactful)].typhoon_name.unique()
```




    array(['MANGKHUT', 'HAIYAN', 'VAMCO', 'RAMMASUN', 'BOPHA', 'GONI'],
          dtype=object)




```python
wind_impact_subset = wind_impact_fixed[wind_impact_fixed.typhoon_name.isin(impactful)]
track_impact_subset = track_impact_fixed[track_impact_fixed.typhoon_name.isin(impactful)]
iwi_impact_subset = iwi_impact_fixed[iwi_impact_fixed.typhoon_name.isin(impactful)]
rwi_impact_subset = rwi_impact_fixed[rwi_impact_fixed.typhoon_name.isin(impactful)]
```


```python
fig, ax = plt.subplots(1,3, figsize=(15,5))
wind_aux = wind_impact_subset[(wind_impact_subset.perc_dmg > 0) & (wind_impact_subset.perc_dmg < 100)]
ax[0].plot(wind_aux.wind_speed, wind_aux.perc_dmg, 'o')
ax[0].set_yscale('log')
ax[0].set_xlabel('Mean Wind Speed [m/s]')
ax[0].set_ylabel('% of bld damaged')
ax[0].set_title('Mean windspeed by ADM3 municipality')

track_aux = track_impact_subset[(track_impact_subset.perc_dmg > 0) & (track_impact_subset.perc_dmg < 100)]
ax[1].plot(track_aux.track_distance, track_aux.perc_dmg, 'o')
ax[1].set_yscale('log')
ax[1].set_xlabel('Mean Track Distance [km]')
ax[1].set_ylabel('% of bld damaged')
ax[1].set_title('Mean Track Distance by ADM3 municipality')

rwi_aux = rwi_impact_subset[(rwi_impact_subset.perc_dmg > 0) & (rwi_impact_subset.perc_dmg < 100)]
ax[2].plot(rwi_aux.scaled_distance, rwi_aux.perc_dmg, 'o')
ax[2].set_yscale('log')
ax[2].set_xlabel('Mean RWI scaled distance')
ax[2].set_ylabel('% of bld damaged')
ax[2].set_title('Mean RWI scaled distance by ADM3 municipality')

# Fit a linear regression line and display coefficients for each plot
for i, data in enumerate([(wind_aux.wind_speed, wind_aux.perc_dmg), (track_aux.track_distance, track_aux.perc_dmg), (rwi_aux.scaled_distance, rwi_aux.perc_dmg)]):
    x, y = data
    coeffs = np.polyfit(x, np.log(y), 1)  # Perform linear fit in log space
    line = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)  # Calculate the linear fit line
    ax[i].plot(x, line, color='red', label=f'Linear Fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    ax[i].legend()

plt.suptitle('Philippines, Impactful typhoons, dmg > 0')
plt.tight_layout()
plt.show()
```



![png](model_correlations_files/model_correlations_27_0.png)




```python
fig, ax = plt.subplots(1,3, figsize=(15,5))
wind_aux = wind_impact_subset[(wind_impact_subset.perc_dmg > 0) & (wind_impact_subset.perc_dmg < 100)]
ax[0].plot(wind_aux.wind_speed, wind_aux.perc_dmg, 'o')
ax[0].set_yscale('log')
ax[0].set_xlabel('Mean Wind Speed [m/s]')
ax[0].set_ylabel('% of bld damaged')
ax[0].set_title('Mean windspeed by ADM3 municipality')

track_aux = track_impact_subset[(track_impact_subset.perc_dmg > 0) & (track_impact_subset.perc_dmg < 100)]
ax[1].plot(track_aux.track_distance, track_aux.perc_dmg, 'o')
ax[1].set_yscale('log')
ax[1].set_xlabel('Mean Track Distance [km]')
ax[1].set_ylabel('% of bld damaged')
ax[1].set_title('Mean Track Distance by ADM3 municipality')

iwi_aux = iwi_impact_subset[(iwi_impact_subset.perc_dmg > 0) & (rwi_impact_subset.perc_dmg < 100)]
ax[2].plot(iwi_aux.IWI, iwi_aux.perc_dmg, 'o')
ax[2].set_yscale('log')
ax[2].set_xlabel('Mean IWI')
ax[2].set_ylabel('% of bld damaged')
ax[2].set_title('Mean IWI by ADM3 municipality')

# Fit a linear regression line and display coefficients for each plot
for i, data in enumerate([(wind_aux.wind_speed, wind_aux.perc_dmg), (track_aux.track_distance, track_aux.perc_dmg)]):
    x, y = data
    coeffs = np.polyfit(x, np.log(y), 1)  # Perform linear fit in log space
    line = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)  # Calculate the linear fit line
    ax[i].plot(x, line, color='red', label=f'Linear Fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    ax[i].legend()

plt.suptitle('Philippines, Impactful typhoons, dmg > 0')
plt.tight_layout()
plt.show()
```



![png](model_correlations_files/model_correlations_28_0.png)




```python
wind_impact_fixed.perc_dmg.hist(bins=50)
plt.yscale('log')
wind_impact_fixed
```



![png](model_correlations_files/model_correlations_29_0.png)



## Define probability of damage based on these results

### Approach 1: regressors on the impactful subset (makes little sense)


```python
from sklearn.linear_model import LinearRegression
# I must include the possibility of existence of 0 damage cells
wind_aux = wind_impact_subset[(wind_impact_subset.perc_dmg < 100)][['typhoon_name', 'Year', 'pcode', 'wind_speed', 'perc_dmg']]
track_aux = track_impact_subset[(track_impact_subset.perc_dmg < 100)][['typhoon_name', 'Year', 'pcode', 'track_distance']]
reduced_dataset = wind_aux.merge(track_aux, on=['typhoon_name', 'Year', 'pcode'])

wind_speed = np.array(reduced_dataset.wind_speed)
track_distance = np.array(reduced_dataset.track_distance)
damage = np.array(reduced_dataset.perc_dmg)


# Fit linear regression models
model_wind_speed = LinearRegression().fit(wind_speed.reshape(-1, 1), damage)
model_track_distance = LinearRegression().fit(track_distance.reshape(-1, 1), damage)

# Define probability density function
# Assign more weight to the wind speed --> the weight comes from the shap values for philippines.
def damage_probability(wind_speed_val, track_distance_val):
    # Predict damage probabilities using linear regression models
    pred_damage_wind_speed = model_wind_speed.predict([[wind_speed_val]])[0]
    pred_damage_track_distance = model_track_distance.predict([[track_distance_val]])[0]

    # Adjust the weighting of wind speed prediction based on its importance compared to track distance
    wind_importance_percentage = 17 #% more important than track_distance
    weighted_pred_damage_wind_speed = pred_damage_wind_speed * (1 + wind_importance_percentage / 100)

    # Combine adjusted wind speed prediction with track distance prediction
    combined_probability = (weighted_pred_damage_wind_speed + pred_damage_track_distance) / 2

    return combined_probability
```

### Approach 2: Logistic regressor (makes total sense)


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# I must include the possibility of existence of 0 damage cells
# I work with alll typhoon events for train test split.

# # impactful events
# wind_aux = wind_impact_subset[(wind_impact_subset.perc_dmg < 100)][['typhoon_name', 'Year', 'pcode', 'wind_speed', 'perc_dmg']]
# track_aux = track_impact_subset[(track_impact_subset.perc_dmg < 100)][['typhoon_name', 'Year', 'pcode', 'track_distance']]
# reduced_dataset = wind_aux.merge(track_aux, on=['typhoon_name', 'Year', 'pcode'])

# # all the other events
# wind_not_impact_subset = wind_impact_fixed[~wind_impact_fixed.typhoon_name.isin(impactful)][['typhoon_name', 'Year', 'pcode', 'wind_speed', 'perc_dmg']]
# track_not_impact_subset = track_impact_fixed[~track_impact_fixed.typhoon_name.isin(impactful)][['typhoon_name', 'Year', 'pcode', 'track_distance']]
# not_impactful_dataset = wind_not_impact_subset.merge(track_not_impact_subset, on=['typhoon_name', 'Year', 'pcode'])

# all_events = pd.concat([reduced_dataset, not_impactful_dataset])
all_events = rwi_impact_fixed[rwi_impact_fixed.perc_dmg < 100]

# Define the dataset and target variable
X = all_events[['wind_speed', 'track_distance', 'scaled_distance']]
y = all_events['perc_dmg']

# Define bins or percentiles for stratification
# For example, you can use percentile-based bins:
percentiles = np.array([0, 0.00009, 1, 10, 50]) #Like the ones we have in the paper for phl strat (more or less)
bins = np.percentile(y, percentiles)

# Perform stratified sampling based on the bins
y_bins = np.digitize(y, bins)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_bins)

# Convert y_train into binary classes
y_train_binary = np.where(y_train > 0, 1, 0)
y_test_binary = np.where(y_test > 0, 1, 0)

# Standardize the features
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)
X_test_scaled = scaler_train.transform(X_test)

# Calculate class weights
class_weights = {0: 1, 1: 3}  # For example, assign 3 times the weight to samples with label 1

# Initialize logistic regression model
log_reg = LogisticRegression(class_weight=class_weights)

# Fit the model to the training data (impactful dataset)
log_reg.fit(X_train_scaled, y_train_binary)

# Predict probabilities for the test data (not impactful dataset)
probabilities = log_reg.predict_proba(X_test_scaled)
```


```python
unique_values, counts = np.unique(y_train_binary, return_counts=True)
print(dict(zip(unique_values, counts)))
```

    {0: 37124, 1: 3006}



```python
unique_values, counts = np.unique(y_test_binary, return_counts=True)
print(dict(zip(unique_values, counts)))
```

    {0: 9294, 1: 739}



```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Predicted probabilities for the test data
predicted_probabilities = log_reg.predict_proba(X_test_scaled)[:, 1]  # Take probabilities for class 1

# Convert probabilities to binary predictions
thres = 0.2 # Set Threshold for damage
predicted_labels = (predicted_probabilities > thres).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test_binary, predicted_labels)

# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=['No Damage', 'Damage'], yticklabels=['No Damage', 'Damage'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix, Threshold dmg. prob. = {}'.format(thres))
plt.show()
```



![png](model_correlations_files/model_correlations_37_0.png)



For the real data, We can have

- No damage: 0% damage at mun level.
- Damage: > 0% damage at mun level.

For the predicted data, we can predict:

- Damage (if prob damage > thres)
- No damage (if prob damage < thres)

### Numerical Example


```python
import warnings

# Suppress the warning
warnings.filterwarnings("ignore")

# Define constant value for scaled_distance
scaled_distance_val = 0.5  # Choose an appropriate value

# Define ranges for wind speed and track distance
wind_speed_range = np.arange(0, 80, 1)
track_distance_range = np.arange(0, 1000, 10)

# Create 2D arrays for wind speed and track distance
wind_speed_grid, track_distance_grid = np.meshgrid(wind_speed_range, track_distance_range)

# Initialize damage probabilities array
damage_probabilities = np.zeros_like(wind_speed_grid, dtype=float)

# Iterate through each combination of wind speed and track distance
for i in range(len(wind_speed_range)):
    for j in range(len(track_distance_range)):
        wind_speed_val = wind_speed_range[i]
        track_distance_val = track_distance_range[j]

        # Predict probability for the current combination
        X_sample = [[wind_speed_val, track_distance_val, scaled_distance_val]]
        X_sample_scaled = scaler_train.transform(X_sample)
        damage_probabilities[j, i] = log_reg.predict_proba(X_sample_scaled)[0][1]
        #damage_probabilities[j, i] = damage_probability(wind_speed_val, track_distance_val)

# Plot the damage probabilities
plt.figure(figsize=(10, 6))
plt.imshow(damage_probabilities, extent=[0, 80, 0, 1000], origin='lower', aspect='auto', cmap='YlOrBr')
plt.colorbar(label='Damage Probability')
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Track Distance [km]')
plt.title('Damage Probability Density \nNumerical example \nScaled RWI distance = {}'.format(scaled_distance_val))
plt.grid(True)
plt.show()

```



![png](model_correlations_files/model_correlations_40_0.png)




```python
# Define constant value for wind_speed
wind_speed_val = 30

# Define ranges for track distance and scaled distance
track_distance_range = np.arange(0, 450, 20)
scaled_distance_range = np.linspace(0, 1, num=300)

# Create 2D arrays for track distance and scaled distance
track_distance_grid, scaled_distance_grid = np.meshgrid(track_distance_range, scaled_distance_range)

# Initialize damage probabilities array
damage_probabilities = np.zeros_like(track_distance_grid, dtype=float)

# Iterate through each combination of track distance and scaled distance
for i in range(len(track_distance_range)):
    for j in range(len(scaled_distance_range)):
        track_distance_val = track_distance_range[i]
        scaled_distance_val = scaled_distance_range[j]

        # Predict probability for the current combination
        X_sample = [[wind_speed_val, track_distance_val, scaled_distance_val]]
        X_sample_scaled = scaler_train.transform(X_sample)
        damage_probabilities[j, i] = log_reg.predict_proba(X_sample_scaled)[0][1]

# Plot the damage probabilities
plt.figure(figsize=(10, 6))
plt.imshow(damage_probabilities, extent=[0, 450, 0, 1], origin='lower', aspect='auto', cmap='YlOrBr')
plt.colorbar(label='Damage Probability')
plt.xlabel('Track Distance [km]')
plt.ylabel('Scaled RWI Distance')
plt.title('Damage Probability Density \nNumerical example \nWindspeed = {}'.format(wind_speed_val))
plt.grid(True)
plt.show()

```



![png](model_correlations_files/model_correlations_41_0.png)




```python
# Define constant value for wind_speed
wind_speed_val = 30

# Define ranges for track distance and scaled distance
track_distance_range = np.arange(0, 450, 20)
scaled_distance_range = np.linspace(0, 10, num=300)

# Create 2D arrays for track distance and scaled distance
track_distance_grid, scaled_distance_grid = np.meshgrid(track_distance_range, scaled_distance_range)

# Initialize damage probabilities array
damage_probabilities = np.zeros_like(track_distance_grid, dtype=float)

# Iterate through each combination of track distance and scaled distance
for i in range(len(track_distance_range)):
    for j in range(len(scaled_distance_range)):
        track_distance_val = track_distance_range[i]
        scaled_distance_val = scaled_distance_range[j]

        # Predict probability for the current combination
        X_sample = [[wind_speed_val, track_distance_val, scaled_distance_val]]
        X_sample_scaled = scaler_train.transform(X_sample)
        damage_probabilities[j, i] = log_reg.predict_proba(X_sample_scaled)[0][1]

# Plot the damage probabilities
plt.figure(figsize=(10, 6))
plt.imshow(damage_probabilities, extent=[0, 450, 0, 10], origin='lower', aspect='auto', cmap='YlOrBr')
plt.colorbar(label='Damage Probability')
plt.xlabel('Track Distance [km]')
plt.ylabel('Scaled RWI Distance')
plt.title('Damage Probability Density \nNumerical example (exaggerating possible Scaled RWI Distance values)\nWindspeed = {}'.format(wind_speed_val))
plt.grid(True)
plt.show()
```



![png](model_correlations_files/model_correlations_42_0.png)



Probability of damage as a function of these features


```python
# Define range of wind speed
wind_speed_range = np.arange(0, 80, 1)

# Define 5 discrete values for track distance
track_distance_values = np.linspace(0, 500, 5)

# Initialize damage probabilities array
damage_probabilities = np.zeros((len(track_distance_values), len(wind_speed_range)), dtype=float)

# Iterate through each track distance value
for idx, track_distance_val in enumerate(track_distance_values):
    # Iterate through each wind speed value
    for i, wind_speed_val in enumerate(wind_speed_range):
        # Predict probability for the current combination
        X_sample = [[wind_speed_val, track_distance_val, scaled_distance_val]]
        X_sample_scaled = scaler_train.transform(X_sample)
        damage_probabilities[idx, i] = log_reg.predict_proba(X_sample_scaled)[0][1]

# Plot the damage probabilities
plt.figure(figsize=(10, 6))
for idx, track_distance_val in enumerate(track_distance_values):
    plt.plot(wind_speed_range, damage_probabilities[idx], label='Track Distance = {} km'.format(int(track_distance_val)))

plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Probability of Damage')
plt.title('Probability of Damage for Different Track Distances')
plt.legend()
plt.grid(True)
plt.show()

```



![png](model_correlations_files/model_correlations_44_0.png)




```python
from pygam import LinearGAM

# Reshape the data for modeling
X_train_reshaped = X_train[['wind_speed', 'track_distance']].values
y_train_reshaped = y_train.values

# Fit the GAM model
gam = LinearGAM().fit(X_train_reshaped, y_train_reshaped)

# Print summary of the model
print(gam.summary())

# Plot the model
fig, axs = plt.subplots(1, 2)

# Wind speed component
XX = gam.generate_X_grid(term=0)
axs[0].plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
axs[0].set_title('Partial Dependence of Wind Speed')
axs[0].set_xlabel('Wind Speed [m/s]')
axs[0].set_ylabel('Partial Dependence')

# Track distance component
XX = gam.generate_X_grid(term=1)
axs[1].plot(XX[:, 1], gam.partial_dependence(term=1, X=XX))
axs[1].set_title('Partial Dependence of Track Distance')
axs[1].set_xlabel('Track Distance [km]')
axs[1].set_ylabel('Partial Dependence')

plt.tight_layout()
plt.show()

```

    LinearGAM
    =============================================== ==========================================================
    Distribution:                        NormalDist Effective DoF:                                     32.2928
    Link Function:                     IdentityLink Log Likelihood:                                -76672.1029
    Number of Samples:                        40130 AIC:                                           153410.7914
                                                    AICc:                                          153410.8484
                                                    GCV:                                                0.1876
                                                    Scale:                                              0.1874
                                                    Pseudo R-Squared:                                    0.029
    ==========================================================================================================
    Feature Function                  Lambda               Rank         EDoF         P > x        Sig. Code
    ================================= ==================== ============ ============ ============ ============
    s(0)                              [0.6]                20           16.0         1.11e-16     ***
    s(1)                              [0.6]                20           16.3         2.39e-03     **
    intercept                                              1            0.0          1.52e-04     ***
    ==========================================================================================================
    Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    WARNING: Fitting splines and a linear function to a feature introduces a model identifiability problem
             which can cause p-values to appear significant when they are not.

    WARNING: p-values calculated in this manner behave correctly for un-penalized models or models with
             known smoothing parameters, but when smoothing parameters have been estimated, the p-values
             are typically lower than they should be, meaning that the tests reject the null too readily.
    None




![png](model_correlations_files/model_correlations_45_1.png)




```python
# Define ranges for wind speed and track distance
wind_speed_range = np.arange(0, 80, 1)
track_distance_range = np.arange(0, 800, 10)  # Adjusted range for compatible dimensions

# Create 2D arrays for wind speed and track distance
wind_speed_grid, track_distance_grid = np.meshgrid(wind_speed_range, track_distance_range)

# Initialize damage probabilities array
damage_probabilities = np.zeros_like(wind_speed_grid, dtype=float)

# Iterate through each combination of wind speed and track distance
for i in range(len(wind_speed_range)):
    for j in range(len(track_distance_range)):
        wind_speed_val = wind_speed_range[i]
        track_distance_val = track_distance_range[j]

        # Predict probability for the current combination
        X_sample = [[wind_speed_val, track_distance_val, scaled_distance_val]]
        X_sample_scaled = scaler_train.transform(X_sample)
        damage_probabilities[j, i] = log_reg.predict_proba(X_sample_scaled)[0][1]

# Now you can stack the grids and flatten them for regression
X = np.column_stack((wind_speed_grid.flatten(), track_distance_grid.flatten()))
y = damage_probabilities.flatten()

# Fit a regression model and obtain the coefficients
regression_model = LinearRegression()
regression_model.fit(X, y)

# Get the coefficients of the linear model
coefficients = regression_model.coef_
intercept = regression_model.intercept_

# Print the coefficients
print("Coefficients:", coefficients)
print("Intercept:", intercept)
```

    Coefficients: [ 0.01376148 -0.00053105]
    Intercept: 0.15887222451273053



```python
beta_0 = regression_model.intercept_
beta_1 = regression_model.coef_[0]
beta_2 = regression_model.coef_[1]

equation = f"Probability of Damage = {beta_0:.4f} + {beta_1:.4f} * Wind Speed + {beta_2:.4f} * Track Distance"
print(equation)
```

    Probability of Damage = 0.1589 + 0.0138 * Wind Speed + -0.0005 * Track Distance


## Test it on VIET typhoons


```python
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString,Polygon, MultiPolygon
import geopandas as gpd
```


```python
input_dir_viet = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_vnm/02_model_features/02_housing_damage/input/"
)
output_dir_viet = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_vnm/02_model_features/02_housing_damage/output/"
)
wind_info_viet = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_vnm/02_model_features/01_windfield"
)
base_info_viet = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_vnm/03_model_input_dataset"
)

# Load typhoon id info
typhoon_ids = pd.read_csv(wind_info_viet/ "typhoons.csv")

# Load Vietnam
vietnam = gpd.read_file(input_dir_viet / "adm2_shp_fixed.gpkg")
vietnam = vietnam.to_crs('EPSG:4326')

# Load grid
grid_land_overlap = gpd.read_file(output_dir_viet / "viet_0.1_degree_grid_land_overlap_new.gpkg")
grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)

# Load damage at grid level
damage_grid_viet = pd.read_csv(output_dir_viet / "building_damage_bygrid_new.csv")

# Number of buildings per id
bld_count = pd.read_csv(input_dir_viet / "vnm_google_bld_grid_count.csv")
bld_count_complete = bld_count.merge(grid_land_overlap, on='id', how='right').fillna(0)
bld_count_complete = bld_count_complete.rename({'count':'numbuildings'}, axis=1)

# Vietnam data
viet_data = pd.read_csv(base_info_viet / "new_model_training_dataset_viet_complete_interpolated_wind.csv")

# Load damage at mun level
damage_mun_viet = pd.read_csv(input_dir_viet / "viet_damage_adm1_level_fixed.csv")

# Load grids per Municipality
ids_mun_viet = pd.read_csv(input_dir_viet / "grid_municipality_info.csv")
```

### Before


```python
# Damage info
damrey = damage_grid_viet[(damage_grid_viet.typhoon_name == 'DAMREY') & (damage_grid_viet.Year == 2017)]
damrey = gpd.GeoDataFrame(damrey.merge(bld_count_complete[['id', 'geometry']], on='id'))
damrey['N_dmg'] = damrey['perc_dmg_grid'] * damrey['numbuildings_z'] /100
# Track path
track = TCTracks.from_ibtracs_netcdf(storm_id='2017304N11127') # Damrey2017
tc_track = track.get_track()
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

# Plot
fig, ax = plt.subplots(1,1, figsize=(5,5))
damrey.plot(ax=ax, column='N_dmg', legend=True, cmap='Reds')
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

ax.set_xlim(100, 120)
ax.set_ylim(7, 24)
ax.set_xticks([])
ax.set_yticks([])

ax.set_title('Vietnam\nDamrey typhoon 2017, \n (houses affected by grid)')
plt.show()
```

    2024-03-11 18:13:53,975 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.




![png](model_correlations_files/model_correlations_52_1.png)



### Now


```python
# Calculate damage probability for each row
damage_probabilities = []
for index, row in viet_data.iterrows():
    wind_speed_val = row['wind_speed']
    track_distance_val = row['track_distance']
    scaled_distance_val = row['scaled_distance']
    # Calculate damage probability using the logistic regression model
    prob_dmg = log_reg.predict_proba(scaler_train.transform([[wind_speed_val, track_distance_val, scaled_distance_val]]))[0][1]
    damage_probabilities.append(prob_dmg)

# Add the calculated probabilities to the DataFrame as a new column
viet_data['prob_dmg'] = damage_probabilities
```


```python
damage_mun_viet[(damage_mun_viet.typhoon_name == 'DAMREY') & (damage_mun_viet.Year == 2017)].region_affected.unique()[0]
```




    "['NC', 'SC', 'C']"




```python
damrey_new = viet_data[(viet_data.typhoon_name == 'DAMREY') & (viet_data.Year == 2017)]
# Add region column
damrey_new = damrey_new.merge(ids_mun_viet[['id', 'region']], left_on='grid_point_id', right_on='id')

# Consider just damage regions
regions = ['NC', 'SC', 'C']
damrey_new.loc[~damrey_new.region.isin(regions), 'prob_dmg'] = 0

damrey_new = gpd.GeoDataFrame(damrey_new[
    ['grid_point_id', 'typhoon_name', 'Year', 'prob_dmg', 'region', 'wind_speed', 'track_distance', 'scaled_distance', 'total_buildings', 'total_buildings_damaged']
    ].merge(grid_land_overlap, left_on='grid_point_id', right_on='id'), geometry='geometry')
```


```python
# Track path
track = TCTracks.from_ibtracs_netcdf(storm_id='2017304N11127') # Damrey2017
tc_track = track.get_track()
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,5))
damrey_new.plot(ax=ax, column='prob_dmg', legend=True, cmap='Reds')
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

ax.set_xlim(100, 120)
ax.set_ylim(7, 24)
ax.set_xticks([])
ax.set_yticks([])

ax.set_title('Vietnam\nDamrey typhoon 2017, \n (Prob of Damage on reported damaged regions)')
plt.show()
```

    2024-03-11 18:22:47,725 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.




![png](model_correlations_files/model_correlations_57_1.png)



We can set a threshold for this to select specific cells


```python
threshold=0.2
damrey_new['with_damage'] = np.where(damrey_new['prob_dmg'] >= threshold, 1, 0)
```


```python
# Track path
track = TCTracks.from_ibtracs_netcdf(storm_id='2017304N11127') # Damrey2017
tc_track = track.get_track()
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,5))
damrey_new.plot(ax=ax, column='with_damage', legend=False, cmap='Reds')
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

ax.set_xlim(100, 120)
ax.set_ylim(7, 24)
ax.set_xticks([])
ax.set_yticks([])

ax.set_title('Vietnam\nDamrey typhoon 2017, \n (Prob of Damage > {}%) \nSelected grids'.format(threshold * 100))
plt.show()
```

    2024-03-12 13:43:02,848 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.




![png](model_correlations_files/model_correlations_60_1.png)



And we distribuite damage acording to the number of houses per grid


```python
# Variables
damrey_new['numbuildings_regions'] = damrey_new[damrey_new['with_damage'] == 1].total_buildings.sum()
damrey_new['frac_bld'] = damrey_new['total_buildings'] / damrey_new['numbuildings_regions']
damrey_new_reduced = damrey_new[damrey_new.with_damage == 1]
damrey_new_reduced_nodmg = damrey_new[damrey_new.with_damage == 0]

# Damage
damrey_new_reduced['N_dmg'] = damrey_new_reduced['frac_bld'] * damrey_new_reduced['total_buildings_damaged']
damrey_new_reduced['perc_dmg'] = damrey_new_reduced['frac_bld'] * damrey_new_reduced['total_buildings_damaged'] * 100 / damrey_new_reduced['numbuildings_regions']

# All country again
damrey_new_complete = pd.concat([damrey_new_reduced, damrey_new_reduced_nodmg]).fillna(0)
```


```python
# Track path
track = TCTracks.from_ibtracs_netcdf(storm_id='2017304N11127') # Damrey2017
tc_track = track.get_track()
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,5))
damrey_new_complete.plot(ax=ax, column='N_dmg', legend=True, cmap='Reds')
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

# Add label to colorbar
cbar = ax.get_figure().get_axes()[1]  # Get the colorbar axis
cbar.set_ylabel('Buildings damaged')  # Add label to the colorbar

ax.set_xlim(100, 120)
ax.set_ylim(7, 24)
ax.set_xticks([])
ax.set_yticks([])

ax.set_title('Vietnam\nDamrey typhoon 2017, \n (Prob of Damage > {}%) \nClassic dissaggregation'.format(threshold * 100))
plt.show()
```

    2024-03-12 13:43:22,180 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.




![png](model_correlations_files/model_correlations_63_1.png)



### Threshold example


```python
wind_thres = 25
```


```python
damrey_thres = viet_data[(viet_data.typhoon_name == 'DAMREY') & (viet_data.Year == 2017)]
# Add region column
damrey_thres = damrey_thres.merge(ids_mun_viet[['id', 'region']], left_on='grid_point_id', right_on='id')

# Add feature
damrey_thres['prob_dmg'] = 1
# Consider just damage regions
regions = ['NC', 'SC', 'C']
damrey_thres.loc[~damrey_thres.region.isin(regions), 'prob_dmg'] = 0

damrey_thres = gpd.GeoDataFrame(damrey_thres[
    ['grid_point_id', 'typhoon_name', 'Year', 'prob_dmg', 'region', 'wind_speed', 'track_distance', 'scaled_distance', 'total_buildings', 'total_buildings_damaged']
    ].merge(grid_land_overlap, left_on='grid_point_id', right_on='id'), geometry='geometry')

damrey_thres.loc[damrey_thres.wind_speed < wind_thres, 'prob_dmg'] = 0

damrey_thres['with_damage'] = np.where(damrey_thres['prob_dmg'] > 0, 1, 0)
```


```python
# Variables
damrey_thres['numbuildings_regions'] = damrey_thres[damrey_thres['with_damage'] == 1].total_buildings.sum()
damrey_thres['frac_bld'] = damrey_thres['total_buildings'] / damrey_thres['numbuildings_regions']
damrey_thres_reduced = damrey_thres[damrey_thres.with_damage == 1]
damrey_thres_reduced_nodmg = damrey_thres[damrey_thres.with_damage == 0]

# Damage
damrey_thres_reduced['N_dmg'] = damrey_thres_reduced['frac_bld'] * damrey_thres_reduced['total_buildings_damaged']
damrey_thres_reduced['perc_dmg'] = damrey_thres_reduced['frac_bld'] * damrey_thres_reduced['total_buildings_damaged'] * 100 / damrey_thres_reduced['numbuildings_regions']

# All country again
damrey_thres_complete = pd.concat([damrey_thres_reduced, damrey_thres_reduced_nodmg]).fillna(0)
```


```python
# Track path
track = TCTracks.from_ibtracs_netcdf(storm_id='2017304N11127') # Damrey2017
tc_track = track.get_track()
points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
track_points = gpd.GeoDataFrame(geometry=points)
tc_track_line = LineString(points)
track_line = gpd.GeoDataFrame(geometry=[tc_track_line])

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,5))
damrey_thres_complete.plot(ax=ax, column='N_dmg', legend=True, cmap='Reds')
track_line.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

# Add label to colorbar
cbar = ax.get_figure().get_axes()[1]  # Get the colorbar axis
cbar.set_ylabel('Buildings damaged')  # Add label to the colorbar

ax.set_xlim(100, 120)
ax.set_ylim(7, 24)
ax.set_xticks([])
ax.set_yticks([])

ax.set_title('Vietnam\nDamrey typhoon 2017, \n (Windspeed > {} m/s grids) \nClassic dissaggregation'.format(wind_thres))
plt.show()
```

    2024-03-14 14:20:10,616 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.




![png](model_correlations_files/model_correlations_68_1.png)
