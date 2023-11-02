# Data models

Here we apply some models to predict the damage caused by each typhoon at grid level.

## Report: codes


### Code 01: Collage data

Input: damage data, wind data, rainfall data, IWI data and topographical data.

In 01_collage_data.ipynb we merge all dataset that we have at grid level. For that, we use the grid_id number.

Output: dataset of all the data mentioned at grid level **complete** and **incomplete**. Where 'complete' implies that we have data for all of the grid cells that overlap with the land and 'incomplete' implies that we're dropping all regions with 0 buildings or not affected by an specific typhoon. Hence, in the complete dataset, the number of datapoints varies by typhoon.


### Code 02: Regression models and binary classification models.

Here we explore:

- How the model performs with just wind and rainfall data?
- What if we add topographical data?
- What if we also add the IWI data?

In every case, since we have not much data, we use LOOCV to train our models. Also, since the data is extremely skewed, we use a stratification method for every regressor model.

In 02.0_model_training-baselines.ipynb we apply a dummy baseline model, a linear regression model, an XG-Bosst model, the 2-Stage XG-Boost model used for the Philippines dataset and a binary classification model with a threshold of 3% damage. We use as features: wind and rainfall data as well as the total number of houses by grid. The target variable is the % of damage by grid cell.


In 02.1_model_training_with_topography.ipynb we apply a dummy baseline model, a linear regression model, an XG-Bosst model, the 2-Stage XG-Boost model used for the Philippines dataset and a binary classification model with a threshold of 3% damage. We use as features: wind, rainfall and topographical data, as well as the total number of houses by grid. The target variable is the % of damage by grid cell.


In 02.2_model_training_with_topography_and_IWI.ipynb we apply a dummy baseline model, a linear regression model, an XG-Bosst model, the 2-Stage XG-Boost model used for the Philippines dataset and a binary classification model with a threshold of 3% damage. We use as features: wind, rainfall, topographical and IWI data, as well as the total number of houses by grid. The target variable is the % of damage by grid cell. Also, here, apart from plotting RMSE per bin, we plot the average error per bin.


### Code 03: 3 class classification method

Here, again we use LOOCV for train/test split and % of housing damage by grid as the target variable. We apply 3-class classification methods (XG-Boost classificator and Logistic Regression). Specifically, we consider 3 cases:

- 0% damage (no damage)
- <1% damage (little damage)
- >1% damage (damage)

Finally, we plot 3x3 confusion matrices.
