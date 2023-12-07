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

### Code 04: Combined model Philippies + Fiji data for training

In 04.0_models_trained_with_phl_data.ipynb we:

- Stratify Fiji and Philippines data separately.
- Using LOOCV, we trained and test on Phlippines dataset using the 2SG-XGBoost model of the Philippines paper. This is just to be sure that we can replicate the result of the paper.
- Then me create a **combined model** using 3 different approaches:

*Approach 1*: Simple model 1

Here we train on Philippines data and we test on Fiji data using a classic XGBoost Regressor model.

*Approach 2*: Simple model 2

Here, using LOOCV for Fiji typhoons we train on Philippines + Fiji typhoons considering a weighted sample XGBoost Regressor model. Here we gave the Fiji typhoons more weight (2:1 ratio) related to the Philippienes ones.

*Approach 3*: Dumb approach. Just skip this one, is not representative. I leave it here for completness.


In 04.1_simple_model_combined_data_tuning_windspeed.ipynb we apply the "Simple model 2" mentioned before but we play with dropping 0 windspeed values on: all typhoons, just Philippines typhoons or not dropping any 0 windspeed value. This is because, in the Philippines paper, we are not considering 0 values of windspeed on grids. We find that dropping all grid cells with 0 windspeed values for the Philippines subset is the correct approach (based on RMSE per bin)


In 04.2_simple_model_combined_data_tuning_weight.ipynb we play with different ratios of weighting samples for the combined model. We dont conclude with ratio is the best one, but a ratio >= 2:1 in favour of the Fiji subset is appropiate to work with in terms on minimizing the RMSE per bin.


In 04.3_simple_model_combined_data_no_rainfall.ipynb we measure the influence of considering and not considering rainfall data in our model. In conclusion, we should better consider this feature.

### Code 05: Combined model Philippies + Fiji datasets.

In 05.0_combined_model.ipynb we aggregate damage at municipality level and display damage maps.

In 05.1_combined_model_feature_importance.ipynb we calculate the feature_importance using the build-in feature from XGBoost for the model with LOOCV (mean values of imoportance per testing typhoon) and for some specific (and impactful) typhoons, like YASA and WINSTON.
