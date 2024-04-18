# The models

## Codes

We load the input datasets and the models from ```utils.py```

### 01 Code: poppulation as target variable

In 01 we run the model for predicting affected population. We train and test on JUST HTI data.

In 01.1 we compute the SHAP values for the features of out model.

In 01.2 we add non-affecting events to the discussion (balanced dataset).

In 01.3 we aggregate every feature to ADM0 level and check the performance of the model in the most simple case possible: a model with just weather features aggregated at ADM0.

### 02 Code: combined model

In 02.0 we run the combined model, predicting building damage for HTI. Here, we infered the buildings affected based on the density people/bld of each grid.

In 02.1 we run the combined model, predicting building damage for HTI. Here, instead of infere the buildings affected based on the density people/bld of each grid, we used the Philippines bld affected / pop affected relations discussed before.
