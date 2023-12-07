---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: env1
    language: python
    name: env1
---

```python
import statistics

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from mlxtend.plotting import plot_confusion_matrix


from utils import get_training_dataset_complete
```

# Data cleaning and stratification

```python
df_complete = get_training_dataset_complete()
df = df_complete.copy()
df = df.rename({'perc_dmg_grid':'percent_houses_damaged', 'total_buildings':'total_houses'}, axis=1)
```

```python
# Set any values of damage houses >100% to 100% .. ok
for r in range(len(df)):
    if df.loc[r, "percent_houses_damaged"] > 100:
        df.at[r, "percent_houses_damaged"] = float(100)
```

```python
#drop windspeed = 0
# df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
# df = df.drop(columns=["grid_point_id", "typhoon_year_y", "typhoon_year"])
```

```python
#what about rainfall data? Lets drop all 0 values.
# df = (df[(df[["rainfall_max_24h"]] != 0).any(axis=1)]).reset_index(drop=True)
# df = (df[(df[["rainfall_max_6h"]] != 0).any(axis=1)]).reset_index(drop=True)
# df.typhoon_name.unique()
```

### Stratification: 3 possible values

```python
fig, ax = plt.subplots(1,2, figsize=(10,6))

ax[0].hist(df.percent_houses_damaged, edgecolor='black')
ax[0].set_xlabel('% Houses Damaged',size=15)
ax[0].set_ylabel('Frequency',size=15)

hist = np.histogram(df.percent_houses_damaged, bins=10 ** np.linspace(0, np.log10(10**1.5), len(df)), density=True)
x = hist[1][:-1]
y = hist[0]

ax[1].plot(x,y, 'ro', alpha=0.4, label='housing damage')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('% Houses Damaged', size=15)
ax[1].set_ylabel('Frequency', size=15) #PDF (% Houses Damaged)
ax[1].grid(c='black', alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.show()
```

```python
# Distribution of information
dmg = np.array(df.percent_houses_damaged.to_list())
offset = 1e-8
dmg_off = dmg + offset
x = list(np.linspace(0,1,101))
info = []
for i in x:
    info.append(np.quantile(dmg_off, i))

plt.plot(x,info)
plt.xlabel('Quantile')
plt.ylabel('Damage [%]')
plt.yscale('log')
plt.title('Damage Offset = {}'.format(offset))
plt.grid()
plt.show()
```

So we have ~50% of the data with 0 damage and ~90% with < 1% damage

```python
conditions = [
    (df['percent_houses_damaged'] == 0), # No damage
    ((df['percent_houses_damaged'] > 0) &
     (df['percent_houses_damaged'] <= 1)), # Little damage
    (df['percent_houses_damaged'] > 1) # Damage
]
values = [0, 1, 2]

df['damage_classification'] = np.select(conditions, values)
```

```python
df['damage_classification'].value_counts()
```

```python
df['damage_classification'].hist()
plt.xlabel('')
plt.ylabel('Frequency')
plt.xticks([0.1, 1.1, 1.9], ['0% damage', '$<$1% damage', '$\geq$ 1% damage'])
plt.title('Distribution of damage')
plt.show()
```

# Define plot function

```python
def cm_plot(cms, accuracy, f1score, model_name):
    overall_confusion_matrix = np.zeros((3, 3))

    for matrix in cms:
        # Ensure that each confusion matrix has the same shape
        if matrix.shape != (3, 3):
            reshaped_cm = np.zeros((3, 3))
            reshaped_cm[:matrix.shape[0], :matrix.shape[1]] = matrix
            matrix = reshaped_cm

        overall_confusion_matrix += matrix

    # Plot Confusion Matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=overall_confusion_matrix,
        show_absolute=True,
        show_normed=True,
        colorbar=False,
        cmap=plt.cm.Greens,
        figsize=(8, 5)
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.xaxis.set(ticks=range(3), ticklabels=("Predicted 0% dmg", "Predicted little dmg", "Predicted dmg"))
    ax.yaxis.set(ticks=range(3), ticklabels=("Actual 0% dmg", "Actual little dmg", "Actual dmg"))
    plt.xticks(rotation=45)
    ax.set_title("Confusion Matrix for {} Model with 3 classes".format(model_name))

    mean_acc = np.mean(accuracy)
    mean_f1 = np.mean(f1score)
    # Add information
    info_text = f"Mean Accuracy: {mean_acc:.2f}\nMean weighted F1 Score: {mean_f1:.2f}"
    ax.text(2.8, -0.2, info_text, fontsize=12, verticalalignment='top',
           bbox={'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'})


    plt.show()

```

# Models

```python
# List of typhoons
typhoons = df.typhoon_name.unique()

# Specify features
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "coast_length",
    "with_coast",
    "mean_altitude",
    "mean_slope",
    "IWI"
]
```

## XGB Classifier -no standardize-

```python
cms_xgb = []
accuracy_xgb = []
f1score_xgb = []
for typhoon in typhoons:

    """      STEP 0: TRAIN TEST SPLIT LOOCV     """

    # Split X and y from dataframe features
    X = df[features]
    y = df["damage_classification"]

    # Split df to train and test (one typhoon for test and the rest of typhoons for train) --LOOCV--
    df_test = df[df["typhoon_name"] == typhoon]
    df_train = df[df["typhoon_name"] != typhoon]

    # Split X and y from dataframe features
    X_test = df_test[features]
    X_train = df_train[features]

    y_train = df_train["damage_classification"]
    y_test = df_test["damage_classification"]

    """      STEP 1: XGB CLASSIFIER      """
    # XGBClassifier
    xgb_model = XGBClassifier(eval_metric=["merror", "mlogloss"], num_class=3)
    xgb_model.fit(X_train, y_train)

    # Make prediction on test data
    y_pred_test = xgb_model.predict(X_test)

    # Confusion matrix
    cm_typhoon = confusion_matrix(y_test, y_pred_test)
    cms_xgb.append(cm_typhoon)

    # Accuracy and F1-score
    acc = accuracy_score(y_test, y_pred_test)
    accuracy_xgb.append(acc)
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    f1score_xgb.append(f1)

```

```python
cm_plot(cms_xgb, accuracy_xgb, f1score_xgb, 'XGBoost')
```

## XGB Classifier (standardize data)

```python
cms_xgb_s = []
accuracy_xgb_s = []
f1score_xgb_s = []
for typhoon in typhoons:

    """      STEP 0: TRAIN TEST SPLIT LOOCV     """

    # Split X and y from dataframe features
    X = df[features]
    y = df["damage_classification"]

    # Split df to train and test (one typhoon for test and the rest of typhoons for train) --LOOCV--
    df_test = df[df["typhoon_name"] == typhoon]
    df_train = df[df["typhoon_name"] != typhoon]

    # Split X and y from dataframe features
    X_test = df_test[features]
    X_train = df_train[features]

    y_train = df_train["damage_classification"]
    y_test = df_test["damage_classification"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    """      STEP 1: XGB CLASSIFIER      """
    # XGBClassifier
    xgb_model = XGBClassifier(eval_metric=["merror", "mlogloss"], num_class=3)
    xgb_model.fit(X_train_scaled, y_train)

    # Make prediction on test data
    y_pred_test = xgb_model.predict(X_test_scaled)

    # Confusion matrix
    cm_typhoon = confusion_matrix(y_test, y_pred_test)
    cms_xgb_s.append(cm_typhoon)

    # Accuracy and F1-score
    acc = accuracy_score(y_test, y_pred_test)
    accuracy_xgb_s.append(acc)
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    f1score_xgb_s.append(f1)
```

```python
cm_plot(cms_xgb_s, accuracy_xgb_s, f1score_xgb_s, 'XGBoost (standardized)')
```

## LogReg classifier

```python
cms_logreg = []
accuracy_logreg = []
f1score_logreg = []

for typhoon in typhoons:
    # STEP 0: TRAIN TEST SPLIT LOOCV

    # Split X and y from dataframe features
    X = df[features]
    y = df["damage_classification"]

    # Split df to train and test (one typhoon for test and the rest of typhoons for train) --LOOCV--
    df_test = df[df["typhoon_name"] == typhoon]
    df_train = df[df["typhoon_name"] != typhoon]

    # Split X and y from dataframe features
    X_test = df_test[features]
    X_train = df_train[features]

    y_train = df_train["damage_classification"]
    y_test = df_test["damage_classification"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 1: Logistic Regression Classifier

    # Logistic Regression classifier
    logistic_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    logistic_model.fit(X_train_scaled, y_train)

    # Make prediction on test data
    y_pred_test = logistic_model.predict(X_test_scaled)

    # Accuracy and F1-score
    acc = accuracy_score(y_test, y_pred_test)
    accuracy_logreg.append(acc)
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    f1score_logreg.append(f1)

    # Confusion matrix
    cm_typhoon = confusion_matrix(y_test, y_pred_test)
    cms_logreg.append(cm_typhoon)

```

```python
cm_plot(cms_logreg, accuracy_logreg, f1score_logreg, 'Logistic Regression')
```

## Random Forest Classifier

```python
cms_random_forest = []
accuracy_random_forest = []
f1score_random_forest = []

for typhoon in typhoons:
    # STEP 0: TRAIN TEST SPLIT LOOCV

    # Split X and y from dataframe features
    X = df[features]
    y = df["damage_classification"]

    # Split df to train and test (one typhoon for test and the rest of typhoons for train) --LOOCV--
    df_test = df[df["typhoon_name"] == typhoon]
    df_train = df[df["typhoon_name"] != typhoon]

    # Split X and y from dataframe features
    X_test = df_test[features]
    X_train = df_train[features]

    y_train = df_train["damage_classification"]
    y_test = df_test["damage_classification"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 1: Random Forest Classifier

    # Random Forest classifier
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=0)
    random_forest_model.fit(X_train_scaled, y_train)

    # Make prediction on test data
    y_pred_test = random_forest_model.predict(X_test_scaled)

    # Accuracy and F1-score
    acc = accuracy_score(y_test, y_pred_test)
    accuracy_random_forest.append(acc)
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    f1score_random_forest.append(f1)

    # Confusion matrix
    cm_typhoon = confusion_matrix(y_test, y_pred_test)
    cms_random_forest.append(cm_typhoon)

```

```python
cm_plot(cms_random_forest, accuracy_random_forest, f1score_random_forest, 'Random Forest')
```
