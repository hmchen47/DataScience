# Example: House Prices


Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices)

[Local notebook](src/a18g-feature-engineering-for-house-prices.ipynb)


## Introduction

+ Project
  + [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
  + [Ames](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


## Step 1 - Preliminaries

+ Setting up environment

  ```python
  import os
  import warnings
  from pathlib import Path

  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import seaborn as sns
  from IPython.display import display
  from pandas.api.types import CategoricalDtype

  from category_encoders import MEstimateEncoder
  from sklearn.cluster import KMeans
  from sklearn.decomposition import PCA
  from sklearn.feature_selection import mutual_info_regression
  from sklearn.model_selection import KFold, cross_val_score
  from xgboost import XGBRegressor

  # Set Matplotlib defaults
  plt.style.use("seaborn-whitegrid")
  plt.rc("figure", autolayout=True)
  plt.rc(
      "axes",
      labelweight="bold",
      labelsize="large",
      titleweight="bold",
      titlesize=14,
      titlepad=10,
  )

  # Mute warnings
  warnings.filterwarnings('ignore')
  ```

+ Data preprocessing
  + pre-processing the data to get it in a form suitable for analysis
  + typical actions to process data
    + __load__ the data from CSV files
    + __clean__ the data to fix any errors or inconsistencies
    + __encode__ the statistical data type (numeric, categories)
    + __impute__ any missing values
  + 3 preprocessing steps after reaing the CSV file: `clean`, `encode` and `impute`
  + creating the data splits:
    + one (`df_train`) for training the model
    + one (`df_test`) for making the predictions 



## Step 2 - Feature  Utility Scores






## Step 3 - Create Features






## Step 4 - Hyperparameter Tuning






## Step 5 - Train Model and Create Submissions








