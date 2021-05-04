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

+ Loading data

  ```python
  def load_data():
      # Read data
      data_dir = Path("../input/house-prices-advanced-regression-techniques/")
      df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
      df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
      # Merge the splits so we can process them together
      df = pd.concat([df_train, df_test])
      # Preprocessing
      df = clean(df)
      df = encode(df)
      df = impute(df)
      # Reform splits
      df_train = df.loc[df_train.index, :]
      df_test = df.loc[df_test.index, :]
      return df_train, df_test
  ```

+ Cleaning data
  
  ```python
  data_dir = Path("data/a18g/")
  df = pd.read_csv(data_dir / "train.csv", index_col="Id")

  df.Exterior2nd.unique()
  # array(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
  #        'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
  #        'AsphShn', 'Stone', 'Other', 'CBlock'], dtype=object)

  def clean(df):
      df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
      # Some values of GarageYrBlt are corrupt, so we'll replace them
      # with the year the house was built
      df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
      # Names beginning with numbers are awkward to work with
      df.rename(columns={
          "1stFlrSF": "FirstFlrSF",
          "2ndFlrSF": "SecondFlrSF",
          "3SsnPorch": "Threeseasonporch",
      }, inplace=True,
      )
      return df
  ```

+ Encoding the statistical data type
  + Pandas w/ Python types corresponding to the standard statistical types (numeric, categories)
  + encoding correct type for each feature $\to$ treating each feature w/ appropriate fucntion
  + applying transformation consistently

  ```python
  # The numeric features are already encoded correctly (`float` for
  # continuous, `int` for discrete), but the categoricals we'll need to
  # do ourselves. Note in particular, that the `MSSubClass` feature is
  # read as an `int` type, but is actually a (nominative) categorical.

  # The nominative (unordered) categorical features
  features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", 
      "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", 
      "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", 
      "SaleType", "SaleCondition"]

  # The ordinal (ordered) categorical features 

  # Pandas calls the categories "levels"
  five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
  ten_levels = list(range(10))

  ordered_levels = {
      "OverallQual": ten_levels,
      "OverallCond": ten_levels,
      "ExterQual": five_levels,
      "ExterCond": five_levels,
      "BsmtQual": five_levels,
      "BsmtCond": five_levels,
      "HeatingQC": five_levels,
      "KitchenQual": five_levels,
      "FireplaceQu": five_levels,
      "GarageQual": five_levels,
      "GarageCond": five_levels,
      "PoolQC": five_levels,
      "LotShape": ["Reg", "IR1", "IR2", "IR3"],
      "LandSlope": ["Sev", "Mod", "Gtl"],
      "BsmtExposure": ["No", "Mn", "Av", "Gd"],
      "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
      "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
      "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
      "GarageFinish": ["Unf", "RFn", "Fin"],
      "PavedDrive": ["N", "P", "Y"],
      "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
      "CentralAir": ["N", "Y"],
      "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
      "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
  }

  # Add a None level for missing values
  ordered_levels = {key: ["None"] + value for key, value in
                    ordered_levels.items()}

  def encode(df):
      # Nominal categories
      for name in features_nom:
          df[name] = df[name].astype("category")
          # Add a None category for missing values
          if "None" not in df[name].cat.categories:
              df[name].cat.add_categories("None", inplace=True)
      # Ordinal categories
      for name, levels in ordered_levels.items():
          df[name] = df[name].astype(CategoricalDtype(levels,
                                                      ordered=True))
      return df
  ```


## Step 2 - Feature  Utility Scores






## Step 3 - Create Features






## Step 4 - Hyperparameter Tuning






## Step 5 - Train Model and Create Submissions








