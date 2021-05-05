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

+ Handling missing values
  + imputing 0 for missing numeric values and `None` for missing categories value
  + probably experimenting w/ other imputation strategies
  + try creating "missing value" indicators as 1 whatever a value was imputed and 0 otherwise

  ```python
  def impute(df):
      for name in df.select_dtypes("number"):
          df[name] = df[name].fillna(0)
      for name in df.select_dtypes("category"):
          df[name] = df[name].fillna("None")
      return df
  ```

+ Loading and processing data splits

  ```python
  Handling missing values

  # Peek at the values
  # display(df_train)
  # display(df_test)

  # Display information about dtypes and missing values
  display(df_train.info())
  # <class 'pandas.core.frame.DataFrame'>
  # Int64Index: 1460 entries, 1 to 1460
  # Data columns (total 80 columns):
  #  #   Column            Non-Null Count  Dtype        #   Column            Non-Null Count  Dtype 
  # ---  ------            --------------  -----        ---  ------            --------------  -----
  #  0   MSSubClass        1460 non-null   category     1   MSZoning          1460 non-null   category
  #  2   LotFrontage       1460 non-null   float64      3   LotArea           1460 non-null   int64   
  #  4   Street            1460 non-null   category     5   Alley             1460 non-null   category
  #  6   LotShape          1460 non-null   category     7   LandContour       1460 non-null   category
  #  8   Utilities         1460 non-null   category     9   LotConfig         1460 non-null   category
  #  10  LandSlope         1460 non-null   category     11  Neighborhood      1460 non-null   category
  #  12  Condition1        1460 non-null   category     13  Condition2        1460 non-null   category
  #  14  BldgType          1460 non-null   category     15  HouseStyle        1460 non-null   category
  #  16  OverallQual       1460 non-null   category     17  OverallCond       1460 non-null   category
  #  18  YearBuilt         1460 non-null   int64        19  YearRemodAdd      1460 non-null   int64   
  #  20  RoofStyle         1460 non-null   category     21  RoofMatl          1460 non-null   category
  #  22  Exterior1st       1460 non-null   category     23  Exterior2nd       1460 non-null   category
  #  24  MasVnrType        1460 non-null   category     25  MasVnrArea        1460 non-null   float64 
  #  26  ExterQual         1460 non-null   category     27  ExterCond         1460 non-null   category
  #  28  Foundation        1460 non-null   category     29  BsmtQual          1460 non-null   category
  #  30  BsmtCond          1460 non-null   category     31  BsmtExposure      1460 non-null   category
  #  32  BsmtFinType1      1460 non-null   category     33  BsmtFinSF1        1460 non-null   float64 
  #  34  BsmtFinType2      1460 non-null   category     35  BsmtFinSF2        1460 non-null   float64 
  #  36  BsmtUnfSF         1460 non-null   float64      37  TotalBsmtSF       1460 non-null   float64 
  #  38  Heating           1460 non-null   category     39  HeatingQC         1460 non-null   category
  #  40  CentralAir        1460 non-null   category     41  Electrical        1460 non-null   category
  #  42  FirstFlrSF        1460 non-null   int64        43  SecondFlrSF       1460 non-null   int64   
  #  44  LowQualFinSF      1460 non-null   int64        45  GrLivArea         1460 non-null   int64   
  #  46  BsmtFullBath      1460 non-null   float64      47  BsmtHalfBath      1460 non-null   float64 
  #  48  FullBath          1460 non-null   int64        49  HalfBath          1460 non-null   int64   
  #  50  BedroomAbvGr      1460 non-null   int64        51  KitchenAbvGr      1460 non-null   int64   
  #  52  KitchenQual       1460 non-null   category     53  TotRmsAbvGrd      1460 non-null   int64   
  #  54  Functional        1460 non-null   category     55  Fireplaces        1460 non-null   int64   
  #  56  FireplaceQu       1460 non-null   category     57  GarageType        1460 non-null   category
  #  58  GarageYrBlt       1460 non-null   float64      59  GarageFinish      1460 non-null   category
  #  60  GarageCars        1460 non-null   float64      61  GarageArea        1460 non-null   float64 
  #  62  GarageQual        1460 non-null   category     63  GarageCond        1460 non-null   category
  #  64  PavedDrive        1460 non-null   category     65  WoodDeckSF        1460 non-null   int64   
  #  66  OpenPorchSF       1460 non-null   int64        67  EnclosedPorch     1460 non-null   int64   
  #  68  Threeseasonporch  1460 non-null   int64        69  ScreenPorch       1460 non-null   int64   
  #  70  PoolArea          1460 non-null   int64        71  PoolQC            1460 non-null   category
  #  72  Fence             1460 non-null   category     73  MiscFeature       1460 non-null   category
  #  74  MiscVal           1460 non-null   int64        75  MoSold            1460 non-null   int64   
  #  76  YrSold            1460 non-null   int64        77  SaleType          1460 non-null   category
  #  78  SaleCondition     1460 non-null   category     79  SalePrice         1460 non-null   float64 
  # dtypes: category(46), float64(12), int64(22)

  display(df_test.info())

  ```

+ Establishing baseline
  + establishing a baseline score to judge feature engineering
  + computing the cross-validated RMSLE score for a feature set
  + using `XGBoost` model
  + probably experimenting other parameters

  ```python
  def score_dataset(X, y, model=XGBRegressor()):
      # Label encoding for categoricals
      #
      # Label encoding is good for XGBoost and RandomForest, but one-hot
      # would be better for models like Lasso or Ridge. The `cat.codes`
      # attribute holds the category levels.
      for colname in X.select_dtypes(["category"]):
          X[colname] = X[colname].cat.codes
      # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
      log_y = np.log(y)
      score = cross_val_score(
          model, X, log_y, cv=5, scoring="neg_mean_squared_error",
      )
      score = -1 * score.mean()
      score = np.sqrt(score)
      return score

  X = df_train.copy()
  y = X.pop("SalePrice")

  baseline_score = score_dataset(X, y)
  print(f"Baseline score: {baseline_score:.5f} RMSLE")
  # Baseline score: 0.14351 RMSLE
  ```


## Step 2 - Feature  Utility Scores

+ Computing mutual information
  + using MI to computer a utility for a feature
  + utility functions: `make_mi_scores` and `plot_mi_scores`
  + some features highly informative while some not informative at all


  ```python
  def make_mi_scores(X, y):
      X = X.copy()
      for colname in X.select_dtypes(["object", "category"]):
          X[colname], _ = X[colname].factorize()
      # All discrete features should now have integer dtypes
      discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
      mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
      mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
      mi_scores = mi_scores.sort_values(ascending=False)
      return mi_scores

  def plot_mi_scores(scores):
      scores = scores.sort_values(ascending=True)
      width = np.arange(len(scores))
      ticks = list(scores.index)
      plt.barh(width, scores)
      plt.yticks(width, ticks)
      plt.title("Mutual Information Scores")

  X = df_train.copy()
  y = X.pop("SalePrice")

  mi_scores = make_mi_scores(X, y)
  mi_scores
  # OverallQual     0.571457  Neighborhood    0.526220
  # GrLivArea       0.430395  YearBuilt       0.407974
  # LotArea         0.394468
  #                   ...   
  # MiscVal         0.000000  MiscFeature     0.000000
  # PoolQC          0.000000  MoSold          0.000000
  # YrSold          0.000000
  ```

+ Computing baseline scores
  + removing the uninformative features
  + computing the mutual information score

  ```python
  def drop_uninformative(df, mi_scores):
      return df.loc[:, mi_scores > 0.0]

  X = df_train.copy()
  y = X.pop("SalePrice")
  X = drop_uninformative(X, mi_scores)

  score_dataset(X, y)
    # 0.14338026718687277
  ```


## Step 3 - Create Features

+ Creating features
  + making feature engineering workflow more modular
  + defining a function to take a prepared dataframe and pass through a pipeline of transformations to get the final feature set
  + [label encoding](https://www.kaggle.com/alexisbcook/categorical-variables) for the categorical features
  + pseudocode

    ```python
    def create_features(df):
        X = df.copy()<br>
        y = X.pop("SalePrice")<br>
        X = X.join(create_features_1(X))<br>
        X = X.join(create_features_2(X))<br>
        X = X.join(create_features_3(X))<br>
        # ...<br>
        return X
    ```





## Step 4 - Hyperparameter Tuning






## Step 5 - Train Model and Create Submissions








