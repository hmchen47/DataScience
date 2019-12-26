# Feature Selection Techniques

[Origin](https://pierpaolo28.github.io/blog/blog30/)

## Introduction

+ choosing the right features in the right format to feed in a model boosting performances leading to the following benefits
  + enable to achieve good model performances using simpler Machine Learning models
  + using simpler ML models, increasing the transparency of model $to$ easier to understand how is making its predictions
  + reduced need to use Ensemble learning techniques
  + reduced need to perform hyperparameter optimization

+ Other common techniques to better use data:
  + [Features selection](https://towardsdatascience.com/feature-selection-techniques-1bfab5fe0784)
  + [Extraction](https://towardsdatascience.com/feature-extraction-techniques-d619b56e31be)

+ Most of the basic feature engineering techniques
  + finding inconsistencies in the data
  + creating new featres by combining/dividing existing ones

+ [Jupyter notebook](https://github.com/pierpaolo28/Artificial-Intelligence-Projects/blob/master/Features%20Analysis/FeatureEngineering.ipynb)

+ Created dataset for analysis

  ```python
  import numpy as np
  import pandas as pd 

  dataset_len = 8000
  dlen = int(dataset_len/2)
  X_11 = pd.Series(np.random.normal(8,2,dlen))
  X_12 = pd.Series(np.random.randint(low=0, high=100, size=dlen))
  X_1 = pd.concat([X_11, X_12]).reset_index(drop=True)
  X_21 = pd.Series(np.random.normal(20,3,dlen))
  X_22 = pd.Series(np.random.normal(9,3,dlen))
  X_2 = pd.concat([X_21, X_22]).reset_index(drop=True)
  X_31 = pd.Series(np.random.normal(8,2,dlen))
  X_32 = pd.Series(np.random.randint(low=0, high=300, size=dlen))
  X_3 = pd.concat([X_31, X_32]).reset_index(drop=True)
  X_4 = pd.Series(np.repeat(['Car', 'Bus', 'Bike', 'Scooter'],
                            dlen/2))
  Y = pd.Series(np.repeat(['True','False'],dlen))
  df = pd.concat([X_1, X_2, X_3, X_4, Y], axis=1)
  df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']
  df['X1'] = df['X1'].mask(np.random.random(df['X1'].shape) < 0.22)
  df[['X3', 'X4']] = df[['X3', 'X4']].mask(np.random.random(df[['X3', 
                          'X4']].shape) < 0.10)
  dates = pd.date_range(start='1/1/2016', periods=len(df), freq='D')
  df.insert(loc= 0, column='Date', value= dates)
  df.head()
  #   	Date	X1	X2	X3	X4	Y
  # 0	2016-01-01	11.798186	16.299548	9.466762	Car	True
  # 1	2016-01-02	6.705743	14.254479	7.405022	Car	True
  # 2	2016-01-03	5.451097	16.471537	7.440786	Car	True
  # 3	2016-01-04	13.364043	22.021941	NaN	Car	True
  # 4	2016-01-05	NaN	19.937920	7.422658	Car	True
  ```

## Log Transformation

+ Log transformation
  + the distributions of the original features get transformed to resemble more closely Gaussian distribution
  + useful w/ models
    + Linear Discriminant Analysis (LDA)
    + Naive Bayes Classifier

+ Applying log transformation to all the numeric features

  ```python
  df[['X1', 'X2', 'X3']]=(df[['X1','X2','X3']]-
                          df[['X1','X2','X3']].min()+
                          1).transform(np.log)
  df.sample(5)
  #             Date        X1        X2        X3      X4         Y
  # 2496  2022-11-01  2.482055  3.270268  2.487769      Bus     True
  #  782  2018-02-21  NaN       3.139792  1.970446      Car     True
  # 3449  2025-06-11  NaN       3.176024  2.672777      Bus     True
  #  148  2016-05-28  2.350232  3.434115  2.216226      Car     True
  # 7505  2036-07-19  NaN       2.248421  NaN       Scooter    False
  ```

## Imputation

+ Imputation: the art of identifying and replacing missing values from a dataset using appropriate values

+ main types of imputation
  + numerical imputation
    + main method: replace missing values w/ the overall mean or mode of the affected column
    + [advanced methods](https://towardsdatascience.com/why-using-a-mean-for-missing-data-is-a-bad-idea-alternative-imputation-algorithms-837c731c1008)
  + categorical imputation
    + commonly replaced using the overall column mode
    + replace the missing values creating a new category and naming it "Unknown" or "Other" if the categorical column structure not well defined

+ Examing which features affected by NaN (Not a Number)

  ```python
  percent_missing = df.isnull().sum() * 100 / len(df)
  missing_values = pd.DataFrame({'NaNs percentage': percent_missing})
  missing_values.sort_values(by ='NaNs percentage' , ascending=False)
  # NaNs percentage
  # X1    21.8000
  # X4    10.4625
  # X3    9.5875
  # Date  0.0000
  # X2    0.0000
  # Y     0.0000
  ```

+ Removing all the rows w/ NaN

  ```python
  # Drop Column if at least 20% of it's elements are Nans
  df = df.loc[:, df.isnull().sum() < 0.2*df.shape[0]]

  percent_missing = df.isnull().sum() * 100 / len(df)
  missing_values = pd.DataFrame({'NaNs percentage': percent_missing})
  missing_values.sort_values(by ='NaNs percentage' , ascending=False)
  #      NaNs percentage
  #X4    10.4625
  #X3    9.5875
  #Date  0.0000
  #X2    0.0000
  #Y     0.0000
  ```

+ Replacing all the NaNs w/ the column mode for both numerical and categorical data

  ```python
  # Using Mean to fill NaNs
  #for i in range(0, len(df.columns)):
  for i in range(1, 3):
      df.iloc[:,i].fillna(df.iloc[:,i].median(), inplace=True)

  df['X4'].fillna(df['X4'].value_counts().idxmax(), inplace=True)

  percent_missing = df.isnull().sum() * 100 / len(df)
  missing_values = pd.DataFrame({'NaNs percentage': percent_missing})
  missing_values.sort_values(by ='NaNs percentage' , ascending=False)
  #         NaNs percentage
  # Date    0.0
  # X2      0.0
  # X3      0.0
  # X4      0.0
  # Y       0.0

  # # Categorical Data
  # df['X4'].fillna(df['X4'].value_counts().idxmax(), inplace=True)

  # percent_missing = df.isnull().sum() * 100 / len(df)
  # missing_values = pd.DataFrame({'NaNs percentage': percent_missing})
  # missing_values.sort_values(by ='NaNs percentage' , ascending=False)

  #Filling all missing values with 0
  df = df.fillna(0)

  percent_missing = df.isnull().sum() * 100 / len(df)
  missing_values = pd.DataFrame({'NaNs percentage': percent_missing})
  missing_values.sort_values(by ='NaNs percentage' , ascending=False)
  #         NaNs percentage
  # Date    0.0
  # X2      0.0
  # X3      0.0
  # X4      0.0
  # Y       0.0
  ```

## Dealing to Dates

+ Date objects: difficult to deal w/ for ML models because of their format

+ Dealing w/ Dates

```python
#Extracting Year
df['year'] = df['Date'].dt.year

#Extracting Month
df['month'] = df['Date'].dt.month

#Extracting the weekday name of the date
df['day'] = df['Date'].dt.day #_name()

df.head()
#         Date        X2        X3  X4      Y   year  month   day
# 0 2016-01-01  2.948228  2.348205  Car   True  2016      1     1
# 1 2016-01-02  2.834804  2.128829  Car   True  2016      1     2
# 2 2016-01-03  2.957205  2.133075  Car   True  2016      1     3
# 3 2016-01-04  3.210623  2.520078  Car   True  2016      1     4
# 4 2016-01-05  3.122827  2.130925  Car   True  2016      1     5
```

## Outliers

+ Outlier
  + a small fraction of data points which are quite distance from the rest of observations in a feature
  + introduced in a dataset mainly because of errors when collecting the data or because of special anomalies which are characteristic of of specific feature

+ Main techniques used to identify outliers
  + Data visualization: determining outlier values by visually inspecting the data distribution
  + Z-score: used if features w/ Gaussian distribution
  + Percentiles: assume that a certain top and bottom percent of data are outlier
  + Capping: replace w/ the highest normal value in the column

+ Other advanced techniques: [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) and [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

+ Testing how many outliers in X2 w/ Z-score and Percentiles methods

  ```python
  print(len(df))
  UpperLimit = df['X2'].mean() + df['X2'].std() * 2
  LowerLimit = df['X2'].mean() - df['X2'].std() * 2
  df2 = df[(df['X2'] < UpperLimit) & (df['X2'] > LowerLimit)]
  print(len(df2))
  UpperLimit = df['X2'].quantile(0.95)
  LowerLimit = df['X2'].quantile(0.05)
  df3 = df[(df['X2'] < UpperLimit) & (df['X2'] > LowerLimit)]
  print(len(df3))
  # 8000
  # 7766
  # 7200
  ```




