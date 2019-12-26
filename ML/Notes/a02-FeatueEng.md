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
  #       Date        X1        X2        X3        X4      Y
  # 2496  2022-11-01  2.482055  3.270268  2.487769  Bus     True
  #  782  2018-02-21  NaN       3.139792  1.970446  Car     True
  # 3449  2025-06-11  NaN       3.176024  2.672777  Bus     True
  #  148  2016-05-28  2.350232  3.434115  2.216226  Car     True
  # 7505  2036-07-19  NaN       2.248421  NaN       Scooter False
  ```



