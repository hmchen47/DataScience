# Hands-on with Feature Selection Techniques

Author: Younes Charfaoui

Organization: Heartbeat

The set of articles in this series:

+ [An Introduction](https://tinyurl.com/y3pkasc6)
+ [Filter Methods](https://tinyurl.com/y58pmgl7)
+ [Wrapper Methods](https://tinyurl.com/y38b36nz)
+ [Embedded Methods](https://tinyurl.com/y5sj9og3)
+ [Hybrid Methods](https://tinyurl.com/y57f97ch)
+ [More Advanced Methods](https://tinyurl.com/y6f38zqs)


## 1. An Introduction

Part 1. The basics of feature selection

+ Feature selection
  + a.k.a. variable selection, attribute selection or subset selection
  + the process by which a data scientist selects automatically and manually a subset of relevant features to use in ML model building
  + selecting the best subset of attributes
    + most important
    + high contribution at the time of prediction making
  + a critical process in any ML pipeline
  + designed to remove irrelevant, redundant, and noisy features
  + preserving a small subset of features from the primary feature space
  + impact: model's performance
  + advantages
    + reducing computational complexity
    + improving model accuracy
    + increasing model interpretability

+ Why feature selection matter
  + not always true: the more data features you have, the better the resulting model is going to be
    + irrelevant features
    + redundant features
    + result: overfitting
  + reasons to select features
    + simple models easier to interpret much easier to understand the output of a model w/ less variables
    + shorter training time: reducing the number of variables $\to$ 
      + reducing the computation cost
      + speeding up model training
      + simpler model tend to have faster prediction times
    + enhanced generalization by reducing overfitting
      + many of the variables just noise w/ little prediction value
      + eliminating these irrelevant noisy features
      + substantially improving the generalization of ML models
    + variable redundancy:
      + redundancy: highly-correlated features providing the same information
      + removing the redundant features w/o losing any information

+ Feature selection vs. feature engineering
  + feature engineering:
    + creating new features
    + helping the ML model make more effective and accurate predictions
  + feature selection:
    + selecting features from the feature pool
    + helping ML models more efficiently make predictions on target variables

+ Feature selection vs. Dimensionality reduction
  + dimensionality reduction
    + tending to lump together w/ feature selection
    + using unsupervised algorithms to reduce the number of feature in a dataset
  + differences
    + feature selection: a process to select and exclude some features w/o modifying them at all
    + dimensionality reduction:
      + modifying or transforming features into a slower dimension
      + creating a whole new feature space that looks approximately like the first one, but smaller in terms of dimensions

+ Procedure of feature selection
  + steps
    + combination of a search technique for proposing a new feature subset
    + an evaluation measuring that scores how well is the different feature subsets
  + computational expensive
  + looking for the best combination of feature subsets from all the available features

+ Feature selection methods
  + filter methods
    + relying on the features' characteristics
    + well-suitable for a quick "screen and removal" of irrelevant features
  + wrapper methods
    + selection of a set of features = a search problem
    + using a predictive ML algorithm to select the best feature subset
    + training a new model on each feature subset $\to$ computationall y expensive
    + providing the best performing feature subset for a given ML algorithm
  + embedded methods
    + taking the interaction of features and models into consideration
    + performing feature selection as part of the model construction process
    + less computationally expensive


## 2. Filter Methods

Part 2. Basic, correlation, and statistical filter methods

### 2.1 Overview of Filter Methods

+ Filter Methods
  + definitions:
    + selecting features from a dataset independently for any ML algorithms
    + relying only on the characteristics of these variables
    + filtered out of the data before learning begins
  + advantages
    + used in any ML algorithm
    + computationally inexpensive
  + good for eliminating irrelevant, redundant, constant, duplicated, and correlated features
  + types
    + univariate
    + multivariate
  + the methods: proposed for univariate and multivariate fileter-based feature selection
    + basic feature methods
    + correlation filter methods
    + statistical & ranking filter methods

+ Univariate of filter methods
  + evaluating and ranking a single feature according to certain criteria
  + treating each feature individually and independently of the feature space
  + procedures
    + ranking features according to certain criteria
    + selecting the highest ranking features according to the those criteria
  + issue: not considering relation to other ones in the dataset

+ Multivariate of filter methods
  + evaluating the entire feature space
  + considering the relations to other ones in the dataset

### 2.1 Basic filter methods

+ Constant features
  + showing single values in all the observations in the dataset
  + no information
  + Python snippet

    ```python
    # import and create the VarianceThreshold object
    from sklearn.feature_selection import VarianceThreshold

    vs_constant = VarianceThreshold(threshold=0)

    # select the numerical columns only
    numerical_x_train_df = x_train_df[x_train_df.select_dtypes([np.number]).columns]

    # fit the object to out data
    vs_constant.fit(numerical_x_train_df)

    # get the constant column names
    constant_columns = [column for column in numerical_x_train_df.columns
      if column not in numerical_x_train_df.columns[vs_constant.get_support()]]

    # detect constant categorical variables
    constant_cat_columns = [column for column in x_train_df.columns
      if (x_train_df[column].dtype == "0" and len(x_train_df[column].unique()) == 1 )]

    # concatenating the two lists
    all_constant_columns = constant_cat_columns + constant_columns

    # drop the constant columns
    x_train_df.drop(labels=all_constant_columns, axis=1, inplace=True)
    x_test_df.drop(labels=all_constant_columns, axis=1, inplace=True)
    ```

+ Quasi-Constant feature
  + a value occupying the majority of the records
  + Python snippet

    ```python
    # make a threshold for quasi constant
    thresshold = 0.98

    # create empty list
    quasi_constant_feature = []

    # loop over all the columns
    for feature in x_train_df.columns:
      # calculate the ratio
      predomiant = (x_train_df[feature].value_counts() / \
        np.float(len(x_train_df))).sort_values(ascending=False).value

      # append the column name if it is bigger than the threshold
      if predomiant >= threshold:
        quasi_constant_feature.append(feature)

    print("Features= {}".format(quasi_constant_feature))

    # drop the quasi constant columns
    x_train_df.drop(labels=duplicated_columns, axis=1, inplace=True)
    x_test_df.drop(labels=duplicated_columns, axis=1, inplace=True)
    ```

+ Duplicated features: Python snippet

  ```python
  #transpose the feature matrice
  train_features_T = x_train_df.T

  # print the number of duplicated features
  print(train_features_T.duplicated().sum())

  # select the duplicated features columns names
  duplicated_columns - train_features_T[train_features_T.duplicated()].index.values

  # drop those columns
  x_train_df.drop(labels=duplicated_columns, axis=1, inplace=True)
  x_test_df.drop(labels=duplicated_columns, axis=1, inplace=True
  ```)

## 2.2 Correlation Filter Methods

+ Correlation
  + correlation:
    + a measure of the linear relationship btw two quantitaive variables
    + a measure of how strongly one variable depndending on another
  + high correlation w/ target
    + a useful property to predict one from another
    + goal: highly correlated w/ the target, especially for linear ML models
  + high correlation btw variables: 
    + providing redundant information in regards to the target
    + making an accurate prediction on the target w/ just one of the redundant variables
    + not adding additional information
    + removing redundant ones to reduce the dimensionality but add noise
  + methods to measure the correlation btw variables
    + Pearson correlation coefficient
    + Spearman's rank correlation coefficient
    + Kendall's rank correlation coefficient

+ Pearson's correlation coefficient
  + popular measure used in ML
  + used to summarize the strength of the linear relationship btw two data variables
  + $r_{xy} \in [-1, 1]$
    + $r_{xy} = 1$: the values of one variable increase as the values of another increases
    + $r_{xy} = -1$: the values of one variable decrease as the values of another decreases
    + $r_{xy} = 0$: no linear correlation btw them, independent
  + assumptions
    + both normally distributed
    + straight-line relationship btw the two variables
    + equally distributed around the regression line
  + Pearson correlation coefficient

    \[ r_{xy} = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^n (x_i - \overline{x})^2} \sqrt{\sum_{i=1}^n (y_i - \overline{y})^2}} \]






