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

[Filter Methods Jupyter Notebook](src/a09-2.FilterMethods.ipynb)


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
    + selecting the highest ranking features according to those criteria
  + issue: not considering relation to other ones in the dataset

+ Multivariate of filter methods
  + evaluating the entire feature space
  + considering the relations to other ones in the dataset

### 2.2 Basic filter methods

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

    # fit the object to the data
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
      predominant = (x_train_df[feature].value_counts() / \
        np.float(len(x_train_df))).sort_values(ascending=False).value

      # append the column name if it is bigger than the threshold
      if predominant >= threshold:
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
  duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

  # drop those columns
  x_train_df.drop(labels=duplicated_columns, axis=1, inplace=True)
  x_test_df.drop(labels=duplicated_columns, axis=1, inplace=True)
  ```

### 2.3 Correlation Filter Methods

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
    + $r_{xy} = 0$: no linear correlation btw them
  + assumptions
    + both normally distributed
    + straight-line relationship btw the two variables
    + equally distributed around the regression line
  + Pearson correlation coefficient

    \[ r_{xy} = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^n (x_i - \overline{x})^2} \sqrt{\sum_{i=1}^n (y_i - \overline{y})^2}} \]

+ Spearman's rank coefficient coefficient
  + variables w/ a nonlinear relationship
  + stronger or weaker across the distribution of the variables
  + a non-parametric test
  + used to measure the degree of association btw two variables w/ a monotonic function
  + $\rho \in [-1, 1]$
  + Pearson correlation assessing linear relationships while Spearman's correlation  assessing monotonic relationship (whether linear or not)
  + suitable for both continuous and discrete ordinal variables
  + not carrying any assumptions about the distribution of the data
  + Spearman's rank correlation coefficient

    \[ \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} \]

+ Kendall's rank correlation
  + a non-parametric test that measures the strength of the ordinal association btw two variables
  + calculating a normalized score for the number of machine or concordant rankings btw the two data samples
  + $\tau \in [-1, 1]$
    + $\tau = 1$ (high): observations w/ a similar rank btw two variables
    + $\tau = -1$ (low): observation w/ a dissimilar rank btw two variables
  + best suitable for discrete data
  + Kendall's rank correlation

    \[ \tau = \frac{2(n_c - n_d)}{n(n-1)} \]

+ Python snippet for correlation coefficients

  ```python
  # creating set to hold the correlated features
  corr_feature = set()

  # create the correlation matrix (default to pearson)
  corr_matrix = x_train_df.corr()

  # optional: display a heatmap of the correlaton matrix
  plt.figure(figsize=(11,11))
  sns.heatmap(corr_matrix)

  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i, j]) > 0.8:
        colname = corr_matrix.column[i]
        corr_feature.add(columns[i])

  x_train_df.drop(labels=corr_features, axis=1, inplace=True)
  x_test_df.drop(labels=corr_features, axis=1, inplace=True)
  ```

  + `dataframe.corr()` function
    + syntax: ` DataFrame.corr(self, method=’pearson’, min_periods=1)`
    + Parameters:
      + `method` :
        + `pearson`: standard correlation coefficient
        + `kendall`: Kendall Tau correlation coefficient
        + `spearman`: Spearman rank correlation
      + `min_periods` : Minimum number of observations required per pair of columns to have a valid result. Currently only available for pearson and spearman correlation
    + Returns: count :y : DataFrame

  + `DataFrame.duplicated()` function
    + Return boolean Series denoting duplicate rows.
    + Syntax: `DataFrame.duplicated(subset=None, keep='first')`

### 2.4 Statistical & Ranking Filter Methods

+ Statistical & Ranking filter
  + evaluating each feature individually
  + evaluating whether the variable to discriminate against the target
  + procedure
    + ranking the feature base on certain criteria or metrics
    + selecting the features w/ the highest rankings
  + methods
    + mutual information
    + Chi-square score
    + ANOVA univariate test
    + Univariate ROC_AUC / RMSE

+ Mutual information
  + a measure of the mutual dependence of two variables
  + measuring the amount of info obtained about one variable through observing the other variable
  + determining how much knowing one variable by understanding another
  + a little bit like correlation, but more general
  + measuring how much info the presence/absence of a feature contributes to making the correct prediction on $Y$
  + MI values
    + $MI = 0$: X and Y independent
    + $MI = E(X)$: $X$ is deterministic of $Y$, $E(X)$ as entropy of $X$
  + Entropy: measuring or quantifying the amount of info within a variable

    \[ I(X; Y) = \sum_{x, y} P_{XY} (x, y) \log \frac{P_{XY}(x, y)}{P_X(x)P_Y(y)} \]

  + Python snippet

    ```python
    # import the required functions and object
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest

    # select the number of features to retain
    select_k = 10

    # get only the numerical features
    numerical_x_train_df = x_train_df[x_train_df.select_dtypes([np.number]).columns]

    # create the SelectKBest
    selection = SelectKBest(mutual_info_classif, k=select_k).fit(numerical_x_train_df, y_train_df)

    # display the retained features
    features = x_train_columns[selection.get_support()]
    print(features)
    ```

+ Chi-squared score
  + commonly used for testing relationships btw categorical variables
  + suitable for categorical variables and binary targets only
  + constraint: non-negative variables and typically boolean, frequencies, or counts
  + simply comparing the observed distribution btw various features in the dataset and the target variable
  + Python snippet

    ```python
    # import the required functions and object
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest

    # change this to how much features to keep from the top ones
    select_k = 10

    # apply the chi2 score on the data and target (target should be binary)
    selection = SelectKBest(chi2, k=select_k).fit(x_train_df, y_train_df)

    # deiplay the k selected features
    features = x_train_df.columns[selection.get_support()]
    ```

+ ANOVA univariate test
  + ANOVA = Analysis Of VAriance
  + similar to chi-squared score
  + measuring the dependence of two variables
  + assumptions
    + linear relationship btw variables and the target
    + both normally distributed
  + suitable for continuous variables and requiring a binary target
  + Python snippet

    ```python
    # import the required functions and object
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import SelectKBest

    # select the number of features to retain
    select_k = 10

    # create the SelectKBest w/ the mutual info strategy
    selection = SelectKBest(f_classif, k=select_k).fit(x_train_df, y_train_df)

    # display the retained features
    features = x_train_df.columns[selection.get_support()]
    print(features)
    ```

+ Univariate ROC-AUC / RMSE
  + using ML models to measure the dependence of two variables
  + suitable for all variables
  + no assumptions about the distribution
  + measuring scenarios
    + regression problem: RMSE
    + classification problem: ROC-AUC
  + procedure
    + build a decision tree using a single variable and target
    + rank features according to the model RMSE or ROC-AUC
    + select the features w/ higher ranking scores
  + Python snippet

    ```python
    # import the DecisionTree Algorithm and evaluation score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score

    # list of the resulting scores
    roc_values = []

    # loop over all features and calculate the score
    for feature in x_train_df.columns:
      clf = DecisionTreeClassifier()
      clf.fit(x_train_df[feature].to_frame(), y_train_df)
      y_scored = clf.predict_proba(x_test_df[feature].to_frame())
      roc_values.append(roc_auc_score(y_test_df, y_scored[:, 1]))

    # create a Pandas Series for visualization
    roc_values = pd.Series(roc_values)
    roc_values.index = x_train_df.columns

    # show the results
    print(roc_values.sort_values(ascending=False))
    ```



## 3. Wrapper Methods

Part 3: [Forward feature selection, backward feature elimination, exhaustive feature selection, and bidirectional search](https://tinyurl.com/y38b36nz)

[Wrapper Methods Jupyter Notebook](src/a09-3.WrapperMethods.ipynb)

### 3.1 Wrapper Methods

+ Wrapper methods
  + disadvantages of filter methods
    + tend to ignore the effect of the selected feature subset on the performance of the algorithm
    + evaluate features individually $\to$ some variables useless for prediction in isolation, but quite useful when combined w/ other variables
  + Definition
    + evaluating a subset of features using aML algorithm algorithm that employs a search strategy to look through the sopace of possible feature subsets
    + evaluating each subset based on the quality of the performance of a given algorithm
  + greedy algorithms
    + aiming to find the best possible combination of features that result in the best performance model
    + computationally expensive
    + impractical in the case of exhaustive search
  + advantages
    + detecting the interaction btw variables
    + finding the optimal feature subset for the desired ML algorithm
  + usually result in better predictive accuracy than filter methods
  + process
    1. search for a subset of features: using a search method to select a subset of features from the availabe ones
    2. build a ML model: choosing ML algorithm trained on the previously-selected subset of features
    3. evaluate model performance: evaluating the newly-trained ML models w/ a chosen metric
    4. repeat: starting against w/ a new subset of features, a new ML model trained, and so on
  + stopping criteria
    + defined by ML learning engineer
    + examples of criteria
      + model performance increases
      + model performance decreases
      + predefined number of features reached
  + search methods
    + forward feature selection: starting w/ no features and adding one at a time
    + backward feature selection: starting w/ all features present and removing one feature at a time
    + exhaustive feature selection: trying all possible feature combinations
    + bidirectional search: both forward and backward feature selection simultaneously to get one unique solution
  + library to install: `mlxtend` containing useful tools for a number of day-to-day data science tasks

### 3.2 Forward Feature Selection

+ Forward feature selection
  + a.k.a. step forward selection, sequential forward feature selection (SFS)
  + an interactive method
  + procedure
    + start by evaluating all feature individually and then select the one that results in the best performance
    + test all possible combinations of the selected feature w/ the remaining features and retain the pair that procedures the best algorithmic performance
    + a loop continues by adding one feature at a time in each iteration until the pre-set criterion reached
  + Python snippet

    ```python
    # import the algorithm to evaluate features
    from sklearn.ensemble import RandomnForestClassifier

    # create the SequentialFeatureSelector object, and configure the parameters
    sfs = SequentialFeatureSelector(RandomForestClassifier(), k_feature=10,
      forward=True, forward=True, floating=Flase, scoring='accuracy', cv=2)

    # fit the object to the training data
    sfs = sfs.fit(x_train, y_train)

    # print the selected feature
    selected_features = x_train.columns[list(sfs.k_feature_idx_)]
    print(selected_features)

    # print the final prediction score
    print(sfs_k_score_)

    # transform to the nrely selected features
    x_train_sfs = sfs.transform(x_train_df)
    x_test_sfs = sfs.transform(x_test_df)
    ```

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `k_features`: the maximum feature to be reached when starting from 0
  + `floating`: using a variant of the step forward selection called step floating forward selection
  + `scorign`: the scoring function to evaluate model performance
  + `cv`: the number of folds of K-fold cross-validation, no cross-validation if cv=0 or False

### 3.3 Backward Feature Elimination

+ Backward feature elimination method
  + a.k.a. step backward selection, sequential backward feature selection  (SBS)
  + starting w/ all the features in the dataset
  + evaluating the performance of the algorithm
  + removing one feature at a time
  + producing the best performing algorithm using an evaluation metric
  + the least significant feature among the remaining available ones
  + removing feature after feature until a certain criterion satisfied
  + Python snippet

    ```python
    # import the algorithm to evaluate features
    frpm sklearn.ensemble import RandomForestClassifier

    # just set forward=False for backward feature selection
    # create the SequentialFeatureSelector object, and configure the parameter
    sbs = SequentialFeatureSelector(RandomForestClassifier(),
      k_features=10, forward=False, floating=False, scoring='accuracy', cv=2)

    # fit the object to the training dataset
    sbs = sbs.fit(x_train_df, y_train)

    # print the selected features
    selected_features = x_train_df.columns[list(sbs.k_feature_idx)]
    print(selected_features)

    #print the final prediction score
    print(sbs.k_score_)

    # transform tot he newly selected features
    x_train_sfs_df = sbs.transform(x_train_df)
    x_test_sfs_df = sbs.transform(x_test_df)
    ```

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `k_features`: the maximum feature to be reached when starting from N
  + `forward`: using the object for step forward or step backward feature selection
  + `floating`: using a variant of the step backward selection called step floating backward selection


### 3.4 Exhaustive Feature Selection

+ Exhaustive feature selection
  + finding the best performing feature subset
  + a brute-force evaluation of feature subsets
  + creating all  the subsets of features from 1 to N
  + building a ML algorithm for each subset and selecting the subset w/ the best performance
  + parameter 1 and N: the minimum number of features and the maximum number of features
  + Python snippet

    ```python
    from mlxtend.feature_selection import ExhaustiveFeatureSelector

    #import the algorithm to evaluate the features
    from sklearn.ensemble import RandomForesetClassifier

    # create the EchaustiveFeatureSelector object
    efs = EchaustiveFeatureSelector(RandomForestClassifier(),
      min_features=4, max_features=10, scoring=roc_auc', cv=2)

    # print the selected features
    selected_features = x_train_df.columns[list(efs.k_feature_idx_)]
    print(selected_features)

    # print the final prediction score
    transform data to the newly selected features
    x_train_sfs = efs.transform(x_train_df)
    x_test_sfs = efs.transform(y_test_df)
    ```

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `min_features`: the lower bound of the number of features to search from
  + `max_features`: the upper bound of the number of features to search from

### 3.5 Limitations and Solutions of Step Forward/Backward Selection

+ limitations of SFS & SBS
  + SFS adding features at each iteration
    + adding up a feature useful in the beginning but non-useful after adding more ones
    + unable to remove the feature
  + SBS removing features at each iteration
    + removing a feature useless in the beginning but useful after removing more ones
    + unable to add the feature in the feature subsets
  + solutions: LRS or sequential floating
    + LRS, or Plus-L, Minus-R: using two parameters L and R (both integer)
    + repeatedly adding and removing features from the solution subset
    + $L > R$: LRS starting from the empty set of features
      + repeatedly adding L features
      + repeatedly removing R features
    + $L < R$: LRS starting from the full set of features
      + repeatedly removing R features
      + repeatedly adding L features
    + compensating for the weaknesses of SFS and SBS w/ some backtracking capabilities
    + constrain: carefully set L and R parameters w/o well-know theory to choose the optimal values



