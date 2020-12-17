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


## 0. Summary

+ Filter Methods
  + Basic feature
    + constant feature
    + quasi-constant feature
    + duplicated features
  + Correlation filter
    + Pearson's correlation coefficient $\left( r_{xy} = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^n (x_i - \overline{x})^2} \sqrt{\sum_{i=1}^n (y_i - \overline{y})^2}} \right)$
    + Spearman's rank correlation coefficient $\left( \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} \right)$
    + Kendall's rank correlation coefficient $\left( \tau = \frac{2(n_c - n_d)}{n(n-1)} \right)$
  + Statistical and Ranking filter
    + Mutual information $\left( \text{Entropy: }I(X; Y) = \sum_{x, y} P_{XY} (x, y) \log \frac{P_{XY}(x, y)}{P_X(x)P_Y(y)} \right)$
    + Chi-squared score 
    + ANOVA univariate test
    + Univariate ROC-AUC / RMSE

+ Wrapper Methods
  + Forward feature addition, sequential forward feature selection (SFS)
  + Backward feature elimination, sequential backward feature selection (SBS)
  + Exhaustive feature selection
  + LRS or Plus-L, Minus-R
  + Sequential floating
    + step floating forward selection (SFFS)
    + step floating backward selection (SFBS)
  + Bidirectional search (BDS)

+ Embedded methods
  + Regularization
  + Tree-based feature importance
    + Feature importance
    + Random forests
  + Permutation importance

+ Hybrid Methods
  + Filter & wrapper
  + Embedded & wrapper
    + Recursive feature elimination
    + Recursive feature addition

+ Advanced methods
  + Dimensionality reduction
    + Principle component analysis (PCA)
    + Linear discriminant analysis (LDA)
  + Heuristic search algorithms
    + Genetic algorithm (GA)
    + Simulation annealing
  + Deep learning
    + Autoencoder for feature selection (AEFS)



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
  + the methods: proposed for univariate and multivariate filter-based feature selection
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
    threshold = 0.98

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
    + a measure of the linear relationship btw two quantitative variables
    + a measure of how strongly one variable depending on another
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
    + syntax: `DataFrame.corr(self, method=’pearson’, min_periods=1)`
    + Parameters:
      + `method` :
        + `pearson`: standard correlation coefficient
        + `kendall`: Kendall Tau correlation coefficient
        + `spearman`: Spearman rank correlation
      + `min_periods` : Minimum number of observations required per pair of columns to have a valid result. Currently only available for Pearson and Spearman correlation
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

### 3.1 Overview of Wrapper Methods

+ Wrapper methods
  + disadvantages of filter methods
    + tend to ignore the effect of the selected feature subset on the performance of the algorithm
    + evaluate features individually $\to$ some variables useless for prediction in isolation, but quite useful when combined w/ other variables
  + Definition
    + evaluating a subset of features using a ML algorithm that employs a search strategy to look through the space of possible feature subsets
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
    1. search for a subset of features: using a search method to select a subset of features from the available ones
    2. build a ML model: choosing ML algorithm trained on the previously-selected subset of features
    3. evaluate model performance: evaluating the newly-trained ML models w/ a chosen metric
    4. repeat: starting against w/ a new subset of features, a new ML model trained, and so on
  + stopping criteria
    + defined by ML engineer
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
    sfs = sfs.fit(x_train_df, y_train_df)

    # print the selected feature
    selected_features = x_train_df.columns[list(sfs.k_feature_idx_)]
    print(selected_features)

    # print the final prediction score
    print(sfs.k_score_)

    # transform to the nrely selected features
    x_train_sfs = sfs.transform(x_train_df)
    x_test_sfs = sfs.transform(x_test_df)
    ```

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `k_features`: the maximum feature to be reached when starting from 0
  + `floating`: using a variant of the step forward selection called step floating forward selection
  + `scoring`: the scoring function to evaluate model performance
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
    from sklearn.ensemble import RandomForestClassifier

    # just set forward=False for backward feature selection
    # create the SequentialFeatureSelector object, and configure the parameter
    sbs = SequentialFeatureSelector(RandomForestClassifier(),
      k_features=10, forward=False, floating=False, scoring='accuracy', cv=2)

    # fit the object to the training dataset
    sbs = sbs.fit(x_train_df, y_train_df)

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
      min_features=4, max_features=10, scoring='roc_auc', cv=2)

    # print the selected features
    selected_features = x_train_df.columns[list(efs.k_feature_idx_)]
    print(selected_features)

    # print the final prediction score
    print(efs.k_score_)

    # transform data to the newly selected features
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

+ LRS, or Plus-L, Minus-R:
  + using two parameters L and R (both integer)
  + repeatedly adding and removing features from the solution subset
  + $L > R$: LRS starting from the empty set of features
    + repeatedly adding L features
    + repeatedly removing R features
  + $L < R$: LRS starting from the full set of features
    + repeatedly removing R features
    + repeatedly adding L features
  + compensating for the weaknesses of SFS and SBS w/ some backtracking capabilities
  + constrain: carefully set L and R parameters w/o well-known theory to choose the optimal values

+ Sequential floating
  + an extension of LRS
  + determining values from the data directly than by adding and removing features for L and R
  + the size of subset floating up and down by adding and removing features
  + context of the floating methods
    + step floating forward selection (SFFS): performing backward steps as long as the objective function increases
    + step floating backward selection (SFBS): performing forward steps as long as the objective function increases
  + algorithm for SFFS
    1. start from an empty set
    2. select the best feature and adding in the feature subset as SFS
    3. select the worse feature from the subset
    4. evaluate and check whether the objective function whether improving or not by deleting the feature.
      + deleting the feature if improving
      + keep the feature if not improving
    5. repeat from step 2 until stop criteria reached
  + algorithm for SBFS
    + start w/ a full one
    + process w/ normal SBS
    + add feature to improve the object function


### 3.6 Other Search Methods

+ Bidirectional Search (BDS)
  + applying SFS and SBS concurrently
  + SFS: performing from the empty set of features
  + SBS: performing from the full set of features
  + issue: converge to different solutions 
  + resolving by the constrains
    + features already selected by SFS not removing by SBS
    + features already removed by SBS not added by SFS


## 4. Embedded Methods

Part 4: [Regularization and tree-based embedded methods](https://tinyurl.com/y5sj9og3)

[Embedded Methods Jupyter Notebook](src/a09-4.EmbeddedMethods.ipynb)

### 4.1 Overview of Embedded Method

+ Embedded methods
  + characteristics of wrapper methods:
    + a good way to ensure the selected features are best for a specific ML model
    + providing better results in terms of performance
    + costing a lot of computation time/resources
  + definition
    + including the feature selection process in ML model training itself
    + result in even better features of the mode in a short amount of time
    + complete the feature selection process within the construction of the ML algorithm itself
    + performing feature selection during the model training
  + advantages
    + solving both model training and feature selection simultaneously
    + taking into consideration the interaction of feature  like wrapper methods do
    + faster like filter methods
    + more accurate than filter methods
    + finding the feature subset for the algorithm being trained
    + much less prone to overfitting
  + process
    + train a ML model
    + derive feature importance from this model, a measure of how much is feature important when making a prediction
    + remove non-important features using the derived feature importance
  + method: regularization & tree-based methods


### 4.2 Regularization Methods

+ Regularization
  + adding a penalty to the different parameters of a model to reduce its freedom
  + penalty applied to the coefficient that multiplies each of the features in the linear model
  + advantages
    + avoid overfitting
    + make the model robust to model
    + improve its generalization
  + main types of regularization for linear models
    + Lasso regression / L1 regularization
      + shrinking some of the coefficients to zero
      + indicating a certain predictor or certain features multiplied by zero to estimate the target
      + not added to the final prediction of the target
      + i.e., features removed due to not contributing to the final prediction
    + ridge regression / L2 regularization
      + not setting the coefficient to zero, but approaching to zero
    + elastic nets / L1/L2 regularization
      + a combination of the L1 and L2
      + incorporating their penalties and ending up w/ features w/ zero as a coefficient
  + Python snippet

    ```python
    # Lasso for regression tasks, and Logistic Regression for Classification tasks
    from sklearn.learn_model import Lasso, LogisticRegression
    from sklearn.feature_selection import SelectFromModel

    # using logistic regression w/ penalty L1
    selection = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
    selection.fit(x_train_df, y_train_df)

    # see the selected features
    selected_features = x_train_df.columns[(selection.get_support())]

    # see the deleted features
    removed_features = x_train_df.columns[(selection.estiamtor_.coef_==0).ravel().tolist()]
    ```

### 4.3 Tree-based Feature Importance

+ Tree-based feature importance
  + tree-based algorithms and models
    + well-established algorithms
    + offering good predictive performance
    + able to provide the feature importance as a way to select features
  + feature importance
    + indicating which variables more important in making accurate predictions on the target variable/class
    + identifying which features mostly used by the ML algorithm to predict the target
  + random forests:
    + providing feature importance w/ straightforward methods
      + mean decrease impurity
      + mean decrease accuracy
    + a group of decision trees
    + each decision tree established over a random extraction of samples and features from the dataset
    + individual unable to see all the features or access all the observations
    + the importance of each feature derived by how "pure" each of the sets is
    + impurity: measure based on the optimal condition chosen
      + classification: typically either the Gini impurity or information gain/entropy
      + regression: variance
  + training a tree
    + feature importance calculated as the decrease in node impurity weighted in a tree
    + the higher the value, the more important the feature
  + Python snippet

    ```python
    from sklearn.ensemble import RandomForestClassifier

    # create the random forest w/ hyperparameters
    model = RandomForestClassifier(n_estimators=340)

    # get the importance of the resulting features
    importances = model.feature_importances_

    # create a dataframe for visualization
    findal_df = pd.DataFrame({"Features": x_train_df.columns, "Importances": importances})
    final_df.set_index("Importances")

    # sort in ascending order to better visualization
    final_df = final_df.sort_values('Importances')

    # plot the feature importance in bars
    final_df.plot.bar()
    ```

  + alternatives: 
    + able to use any other tree-based algorithm the same way =
    + best tree model types: gradient boosting algorithms (like XH=GBoost, CatBoost, and any more)
    + providing accurate feature importance


## 5. Hybrid Methods

Part 5: [Combining filter, wrapper, and embedded feature selection methods](https://tinyurl.com/y57f97ch)

[Hybrid Method Jupyter Methods](src/a09-5.HybridMethods.ipynb)


### 5.1 Overview of Hybrid Methods

+ Hybrid methods
  + definitions
    + combining the difference approaches to get the best possible feature subset
    + combinations of approaches up to engineer
    + taking the best advantages from other feature selection methods
  + advantages
    + high performance and accuracy
    + better computational complexity than wrapper methods
    + more flexible and robust models against high dimensional data
  + process
    + depending on what engineers choose to combine
    + main priority: select the methods to use, then follow their processes


### 5.2 Filter & Wrapper Methods

+ Hybrid method: Filter + Wrapper methods
  + filter method
    + ranking methods, including mutual information and Chi-square score
    + order features independently involving any learning algorithm
    + the best features selected from the ranking list
  + procedure
    + using ranking methods to generate a feature list
    + using the top k features from this list to perform wrapper method (SFS or SBS)
  + advantages
    + reducing the feature space of datset using these filter-based rangers
    + improving the time complexity of the wrapper methods

### 5.3 Embedded & Wrapper Methods

+ Hybrid method: Embedded & Wrapper methods
  + embedded methods
    + establishing feature importance
    + used to select top features
  + procedure:
    + embedded methods to get top k features
    + performing a wrapper methods search
  + methods:
    + recursive feature elimination
    + recursive featire addition

### 5.4 Recursive feature elimination

+ Recursive feature elimination
  + procedure
    1. train a model on all the data features
      + possible candidate: tree-based model, lasso, logistic regression, or others offering feature importance
      + evaluating its performance on a suitable metric
    2. derive the feature importance to rank features accordingly
    3. delete the least important feature and re-train the model on the remaining ones
    4. use the previous evaluation metric to calculate the performance of the resulting model
    5. test whether the evaluation metric decreases by an arbitrary threshold to remain or remove
    6. repeat step 3~5 until all features removed
  + difference w/ SBS
    + SBS: eliminate all the features to determine which one is the least important
    + recursive feature elimination:
      + getting this info from the ML models derived importance
      + removing the feature only once rather than removing all the features at each step
  + faster than pure wrapper methods and better than pure embedded methods
  + limitations:
    + use an arbitrary threshold value to decided whether to keep a feature or not
    + the smaller this threshold value, the more features will be included in the subset and vice versa
  + Python snippet to select the best features

    ```python
    from sklearn.feature_selection import RFECV

    # use any other model selected
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estiamtors=411)

    # build the REF w/ cv option
    ref = REFCV(model, min_features_to_selecct=3, step=1, cv=5, scoring='accuracy')

    # fit the RFE to our data
    selecton = rfe.fit(x_train_df, y_train_df)

    # print the selected features
    print(x_train_df.columns[selection.support_])
    ```

  + `RFECV`: recursive feature elimination w/ corss-validation
    + `min_features_to_select`: int (default=1) <br/>The minimum number of features to be selected. This number of features will always be scored, even if the difference between the original feature count and `min_features_to_select` isn’t divisible by `step`.
    + `step`: int or float, optional (default=1)
      + $\ge 0$: correspond to the (integer) number of features to remove at each iteration
      + $\in (0.0, 1.0)$: correspond to the percentage (rounded down) of features to remove at each iteration
    + `cv`: int, cross-validation generator or an iterable, optional<br/>Determines the cross-validation splitting strategy.
      + `None`: using the default 5-fold cross-validation
      + `integer`: specifying the number of folds
      + CV splitter
        + `cross-validation generator`: a non-estimator family of classes used to split a dataset into a sequence of train and test portions , by providing split and get_n_splits methods
        + `cross-validation estimator`: an estimator that has built-in cross-validation capabilities to automatically select the best hyper-parameters
        + `scorer`: a non-estimator callable object which evaluates an estimator on given test data, returning a number
      + an iterable yielding (train, test) splits as arrays of indices
    + `scoring`: string, callable or None, optional, (default=None)<br/>A string or a scorer callable object / function with signature `scorer(estimator, X, y)`
  + Python snippet for recursive feature elimination

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # array to hold the feature to be removed
    model_all_features = RandomForestClassifier(n_estimators=221)
    model_all_features.fit(x_train_df, t_train_df)

    # get the 1st score of all the features (able to use own metric)
    y_pred_test_df = model_all_features.predict(x_test_df)
    auc_score_all = roc_auc_score(y_test_df, y_pred_test_df)

    # loop over all the feature to do recursive feature elimination
    for feature in x_train_df.columns:
      model = RandomForestClassifier(n_estiamntors=221)

      # delete the current feature
      x_train_rfe_df = x_train_df.drop(features_to_remove + [feature], axis=1)
      x_test_rfe_df = x_test_df.drop(features_to_remove + [feature], axis=1)

      # fit model w/ all variables minus the removed features and the feature to be evaluated
      mode.fit(x_train_rfe_df, y_train_df)
      y_pred_test_df = model.predict(x_test_rfe_df)
      auc_score_int = roc_auc_score(y_test_df, y_pred_test_df)

      # determine the drop in the roc_auc
      diff_auc = auc_score_all - auc_score_int

      # compare the drop in the roc-auc w/ the threshold
      if diff_auc < threshold:
        # feature, require to set the new roc to the
        # one based on the remaining feature
        auc_score_all = auc_score_int

        # and append the feature to remove to this list
        feature_to_remove.append(feature)

    # print the features that need removing
    print(features_to_remove)
    features_to_keep = [x for x in_train_df.columns \
      if x not in features_to_remove]

    # print the features to keep
    print('total features to keep: ', len(features_to_keep))
    ```

### 5.5 Recursive Feature Addition

+ Recursive feature additions
  + starting w/ no features and adding one feature at the time
  + procedure
    1. train a model on all the data and derive the feature importance to rank it accordingly
      + possible models: tree-based model, Lasso, logistic regression, or others offering feature importance
    2. from the initial model, create another w/ the most important feature and evaluate it w/ an evaluation metric of your choice
    3. add another important feature and use it to re-train the model, along w/ any feature from the previous step
    4. use the previous evaluation metric to calculate the performance of the resulting model
    5. test whether the evaluation metric increases by an arbitrary-set threshold
    6. repeat step 3-5 until features added
  + Python snippet

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # array to hold the feature to be kept
    features_to_keep = [x_train_df.columns[0]]

    # set this value accordingly
    threshold = 0.002

    # create preferred model and fit it to the training data
    model_one_feature_df = RandomForestClassifier(n_estimators=332)
    model_one_feature_df.fit(x_train_df[[x_train_df.columns[0]]])

    # evaluate against metric selected
    y_pred_test_df = model_one_feature_df.predict(x_test_df[[x_train_df.columns[0]]])
    auc_roc_all = roc_auc_score(y_test_df, y_pred_test_df)

    # start iterating from the feature
    for feature in x_train_df.columns[1:]:
      model = RandomForestClassifier(n_estimators=332)

      # fit model w/ the selected features and the feature to be evaluated
      model.fit(x_train_df[features_to_keep + [featue]], y_train_df)
      y_pred_test_df = model.predict(x_test_df[features_to_keep + [feature]])
      auc_roc_int = roc_auc_score(y_test_df, y_pred_test_df)

      # determine the drop in the roc-auc
      diff_auc = auc_score_int - auc_score_all

      # compare the drop in roc-auc w/ the threshold
      if diff_auc >= threshold:
        # if the increases in the roc is bigger than the threshold
        # keep the feature and re-adjust the roc-auc to the new
        # value considering the added feature
        auc_score_all = auc_score_int
        featue_to_keep.append(feature)

    # print the feature to keep
    print(features_to_keep)
    ```


## 6. Advanced Methods

Part 6: [Dimensionality reduction, genetic algorithms, permutation importance, and deep learning approaches](https://tinyurl.com/y6f38zqs)


### 6.1 Dimensionality Reduction

+ Dimensionality reduction vs. feature selection
  + both tried to reduce the number of features
  + feature selection: select and exclude some features w/o making any transformation
  + dimensionality reduction: transform features into a lower dimension

+ Principal component analysis (PCA)
  + a dimensionality reduction technique
  + using linear algebra to transform a dataset into a compressed form
  + starting by calculating the Eigen decomposition (or singular value decomposition, SVD) of the covariance matrix of the features
  + procedure
    + searching the correlation btw features
    + building new features that preserve the same explained variance of the original ones
  + resulting in a lower-dimensional projection of the data, the __maximal data variance__
  + measuring the importance of a given variable
  + observing how much its contributing to the reduced feature space that PCA obtains
  + feature selection w/ PCA
    + calculating the explained variance of each feature
    + using it as feature importance to rank variable accordingly
  + Python snippet

    ```python
    from sklearn.decomposition import PCA

    # create th PCA onject w/ all the features
    pca = PCA(n_estimators=x_train_df.shape[1])

    # fit the object to our data
    fit = pac.fit(x_train_df)

    # print out the results
    print(fit.explained_variance_ratio)
    print(fit.components) 
    ```

  + alternative: linear discriminant analysis (LDA)
  

### 6.2 Heuristic Search Algorithms

+ Heuristic search
  + attempt to perform optimization tasks in an iterative fashion to find an approximate solution if classical models failed to find an exact solution
  + not always find the best or even the optimal solution, but find a good or acceptable solution within a reasonable amount of time and memory space
  + performing a heuristic search across feature subsets to find the best one

+ Genetic algorithms (GA)
  + global optimization techniques for searching very large spaces
  + sort of randomized search
  + inspired by the biological mechanisms of natural selection and reconstruction
  + working throughout populations of possible solutions (generations)
  + each solution in the search space represented as a finite length string (chromosome) over some finite set of symbols
  + using an objective (or fitness) function to evaluate the suitability of each solution
  + feature selection
    + each chromosome representing a feature subset
    + represented w/ binary encoding: feature w/ 1 to choose and 0 to eliminate from
  + conducting many iterations to create a new generation (new feature subset) of possible solutions from the current generations using a few oprtors
  + operators
    + __selection:__
      + probabilististically filtering out solutions that perform poorly
      + choosing high performing solutions to exploit
    + __cross over:__
      + the GA way to explore new solutions and exchange info btw strings
      + applied to selected pairs of chromosomes randomly
      + the probability equal to a given crossover rate
      + generating new chromosomes that hopefully will retain good features from the previous generations
    + __mutation:__
      + protecting GAs against the irrecoverable loss of good solution features
      + changing a symbol of some chromosomes
      + changing ratio as a probability equal to a very low given mutation rate to restore lost genetic material
  + advantages
    + working w/ a population of solutions
    + more effective to escape local minima
  + procedures
    1. initializing a population w/ randomly-generated individuals
      + different feature subsets
      + creating a machine learning algorithm
    2. evaluating the fitness of each feature subset w/ an evaluation metric of choice depending on the chosen algorithm
    3. reproducing high-fitness chromosomes (feature subsets) in the new population
    4. removing poor-fitness chromosomes (selection)
    5. constructing new chromosomes (crossover)
    6. recovering lost features (mutation)
    7. repeating step 2~6 until a stopping criterion met (or the number of iterations)
  + open-source implementation: `sklearn-genetic`
  + Python snippet

    ```python
    from genetic_selection import GeneticSelectionCV

    # import preferred ml model
    from sklearn.ensemble import RandomForesetClassifier

    # build the model w/ perferrred hyperparameters
    model = RandomForesetClassifier(n_estimators=231)

    # create the GeneticSelection search w/ the different parameters available
    selection = GeneticSelectionCV(model, cv=5, scoring='accuracy', 
      max_features=12, n_population=120, crossover_proba=0.5, 
      mutation_proba=0.2, n_generations=50, crossover_inndependent_proba=0.5,
      mutation_independent_proba=0.05, n_gen_no_chane=10, n_jobs=-1)

    # fit the GA search to data
    selection = selection.fit(x_train_df, y_train_df)

    # print result
    print(selection.support_)
    ```

  + parameters of GeneticSelectionCV
    + __estiamtor__: model used to evaluate the suitability of the feature subset, alongside an evaluation metric
    + __cv__: int, generator, or an iterable used to determine the cross-validation splitting strategy
    + __scoring__: the evaluation metric
      + the fitness function to evaluate the performance of a chromosome
      + ML model's performance against a subset of features
    + __max_features__: determine the maximum number of features selected
    + __n_population__: number of populations for the genetic algorithm, different feature subsets
    + __crossover_proba__: probability value of a crossover operation for the genetic algorithm
    + __mutation_proba__: probability value of a mutation operation for the genetic algorithm
    + __n_generation__: an integer describing the number of generations for the genetic algorithm - the stopping criterion
    + __crossover_independent_proba__:
      + the independent probability for each attribute to be exchanged
      + offering much more flexibility for the generic algorithm to search
    + __mutation_independent_proba__: the independent probability for each attribute to be mutated by the generic algorithm
    + __n_gen_no_change__: the number of generations needs to terminate the search if no change w/ the best individuals
  
+ Simulation annealing
  + a heuristic approach to search for the feature subsets
  + a global search algorithm allowing a suboptimal solution to be accepted in the hope that better solution will show up eventually


### 6.3 Feature Importance w/ Permutation Importance

+ Permutation Importance
  + basically randomly shuffle the values of a feature (w/o touching the other variables of the targets) to see how this permutation affects the performance metric of the ML model
  + the choice of metric upon the engineer
  + measuring the importance of a feature by measuring the increase in the model's prediction error after permutation the feature values
  + breaking the relationship btw the feature and the true outcome
    + important feature:
      + shuffling values increasing the model error
      + relying on the feature for the prediction
    + non-important feature:
      + shuffling values leaving the model error unchanged
      + model ignoring the feature for the prediction
  + Python snippet

    ```python
    # import whatever algorithm chooses
    from sklearn.emsenble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # build model and train it
    model = RandomForestClassifier(n_estimator=221)
    model.fit(x_train_df, y_train_df)

    # get the default score
    train_auc = roc_auc_score(y_train_df, model.predict(x_train_df))
    feature_dict_score = {}

    # loop over all the feature
    for feature in x_train_df.columns:
      # copy the dataset to do some permutation
      x_train_copy_df = x_train_df.copy().reset_index(drop=True)
      y_train_copy_df = y_train_df.copy().reset_index(drop=True)

      # shuffle an individual feature
      x_train_copy_df[feature] = x_train_copy_df[feature].sample(
        feac=1, random_sate=random_state
      ).reset_index(drop=true)

      # make a prediction w/ permuted feature and calculate roc auc
      shuffle_auc = roc_auc_score(y_train_copy_df, 
        model.predict(x_train_copy_df))

      # save the drop in dictionary
      feature_dict_scores[feature] = (train_auc - shuffle_aux)

    # print the resulting dictionary, features => how much it drop the score
    print(feature_dict_scores)
    ```

  + model-agnostic permutation feature importance: a more advanced technique in permutation importance


### 5.4 Deep Learning

+ Deep learning
  + involving the use of NN to build high-performing ML models
  + NN able to learn nonlinear relationships among features
  + most traditional embedded-based methods only exploring linear relationships across features

+ Autoencoders
  + able to derive feature importance from NN to help to select good feature subsets
  + procedure
    + learning to compress and encode data
    + learning how to reconstruct the data back from the reduced encoded representation
  + goal: resulting representation as close as possible to the original input
  + taking the feature space and reducing its dimensionality $\to$ reconstructing inputs from its reduced format
  + principle: by reducing the data dimensions
    + learn how to ignore the noise
    + which feature best helping to reconstruct the data
  + Auttoencoder for Feature Selection (AEFS)
    + [Autoencoder Inspired Unsupervised Feature Selection]((https://arxiv.org/abs/1710.08310))
    + a solution for feature selection
    + uncovering existing nonlinear relationships btw features
    + using a single-layer autoencoder to reconstruct data

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/y6f38zqs')"
        src    ="https://tinyurl.com/yykc9cjf"
        alt    ="Signle-leayer Autoencoder"
        title  ="Signle-leayer Autoencoder"
      />
    </figure>

    + after reconstructing the data
      + use the first weight matrix of the autoencoder that connects the input feature layer to the reduced layer
      + squared weight ($w^2$) $\to 0$: feature contributing little to the representation of others
      + significant corresponding weight: important feature in the representation of other features
  + Python snippet

    ```python
    # import the keras library
    from keras.layers import Dense
    from keras.models import Sequential

    # the dimension of the encoding layer
    encoding_dim = 16

    # create the autoncoder
    model = Sequential()

    # add the encoding layer
    model.add(Dense(encoding_dim, activation='relu', 
      input_shape=(x_train_df.shape[1],)))

    # complie the model, use whatever optimizer
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')

    # fit model to the training data
    model.fit(x_train_df, y_train_df, epochs=50, batch_size=64,
      shuffle=True, validation_data=(x_train_df, x_test_df))

    # get the first layer weights
    weights = model.layers[1].get_weights()[0]

    # get the feature importance
    print(weights.sum(axis=1))
    ```

  + limitation: a simple single-layer autoencoder unable to model complex nonlinear feature dependencies
  + improvement: [Deep Feature Selection using a Teacher-Student Network](https://arxiv.org/abs/1903.07045)







