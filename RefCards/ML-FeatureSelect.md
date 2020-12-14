# Feature Selection for Machine Learning


## Overview

+ [Feature selection](../Notes/a09-FeatureSelect.md#1-an-introduction)
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

+ [Why feature selection matter](../Notes/a09-FeatureSelect.md#1-an-introduction)
  + not always true: the more data features you have, the better the resulting model is going to be
    + irrelevant features
    + redundant features
    + result: overfitting
  + reasons to select features
    + simple models easier to interpret much easier to understand the output of a model w/ less variables
    + shorter training time
    + enhanced generalization by reducing overfitting
    + variable redundancy

+ [Feature selection vs. feature engineering](../Notes/a09-FeatureSelect.md#1-an-introduction)
  + feature engineering:
    + creating new features
    + helping the ML model make more effective and accurate predictions
  + feature selection:
    + selecting features from the feature pool
    + helping ML models more efficiently make predictions on target variables

+ [Feature selection vs. Dimensionality reduction](../Notes/a09-FeatureSelect.md#1-an-introduction)
  + dimensionality reduction
    + tending to lump together w/ feature selection
    + using unsupervised algorithms to reduce the number of feature in a dataset
  + differences
    + feature selection: a process to select and exclude some features w/o modifying them at all
    + dimensionality reduction:
      + modifying or transforming features into a slower dimension
      + creating a whole new feature space that looks approximately like the first one, but smaller in terms of dimensions

+ [Procedure of feature selection](../Notes/a09-FeatureSelect.md#1-an-introduction)
  + process
    + combination of a search technique for proposing a new feature subset
    + an evaluation measuring that scores how well is the different feature subsets
  + computational expensive
  + looking for the best combination of feature subsets from all the available features

+ [Feature selection methods](../Notes/a09-FeatureSelect.md#1-an-introduction)
  + filter methods
  + wrapper methods
  + embedded methods


## Filter Methods

### Overview of Filter Methods

+ [Filter Methods](../Notes/a09-FeatureSelect.md#21-overview-of-filter-methods)
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

+ [Univariate of filter methods](../Notes/a09-FeatureSelect.md#21-overview-of-filter-methods)
  + evaluating and ranking a single feature according to certain criteria
  + treating each feature individually and independently of the feature space
  + procedures
    + ranking features according to certain criteria
    + selecting the highest ranking features according to those criteria
  + issue: not considering relation to other ones in the dataset

+ [Multivariate of filter methods](../Notes/a09-FeatureSelect.md#21-overview-of-filter-methods)
  + evaluating the entire feature space
  + considering the relations to other ones in the dataset


### Basic filter methods

+ [Constant features](../Notes/a09-FeatureSelect.md#22-basic-filter-methods)
  + showing single values in all the observations in the dataset
  + no information
  + Python: `from sklearn.feature_selection import VarianceThreshold`

+ [Quasi-Constant feature](../Notes/a09-FeatureSelect.md#22-basic-filter-methods)
  + a value occupying the majority of the records
  + hyperparameter: threshold

+ [Duplicated features: Python snippet](../Notes/a09-FeatureSelect.md#22-basic-filter-methods)

  ```python
  # select the duplicated features columns names
  duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
  ```

### Correlation Filter Methods

+ [Correlation](../Notes/a09-FeatureSelect.md#23-correlation-filter-methods)
  + correlation:
    + a measure of the linear relationship btw two quantitative variables
    + a measure of how strongly one variable depending on another
  + high correlation w/ target
    + a useful property to predict one from another
    + goal: highly correlated w/ the target, especially for linear ML models
  + high correlation btw variables:
    + providing redundant information in regards to the target
    + making an accurate prediction on the target w/ just one of the redundant variables
    + removing redundant ones to reduce the dimensionality but add noise
  + methods to measure the correlation btw variables
    + Pearson correlation coefficient
    + Spearman's rank correlation coefficient
    + Kendall's rank correlation coefficient

+ [Pearson's correlation coefficient](../Notes/a09-FeatureSelect.md#23-correlation-filter-methods)
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

+ [Spearman's rank coefficient coefficient](../Notes/a09-FeatureSelect.md#23-correlation-filter-methods)
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

+ [Kendall's rank correlation](../Notes/a09-FeatureSelect.md#23-correlation-filter-methods)
  + a non-parametric test that measures the strength of the ordinal association btw two variables
  + calculating a normalized score for the number of machine or concordant rankings btw the two data samples
  + $\tau \in [-1, 1]$
    + $\tau = 1$ (high): observations w/ a similar rank btw two variables
    + $\tau = -1$ (low): observation w/ a dissimilar rank btw two variables
  + best suitable for discrete data
  + Kendall's rank correlation

    \[ \tau = \frac{2(n_c - n_d)}{n(n-1)} \]

+ Python related functions
  + `dataframe.corr()` function
  + `DataFrame.duplicated()` function

### Statistical & Ranking Filter Methods

+ [Statistical & Ranking filter](../Notes/a09-FeatureSelect.md#24-statistical--ranking-filter-methods)
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

+ [Mutual information](../Notes/a09-FeatureSelect.md#24-statistical--ranking-filter-methods)
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

  + Python: `from sklearn.feature_selection import mutual_info_classif`

+ [Chi-squared score](../Notes/a09-FeatureSelect.md#24-statistical--ranking-filter-methods)
  + commonly used for testing relationships btw categorical variables
  + suitable for categorical variables and binary targets only
  + constraint: non-negative variables and typically boolean, frequencies, or counts
  + simply comparing the observed distribution btw various features in the dataset and the target variable
  + Python: `from sklearn.feature_selection import chi2`

+ [ANOVA univariate test](../Notes/a09-FeatureSelect.md#24-statistical--ranking-filter-methods)
  + ANOVA = Analysis Of VAriance
  + similar to chi-squared score
  + measuring the dependence of two variables
  + assumptions
    + linear relationship btw variables and the target
    + both normally distributed
  + suitable for continuous variables and requiring a binary target
  + Python: `from sklearn.feature_selection import f_classif`

+ [Univariate ROC-AUC / RMSE](../Notes/a09-FeatureSelect.md#24-statistical--ranking-filter-methods)
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
  + Python: `from sklearn.tree import DecisionTreeClassifier`

## Wrapper Methods
### Overview of Wrapper Methods

+ [Wrapper methods](../Notes/a09-FeatureSelect.md#31-overview-of-wrapper-methods)
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

### Forward Feature Selection

+ [Forward feature selection](../Notes/a09-FeatureSelect.md#32-forward-feature-selection)
  + a.k.a. step forward selection, sequential forward feature selection (SFS)
  + an interactive method
  + procedure
    + start by evaluating all feature individually and then select the one that results in the best performance
    + test all possible combinations of the selected feature w/ the remaining features and retain the pair that procedures the best algorithmic performance
    + a loop continues by adding one feature at a time in each iteration until the pre-set criterion reached

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `k_features`: the maximum feature to be reached when starting from 0
  + `floating`: using a variant of the step forward selection called step floating forward selection
  + `scoring`: the scoring function to evaluate model performance
  + `cv`: the number of folds of K-fold cross-validation, no cross-validation if cv=0 or False

### Backward Feature Elimination

+ [Backward feature elimination method](../Notes/a09-FeatureSelect.md#33-backward-feature-elimination)
  + a.k.a. step backward selection, sequential backward feature selection  (SBS)
  + starting w/ all the features in the dataset
  + evaluating the performance of the algorithm
  + removing one feature at a time
  + producing the best performing algorithm using an evaluation metric
  + the least significant feature among the remaining available ones
  + removing feature after feature until a certain criterion satisfied

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `k_features`: the maximum feature to be reached when starting from N
  + `forward`: using the object for step forward or step backward feature selection
  + `floating`: using a variant of the step backward selection called step floating backward selection


### Exhaustive Feature Selection

+ [Exhaustive feature selection](../Notes/a09-FeatureSelect.md#34-exhaustive-feature-selection)
  + finding the best performing feature subset
  + a brute-force evaluation of feature subsets
  + creating all the subsets of features from 1 to N
  + building a ML algorithm for each subset and selecting the subset w/ the best performance
  + parameter 1 and N: the minimum number of features and the maximum number of features

+ `mlxtend.feature_selection.SequentialFeatureSelector()` function
  + `min_features`: the lower bound of the number of features to search from
  + `max_features`: the upper bound of the number of features to search from

### Limitations and Solutions of Step Forward/Backward Selection

+ [Limitations of SFS & SBS](../Notes/a09-FeatureSelect.md#35-limitations-and-solutions-of-step-forwardbackward-selection)
  + SFS adding features at each iteration
    + adding up a feature useful in the beginning but non-useful after adding more ones
    + unable to remove the feature
  + SBS removing features at each iteration
    + removing a feature useless in the beginning but useful after removing more ones
    + unable to add the feature in the feature subsets
  + solutions: LRS or sequential floating

+ [LRS, or Plus-L, Minus-R](../Notes/a09-FeatureSelect.md#35-limitations-and-solutions-of-step-forwardbackward-selection)
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

+ [Sequential floating](../Notes/a09-FeatureSelect.md#35-limitations-and-solutions-of-step-forwardbackward-selection)
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


### Bidirectional Search

+ [Bidirectional Search (BDS)](../Notes/a09-FeatureSelect.md#36-other-search-methods)
  + applying SFS and SBS concurrently
  + SFS: performing from the empty set of features
  + SBS: performing from the full set of features
  + issue: converge to different solutions 
  + resolving by the constrains
    + features already selected by SFS not removing by SBS
    + features already removed by SBS not added by SFS


## Embedded Methods

### Overview of Embedded Method

+ [Embedded methods](../Notes/a09-FeatureSelect.md#41-overview-of-embedded-method)
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


### Regularization Methods

+ [Regularization](../Notes/a09-FeatureSelect.md#42-regularization-methods)
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
  + Python: `from sklearn.feature_selection import SelectFromModel`

### Tree-based Feature Importance

+ [Tree-based feature importance](../Notes/a09-FeatureSelect.md#43-tree-based-feature-importance)
  + tree-based algorithms and models
    + well-established algorithms
    + offerring good predictive performance
    + able to provide the featire importance as a way to select features
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
  + alternatives:
    + able to use any other tree-based algorithm the same way =
    + best tree model types: gradient boosting algoritms (like XH=GBoost, CatBoost, and any more)
    + providing accurate feature importance


## Hybride Methods

### Overview of Hybrid Methods

+ [Hybrid methods](../Notes/a09-FeatureSelect.md#51-overview-of-hybrid-methods)
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


### Filter & Wrapper Methods

+ [Hybrid method: Filter + Wrapper methods](../Notes/a09-FeatureSelect.md#52-filter--wrapper-methods)
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

### Embedded & Wrapper Methods

+ [Hybrid method: Embedded & Wrapper methods](../Notes/a09-FeatureSelect.md#53-embedded--wrapper-methods)
  + embedded methods
    + establishing feature importance
    + used to select top features
  + procedure:
    + embedded methods to get top k features
    + performing a wrapper methods search
  + methods:
    + recursive feature elimination
    + recursive featire addition

### Recursive feature elimination

+ [Recursive feature elimination](../Notes/a09-FeatureSelect.md#54-recursive-feature-elimination)
  + procedure
    1. train a model on all the data features
      + possible candidate: tree-based model, lasso, logistic regression, or others offerring feature importance
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
  + Python: `from sklearn.feature_selection import RFECV`

+ [`RFECV`: recursive feature elimination w/ corss-validation](../Notes/a09-FeatureSelect.md#54-recursive-feature-elimination)
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

+ [Python snippet](../Notes/a09-FeatureSelect.md#54-recursive-feature-elimination)


### Recursive Feature Addition

+ [Recursive feature additions](../Notes/a09-FeatureSelect.md#55-recursive-feature-addition)
  + starting w/ no features and adding one feature at the time
  + procedure
    1. train a model on all the data and derive the feature importance to rank it accordingly
      + possible models: tree-based model, Lasso, logistic regression, or others offerring feature importance
    2. from the initial model, create another w/ the most important feature and evaluate it w/ an evaluation metric of your choice
    3. add another important feature and use it to re-train the model, along w/ any feature from the previous step
    4. use the previous evaluation metric to calculate the performance of the resulting model
    5. test whether the evaluation metric increases by an arbitrary-set threshold
    6. repeat step 3-5 until features added
  + Python snippet


## Advanced Methods

### Dimensionality Reduction

+ [Dimensionality reduction vs. feature selection](../Notes/a09-FeatureSelect.md#61-dimensionality-reduction)
  + both tried to reduce the number of features
  + feature selection: select and exclude some features w/o making any transformation
  + dimensionality reduction: transform features into a lower dimension

+ [Principal component analysis (PCA)](../Notes/a09-FeatureSelect.md#61-dimensionality-reduction)
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
  + Python: `from sklearn.decomposition import PCA`
  + alternative: linear discriminant analysis (LDA)
  

### Heuristic Search Algorithms

+ [Heuristic search](../Notes/a09-FeatureSelect.md#62-heuristic-search-algorithms)
  + attempt to perform optimization tasks in an iterative fashion to find an approximate solution if classical models failed to find an exact solution
  + not always find the best or even the optimal solution, but find a good or acceptable solution within a reasonable amount of time and memory space
  + performing a heuristic search across feature subsets to find the best one

+ [Genetic algorithms (GA)](../Notes/a09-FeatureSelect.md#62-heuristic-search-algorithms)
  + global optimization techniques for searching very large spaces
  + sort of randomized search
  + inspired by the biological mechanisms of natural selection and reconstruction
  + working throughout populations of possible solutions (generations)
  + each solution in the search space represented as a finite length string (chromosome) over some finite set of symbols
  + using an objective (or fitness) function to evaluate the suitability of each solution
  + feature selection
    + each chromosome representing a feature subset
    + represented w/ binary encoding: feature w/ 1 to choose and 0 to eliminate from
  + conducting many iterations to create a new generation (new feature subset) of possible solutions from the current generations using a few operators
  + operators
    + __selection:__
      + probabilististically filtering out solutions that perform poorly
      + choosing high perfroming solutions to exploit
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
  + Python: `from genetic_selection import GeneticSelectionCV`
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
    + __mutation_independent_proba__: the independent probabiility for each attribute to be mutated by the generic algorithm
    + __n_gen_no_change__: the number of generations needs to terminate the search if no change w/ the best individuals
  
+ Simulation annealing
  + a heuristic approach to search for the feature subsets
  + a global search algorithm allowing a suboptimal solution to be accepted in the hope that better solution wiill show up eventually


### Feature Importance w/ Permutation Importance

+ [Permutation Importance](../Notes/a09-FeatureSelect.md#63-feature-importance-w-permutation-importance)
  + basically randomly shuffle the values of a feature (w/o touching the other variables of the targets) to see how this permutation affects the performance metric of the ML meodel
  + the choice of metric upon the engineer
  + measuring the importance of a featrue by measuring the increase in the model's prediction error after permutation the feature values
  + breaking the relationship btw the feature and the true outcome
    + important feature:
      + shuffling values increasing the model error
      + relying on the feature for the prediction
    + non-important feature:
      + shuffling values leaving the model error unchanged
      + model ignoring the feature for the prediction
  + model-agnostic permutation feature importance: a more advanced technique in permutation importance


### Deep Learning

+ [Deep learning](../Notes/a09-FeatureSelect.md#54-deep-learning)
  + involving the use of NN to build high-performing ML models
  + NN able to learn nonlinear relationships among features
  + most traditional embedded-based methods only exploring linear relationships across features

+ [autoencoders](../Notes/a09-FeatureSelect.md#54-deep-learning)
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
    + after reconstructing the data
      + use the first weight matrix of the autoencoder that connects the input feature layer to the reduced layer
      + squared weight ($w^2$) $\to 0$: feature contributing little to the representation of others
      + significant corresponding weight: important feature in the representation of other features
  + limitation: a simple single-layer autoencoder unable to model complex nonlinear feature dependencies
  + improvement: [Deep Feature Selection using a Teacher-Student Network](https://arxiv.org/abs/1903.07045)







