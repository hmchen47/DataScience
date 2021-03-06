# Feature Engineering - Machine Learning


## Summary

+ Imputation
  + numerical variable
    + mean or median imputation
    + arbitrary value imputation
    + end of tail imputation
  + categorical variables
    + frequent category imputation
    + add a missing category
  + both
    + complete case analysis
    + add a missing indicator
    + random sample imputation
    + iterative imputation (multivariate)
    + K-nearest neighbor imputation

+ techniques for data encoding (categorical variable)
  + traditional techniques
    + one-hot encoding
    + count or frequency encoding
    + ordinal or label encoding
  + monotonic relationship
    + ordered label encoding
    + mean (target) encoding
    + probability ratio encoding
    + weight of evidence $\left(\text{WOE} = \ln \left(\frac{p(1)}{p(0)}\right)\right)$
    + Catboost encoder - target-based
    + Leave-one-out encoder (LOO/LOOE) - target-based
    + James-Stein Encoder $\left( \widehat{x}^k = (1-B) \cdot \frac{n^+}{n} + B \cdot \frac{y^+}{y} \right)$ - target-based
  + alternative techniques
    + rare labels encoding
    + binary encoding

+ Most common-used transformer
  + logarithmic transformation $\left(f(x) = \ln(x), x > 0 \right)$
  + square root transformation $\left(f(x) = \sqrt{x}, x \ge 0 \right)$
  + reciprocal transformation $\left(f(x) = \frac1x, x \ne 0 \right)$
  + exponential or power transformation $\left(f(x) = x^2, x^3, \dots, x^n, \exp(x) \right)$
  + Box-Cox transformation $\left(x_i^(\lambda) = \begin{cases} (x_i^\lambda -1) / \lambda & \text{if } \lambda =\ne 0 \\ \ln(x_i) & \text{if } \lambda = 0 \end{cases} \right)$
  + Yeo-Johnson transformation $\left(x_i^{(\lambda)} = \begin{cases} [(x_i + 1)^\lambda - 1] / \lambda & \text{if } \lambda \ne 0, x_i \ge 0, \\ \ln(x_i) + 1 & \text{if } \lambda = 0, x_i \ge 0, \\ -[(-x_i + 1)^{2-\lambda} - 1]/(2-\lambda) & \text{if } \lambda \ne 2, x_i < 0, \\  -\ln(-x_i + 1) & \text{if } \lambda = 2, x_i < 0 \end{cases}\right)$


+ Variable discretization approaches
  + supervised approach
    + discretization w/ decision tree
  + unsupervised approaches
    + equal-width discretization ($\text{width} = \frac{\max(x) - \min(x)}{N}$)
    + equal-frequency discretization
    + K-means discretization
  + other
    + custom discretization

+ Outlier Detection
  + visualization plots like box plot and scatter plot
  + normal distribution ($\mu \pm 3 \times \text{s.d.}$)
  + Inter-quantal range proximity rule (upper bound = $Q_3(x) + 1.5 \times \text{IQR}$, lower bound = $Q_1(x) - 1.5 \times \text{IQR}$)
  + Density-Based Spatial Clustering of Application w/ Noise (DBSCAN)
  + Isolation Forest - tree-based
  + Local Outlier Factor (LOF)

+ Handling outliers
  + trimming: simply removing the outliers from dataset
  + imputing: treating outliers as missing data and applying missing data imputation techniques
  + discretization: placing outliers in edge bins w/ higher or lower values of the distribution
  + censoring: capping the variable distribution at the maximum and minimum values

+ Scaling methods
  + mean normalization $\left(\overline{x} = \frac{x - \mu}{\max(x) - \min(x)]} \right)$
  + standardization $\left(\overline{x} = \frac{x - \mu}{\text{std}(x)} \right)$
  + robust scaling (scaling to median and IQR) $\left(\overline{x} = \frac{x - \text{median}(x)}{Q_3(x) - Q_1(x)} \right)$
  + robust to maximum and minimum $\left(\overline{x} = \frac{x - \min(x)}{\max(x) - \min(x)} \right)$
  + scale to absolute maximum $\left(\overline{x} = \frac{x}{\max(x)}\right)$
  + scale to unit norm $\left(\overline{x} = \frac{x}{\|x\|} \right)$


## Overview

### General 

+ [Feature engineering](../Notes/a08-FeatureEng.md#1-broad-introduction)
  + the process of using data domain knowledge to create features or variables
  + purpose: making ML algorithms effectively
  + very time consuming process
  + a number of processes
    + filling missing values within a variable
    + encoding categorical variables into numbers
    + variable transformation
    + creating or extracting new features from the ones available in the dataset

+ [Feature engineering](../Notes/a18a-FeatureEng.md)
  + simply making data better suited to the problem at hand

+ [Purposes of feature engineering](../Notes/a08-FeatureEng.md#1-broad-introduction)
  + raw data: messy and unsuitable for training a model
  + solution: data exploration and cleaning
  + involving changing data types and removing or imputing missing values
  + requirements: a certain understanding of the data acquired through exploration
  + solving these challenges and building high-performing models
  + solutions
    + removing outliers or specific features
    + creating features from the data that represent the underlying problem better
  + algorithms often hinging on how the input features engineered

+ [Reasons for feature engineering](../Notes/a18a-FeatureEng.md)
  + improving a model's predictive performance
  + reducing computational or data needs
  + improving interpretability of the results

+ [Principle of feature engineering](../Notes/a18a-FeatureEng.md#a-guiding-principle-of-feature-engineering)
  + useful feature: relationship to the target that your model is able to learn
  + linear model: transforming the features to make features' relationship to the target linear
  + key idea: a transformation applied to a feature becoming in essence a part of model itself
  + high return on time invested in feature engineering

+ [Tips to discovering new features](../Notes/a18c-CreateFeat.md#introduction)
  + understand the features: referring to data documentation if available
  + acquire domain knowledge: research the problem domain
  + study previous work
  + use data visualization:
    + revealing pathologies in the distribution of a feature
    + simplifying complicated relationships
    + a must step for feature engineering process

+ [Feature Engineering vs. Feature Selection](../Notes/a08-FeatureEng.md#1-broad-introduction)
  + feature engineering:
    + <mark style="background-color: lightpink;">creating new features</mark> from the existing ones
    + helping ML model more effective and accurate predictions
  + feature selection
    + selecting from the feature pool
    + helping ML models to predict on target variables more efficiently
    + typical ML pipeline: completing feature engineering then feature selection

+ [Categorical encoding w/ category_encoders lib](../Notes/a16-FeatureEng.md#beyond-one-hot-17-ways)
  + the process of transforming a categorical column into one (or more) numerical column(s)
  + Python library: [category_encoders](https://github.com/scikit-learn-contrib/category_encoders)

    ```python
    !pip install category_encoders
    
    import category_encoders as ce

    ce.OrdinalEncoder().fit.transform(x)
    ```

  + classification of endcoders

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://tinyurl.com/38swc7jm"
        alt    = "Classification of encoding methods fpr categorical variables"
        title  = "Classification of encoding methods fpr categorical variables"
      />
    </figure>

  + [local demo notebook](../Notes/a16-FeatureEng.md#beyond-one-hot-17-ways/src/a16-beyond_one_hot.ipynb)


### Variables Types

+ [Variables](../Notes/a08-FeatureEng.md#2-variables-types)
  + any characteristic, number, or quantity measured or counted
  + major types of variables
    + numerical variables
    + categorical variables
    + datetime variables
    + mixed variables
  + get the type of each variable from a Pandas dataframe

+ [Numerical variables](../Notes/a08-FeatureEng.md#2-variables-types)
  + (predictably) numbers
  + categories of numerical variables
    + continuous variables
    + discrete variables

+ [Continuous variables](../Notes/a08-FeatureEng.md#2-variables-types)
  + an uncountable set of values
  + probably containing any value within a given range
  + visualization
    + density plot
    + histogram
    + box plot
    + scatter plot

+ [Discrete variables](../Notes/a08-FeatureEng.md#2-variables-types)
  + a finite number of values
  + integers, counts
  + visualization
    + count plot
    + pie chart

+ [Categorical variables](../Notes/a08-FeatureEng.md#2-variables-types)
  + selected from a group of categories
  + a.k.a. labels
  + categories: ordinal & nominal
  + ordinal variables: variables existed within meaningfully ordered categories
  + nominal variables: not a natural order in the labels
  + in some scenarios, categorical variables coded as numbers when the data was recorded

+ [Dates and Times](../Notes/a08-FeatureEng.md#2-variables-types)
  + particular type of categorical variable
  + containing __dates__, __time__, or __data and time__
  + usually __not working__ w/ datetime variables in their __raw format__
    + date variables containing a considerable number of different categories
    + able to extract much more information from datetime variables by preprocessing them correctly
  + date variable issues
    + containing dates not present in the dataset used to train the learning model
    + containing dates placed in the future, w.r.t. the dates in the training dataset
  
+ [Mixed variables](../Notes/a08-FeatureEng.md#2-variables-types)
  + containing both numbers and labels
  + occurring in a given dataset, especially when filling its values


### Common Issues in Datasets

+ [General issues](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + missing data
  + categorical variable - cardinality
  + categorical variable - rare labels
  + linear model assumptions
  + variable distribution
  + outliers
  + feature magnitude

+ [Missing data](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + when no data stored for a particular observation in  variable
  + basically just the absence of data
  + data missing for multiple reasons: lost & not exist
  + many features not mandatory
  + solution: missing data imputation techniques
  + issues:
    + probably distort the original variable distribution
    + alter the way variables interact w/ each other
    + affect the machine learning model's performance $\gets$ many models make assumptions about the variable distribution
  + carefully choosing the right missing data imputation technique
  + main mechanisms lead to missing data
    + __missing data completely at random (MCAR)__
    + __missing data at random (MAR)__: the probability of an observation being missing depends on available information
    + __missing data not at random (MNAR)__: a mechanism or a reason why values introduced in the dataset
  + __labels__: the values of a categorical variable selected from a group of categories
  + __cardinality__: the number of different labels
  + cardinality on models: issues w/ multiple labels in a categorical variable
  + high cardinality

+ [Categorical variable - rare labels](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + rare labels: appear only in a small proportion of the observation in a dataset
  + impacts and considerations on rare labels
    + causing overfitting and generalization problems
    + hard to understand the role of the rare label in the final prediction
    + removing rare labels may improve model performance

+ [Linear model assumptions](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + linearity
    + the relationship btw the variables ($X$s) and the target ($Y$) is linear and accessed w/ scatter plot

      \[ Y = \beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 + \cdots + \beta_n \cdot x_n \]

    + homoscedasticity
      + homogeneity of variance: independent variables w/ the sam variance
      + tests and plots to determine homogeneity
        + residual plot
        + Levene's test
        + Barlett's test
      + not homoscedasticity:
        + performing non-linear transformations (e.g., logarithm-transformation)
        + feature scaling to improve the homogeneity of variance
  + normality
    + assessment: histograms and Q-Q plots
    + Q-Q plot: if variable is normally distributed,, the values of the variable falls in a 45-degree line when plotted against the theoretical quantities
    + variable not normal distribution: non-linear transformation to fix
  + independent: observations independent of each other

+ [Variable/Probability distribution](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + a function describing the likelihood of obtaining the possible values that a variable can take
  + properties of probability distribution
    + $P(x)$: the likelihood that the random variable takes a specific value of $x$
    + unitary: $\sum P(x) = 1$
    + non-negative: $P(x) \in [0, 1]$
  + different probability distributions
    + discrete, like Binomial and Poisson
    + continuous, like Gaussian, skewed, and many others
  + distributions and model performance
    + linear models assumption: the independent variables w/ normal distribution
    + other models not assume the distribution of variables, a better spread of the values may improve their performance

+ [Outliers](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + a data point significantly different from the remaining data
  + Ref: [How to Make Your Machine Learning Models Robust to Outliers](https://tinyurl.com/y3p38u6t)
  + an observation deviateing so much from the other observations
  + should outliers be removed?
    + depending on context
    + deserve special attention
    + simply ignore entirely
  + algorithms susceptible to outliers mostly linear models
  + detecting outliers
    + using an extreme value analysis w/ a normal distribution to detect outliers
    + approximately 99% of the observations of a normally-distributed variable lie within the $\text{mean} \pm 3 \cdot \text{standard deviations}$
    + values outside mean $\pm 3 \times$ standard deviations considered outliers
  + visualization: box plot

+ [Feature magnitude](../Notes/a08-FeatureEng.md#3-common-issues-in-datasets)
  + examples
    + a dataset w/ a column for age and another one for income
    + the number of rooms in a given house and its price
  + feature magnitude matters
    + the scale of the variable directly influences the regression coefficient
    + variable w/ a more significant magnitude range (e.g., income) dominate over the ones w/ a smaller magnitude range (age)
    + gradient descent converges faster when features are on similar scale
    + features scaling helps secrease he time to find support vectors for SVMs
    + Euclidean distances are sensitive to feature magnitude
  + models affected by feature magnitude
    + linear and logistic regression
    + neural networks
    + support vector machines
    + K-nearest neighbors
    + K-mean clustering
    + linear discriminant analysis (LDA)
    + principle component analysis (PCA)
  + tree-based models insensitive to feature magnitude
    + classification and regression tress
    + random forests
    + gradient-boosted trees


### Mutual Information

+ [Handling features](../Notes/a18b-MutualInfo.md#introduction)
  + issue: hundreds and thousands of features w/o description
  + procedure to resolve
    + constructing a ranking w/ a __feature utility metric__, a function measuring associatiions btw a feature and a target
    + choosing a smaller set of the most useful features to develop initially and having more confidence to spend time on them

+[Mutual information](../Notes/a18b-MutualInfo.md#introduction)
  + metric used to measure associations btw a feature and a target
  + a lot like correlation to measure a relationship btw two quantities
  + MI detecting any kind of relationship while correlation only detecting linear relationship
  + a great general-purpose metric and specially useful at the start of feature development
  + advantages
    + easy to use and interpret
    + computationally efficient
    + theoretically well-founded
    + resistant to overfitting
    + able to detect any kind of relationship

+ [Mutual information and measurement](../Notes/a18b-MutualInfo.md#introduction)
  + MI describing relationships in terms of _uncertainty_
  + __mutual information (MI)__ btw two quantities: a measure of the extent to which knowledge of one quantity reduces uncertainty about the other
  + scikit-learn algorithm for MI
    + two mutual information metrics in `feature_selection` module
    + continuous features
      + `float` dtype
      + real value targets: `mutual_info_regression`
    + categorical features
      + `object` or `categorical` dtype
      + treated as discrete by giving them a label encoding
      + categorical targets: `mutual_info_classif`
  + data visualization: a great toolbox for feature ranking, e.g., bar chart

+ [Mutual information scores](../Notes/a18b-MutualInfo.md#interpreting-mutual-information-scores)
  + MI = 0.0
    + least possible value
    + independent: unable to tell anything about the other
  + MI maximum value
    + theory: no upper bound
    + practice: MI > 2.0 uncommon
    + MI: a logarithm quantity

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/mutual-information')"
        src    = "https://i.imgur.com/Dt75E1f.png"
        alt    = "Left: Mutual information increases as the dependence between feature and target becomes tighter. Right: Mutual information can capture any kind of association (not just linear, like correlation.)"
        title  = "Left: Mutual information increases as the dependence between feature and target becomes tighter. Right: Mutual information can capture any kind of association (not just linear, like correlation.)"
      />
    </figure>

+ [Considerations when using mutual information](../Notes/a18b-MutualInfo.md#interpreting-mutual-information-scores)
  + relative potential:
    + MI helping to understand the relative potential of a feature
    + the potential as a predictor of the target
  + univariate metric
    + possible for a feature very informative when interacting w/ other features
    + not so informative for the feature itself
    + MI unable to detect interaction btw features
  + feature and model
    + the usefulness of feature depending on the model it used w/
    + feature probably only useful to the extent related to the target
    + a feature w/ high MI score $\nRightarrow$ model able to do anything w/ that information




## Imputing Missing Values

### Overview

+ [Data imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + the act replacing missing data w/ statistical estimates of missing values
  + goal: producing a complete dataset to use in the process of training ML models
  + python library:
    + `sklearn.impute`: transformers for missing value imputation
    + `feature-engine`: simplify the process of imputing missing values
  + classification of methods
    + numerical variable
      + mean or median imputation
      + arbitrary value imputation
      + end of tail imputation
    + categorical variables
      + frequent category imputation
      + add a missing category
    + both
      + complete case analysis
      + add a missing indicator
      + random sample imputation

### Mean and Median Imputation

+ [Mean and median imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + replacing all occurrences of missing values (NA) within a variable w/ the mean and median of the variable
  + scenarios
    + suitable for numerical variables
    + missing completely at random (MCAR)
    + more than 5% of the variable containing missing data
  + applied to both training and test sets
  + considerations
    + normal distribution: the mean and median approximately the same
    + skewed distribution: median as a better representation
  + assumption
    + missing data at random
    + missing observations most likely like the majority of the observations in the variable
  + advantages
    + easy to implement
    + easy way of obtaining complete datasets
    + used in production
  + limitations
    + distortion of the original variable distribution and variance
    + distortion of the covariance w/ the remaining dataset variable
    + the higher the percentage of missing values, the higher the distortions
  + Python: `from sklearn.impute import SimpleImputer`

### Arbitrary Value Imputation

+ [Arbitrary value imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + replacing all occurrences of missing values (NA) within a variable w/ an arbitrary value
  + arbitrary value different from the mean and median and not within the normal values of the variable
  + typical arbitrary values: 0, 999, -999, (or other combinations of 9s) or -1 (positive distribution)
  + scenarios
    + suitable for numerical and categorical variables
  + assumptions
    + no missing data at random
  + advantages
    + easy to implement
    + a fast way to obtain complete datasets
    + used in production, i.e., during model deployment
    + capturing the importance of a value being "missing", if existed
  + limitations
    + distortion of the original variable distribution and variance
    + distortion of the covariance w/ the remaining dataset variable
    + arbitrary value at the end of the distribution $\to$ mask or create outliers
    + carefully not choose an arbitrary value too similar to the mean or median (or any other typical value of the variable distribution)
    + the higher the percentage of NA, the higher the distortion
  + Python: `from sklearn.impute import SimpleImputer`


### End of Tail Imputation

+ [End of tail imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + roughly equivalent to arbitrary value imputation
  + automatically selecting the arbitrary values at the end of the variable distributions
  + scenarios
    + suitable for numerical variables
  + ways to select arbitrary values
    + normal distribution: using the $\mu \pm 3 \cdot \text{s.d.}$
    + skewed distribution: using the IQR proximity rule
  + replacing missing data calculated only on the train set
  + normal distribution
    + most of the observation (~99%) of a normally-distributed variable lie within $\pm 3 \times$ s.d.
    + the selected value = $\mu \pm 3 \times$ s.d.
  + skewed distribution
    + general approach: calculate the quantiles and the inter-quantile range (IQR)
      + IQR = 75th Quantile - 25th Quantile
      + upper limit = 75th Quantile + IQR x 3
      + lower limit = 25th Quantile - IQR x 3
    + selected value for imputation: upper limit or lower limit
  + Python: `from feature_engine.missing_imputers import EndTailImputer`


### Frequent Category Imputation

+ [Frequent category imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + a.k.a. mode imputation
  + replacing all occurrences of missing values (NA) within a variable w/ the mode, or the most frequent value
  + scenarios
    + suitable for numerical and categorical variables
    + in practice, using the technique w/ categorical variables
    + using w/ data as missing complete at random (MCAR)
    + no more than 5% of the variable contains missing data
  + applied only to train and test sets
  + assumption
    + missing data at random
    + missing observations most likely like the majority of the observations (i.e., the mode)
  + advantages
    + easy to implement
    + a fast way to obtain a complete dataset
    + used in production
  + limitations
    + distort the relation of the most frequent label w/ other variables within dataset
    + may lead to an over-representation of the most frequent label if a lot of missing observations existed
  + Python: `from sklearn.impute import SimpleImputer`


### Missing Category Imputation

+ [Missing category imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + treating missing data as an additional label or category of the variable
  + create a new label or category by filling the missing observations w/ a Missing category
  + most widely used method of missing data imputation for categories variables
  + advantages
    + easy to implement
    + fast way of obtaining complete datasets
    + integrated into production
    + capturing the importance of "missingness"
    + no assumption mad on the data
  + limitations: small number of missing data $\to$ creating an additional category just adding another rare label to the variable
  + Python: `from sklearn.impute import SimpleImputer`

### Complete Case Analysis

+ [Complete case analysis (CCA)](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + discarding observations where values in any of the variables are missing
  + keep only those observations for which there's information in all of the dataset variables
  + observations w/ any missing data excluded
  + scenarios
    + missing data complete at random (MCAR)
    + no more than 5% of the total dataset containing missing data
  + assumption: missing data at random
  + advantages
    + simple to implement
    + no data manipulation required
    + preserving the distribution of the variables
  + limitation
    + excluding a significant fraction of the original dataset (if missing data significant)
    + excluding informative observations for the analysis (if data not missing at random)
    + create a biased dataset if the complete cases differ from the original data (if MAR or MNAR)
    + used  in production $\to$ not knowing how to handle missing data
  + Python: `data.dropna(inplace=True)`


### Missing Indicator

+ [Missing indicator](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + an additional binary variable indicating whether the data was missing for an observation (1) or not (0)
  + goal: capture observations where data is missing
  + used together w/ methods assuming MAR
    + mean, median, mode imputation
    + random sample imputation
  + scenario: suitable for categorical and numberic variables
  + assumptions
    + NOT missing at random
    + predictive missing data
  + advantages
    + easy to implement
    + capture the importance of missing data
    + integrated into production
  + limitations
    + expanding the feature space
    + original variable still requiring to be imputed
    + many missing indicators may end up being identical or very highly corrrelated
  + Python: `from sklearn.imput import MissingIndicator`


### Random Sample Imputation

+ [Random sample imputation](../Notes/a08-FeatureEng.md#4-imputing-missing-values)
  + taking a random observation from the pool of available observations of the variable and using those randomly selected values to fill in the missing one
  + scenario:
    + suitable for numerical and categorical variables
  + assumptions
    + missing data at random
    + replacing the missing values within the same distribution of the original value
  + advantages
    + easy to implement
    + a fast way of obtaining complete dataset
    + used in production
    + preserving the variance of the variable
  + limitations
    + randomness
    + relationship btw imputed variables and other variables probably affected if a lot of missing values
    + requiring massive memory for deployment to store the original training set to extract values from and replace the missing values w/ the randomly selected values
  + Python: `from feature_engine.missing_data_imputer import  RandomSampleImputer`


### Iterative Imputation

+ [Iterative imputation](../Notes/a08-FeatureEng.md#112-advanced-missing-value-imputation)
  + a multivariate imputer that estimates feature from all the other ones in a round-robin manner
  + using a strategy for imputing missing values by modeling each feature w/ missing values as a function of other features
  + dtermining misssing values by discovering patterns from its neighbors
  + using round-robin at each step
    1. choosing a feature as output $y$ and all the other feature columns as imput $x$
    2. training a regressor and fitting it on $(x, y)$ for known $y$
    3. the regressor used to predict the missing values of $y$
    4. repeating until the defined `max_iteration`  reached
  + `IterativeImputer` still experimental in Sklearn
  + Python: `from sklearn.experimental import enable_iterative_imputer` & `from sklearn.imputer import IterativeImputer`


### K-Nearest Neighbor Imputation

+ [K-nearest neighbor (KNN) imputing](../Notes/a08-FeatureEng.md#112-advanced-missing-value-imputation)
  + using the famous KNN algorithm to predict the missing values from the neighbors
  + any point value approximated by the nearest point values of ofther variables
  + Python: `from sklearn.impute import KNNImputer`


## Encoding Categorical Variables

### Overview
 
+ [Categorical encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + permanently replacing category strings w/ numerical representations
  + goal: producing variables used to train machine learning models and build predictive features from categories
  + techniques for data transformation
    + traditional techniques
      + one-hot encoding
      + count or frequency encoding
      + ordinal or label encoding
    + monotonic relationship
      + ordered label encoding
      + mean encoding
      + probability ratio encoding
      + weight of evidence
    + alternative techniques
      + rare labels encoding
      + binary encoding
  + Python library: category_encoders - containing a lot of basic and advanced methods for categorical variable encoding

+ [Supervised feature encoding engineering](../Notes/a18f-TargetEnc.md#target-encoding-1)
  + a method of encoding categories as integer number
  + example: one-hot or label encoding

+ [Target encoding](../Notes/a18f-TargetEnc.md#target-encoding-1)
  + any kind of encoding replacing a feature's categories w/ some number derived from the target
  + simple and effect version: applying a group aggregation, like the mean
  + Automobiles: average price of each vehicle's make

    ```python
    autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
    ```

  + mean encoding: applying a group aggregation w/ mean
  + other encodings: likelihood encoding, impact encoding, and leave-one-out encoding




### One-Hot Encoding

+ [One-hot encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + consisting of encoding each categorical variable w/ a set of boolean variables, that take values of __0__ or __1__
  + the value indicating if a category is present for each observation
  + multiple variants
  + one-hot encoding into $k-1$ variables
    + creating $k-1$ binary variables, where $k$ is the number of distinct categories
    + using one less dimension and still represent the data fully
    + e.g., medical test w/ $k=2$ (positive/negative), creating only one ($k - 1 =1$) binary variable
    + most ML algorithms considering the entire dataset while training
    + encoding categorical variables into $k-1$ binary values better $\to$ avoid introducing redundant information
  + one-hot encoding into $k$ variables:
    + occasions better to encode variables into $k$ variables
      + building tree-based algorithms
      + making feature selection w/ recursive algorithms
      + interested in determining the importance of every single category
  + one-hot encoding of most frequent categories
    + only considering the most frequent categories in a variable
    + avoid overextending the feature space
  + advantages
    + not assuming the distribution of categories of the categorical variable
    + keeping all the information of the categorical variable
    + suitable for linear models
  + limitations
    + expanding the feature space
    + not adding extra information while encoding
    + many dummy variables probably identical $\to$ introducing redundant information
  + Python: `data_with_k_df = pd.get_dummies(data_df)`


### Integer (Label) Encoding

+ [Integer (Label) Encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + replacing the categories w/ digits from $1$ to $n$ (or $0$ to $n-1$, depending on the implementation)
  + $n$: the number of the variable's distinct categories (the cardinality)
  + the number assigned arbitrary
  + advantages
    + straightforward to implement
    + not expanding the feature space
    + working well enough w/ tree-based algorithms
    + allowing agile benchmarking of ML models
  + limitations
    + not adding extra information while encoding
    + not suitable for linear models
    + not handling new categories in the test set automatically
    + creating an order relationship btw the categories


### Count or Frequency Encoding

+ [Count or frequency encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + replacing categories w/ the count or percentage that show each category in the dataset
  + capturing the representation of each label in the dataset
  + advantages
    + straightforward to implement
    + not expanding the feature space
    + working well w/ tree-based algorithms
  + limitations
    + not suitable for linear models
    + not handling new categories in the test set automatically
    + losing valuable information if there are two different categories w/ the same amount of observations count

### Ordered Label Encoding

+ [Ordered label encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + replacing categories w/ integers from 1 to n
  + $n$: the number of distinct categories in the variable (the cardinality)
  + using the target mean information of each category to decide how to assign these numbers
  + advantages
    + straightforward to implement
    + not expanding the feature space
    + creating a monotonic relationship btw categories and the target
  + limitation: probably leading to overfitting


### Mean (Target) Encoding

+ [Mean (target) encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + replacing the category w/ the mean target value for that category
  + procedure
    + grouping each category alone
    + for each group, calculating the mean of the target in the corresponding observations
    + assigning mean to that category
    + encoded the category w/ the mean of the target
  + advantages
    + straightforward to implement
    + not expanding the feature space
    + creating a monotonic relationship btw categories and the target
  + limitations
    + probably leading to overfitting
    + probably leading to a possible loss of value if two categories have the same mean as the target


### Smoothing

+ [Issues of encoding](../Notes/a18f-TargetEnc.md#smoothing)
  + unknown categories
    + creating a special risk of overfitting
    + required to be trained on an independent "encoding" split
    + imputation: filling in missing values for any categories
  + rare categories
    + any statistics on this group unlikely very accurate
    + solution: smoothing

+ [Smoothing technique](../Notes/a18f-TargetEnc.md#smoothing)
  + blending the in-category average w/ the overall average
  + rare categories: less weight on their category average
  + missing categories: the overall average
  + pseudocode

    <code>encoding = weight * in_category + (1 - weight) * overall </code>

  + weight
    + a value btw 0 and 1 calculated from the catgory frequency
    + determining weight by computing __m-estimate__: $\text{weight } = n / (n + m)$
      + $n$: the total number of times the category occurred in the data
      + $m$: hyperparameter to determine the "smoothing factor"
    + value for $m \to$ how noisy expecting the categories to be
      + target values varying a great deal $\implies$ choosing a larger value for $m$
      + target values relatively stable $\implies$ choosing a smaller value
  + larger values of $m$ $\to$ more weight on the overall estimate

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/target-encoding')"
        src    = "https://i.imgur.com/1uVtQEz.png"
        alt    = "M-estimate w/ categories count and smoothing factor"
        title  = "M-estimate w/ categories count and smoothing factor"
      />
    </figure>

+ [Use cases for target encoding](../Notes/a18f-TargetEnc.md#smoothing)
  + high-cardinality features:
    + a feature w/ large number of categories: troublesome to encode
    + one-hot encoding:
      + generating too many features and alternative
      + not appropriate for that feature
    + target encoding: deriving numbers for the categories w/ the relationship w/ the target
  + domain-motivated feature
    + prior experience: categorical feature probably not so important even if scored poorly w/ a feature metric
    + target encoding revealing a feature's true information


### Weighted of Evidence Encoding

+ [Weight of evidence encoding (WOE)](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + used to encode categorical variables for classification
  + apply the natural logarithm ($\ln$) of the probability that the target equals 1 divided by the probability of the target values 0
  + math formula

    \[ \text{WOE} = |\ln(p(1)/p(0))| \]

    + $p(1)$: the probability of the target being 1
    + $p(0)$: the probability of the target being 0
  + WOE value
    + WOE > 0: the probability of the target being 0 is more significant
    + WOE < 0: the probability of the target being 1 is more significant
  + creating an excellent visual representation of the variable
  + observation: category favoring the target being 0 or 1
  + advantages
    + creating a monotonic relationship btw the target and the variables
    + ordering the categories on the 'logistic' scale, nature for logistic regression
    + comparing the transformed variables because they are on the same scale $\to$ determine which one is more predictive
  + limitations
    + probably lead to overfitting
    + not defined when the denominator is 0

### Probability Ratio Encoding

+ [Probability ratio encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + suitable for classification problems only, where the target is binary
  + similar to WOE, but not applying the natural logrithm
  + each category, the mean of the target = 1
    + $P(1)$: the probability of the target being 1
    + $P(0)$: the probability of the target being 0
  + calculating the ratio = P(1)/P(0) and replacing the categories by that ratio
  + advantages
    + capturing information within the category, and therefore creating more predictive features
    + creating a montonic relationship btw the variables and the target, suitable for linear mdoels
    + not expanding the feature space
  + limitations
    + likely to cause overfitting
    + not defined when the denominator is 0


### Rare Label Encoding

+ [Rare label encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + rare label: appearing only in a tiny proportion of the observations in a dataset
  + causing some issues, especially w/ overfitting and generation
  + solution: group those rare labels into a new category like other or rare


### Binary Encoding

+ [Binary encoding](../Notes/a08-FeatureEng.md#5-encoding-categorical-variables)
  + using binary code
  + procedure
    + converting each integer to binary code
    + each binary digits gets one column in the dataset
  + $n$ unique categories $\implies$ binary encoding results in the only $\log_2 n$ features
  + advantages
    + straightforward to implement
    + not expanding the feature space too much
  + limitations
    + exposing the loss of info during encoding
    + lacking the human-readable sense
  + Python: `from category_encoders import BinaryEncoder`


### Catboost Encoder

+ [Catboost encoder](../Notes/a08-FeatureEng.md#111-advanced-categorical-encoding)
  + similar to target encoding
  + replacing the category w/ the mean target value for that category
  + the order of observations in the dataset
  + the target probability: calculated only from the rows before it
  + similar to leave-one-out encoding but calculated the values
  + procedure
    + repeating training numerous times on shuffled copies of the dataset
    + averaging the results
  + Python: `from category_encoder import CatBoostEncoder`

### Leave-One-Out Encoder

+ [Leave-one-out encoder (LOO/LOOE)](../Notes/a08-FeatureEng.md#111-advanced-categorical-encoding)
  + an example of target-based encoding
  + preventing target data leakage, unlike other target-based methods
  + consisting of calculating the mean target of a given category $k$ for observation $j$ w/o using the corresponding target of $j$
  + calculating the per-category means w/ the typical target encoder
  + Python: `from category_encoder import LeaveOneOutEncoder`


### James-Stein Encoder

+ [James-Stein encoder](../Notes/a08-FeatureEng.md#111-advanced-categorical-encoding)
  + another example of a target-based encoder, defined for normal distribution
  + shrinking the average toward the overall average
  + intended to improve the estimation of the category's mean target by shrinking them towards a more median average
  + getting the mean target for category $k$

    \[ \widehat{x}^k = (1 - B) * \frac{n^+}{n} + B * \frac{y^+}{y} \]

    + $\frac{n^+}{n}$: the estimation of the category's mean target
    + $\frac{y^+}{y}$: the central average of the mean target
    + $B$: a hyperparameter, representing the power of shrinking
  + Python: `from category_encoders import JamesSteinEncoder`


## Transforming Variables


### Overview

+ [Transforming variables](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + assumption of linear and logistic regression: normal distribution w/ variable
  + in practice, real datasets following more a skewed distribution
  + purpose:
    + mapping skewed distribution to a normal distribution
    + increasing the performance of models
  + tools to estimate normality: histogram and Q-Q plot
  + most common-used methods
    + logarithmic transformation
    + square root transformation
    + reciprocal transformation
    + exponential or power transformation
    + Box-Cox transformation
    + Yeo-Johson transformation
  + Python: `from sklearn.preprocessing import FunctionTransformer`

+ [Q-Q plot](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + variable following a normal distribution $\implies$ the variable's values fall in a 45-degree line against the theoretical quantiles
  + Python snippet

    ```python
    # import the libraries
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pandas as pd

    # read data
    data_df = pd.read_csv("dataset.csv")

    # create and show the plot
    stats.probplot(data_df["variable"], dist="norm", plot=plt)
    plt.show()
    ```

+ [Representing feature relationships](../Notes/a18c-CreateFeat.md#mathematical-transforms)
  + relationship among numerical features usually expressed mathematical formulas
  + ratio:
    + features describing a car's engine in Automobile dataset
    + a variety of formulas for creating potentially useful new feature
    + e.g., `stroke ratio`: a measure of how efficient an engine vs how performant
  + combination
    + complicated formulation among features
    + the more complicated combination is, the more difficult it will be for a model learn
    + e.g., engine;s "displacement" as a measure of its power
  + data visualization
    + able to suggest transformations
    + often a "reshaping" of a feature through powers or logarithms
    + e.g., highly skewed distribution of `Windspeed` in US Accidents

+ [Counting features](../Notes/a18c-CreateFeat.md#counts)
  + features describing presence or absence
  + representing such features w/ binary (1 for presence , 0 for Absence) or Boolean (True or False)
  + dealing such features in sets
  + new "counts" features: aggregating such features
  + able to create Boolean values w/ dataframe built-in methods


+ Manipulating structure data
  + complex strings usually broken into simpler pieces
  + common examples of structure data
    + ID numbers: `'123-45-6789'`
    + Phone numbers: `'(999) 555-0123'`
    + Street addresses: `'8241 Kaggle Ln., Goose City, NV'`
    + Internet addresses: `'http://www.kaggle.com'`
    + Product codes: `'0 36000 29145 2'`
    + Dates and times: `'Mon Sep 30 07:06:05 2013'`
  + able to apply string methods, like `split`, directly to columns
  + able to join simple features into a composed feature



### Logarithmic Transformation

+ [Logarithmic transformation](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + formula: $ f(x) = \ln(x), x > 0$
  + simplest and most popular among the different types of transformations
  + involving a substantial transformation that significantly affects distribution shape
  + making extremely skewed distribution less skewed, especially for right-skewed distributions
  + constraint: only for __strictly positive__ numbers
  + Python: `logarithm_transformer = FunctionTransformer(np.log, validate=True)`


### Square Root Transformation

+ [Square root transformation](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + formula: $f(x) = \sqrt{x}, x \ge 0$
  + simple transformation w/ average effect on distribution shape
  + weaker than logarithmic transformation
  + used for reducing right-skewed distributions
  + advantage: able to apply to zero values
  + constrain: only for positive numbers
  + Python: `sqrt_transformer = FunctionTransformer(np.sqrt, validate=True)`  
  + alternative: cubic root function


### Recipocal Transformation

+ [Recipocal transformation](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + formula: $f(x) = \frac{1}{x}, x \ne 0$
  + a powerful transformation w/ a radical effect
  + positive reciprocal: reversing the order among values of the same sign $\to$ large values $\to$ smaller
  + negative reciprocal: preserving the order among values of the same ign
  + constraint: not defined for zero
  + Python: `recipocol_transformer = FunctionTransformer(np.recipocol, validate=True)`
  + alternative; negative reciprocal function


### Exponential or Power Transformation

+ [Exponential or Power transformation](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + formula:

    \[ \begin{align*} f(x) &= x^2 \\ g(x) &= x^3 \\ h(x) &= x^n \\ k(x) &= \exp(x) \end{align*} \]

  + a reasonable effect on distribution shape
  + applying power transformation (power of two usually) to reduce left skewness
  + Python: `exponential_transformer = FunctionTransformer(lambda x: x**(3), validate=True)`


### Box-Cox Transformation

+ [Box-Cox transformation](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + formula: ($x_i > 0$)

    \[ x_i^{(\lambda)} = \begin{cases} 
      \frac{x_i^{\lambda}-1}{\lambda} & \text{if } \lambda \ne 0, \\ 
      \ln(x_i) & \text{if } \lambda = 0
    \end{cases} \]

  + one of the most successful transformations
  + evolution of the exponential transformation by looking through various exponents instead of trying them manually
  + process
    + searching and evaluating all the other transformations and choosing the best one
    + hyperparameter ($\lambda$): varying over the range (-5, 5)
    + examining all values of $\lambda$
    + choosing the optimal value (resulting in the best approximation to a normal distribution)
  + constraint: only for positive number
  + Python: `boxcox_transformer = PowerTransformer(method='box-cox', standardize=False)`



### Yeo-Johnson Transformation

+ [Yeo-Johnson transformation](../Notes/a08-FeatureEng.md#6-transforming-variables)
  + formula

    \[ x_i^{(\lambda)} = \begin{cases}
      [(x_i + 1)^\lambda - 1] / \lambda & \text{if } \lambda \ne 0, x_i \ge 0, \\
      \ln(x_i) + 1 & \text{if } \lambda = 0, x_i \ge 0, \\
      -[(-x_i + 1)^{2-\lambda} - 1]/(2-\lambda) & \text{if } \lambda \ne 2, x_i < 0, \\
      -\ln(-x_i + 1) & \text{if } \lambda = 2, x_i < 0
    \end{cases} \]

  + an adjustment to the Box-Cox transformation
  + able to apply to negative numbers
  + Python: `yeo_johnson_transformer = PowerTransformer(method='yeo-johnson', standardize=False)`


## Group Transforms

+ [Group transforms](../Notes/a18c-CreateFeat.md#group-transforms)
  + aggregating information across multiple rows grouped by some category
  + good practice: category interaction $\to$ group transform over the category
  + aggregation function to combine two features
    + grouping categorical feature
    + aggregating feature values
  + built-in dataframe method as aggregation function, e.g., `mean`, `max`, `min`, `median`, `var`, `std`, `count`
  + preventing inappropriate data splitting
    + using training and validation splits to preserve their independence
    + best practice
      + creating a grouped feature using only the training set
      + joining it to the validation set
      + using the validation set's `merge` set after creating a unique set of values w/ `drop_duplicates` on the training set

+ [Tips for creating features](../Notes/a18c-CreateFeat.md#group-transforms)
  + linear models
    + learning sums and differences naturally
    + unable to learn anything more complex
  + ratio:
    + difficult for most models to learn
    + ratio combinations leading to some easy performance gains
  + normalization
    + linear models and Neural Nets generally doing better w/ normalized features
    + NN: features scaled to values not too far from 0
    + tree-based models also beneficial from normalization but limited
  + tree models
    + learning to approximate almost any combination of features
    + combination especially important when limited data
  + counts:
    + especially helpful for tree models
    + tree models w/o natural way of aggregating information across many features at once




## Variable Discretization

### Overview

+ [Variable Discretization](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + transforming a continuous variable into a discrete one
  + essentially creating a set of contiguous intervals spanning the variable's value range
  + binning = discretization, bin = interval
  + approaches
    + supervised approach
      + discretization w/ decision tree
    + unsupervised approaches
      + equal-width discretization
      + equal-frequency discretization
      + K-means discretization
    + other
      + custom discretization

+ [Using the newly-created discrete variable](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + usually encoding w/ ordinal, i.e., integer encoding as 1, 2, 3, tec.
  + two major methods
    + using the value of the interval straight away if using intervals as numbers
    + treating numbers as categories, applying any of the encoding technique that creates a monotone relationship w/ the target
  + advantageous way of encoding bins: treating bins as categories to use an encoding technique that creates a monotone relationship w/ the target

### Equal-Width Discretization

+ [Equal-width discretizatoin](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + the most simple form of discretization
  + dividing the range of possible values into $N$ bins of the same width
  + width of intervals: $\text{width} = \frac{\max - \min}{N}$
  + $N$ parameter:
    + the number of intervals
    + determined experimentally - no rules of thumb here
  + considerations
    + not improving the values spread
    + handling outliers
    + creating a discrete variable
    + useful when combined w/ categorical encoding
  + Python: `from sklearn.preprocessing import KBinsDiscretizer`


### Equal-Frequency Discretization

+ [Equal-frequency discretization](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + dividing the scope of possible values of the variable into $N$ bins
  + each bin holding the same number (or approximately the same number) of observation
  + considerations
    + the interval boundaries corresponding to quantile
    + improving the value spread
    + handling outliers
    + disturbing the relationship w/ the target
    + useful when combined w/ categorical encoding
  + Python: `from sklearn.preprocessing import KBinsDiscretizer`


### K-Means Discretization

+ [K-means discretization](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + consisting of applying k-means clustering to the continuous variable
  + bin = cluster
  + reviewing the k-mean algorithms
    1. creating $K$ random points, center of cluster
    2. associating every data point w/ the closest center (using some distance metric, like Euclidean distance)
    3. re-computing each center position in the center of its associated points
    4. repeat step 2 & 3 until convergence
  + tutorials about k-means
    + [Mathematics behind K-means](https://tinyurl.com/yy34qu7y)
    + [K-Means using Sklearn and Python](https://tinyurl.com/yybc7swq)
    + [Visualizing K-means](https://tinyurl.com/y32ru3yn)
  + considerations
    + not improving the values spread
    + handling outliers, though outliers may influence the centroid
    + creating a discrete variable
    + useful when combined w/ categorical encoding
  + Python: `from sklearn.preprocessing import KBinDiscretizer`

+ [Unsupervised learning algorithms](../Notes/a18d-KMeans.md)
  + not making use of a target
  + purpose:
    + learning some property of the data
    + representing the structure of the features in a certain way
  + a "feature discovery" technique in terms of feature engineering

+ [Clustering](../Notes/a18d-KMeans.md)
  + the assigning of data points to groups
  + group based on how similar the points to each other
  + making "birds of a feather flock together"
  + used for feature engineering: an attempt to discover
    + groups of customers representing a market segment
    + geographic areas sharing similar weather patterns
  + adding a feather of cluster labels $\to$ untangle complicated relationships of space and proximity

+ [Feature w/ clustered labels](../Notes/a18d-KMeans.md#cluster-labels-as-a-feature)
  + clustering: like a traditional "binning" or "[discretization](https://bit.ly/2PBQ1VB)" transform
  + multiple features:
    + a.k.a. vector quantization
    + multi-dimensional binning
  + motivation for adding cluster labels
    + clusters breaking up complicated relationships across features in simple chunks
    + applying divided and conquer strategy to handle different clusters
    + learning the simpler chunks one-by-one instead learning the complicated one

+ [Clustering algorithms](../Notes/a18d-KMeans.md#k-means-clustering)
  + classification
    + how they measure "similarity" or "proximity"
    + what kinds of features working with
  + k-means: intuitive and easy to apply in a feature engineering context
  + selection of algorithm: depending on application

+ [K-means clustering](../Notes/a18d-KMeans.md#k-means-clustering)
  + measuring similarity using ordinary straight-line distance (Euclidean distance)
  + creating clusters by placing a number of points, called __centroids__, inside the feature space
  + each point assigning to the cluster whatever centroid it closest to
  + $k$: the parameter about how many centroids
  + Voronoi tessallation
    + imaging each centroid capturing points through a sequence of radiating circles
    + a line formed w/ the overlapped sets of circles from competing centroids
    + analogy: which cluster to assigned w/ future data

+ [K-means w/ scikit-learn's implementation](../Notes/a18d-KMeans.md#k-means-clustering)
  + hyperparameters: `n_clusters`, `max_iter`, `n_init`
  + procedure
    + init: randomly initializing `n_cluster` centroids
    + assign points to the nearest cluster centroid
    + move each centroid to minimize the distance to its points
    + repeat the above 2 steps until the centroids converged or reaching the maximum iteration (`max_iter`)
  + issue: initial random position of the centroids $\to$ poor clustering
  + solution:
    + repeat the algorithm a number of times (`n_init`)
    + return the clustering w/ the least total distance btw each point and its centroid, the optimal clustering
  + increasing the `max_iter` for a large number of clusters
  + increasing `n_init` for a complex dataset
  + sensitive to scale:
    + rescale or normalize data w/ extreme values
    + depending on domain knowledge and predicting target
    + rule of thumb: feature
      + ready directly comparable, e.g., test result at different time $\to$ not rescale
      + not on comparable scales, e.g., height and weight $\to$ usually benefit for rescale
      + not clear $\to$ use common sense
    + features w/ larger values weighted more heavily
    + comparing different schemes through cross-validation probably helpful
  + best partitioning for a set of features depending on
    + model used
    + what to predict



### Discretization w/ Decision Trees

+ [Discretization w/ Decision Trees](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + consisting of a decision tree to identify the optimal bins
  + a decision tree making a decision $\to$ assigning an observation to one of $N$ end leaves
  + generating a discrete output, the predictions at each of its $N$ leaves
  + procedure
    + training a decision tree of limited depth (2, 3, or 4) using only the variable to discretize and the target
    + replacing the variable's value w/ the output returned by the tree
  + considerations
    + not improving the values spread
    + handling outliers since trees are robust to outliers
    + creating a discrete variable
    + prone to overfitting
    + cost some time to tune the parameters effectively (e.g., tree depth, the minimum nimber of samples in one partition minimum info gain)
    + observations within each bin more similar to each other
    + creating a monotonic relationship
  + Python: `from sklearn.proprecessing import DecisionTreeClassifier`


### Custom Discretization

+ [Custom discretization](../Notes/a08-FeatureEng.md#7-variable-discretization)
  + engineering variables in a custom environment (i.e., for a particular business use case)
  + determining the intervals where the variable divided so that it makes sense for the business
  + example: Age divided into groups like [0-10] as kids, [10-25] as teenagers, and so on
  + Python: `labels = ['0-10', '10-25', '25-65', '>65']`



### Principal Component Analysis

+ [Principal Component Analysis and feature engineering](../Notes/a18e-PCA.md#introduction)
  + a partitioning of the variation in the data
  + a great tool to help to discover important relationship in the data
  + used to create more informative features
  + typically applied to [standardized](https://bit.ly/2S9yMM2) data
  + variation meaning
    + standardized data: correlation
    + non-standardized data: covariance

+ [Visualization for Principal Component Analysis](../Notes/a18e-PCA.md#principal-component-analysis)
  + axes of variation
    + describing the ways the abalone tend to different from one another
    + axes: perpendicular lines along the natural dimensions of the data
    + each axis for one original feature
  + idea of PCA: instead of describing the data w/ the original features, describing it w/ axes of variation
  + dataset: [Abalone data set](https://www.kaggle.com/rodolfomendes/abalone-dataset)
    + physical measurements taken from several thousand Tasmanian abalone
    + only focusing on `Height` and `Diameter` of their shells
  + axes of variation for abalone
    + Size component
      + the longer axis
      + small height and small diameter (lower left) contrasted w/ large height and large diameter (upper right)
    + Shape component
      + the shorter axis
      + small height and large diameter (flat shape) contrasted w/ large height and small diameter (round shape)

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://www.kaggle.com/ryanholbrook/principal-component-analysis" ismap target="_blank">
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/rr8NCDy.png"
          alt   = "Axes of variation with abalone"
          title = "Axes of variation with abalone"
        >
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/XQlRD1q.png"
          alt   = "The principal components become the new features by a rotation of the dataset in the feature space."
          title = "The principal components become the new features by a rotation of the dataset in the feature space."
        >
      </a>
    </div>

+ [PCA as new features](../Notes/a18e-PCA.md#principal-component-analysis)
  + new features PCA: liner combinations (weighted sums) of the original features

    <code> df["Size"] = 0.707 * X["Height"] + 0.707 * X["Diameter"]</code><br>
    <code> df["Shape"] = 0.707 * X["Height"] - 0.707 * X["Diameter"] </code>

    + principal components of the data: `Size`, `Shape`
    + loadings: weights, 0.707
  + number of principal components = features in the original dataset
  + component's loadings expressed through signs and magnitudes
    + table of loadings

      <table>
        <thead>
          <tr><th>Features \ Components</th><th>Size (PC1)</th><th>Shape (PC2)</th></tr>
        </thead>
        <tbody>
          <tr style="text-align: right;"><td>Height</td><td>0.707</td><td>0.707</td></tr>
          <tr style="text-align: right;"><td>Diameter</td><td>0.707</td><td>-0.707</td></tr>
        </tbody>
      </table>

    + `Size` component: `Height` and `Diameter` varying in the same direction (same sign)
    + `Shape` component: `Height` and `Diameter` varying in opposite direction (opposite sign)
    + all loadings w/ the same magnitude $\to$ features contributing equally

+ [Percent of explained variance](../Notes/a18e-PCA.md#principal-component-analysis)
  + PCA represents the amount of variation in each component
  + more variation in the data along the `Size` component than along the `Shape` component
  + making the precise comparison though each component's percent of explained variation
  + `Size` component: the majority of variation btw `Height` and `Diameter`
  + the amount of variance in a component
    + not necessarily correspond to how good it is as a predictor
    + depending on what to predict

+ [Ways to use PCA for feature engineering](../Notes/a18e-PCA.md#pca-for-feature-engineering)
  + use as a __descriptive technique__
    + computing the MI scores for the components
    + what kind of variation most predictive of the target
    + ideas for kinds of features to create
      + `Size`: product of `Height` and `Diameter`
      + `Shape`: ratio of `Height` and `Diameter`
    + try clustering on one or more of the high scoring components
  + use __components__ themselves as features
    + the components exposing the variational structure of the data directly
    + often more informative than the original features
    + use cases
      + __dimensionality reduction__
        + highly redundant features, in particular, multicolinear
        + partitioning out the redundancy into one or more near-zero variance components
      + __anomaly detection__
        + unusual variation often w/ the low-variance components
        + unusual variation: not apparent from the original features
        + components highly informative in an anomaly or outlier detection task
      + __noise reduction__
        + sensor reading often w/ common background noise
        + able to collect the (informative) signal into a smaller number of features while leaving out the noise
        + boosting the signal-to-noise ratio
      + __decorrelation__
        + ML sometimes struggling w/ highly-correlated features
        + transforming correlated features into uncorrelated components

+ [PCA best practices](../Notes/a18e-PCA.md#pca-for-feature-engineering)
  + only working w/ numeric features, including continuous quantities or counts
  + sensitive to scale: standardizing data before applying PCA
  + removing or constraining outliers for undue influence on the results





## Handling Outliers


### Overview

+ [Outliers](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + a data point significantly different from the remaining data
  + an observation deviating so much from the other observations
  + arousing suspicion that a different mechanism produced it
  + handling outliers
    + trimming: simply removing the outliers from dataset
    + imputing: treating outliers as missing data and applying missing data imputation techniques
    + discretization: placing outliers in edge bins w/ higher or lower values of the distribution
    + censoring: capping the variable distribution at the maximum and minimum values


### Detection

+ [Detecting Outliers](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + using visualization plots like box plot and scatter plot
    + box plot: black points as outliers
    + scatter plot: most points located in center but one far from center might be outlier
  + using a normal distribution (mean and s.d.)<br/>
    about 99.7% of the data lie within 3 s.d. of the mean


### IQR Proximity Rule

+ [Inter-quantal range proximity rule](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + Interquartile range (IQR)
    + used to build boxplot graphs
    + dividing data into four parts and each part is a quartile
    + IQR = he difference between the 3rd quantile Q3 (75%) and the 1st quantile or Q1 (25%)
  + outliers defined w/ IQR
    + below Q1 - 1.5 x IQR
    + above Q3 + 1.5 x IQR


### DBSCAN

+ [Density-Based Spatial Clustering of Application w/ Noise (DBSCAN)](../Notes/a08-FeatureEng.md#113-advanced-outlier-detection)
  + a clustering algorithm used to group points in the same clusters
  + choosing two htperparameters
    + `epsilon` > 0 for the distances btw points: the maximum distance btw two examples for one to be considered in the neighborhood of the other
    + `min_samples` $\in \Bbb{N}$: serving as the number of samples in a neighborhood for a point to be considered as a core point
  + algorithm
    1. randomly selecting a point not assigned to a cluster
    2. determining if it belongs to a cluster by seeing if there are at least `min_samples` points around it within `epsilon` distance
    3. creating a cluster of this point w/ all other samples within `epsilon` distance to it
    4. finding all points that are within `epsilon` distance of each point in that cluster and adding them to the same cluster
    5. finding all points that are within `epsilon` distance of all recently added points and adding these to the same cluster
    6. repeating steps 1~5
  + all points not reachable from any other point aare considered __outliers__
  + Python: `from sklearn.cluster import DBSCAN`


### Isolation Forests

+ [Isolation forests](../Notes/a08-FeatureEng.md#113-advanced-outlier-detection)
  + built on the foundation of decision trees and using tree assemble methods
  + algorithm examining how quickly a point isolated
  + normal point: more partition to isolate
  + outliers
    + isolated quickly in the first splits
    + less frequent than regular observations
    + lying further away from the regular observations in the feature space
    + w/ random partitioning identified closer to the root of the tree
  + Python: `from sklearn.ensemble import IsolatedForest`


### Local Outlier Factor

+ [Local outlier factor (LOF)](../Notes/a08-FeatureEng.md#113-advanced-outlier-detection)
  + measuring the local variation of density of a given sample taking just its neighbors into considerations and not the global data distribution
  + outlier: the density around that points is significantly different from the density around its neighbors
  + algorithm
    1. calculating the distances btw a randomly selected point and every other point
    2. finding the farest $k$ cloest point (`k-th` nearest-neighbor)
    3. fidning the other $k$ closest points, like a normal KNN
    4. calculating the point density (local reachability density) using the inverse of the average distance btw that point and its neighbors (the lower the density, the farther the point is from its neighbors)
    5. calculating the LOF, essentially the average local reachability density of the neighbors divided by the point's own local reachability density
  + imterpretation of the final LOF score
    + LOF(k) = 1: similar density as neighbors
    + LOF(k) < 1: higher density than neighbors (inlier)
    + LOF(k) > 1: lower density than neighbors (outlier)
  + Python: `from sklearn.neighbor import LocalOutlierFactor`


### Trimming

+ [Trimming outliers](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + merely removing outliers from the dataset
  + deciding on a metric to determine outliers
  + considerations
    + fast method
    + removing a significant amount of data
  + Python: `outliers = np.where(data_df[variable] > upper, True, np.where(data_df[variable] < lower, True, False))` & `data_df = data_df.loc[~(outliers,)]`


### Censoring

+ [Censoring outliers](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + setting the maximum and/or the minimum of the distribution at any arbitrary value
  + values bigger or smaller than the arbitrarily chosen value are replaced by the value
  + concerns about capping
    + not removing data
    + distorting the distributions of the variables
  + arbitrarily replacing the outliers
  + inter-quantal range proximity rule
  + Gaussian approximation
  + using quantiles


### Imputer

+ [Imputing outliers](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + treating outliers as missing data
  + refer to [imputing variable techniques](#imputing-missing-values)


### Transformation

+ [Transforming outliers](../Notes/a08-FeatureEng.md#8-handling-outliers)
  + applying some mathematical transformations, such as log transformation
  + refer to [transforming variables](#transforming-variables)



## Feature Scaling


### Overview

+ [Feature scaling](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + methods used to normalize the range w/ values of independent variables
  + ways to set the feature value range within a similar scale
  + concerns
    + the scale of the variable directly influencing the regression coefficient
    + variable w/ a more significant magnitude dominate over the ones w/ a smaller magnitude range
    + gradient descent converges faster when features are on the same scales
    + feature scaling helps decrease the time to find support vector of SVMs
    + Euclidean distances are sensitive to feature magnitude
  + algorithms sensitive to feature magnitude
    + linear and logistic regression
    + Neural networks
    + support vector machine
    + KNN
    + K-means clustering
    + linear discriminant analysis (LDA)
    + principle component analysis (PCA)
  + algorithm insensitive to feature magnitude
    + classification and regression trees
    + random forest
    + gradient boosted trees
  + scaling methods
    + mean normalization
    + standardization
    + robust scaling (scaling to median and IQR)
    + robust to maximum and minimum
    + scale to absolute maximum
    + scale to unit norm


### Mean Normalization

+ [Mean normalization](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + centering the variable at 0 and rescaling the variable's value range to the range -1 and 1
  + scaling formula:

    \[ \overline{x} = \frac{X - \text{mean}(X)}{\max(X) - \min(X)} \]

  + not normalizing the variable distribution
  + characteristics
    + centering the mean at 0
    + different resulting variance
    + modifying the shape of original distribution
    + normalizing the minimum and maximum values w/ the range [-1, 1]
    + preserving outliers if existed


### Standardization

+ [Standardization](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + centering the variable at 0 and standarizing the variance to 1
  + scaling formula:

    \[ \overline{x} = \frac{X - \text{mean}(X)}{\text{std}(X)} \]

  + not normalized the variable distribution
  + characteristics
    + scaling the variance at 1
    + centering the mean at 0
    + preserving the shape of the original distribution
    + preversing outliers if existed
    + minimum and maximum values varying
  + Python: `import sklearn.preprocessing import StandardScaler`


### Robust Scaling

+ [Robust scaling (scaling to median and IQR)](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + using median instead of mean
  + scaling formula

    \[ \overline{x} = \frac{X - \text{median}(X)}{Q3(x) - Q1(x)} \]

  + characteristics
    + centering the median at 0
    + resulted variance varying across variables
    + not preserving the shape of the original distribution
    + minimum and maximum values varying
    + robust to outliers
  + Python: `from sklearn.preprocessing import RobustScaler`


### Min-Max Scaling

+ [Min-Max scaling](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + compressing the value between 0 and 1
  + scaling formula:

    \[ \overline{x} = \frac{X - \min(X)}{\max(X) - \min{X}} \]

  + not normalizing the variable distribution
  + characteristics
    + not centering the mean at 0
    + making the variance vary across variables
    + not maintaining the shape of the original distribution
    + maximum and minimum values in the range of [0, 1]
    + sensitive to outliers
  + Python:`from sklearn.preprocessing import MinMaxScaler`


### Maximum Absolute Scaling

+ [Maximum absolute scaling](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + scaling the variable btw -1 and 1
  + scaling formula:

    \[ \overline{x} = \frac{X}{\max(X)} \]

  + characteristics
    + the resulting mean not centered
    + not scaling the variance
    + sensitive to outliers
  + Python: `from sklearn.preporcessing import MaxAbsScaler`


### Scaling to Vector Unit Norm

+ [Scaling to vector unit norm](../Notes/a08-FeatureEng.md#9-feature-scaling)
  + scale to vector unit norm
  + scaling formula:

    \[ \overline{x} = \frac{X}{\|X\|} \]
  
  + the distance measure for unit norm
    + [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) (L2 norm): $L2(X) = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$
    + [Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) (L1 norm): $L1(X) = |x_1| + |x_2| + \cdots + |x_n|$
  + characteristics
    + the length of the resulting vector is 1
    + normalizing the feature vector and not the observation vector
    + sensitive to outlier
    + recommended for text classification and clustering
  + Python: `from sklearn.preprocessing import Normalizer`


## Handling Date-Time and Mixed Variables


### Date and Time Variables

+ [Engineering variables of date and time](../Notes/a08-FeatureEng.md#10-handling-date-time-and-mixed-variable)
  + date and time: good resource of information
  + each number corresponding to a specific part of the date and time
  + date-time varables in many formats, e.g.,
    + time of birth: 19:45:57
    + birthday date: 16-08-1995, 16-04-1997
    + invoice date: 03-06-2020 19:47:29
  
+ Time/date components: `pd.Series.dt`

  <table class="colwidths-given table"><table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 55vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y3fa59ex">Date / Time Components</a></caption>
  <thead>
  <tr style="font-size: 1.2em; vertical-align:middle"">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Property</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr><td>year</td><td>The year of the datetime</td>  </tr>
    <tr><td>month</td><td>The month of the datetime</td>  </tr>
    <tr><td>day</td><td>The days of the datetime</td>  </tr>
    <tr><td>hour</td><td>The hour of the datetime</td>  </tr>
    <tr><td>minute</td><td>The minutes of the datetime</td>  </tr>
    <tr><td>second</td><td>The seconds of the datetime</td>  </tr>
    <tr><td>microsecond</td><td>The microseconds of the datetime</td>  </tr>
    <tr><td>nanosecond</td><td>The nanoseconds of the datetime</td>  </tr>
    <tr><td>date</td><td>Returns datetime.date (does not contain timezone information)</td>  </tr>
    <tr><td>time</td><td>Returns datetime.time (does not contain timezone information)</td>  </tr>
    <tr><td>timetz</td><td>Returns datetime.time as local time with timezone information</td>  </tr>
    <tr><td>dayofyear</td><td>The ordinal day of year</td>  </tr>
    <tr><td>weekofyear</td><td>The week ordinal of the year</td>  </tr>
    <tr><td>week</td><td>The week ordinal of the year</td>  </tr>
    <tr><td>dayofweek</td><td>The number of the day of the week with Monday=0, Sunday=6</td>  </tr>
    <tr><td>weekday</td><td>The number of the day of the week with Monday=0, Sunday=6</td>  </tr>
    <tr><td>quarter</td><td>Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.</td>  </tr>
    <tr><td>days_in_month</td><td>The number of days in the month of the datetime</td>  </tr>
    <tr><td>is_month_start</td><td>Logical indicating if first day of month (defined by frequency)</td>  </tr>
    <tr><td>is_month_end</td><td>Logical indicating if last day of month (defined by frequency)</td>  </tr>
    <tr><td>is_quarter_start</td><td>Logical indicating if first day of quarter (defined by frequency)</td>  </tr>
    <tr><td>is_quarter_end</td><td>Logical indicating if last day of quarter (defined by frequency)</td>  </tr>
    <tr><td>is_year_start</td><td>Logical indicating if first day of year (defined by frequency)</td>  </tr>
    <tr><td>is_year_end</td><td>Logical indicating if last day of year (defined by frequency)</td>  </tr>
    <tr><td>is_leap_year</td><td>Logical indicating if the date belongs to a leap year</td>  </tr>
  </tbody>
  </table>

### Mixed Variables

+ [Engineering mixed variables types](../Notes/a08-FeatureEng.md#10-handling-date-time-and-mixed-variable)
  + solution: extracting the categorical part in one variable and the numerical part in a different variable
  + two special formats in a mixed variable
    + different observations
    + same observation

+ [Labels and numbers in different observations](../Notes/a08-FeatureEng.md#10-handling-date-time-and-mixed-variable)
  + either numbers or labels in their values
  + example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick="window.open('https://tinyurl.com/y6ss46t3')"
        src    ="https://tinyurl.com/yxq3bunr"
        alt    ="Example of mixed numerical & labels w/ different observations"
        title  ="Example of mixed numerical & labels w/ different observations"
      />
    </figure>

  + resulted in a lot of nan values $\to$ applying missing data techniques
  + Python: `data_df[mixed_num] = pd.to_number(data_df[mixed], errrors='coerce', downcast='integer')`

+ [Labels and numbers in the same observation](../Notes/a08-FeatureEng.md#10-handling-date-time-and-mixed-variable)
  + variables containing both numbers and labels in their values
  + tricky to extract categorical and numerical values
  + depending on a number of factors, e.g., number of letters, locations, etc.
  + Python: `data_df[mixed_num] = data_df[mixed].str.extract('(\d+)')`
  + Regular Expression (regex): detect patterns in mixed variables and easily extract categorical and numerical parts


### Periodicity

+ [Cyclical feature problem](../Notes/a08-FeatureEng.md#10-handling-date-time-and-mixed-variable)
  + cyclical or periodic data
    + data following a cycle
    + e.g., hours, minutes, seconds, days of the month, days of weeks, and months
  + preserving the cyclical info in datasets for models to learn accurate and behave correctly
  + one solution: projecting the cyclical feature on a circle, specifically the __unit circle__
  + unit circle: using $\cos$ and $\sin$ functions to express the periodicity
  + the $\sin$ and $\cos$ as new created features to transform the cyclical feature
  + Python: `data_df['payment_hour_sin'] = np.sin(data_df['payment_hour'] * (2. * np.pi / 24.))` & `data_df['payment_hour_cos'] = np.cos(data_df['payment_hour'] * (2. * np.pi / 24.))`


## Advanced Feature Engineering

### Automated feature engineering

+ [Deep feature synthesis](../Notes/a08-FeatureEng.md#114-automated-feature-engineering)
  + automatically generating features for relational dataset
  + relationships in the data to a base field
  + sequentially applying mathematical functions along that path to create the final feature

+ [Featuretools](../Notes/a08-FeatureEng.md#114-automated-feature-engineering)
  + an open-source framework for implementing automated feature engineering
  + a comprehensive tool intended to make the feature generation process fast-forward
  + components
    + deep feature synthesis: the backbone of featuretools
    + entities: multiple entities result in an EntitySet
    + feature primitives: Deep Feature Synthesis applied to EntitySet - transfrmations or aggregations like count or average

### Geospatial data

+ [Geospatial feature](../Notes/a08-FeatureEng.md#115-engineering-geospatial-data)
  + represented as longitude and latitude
  + features influcing predictive model's results by a large margin if well-engineered
  + procedure
    + visualizing the features to obtain valuable insight
    + exploring different methods to extract and design new features


### Resampling imblanced data

+ [Resampling](../Notes/a08-FeatureEng.md#116-resampling-imblanced-data)
  + issue: classes not represented equally
  + causing problems for some algorithms
  + resampling engineering and reducing this effect on machine learning algorithms


## Examples

+ [Concrete Formulations - Counting Feature Used](../Notes/a18a-FeatureEng.md#example---concrete-formulations)
  + task: illustrating how adding a few synthetic to dataset to improve the predictive performance of a random forest model
  + dataset: [Concrete](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength)
    + containing a variety of concrete formulations and the resulting product's comprehensive strength
    + comprehensive strength: a measure of how much load that kind of concrete can bear

+ [1985 Automobiles - Mutual Information](../Notes/a18b-MutualInfo.md#example---1985-automobiles)
  + dataset: [Automobile dataset](https://www.kaggle.com/toramky/automobile-dataset)
  + goal: predicting a car's `price` (the target) from 23 pf the car's features
  + task: ranking the features w/ __mutual information__ and investigating the results by data visualization

+ [Ames House Price - Mutual Information](/Notes/a18b-MutualInfo.md#exercise-mutual-information)
  + dataset: [Ames data set](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
  + task: identify initial set of features w/
    + mutual information
    + interaction plots

+ [Ames House Price - creating features](../Notes/a18c-CreateFeat.md#exercise)
  + dataset: [Ames dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
  + task: developing mathematical transforms
    + features describing areas
    + same units (square-feet)
    + using `XGBoots` (a tree-based model) $\to$ focus on ratios and sums
  + create mathematical transforms
    + `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
    + `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
    + `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`

+ [California Housing - K-Means](/Notes/a18d-KMeans.md#example---california-housing)
  + data set: [California Housing](https://www.kaggle.com/camnugent/california-housing-prices)
    + `Latitude` and `Longitude`: natural candidates for k-means clustering
    + `MedInc`: creating economic segments in different regions of California
  + training w/ K-means

+ [1985 Automobiles - PCA](/Notes/a18e-PCA.md#example---1985-automobiles)
  + dataset: [Automobile](https://www.kaggle.com/toramky/automobile-dataset)
  + task: descriptive technique to discover features

+ [PCA for feature engineering](../Notes/a18e-PCA.md#exercise)
  + dataset: [Ames](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
  + task:
    + using PCA results to discover one or more new features
    + new features to improve the performance of the model
      + inspired by the loadings
      + using the components themselves as features


+ [MovieLens1M - Target Encoding](../Notes/a18f-TargetEnc.md#smoothing)
  + dataset: [MovieLens1M](https://www.kaggle.com/grouplens/movielens-20m-dataset)
    + 1 million movie rating by users of the MovieLens website
    + features describing each user and movie
  + tasks:
    + identifying features for encoding
    + applying M-estimate encoding

+ [Ames - Target Encoding](../Notes/a18f-TargetEnc.md#exercise)
  + dataset: [Ames](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
  + task: encode `Ratings` w/ `SalePrice`

+ [House Prices - XGBoost](../Notes/a18g-HousePrices.md)
  + dataset: [Ames](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
  + task: predict `SalePrice` w/ XGBoost



