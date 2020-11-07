# Hands-on with Feature Engineering Techniques


Author: Younes Charfaoui

Organization: Heartbeat

The set of articles in this series:

+ [Broad Introduction](https://tinyurl.com/y3br6dyb)
+ [Variable Types](https://tinyurl.com/y6egl9qc)
+ [Common Issues in Datasets](https://tinyurl.com/y2ngptgp)
+ [Imputing Missing Values](https://tinyurl.com/y6aye4ar)
+ [Categorical Variables Encoding](https://tinyurl.com/y6yq38cg)
+ [Transforming Variables](https://tinyurl.com/y2mhvfrm)
+ [Variables Discretization](https://tinyurl.com/y2qzj6q7)
+ [Handling Outliers](https://tinyurl.com/y4rljbj3)
+ [Feature Scaling](https://tinyurl.com/y6s3c788)
+ [Handling Time-Date and Mixed Variables](https://tinyurl.com/y6ss46t3)
+ [Advanced Feature Engineering Methods](https://tinyurl.com/y4op33zr)


## 1. Broad Introduction

+ Feature engineering
  + the process of using data domain knowledge to create features or variables
  + purpose: making ML algorithms effectively
  + very time consuming process
  + a number of processes
    + filling missing values within a variable
    + encoding categorical variables into numbers
    + variable transformation
    + creating or extracting new features from the ones available in the dataset

+ Purposes of feature engineering
  + raw data: messy and unsuitable for training a model
  + solution: data exploration and cleaning
  + involving changing data types and removing or imputing missing values
  + requirements: a certain understanding of the data acquired through exploration
  + solving these challenges and building high-performing models
  + solutions
    + removing outliers or specific features
    + creating features from the data that represent the underlying problem better
  + algorithms often hinging on how the input features engineered

+ Feature Engineering vs. Feature Selection
  + feature engineering:
    + <mark style="background-color: lightpink;">creating new features</mark> from the existing ones
    + helping ML model more effective and accurate predictions
  + feature selection
    + selecting from the feature pool
    + helping ML models to predict on target variables more efficiently
    + typical ML pipeline: completing feature engineering then feature selection
  

## 2. Variables Types

+ Variables
  + any characteristic, number, or quantity measured or counted
  + major types of variables
    + numerical variables
    + categorical variables
    + datetime variables
    + mixed variables
  + get the type of each variable from a Pandas dataframe

    ```python
    # import the pandas library
    import pandas as pd

    # read data
    data_df = pd.read_csv("dataset.csv")
    ```

+ Numerical variables
  + (predictably) numbers
  + categories of numerical variables
    + continuous variables
    + discrete variables
  + continuous variables
    + an uncountable set of values
    + probably containing any value within a given range
    + visualization
      + density plot
      + histogram
      + box plot
      + scatter plot
    + [example](src/a08-ex01-VisualNumerical.py):

      <figure style="margin: 0.5em; text-align: center;">
        <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
          onclick="window.open('https://tinyurl.com/y6egl9qc')"
          src    ="https://miro.medium.com/max/875/1*9N8ljqnCENz_IRr14dkmVg.png"
          alt    ="Visualization of continuous numerical variables: (a) box plot, (b) histogram, (c) density plot, & (d) scatter plot"
          title  ="Visualization of continuous numerical variables: (a) box plot, (b) histogram, (c) density plot, & (d) scatter plot"
        />
      </figure>

  + discrete variables
    + a finite number of values
    + integers, counts
    + visualization
      + count plot
      + pie chart
    + [example](src/a08-ex02-DiscreteVars.py)

      <figure style="margin: 0.5em; text-align: center;">
        <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
          onclick="window.open('https://tinyurl.com/y6egl9qc')"
          src    ="https://tinyurl.com/yxkpt2jp"
          alt    ="Visualization of discrete numerical variables: (a) density plot, (b) histogram"
          title  ="Visualization of discrete numerical variables: (a) density plot, (b) histogram"
        />
      </figure>

+ Categorical variables
  + selected from a group of categories
  + a.k.a. labels
  + categories
    + ordinal variables
    + nominal variables
  + ordinal variables
    + variables existed within meaningfully ordered categories
    + examples
      + student grades on an exame (A, B, C, or F)
      + days of the week (Sunday, Monday, Tuesday, ...)
  + nominal variables
    + not a natural order in the labels
    + e.g., country of birth - categorical but order not matter
  + in some scenarios, categorical variables coded as numbers when the data was recorded

+ Dates and Times
  + particular type of categorical variable
  + containing __dates__, __time__, or __data and time__
  + usually __not working__ w/ datetime variables in their __raw format__
    + date variables containing a considerable number of different categories
    + able to extract much more informa tion from datetime variables by pre-processing them correctly
  + date variable issues
    + containing dates not present in the dataset used to train the learning model
    + containing dates placed in the future, w.r.t. the dates in the training dataset
  
+ Mixed variables
  + containing both numbers and labels
  + occurring in a given dataset, especially when filling its values
  + example 
    + number not able to be retrieved for a variety of reasons, e.g., survey of income of a person
    + returning a label to represent the reason behind the issue, e.g., ERROR_OMMIT for client omit to answer


## 3. Common Issues in Datasets

+ General issues
  + missing data
  + categorical variable - cardinality
  + categorical variable - rare labels
  + linear model assumptions
  + variable distribution
  + outliers
  + feature magnitude

+ Missing data
  + when no data stored for a particular observation in  variable
  + basically just the absence of data
  + data missing for multiple reasons
    + lost: forgotten, omitted, lost, or not stored properly
    + not exist: e.g., a variable created from the di vision of 2 variables, and the denominator takes 0
  + many features not mandatory
  + solution: missing data imputation techniques
  + issues:
    + probably distort the original variable distribution
    + alter the way variables interact w/ each other
    + affect the machine learning model's performance $\gets$ many models make assumptions about the variable distribution
  + carefully choosing the right missing data imputation technique
  + main mechanisms lead to missing data
    + __missing data completely at random (MCAR)__
      + the probability of being missing: same for all the observation
      + no relationship: missing data and any other values
      + observed or missing, within the dataset
      + disregarding those cases not bias the inferences made
    + __missing data at random (MAR)__: the probability of an observation being missing depends on available information
    + __missing data not at random (MNAR)__: a mechanism or a reason why values introduced in the dataset
  + __labels__: the values of a categorical variable selected from a group of categories
  + __cardinality__: the number of different labels
  + cardinality on models: issues w/ multiple labels in a categorical variable
  + high cardinality
    + transform those labels into numbers
      + categories encoded as numbers
      + encoding techniques impact feature space and variable interactions
    + uneven distribution btw the train and test sets
      + some labels may appear only in the train set
      + over-fitting
    + labels may appear only in the test set: model not probably knowing how to iterpret these labels
    + too many labels tend to dominate over those w/ fewer labels
      + in particular, tree-based algorithms
      + a significant number of labels within a variable may introduce noise in the dataset
      + reducing cardinality may help improve model performance
    + models learn from the labels seen in the training set but not new, unseen labels unable to perform any calculations, resulting in errors

+ Categorical variable - rare labels
  + rare labels: appear only in a small proportion of the observation in a dataset
  + impacts and considerations on rare labels
    + causing overfitting and generalization problems
    + hard to understand the role of the rare label in the final prediction
    + removing rare labels may improve model performance
  + visualization: count plot

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/y2ngptgp')"
        src    ="https://tinyurl.com/y4qcft5t"
        alt    ="Count plot w/ image labels"
        title  ="Count plot w/ image labels"
      />
    </figure>

+ Linear model assumptions
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

      <figure style="margin: 0.5em; text-align: center;">
        <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
          onclick="window.open('https://tinyurl.com/y2ngptgp')"
          src    ="https://tinyurl.com/y3plgsbg"
          alt    ="Count plot w/ image labels"
          title  ="Count plot w/ image labels"
        />
        <figcaption>Q-Q Plot Example</figcaption>
      </figure>

    + Q-Q plot: if variable is normally distributed,, the values of the variable falls in a 45-degree line when plotted against the theoretical quantities
    + variable not normal distribution: non-linear transformation to fix
  + independent: observations independent of each other

+ Variable/Probability distribution
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

+ Outliers
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

+ Feature magnitude
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



## 4. Imputing Missing Values

+ Data imputation
  + the act replacing missing data w/ statistical estimates of missing values
  + goal: producing a complete dataset to use in the process of training ML models
  + python library: `feature-engine` to simply the process of imputing missing values
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

+ Mean and median imputation
  + replacing all occurrences of missing values (NA) within a variable w/ the mean and median of the variable
  + scenario
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
  + pros
    + easy to implement
    + easy way of obtaining complete datasets
    + used in production
  + example
    + data (Age): (29 $\to$ 29), (43 $\to$ 43), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lighgreen;">36.2</span>), (25 $\to$ 25), (34 $\to$ 34), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lighgreen;">36.2</span>), (50 $\to$ 50)
    + python code

      ```python
      from sklearn.impute import SimpleImputer

      # create the imputer, the strategy can be mean and median
      imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

      # apply the transformation to the train and test
      train_df = imputer.transform(train_df)
      test_df = imputer.transform(test_df)
      ```



