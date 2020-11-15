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
  + advantages
    + easy to implement
    + easy way of obtaining complete datasets
    + used in production
  + limitations
    + distortion of the original variable distribution and variance
    + distortion of the covariance w/ the remaining dataset variable
    + the higher the percentage of missing values, the higher the distortions
  + example
    + data (Age): (29 $\to$ 29), (43 $\to$ 43), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">36.2</span>), (25 $\to$ 25), (34 $\to$ 34), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">36.2</span>), (50 $\to$ 50)
    + Python snippet

      ```python
      from sklearn.impute import SimpleImputer

      # create the imputer, the strategy can be mean and median
      imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

      # apply the transformation to the train and test
      train_df = imputer.transform(train_df)
      test_df = imputer.transform(test_df)
      ```

+ Arbitrary value imputation
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
  + example: using 999 as an arbitrary value
    + data (Age): (29 $\to$ 29), (43 $\to$ 43), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">99</span>), (25 $\to$ 25), (34 $\to$ 34), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">99</span>), (50 $\to$ 50)
    + Python snippet

    ```python
    from sklearn.impute import SimpleImputer

    # create the imputer, w/ fill value 999 as the arbitrary value
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=999)

    # apply the transformation to the train and test
    train_df = imputer.transform(train_df)
    test_df = imputer.transform(test_df)
    ```

+ End of tail imputation
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
  + example:
    + data (Age): (29 $\to$ 29), (43 $\to$ 43), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">50</span>), (25 $\to$ 25), (34 $\to$ 34), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">50</span>), (50 $\to$ 50)
    + Python snippet

      ```python
      from feature_engine.missing_imputers import EndTailImputer

      # create the imputer
      imputer = EngTailImputer(distribution='gaussian', tail='right')

      # fit the imputer to the train set
      imputer.fit(train_df)

      # transform the data
      train_t_df = imputer.transform(train_df)
      test_t_df = imputer.transform(test_t_df)
      ```

+ Frequent category imputation
  + a.k.a. mode imputation
  + replacing all occurrences of missing values (NA) within a variable w/ the mode, or the most frequent value
  + scenario
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
  + example:
    + data (Gender): (Male $\to$ Male), (Male $\to$ Male), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">Male</span>), (Female $\to$ Female), (Male $\to$ Male), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">Male</span>), (Femal $\to$ Female)
    + Python snippet 

      ```python
      from sklearn.impute import SimpleImputer

      # create the imputer, w/ the most frequent as strategy to fill missing
      imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

      # fit the imputer to  the train data
      imputer.fit(train_df)

      # apply the transformation to the train and test
      train_df = imputer.transform(train_df)
      test_df = imputer.transform(test_df)
      ```

+ Missing category imputation
  + treating missing data as an additional label or category of the variable
  + create a new label or category by filling the missing observations w/ a Missing category
  + most widely used method of missing data imputation for categories variables
  + advantages
    + easy to implement
    + fast way of obtaining complete datasets
    + integrated into production
    + capturing the importance of "missingness"
    + no assumption mad on the data
  + limitations
    + small number of missing data $\to$ creating an additional category just adding another rare label to the variable
  + example: fill missing data w/ a new category, "Missing"
    + data (Gender): (Male $\to$ Male), (Male $\to$ Male), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">Missing</span>), (Female $\to$ Female), (Male $\to$ Male), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">Missing</span>), (Femal $\to$ Female)
    + Python snippet

      ```python
      from sklearn.impute import SimpleImputer

      # create the imputer, w/ most frequent as strategy to fill missing data
      imputer = SimpleImnputer(missing_value=np.nan, strategy='constant', fill_value="Missing")

      # fit the imputer to the train data
      # make sure to select only the categorical variable in the following train and test sets
      imputer.fit(train_df)

      # apply the transformation to the train and test
      train_df = imputer.transform(train_df)
      test_df = imputer.transform(test_df)
      ```

+ Complete case analysis (CCA)
  + discarding observations where values in any of the variables are missing
  + keep only those observations for which there's information in all of the dataset variables
  + observations w/ any missing data excluded
  + scenario
    + missing data complete at random (MCAR)
    + no more than 5% of the total dataset containing missing data
  + assumption 
    + missing data at random
  + advantages
    + simple to implement
    + no data manipulation required
    + preserving the distribution of the variables
  + limitation
    + excluding a significant fraction of the original dataset (if missing data significant)
    + excluding informative observations for the analysis (if data not missing at random)
    + create a biased dataset if the complete cases differ from the original data (if MAR or MNAR)
    + used  in production $\to$ not knowing how to handle missing data
  + example:
    + data (Gender): (Male $\to$ Male), (Male $\to$ Male), (<span style="color: pink;"> NA </span> $\to$ ), (Female $\to$ Female), (Male $\to$ Male), (<span style="color: pink;"> NA </span> $\to$ ), (Femal $\to$ Female)
    + Python snippet

      ```python
      # read data and apply the method
      data.dropna(inplace=True)
      ```

+ Missing indicator
  + an additional binary variable indicating whether the data was missing for an observation (1) or not (0)
  + goal: capture observations where data is missing
  + used together w/ methods assuming MAR
    + mean, median, mode imputation
    + random sample imputation
  + scenario
    + suitable for categorical and numberic variables
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
  + example:
    + data (Gender $\to$ Gender, Missing): (Male $\to$ Male, <span style="color: pink;">False</span>), (Male $\to$ Male, <span style="color: pink;">False</span>), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">Female, True</span>), (Female $\to$ Female, <span style="color: pink;">False</span>), (Male $\to$ Male, <span style="color: pink;">False</span>), (<span style="color: pink;"> NA </span> $\to$ <span style="color: lightgreen;">Male, True</span>), (Femal $\to$ Female, <span style="color: pink;">False</span>)
    + Python snippet

      ```python
      from sklearn.imput import MissingIndicator

      # create the object w/ missing only columns
      indicator = MissingIndicator(error_on_new=True, features='missing-only')
      indicator.fit(train_df)

      # print the column of the missing data
      print(train_df.columns[indicator.features_])

      # create a column name for each of the new Missing indicators
      indicator_columns = [column + '_NA_IND' for column in train_df.columns[indicator.features_]]
      indicator_df = pd.DataFrame(temporary, columns=indicator_columns)

      # create the final train data
      train_df = pd.concat([train_df.reset_index(), indicator_df], axis=1)

      # now the same for the test set
      temporary = indicator.transform(test_df)
      indicator_df = pd.DataFrame(temporary, columns=indicator_columns)

      # create the final test data
      test = pd.concat([X_test.reset_index(), indicator_df], axis=1)


+ Random sample imputation
  + taking a random observaton from the pool of available observations of the variable and using those randomly selected values to fill in the missing one
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
  + example:
    + data (Gender, Age): ((Male, 29) $\to$ (Male, 29)), ((Male, <span style="color: pink;">NA</span>) $\to$ (Male, <span style="color: lightgreen;">34</span>)), (  (<span style="color: pink;">NA</span>, 43) $\to$ (<span style="color: lightgreen;">Female</span>, 43)), ((Female, 25) $\to$ (Femal, 25)), ((Male, 34) $\to$ (Male, 34)), ((<span style="color: pink;">NA</span>, 50) $\to$ (<span style="color: lightgreen;">Male</span>, 50)), ((Femal, <span style="color: pink;">NA</span>) $\to$ (Female, <span style="color: lightgreen;">25</span>))
    + Python snippet

      ```python
      from feature_engine.missing_data_imputer import  RandomSampleImputer

      # create the imputer
      imputer = RandomSampleImputer(random_state=29)

      # fit the imputer to the train set
      imputer.fit(train_df)

      # transform the data
      train_t_df = imputer.transform(train_df)
      test_t_df = imputer.transform(test_df)

## 5. Encoding Categorical Variables
 
+ Categorical encoding
  + permanently replacing category strings w/ numerical representations
  + goal: producing variables used to train machine learning models and build predictive features from categories
  + techniques for data transformation
    + traditional techniques
      + one-hot encoding
      + count or frequency encoding
    + monotonic relationship
      + ordered label encoding
      + mean encoding
      + probability ratio encoding
      + weight of evidence
    + alternative techniques
      + rare labels encoding
      + binary encoding
  + Python library: category_encoders - containing a lot of basic and advanced methods for categorical variable encoding
  + typical sample data (Color, Target): 

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 1 0vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y6ccmhqp"
        alt    ="Example of categorical encoding"
        title  ="Example of categorical encoding"
      />
    </figure>

+ One-hot encoding
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
    + data example:

      <figure style="margin: 0.5em; text-align: center;">
        <img style="margin: 0.1em; padding-top: 0.5em; width: 4 0vw;"
          onclick="window.open('https://tinyurl.com/y6yq38cg')"
          src    ="https://tinyurl.com/yysvncj6"
          alt    ="Example of one-hot encoding into $k$ variables"
          title  ="Example of one-hot encoding into $k$ variables"
        />
      </figure>

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
  + Python snippet

    ```python
    # import the pandas library
    import pandas as pd

    # read dataset
    data_df = pd.read_csv("dataset.csv")

    # perform one hot encoding w/ k
    data_with_k_df = pd.get_dummies(data_df)

    # perform one hot encoding w/ k-1, it automatically drop the first
    data_with_k_one_df = pd.get_dummies(data_df, drop_first=True)
    ```

+ Integer (Label) Encoding
  + replacing the categories w/ digits from $1$ to $n$ (or $0$ to $n-1$, depending on the implementation)
  + $n$: the number of the variable's distinct categories (the cardinality)
  + the number assigned arbitrary
  + data example:

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 4 0vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y23xs9ug"
        alt    ="Example of integer (label) encoding"
        title  ="Example of integer (label) encoding"
      />
    </figure>

  + advantages
    + straightforward to implement
    + not expanding the feature space
    + working well enough w/ tree-based algorithms
    + allowing agile benchmarking of ML models
  + limitations
    + not adding extra information while encoding
    + not suitable for linear models
    nor handling new categories in the test set automatically
    + creating an order relationship btw the categories
  + Python snippet

    ```python
    import pandas as pd

    # get data
    data_df = pd.read_csv("dataset.csv")

    # function to find the different enumeration of variable
    def create_category_mapping(data_df, variable):
      return {K: i for i, k in enumerate(data_df[variable].unique(), 0)}

    # function to apply the encoding on the variable
    def label_encode(train_df, test_df, variable, ordinal_mapping):
      train_df[variable] = train_df[variable].map(ordinal_mapping)
      test_df[variable] = test_df[variable].map(ordinal_mapping)

    # check that data contains only
    for variable in data_df.columns:
      mappings = create_category_mapping(data_df, variable)
      label_encode(train_df, test_df, variable, mappings)
    ```

+ Count or frequency encoding
  + replacing categories w/ the count or percentage that show each category in the dataset
  + capturing the representation of each label in the dataset
  + example: 

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y2d2llys"
        alt    ="Example of count or frequency encoding"
        title  ="Example of count or frequency encoding"
      />
    </figure>

  + advantages
    + straightforward to implement
    + not expanding the feature space
    + working well w/ tree-based algorithms
  + limitations
    + not suitable for linear models
    + not handling new categories in the test set automatically
    + losing valuable information if there are two different categories w/ the same amount of observations count
  + Python snippet

    ```python
    import pandas as pd

    # get data
    data_df = pd.read_csv("dataset.csv")

    # loop to find the different count of categories in a dict
    # and apply them to the variable in the train and test set
    for variable in train_df.columns:
      count_dict = train[variable].value_counts().to_dict()
      train_df[variable].map(count_map)
      test_df[variable].map(count_map)
    ```

+ Ordered label encoding
  + replacing categories w/ integers from 1 to n
  + $n$: the number of distinct categories in the variable (the cardinality)
  + using the target mean information of each category to decide how to assign these numbers
  + data example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y2tm9sm8"
        alt    ="Example of ordered label encoding"
        title  ="Example of ordered label encoding"
      />
    </figure>

  + advantages
    + straightforward to implement
    + not expanding the feature space
    + creating a monotonic relationship btw categories and the target
  + limitation: probably leading to overfitting
  + Python snippet

    ```python
    import pandas as pd

    # get data
    data_df = pd.read_csv("dataset.csv")

    # get your target variable name
    target = "your target variable name"

    # generate the order list of labels, then apply it to the variable
    for variable in train_df.columns:
      labels = train_df.groupby([variable])[target].mean().sort_values().index
      mappings = {x: i for i, x in enumerate(labels, 0)}

      # apply the encoding to the train and test sets
      train_df[variable] = train_df[variable].map(mapping)
      test_df[variable] = test_df[variable].map(mapping)
    ```

+ Mean (target) encoding
  + replacing the category w/ the mean target value for that category
  + procedure
    + grouping each category alone
    + for each group, calculating the mean of the target in the corresponding observations
    + assigning mean to that category
    + encoded the category w/ the mean of the target
  + data example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/yxsnfdfq"
        alt    ="Example of Mean (target) encoding"
        title  ="Example of Mean (target) encoding"
      />
    </figure>

  + advantages
    + straightforward to implement
    + not expanding the feature space
    + creating a monotonic relationship btw categories and the target
  + limitations
    + probably leading to overfitting
    + probably leading to a possible loss of value if two categories have the same mean as the target
  + Python snippet

    ```python
    import pandas as pd

    # get data 
    data_df = pd.read_csv("dataset.csv")

    # get target variable name
    target = "your target variable name"

    # loop over the categorical columns to apply the encoding
    for variable in train_df.columns:
      # create dictionary of category: mean values
      dict = train_df.groupby([variable])[target].mean().to_dict()

      # apply the encoding to the train and test sets
      train_df[variable] = train_df[variable].map(dict)
      test_df[variable] = test_df[variable].map(dict)
    ```

+ Weight of evidence encoding (WOE)
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
  + data example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y22c9oep"
        alt    ="Example of Weight of evidence encoding (WOE)"
        title  ="Example of Weight of evidence encoding (WOE)"
      />
    </figure>

  + advantages
    + creating a monotonic relationship btw the target and the variables
    + ordering the categories on the 'logistic' scale, nature for logistic regression
    + comparing the transformed variables because they are on the same scale $\to$ determine which one is more predictive
  + limitations
    + probably lead to overfitting
    + not defined when the denominator is 0
  + Python snippet

    ```python
    import pandas as pd
    import numpy as np

    # get data
    data_df = pd.read_csv("dataset.csv")

    # get target variable name
    target = "your target variable name"

    # loop over all the categorical variables
    for variable in train_df.columns:
      # calculating the mean of target for each category
      # probability of events or P(1)
      dataframe = pd.DataFrame(train_df.groupby([variable])[target].mean())

      # calculating the non target probability
      # probability of non-events or p(0)
      dataframe['ratio'] =np.log(dataframe[target] / dataframe['non-target'])
      ratio_mapping = dataframe['ratio'].to_dict()

      # applying the WOE
      train_df[variable] = train_df[variable].map(ratio_mapping)
      test_df[variable] = test_dff[variable].map(ratio_mapping)
    ```

+ Probability ratio encoding
  + suitable for classification problems only, where the target is binary
  + similar to WOE, but not applying the natural logrithm
  + each category, the mean of the target = 1
    + $P(1)$: the probability of the target being 1
    + $P(0)$: the probability of the target being 0
  + calculating the ratio = P(1)/P(0) and replacing the categories by that ratio
  + data example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y3q7kzew"
        alt    ="Example of Probability ratio encoding"
        title  ="Example of Probability ratio encoding"
      />
    </figure>

  + advantages
    + capturing information within the category, and therefore creating more predictive features
    + creating a montonic relationship btw the variables and the target, suitable for linear mdoels
    + not expanding the feature space
  + limitations
    + likely to cause overfitting
    + not defined when the denominator is 0
  + Python snippet

    ```python
    import pandas as pd
    import numpy as np

    # get data
    data_df = pd.read_csv("dataset.csv")

    # get target variable anme
    target = "your target variable name"

    # loop over all the categorical variables
    for variable in train_df.columns:
      # calculating the mean of target for each category
      # probability of events or p(1)
      dataframe = pd.DataFrame(train_df.groupby([variable])[target].mean())

      # calculating the non target probability
      # probability of non-events or p(0)
      dataframe['non-target'] = 1 - dataframe[target]

      # calculating the ratio
      dataframe['ratio'] = dataframe[target] / dataframe['non-target']
      ratio_mapping = dataframe['ratio'].to_dict()

      # applying the probability ratio encoding
      train_df[variable] = train_df[variable].map(ratio_mapping)
      test_df[variable] = test_df[variable].map(ratio_mapping)
    ```

+ Rare label encoding
  + rare label: appearing only in a tiny proportion of the observations in a dataset
  + causing some issues, especially w/ overfitting and generation
  + solution: group those rare labels into a new category like other or rare
  + data example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/yyxdzpt2"
        alt    ="Example of rare label encoding"
        title  ="Example of rare label encoding"
      />
    </figure>

  + Python snippet

    ```python
    import pandas as pd
    import numpy as np

    # get dat
    data_df = pd.read_csv("dataset.csv")

    # define threshold here
    threshold = 0.05

    # loop over all the categorical variables
    for variable in train_df.columns:
      # locate all the categories that are not rare
      counts = train_df.groupby([variable])[variable].count() / len(train_df)
      frequent_labels = [x for x in counts.loc[counts > threshold].index.values]

      # change the rare category names w/ the word rare, and thus encoding it
      train_df[variable] = np.where(train_df[variable].isin(frequent_labels), \
        train_df[variable], 'Rare')
      test_df[variable] = np.where(test_df[variable].isin(frequent_labels), \
        test_df[variable], 'Rare')
    ```

+ Binary encoding
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
  + data example

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 40vw;"
        onclick="window.open('https://tinyurl.com/y6yq38cg')"
        src    ="https://tinyurl.com/y2uazs6q"
        alt    ="Example of binary encoding"
        title  ="Example of binary encoding"
      />
    </figure>

  + Python snippet

    ```python
    import pandas as pd
    from category_encoders import BinaryEncoder

    # get data
    data_df = pd.read_csv("dataset.csv")

    # split into x and y
    x_train_df = data_df.drop('target', axis=1)
    y_train_df = data_df['target']

    # create an encoder object - it will apply on all strings column
    binary = BinaryEncoder()

    # fit and transform to get encoded data
    binary.fit_transform(x_train_df, y_train_df)
    ```

## 6. Transforming Variables

+ Transforming variables
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
  + common Python snippet for transformations

    ```python
    # import the library
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create columns variables to hold the columns that need transformation
    cols = ['col1', 'col2', 'col3', ...]
    ```

+ Q-Q plot
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

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick="window.open('https://tinyurl.com/y2mhvfrm')"
        src    ="https://tinyurl.com/y5y6lssv"
        alt    ="Q-Q Plot"
        title  ="Q-Q Plot"
      />
    </figure>
  

+ Logarithmic transformation 
  + formula: $ f(x) = \ln(x) $
  + simplest and most popular among the different types of transformations
  + involving a substantial transformation that significantly affects distribution shape
  + making extremely skewed distribution less skewed, especially for right-skewed distributions
  + constraint: only for __strictly positive__ numbers
  + Python snippet

    ```python
    # create the function transformer object w/ logarithm transformation
    lagrithm_transformer = FunctionTransformer(np.log, validate=True)

    # apply the transformation to data
    data_new_df = logarithm_transformer.transformer(data_df[cols])
    ```

+ Square root transformation
  + formula: $f(x) = \sqrt{x}$
  + simple transformation w/ average effect on distribution shape
  + weaker than logarithmic transformation
  + used for reducing right-skewed distributions
  + advantage: able to apply to zero values
  + constrain: only for positive numbers
  + Python snippet

    ```python
    #create the fucntion transformaer object w/ square root transformation
    sqrt_transformer = FunctionTransformer(np.sqrt, validate=True)

    # apply the transformation to data
    data_new = sqrt_transformer.transformer(data_df[cols])
    ```
  
  + alternative: cubic root fucntion

+ Reciprocal transformation
  + formula: $f(x) = \frac{1}{x}$
  + a powerful transforamtion w/ a radical effect
  + positive reciprocal: reversing the order among values of the same sign $\to$ large values $\to$ smaller
  + negative reciprocal: preserving the order among values of the same ign
  + constraint: not defined for zero
  + Python snippet

    ```python
    #create the function transformer object w/ reciprocal transformation
    reciprocol_transformer = FunctionTransformer(np.reciprocol, validate=True)

    # apply the transformation to data
    data_new_df = reciprocol_transformer.transform(data_df[cols])
    ```
  
  + alternative; negative reciprocal fucntion

+ Exponential or Power transformation
  + formula

    \[ \begin{align*} f(x) &= x^2 \\ g(x) &= x^3 \\ h(x) &= x^n \\ k(x) &= \exp(x) \end{align*} \]

  + a reasonable effect on distribution shape
  + applying power transformation (power of two usually) to reduce left skewness
  + using any exponent in the transformation, even using the $\exp()$ fucntion
  + Python snippet

    ```python
    # create the fucntion transformer object w/ exponent transformation
    # using x^3 is arbitrary here, able to choose any exponent
    exponential_transformer = FunctionTransformer(lambda x: x**(3), validate=True)

    # apply the transformation to data
    data_new_df = exponential_transformer.transform(data_df[cols])
    ```

+ Box-Cox transformation
  + formula

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
  + Python snippet

    ```python
    # create the power transformer object w/ method 'box-cox'
    boxcox_transformer = PowerTransformer(method='box-cox', standardize=False)

    # apply the transformation to data
    data_new_df = boxcox_transformer.transfor(data_df[cols])
    ```

+ Yeo-Johnson transformation
  + formula

    \[ x_i^{(\lambda)} = \begin{cases}
      [(x_i + 1)^\lambda - 1] / \lambda & \text{if } \lambda \ne 0, x_i \ge 0, \\
      \ln(x_i) + 1 & \text{if } \lambda = 0, x_i \ge 0, \\
      [(x_i + 1)^{2-\lambda} - 1]/(2-\lambda) & \text{if } \lambda \ne 2, x_i < 0, \\
      -\ln(-x_i + 1) & \text{if } \lambda = 2, x_i < 0
    \end{cases} \]

  + an adjustment to the Box-Cox transformation
  + able to apply to negative numbers
  + Python snippet

    ```python
    # create the power transformer object w/ method 'yeo-johnson'
    yeo_johnson_transformer = PowerTransformer(method='yeo-johnson', standardize=False)

    # apply the transformation to data
    data_new_df = yeo_johnson_transformer.transform(data_df[cols])
    ```


## 7. Variable Discretization

+ Variable Discretization
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

+ Equal-width discretizatoin
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
  + Python snippet

    ```python
    import pandas as pd
    from sklearn.preprocessing import KBinsDiscretizer

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create the discretizer object w/ strategy uniform and 8 bins
    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal' strategy='uniform')

    # fit the discretizer to the train set
    discretizer.fit(train)

    # apply the discretization
    train_df = discretizer.transform(train_df)
    test_df = discretizer.transform(test_df)
    ```

+ Equal-frequency discretization
  + dividing the scope of possible values of the variable into $N$ bins
  + each bin holding the same number (or approximately the same number) of observation
  + considerations
    + the interval boundaries corresponding to quantiles
    + improving the value spread
    + handling outliers
    + disturbing the relationship w/ the target
    + useful when combined w/ categorical encoding
  + Python snippet

    ```python
    import pandas as pd
    from sklearn.preprocessing import KBinsDiscretizer

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create the discretizer object w/ strategy quantile and 8 bins
    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')

    # fit the discretizer to the train set
    discterizer.fit(data_df)

    # apply the discretization
    train_df = discretizer.transformer(train_df)
    test_df = discretizer.transformer(test_df)
    ```

+ K-means discretization
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
  + Python snippet

    ```python
    import pandas as pd
    from sklearn.preprocessing import KBinDiscretizer

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create the discretizer object w/ strategy kmeans and 8 bins
    discretizer = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')

    # fit the discretizer to the train set
    discretizer.fit(data_df)

    # apply the discretization
    train_df = discretizer.transformer(data_df)
    test_df = discretizer.transformer(test_df)
    ```

+ Discretization w/ Decision Trees
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
  + Python snippet

    ```python
    import panadas as pd
    from sklearn.proprecessing import DecisionTreeClassifier

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create variables for the target and the variable to discretize
    x_variable = train_df['variable']
    target = train_df['target']

    # build the decision tree w/ max depth of choice
    # {depth pf 2 will create two splits, and 4 different bins for discretization}
    decision_tree = DecisionTreeClassifier(max_depth=2)

    # start the learning process
    decision_tree.fit(x_variable, target)

    # apply the discretization to the variable
    train_df['variable'] = decision_tree.predict_proba(train_df['variable'].to_frame()[:, 1])
    test_df['variable'] = decision_tree.predict_prob(test_df['variable'].to_frame()[:, 1])
    ```

+ Using the newly-created discrete variable
  + usually encoding w/ ordinal, i.e., integer encoding as 1, 2, 3, tec.
  + two major methods
    + using the value of the interval straight away if using intervals as numbers
    + treating numbers as categories, applying any of the encoding technique that creates a montone relationship w/ the target
  + advantageous way of encoding bins: treating bins as categories to use an encoding technique that creates a monotone relationship w/ the target

+ Custom discretization
  + engineering variables in a custom environment (i.e., for a particular business use case)
  + determining the intervals where the variable divided so that it makes sense for the business
  + example: Age divided into groups like [0-10] as kids, [10-25] as teenagers, and so on
  + Python snippet

    ```python
    import pandas as pd

    # bins intervals
    labels = ['0-10', '10-25', '25-65', '>65']

    # discretization w/ pandas
    train_df['age'] = pd.cut(train_df.age, bins=bins, labels=labels, include_lowest=True)
    test_df['age'] = pd.cut(test_df.age, bins=bins, labels=labels, include_lowest=True)
    ```

## 8. Handling Outliers

+ Outliers
  + a data point significantly different from the remaining data
  + an observation deviating so much from the other observations
  + arousing suspicion that a different mechanism produced it
  + handling outliers
    + trimming: simply removing the outliers from dataset
    + imputing: treating outliers as missing data and applying missing data imputation techniques
    + discretizaton: placing outliers in edge bins w/ higher or lower values of the distribution
    + censoring: capping the variable distribution at the maximum and minimum values

+ Detecting Outliers
  + using visualization plots like box plot and scatter plot
    + box plot: black points as outliers
    + scatter plot: most points located in center but one far from center might eb outlier

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/y4rljbj3" ismap target="_blank">
        <img style="margin: 0.1em;" height=150
          src  ="https://tinyurl.com/yxblyfx8"
          alt  ="Example of Box plot"
          title="Example of Box plot"
        >
        <img style="margin: 0.1em;" height=150
          src  ="https://miro.medium.com/max/464/1*Fh1snT0LP1WP586tV1WImA.png"
          alt  ="Example of scatter plot"
          title="Example of scatter plot"
        >
      </a>
    </div>

  + using a normal distribution (mean and s.d.)<br/>
    about 99.7% of the data lie within 3 s.d. of the mean

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick="window.open('https://tinyurl.com/y4rljbj3')"
        src    ="https://tinyurl.com/y4zdoyry"
        alt    ="Example of normal distribution w/ standard deviations"
        title  ="Example of normal distribution w/ standard deviations"
      />
    </figure>

+ Inter-quantal range proximity rule
  + Interquartile range (IQR)
    + used to build boxplot graphs
    + dividing data into four parts and each part is a quartile
    + IQR = he difference between the 3rd quantile Q3 (75%) and the 1st quantile or Q1 (25%)
  + outliers defined w/ IQR
    + below Q1 - 1.5 x IQR
    + above Q3 + 1.5 x IQR

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/y4rljbj3')"
        src    ="https://tinyurl.com/yyj72x5m"
        alt    ="Example of IQR from Wikipedia"
        title  ="Example of IQR from Wikipedia"
      />
    </figure>

+ Trimming outliers
  + merely removing outliers from the dataset
  + deciding on a metric to determine outliers
  + considerations
    + fast method
    + removing a significant amount of data
  + Python snippet

    ```python
    import pandas as pd
    import numpy as np
 
    # load data
    data_df = pd.read_csv("dataset.csv")

    for variable in data_df.columns:
      # calculate the IQR
      IQR = data_df[variable].quantile(0.75) - data_df[variable].quantile(0.25)

      # calculate the boundaries
      lower = data_df[variable].quantile(0.25) - (IQR * 1.5)
      upper = data_df[variable].quantile(0.75) - (IQR * 1.5)

      # find the outlier
      outliers = np.where(data_df[variable] > upper, True, \
        np.where(data_df[variable] < lower, True, False))

      # remove outliers from data
      data_df = data_df.loc[~(outliers,)]

+ Censoring outliers
  + setting the maximum and/or the minimum of the distribution at any arbitrary value
  + values bigger or smaller than the arbitrarily chosen value are replaced by the value
  + concerns about capping
    + not removing data
    + distorting the distributions of the variables
  + arbitrarily replacing the outliers

    ```python
    import pandas as pd
    import numpy as np

    # load data
    data_df = pd.read_csv("dataset.csv')

    for variable in data_df.columns:
      # create boundaries (age for example)
      lower = 10
      upper = 89

      # replacing the outliers
      data_df[variable] = np.where(data_df[variable] > upper, upper,  \
        np.where(data_df[variable] < lower, data[variable]))
    ```

  + inter-quantal range proximity rule

    ```python
    import pandas as pd
    import numpy as np

    #load data
    data_df = pd.read_csv("dataset.csv")

    for variable in data_df.columns:
      # calculate the IQR
      IQR = data_df[variable].quantile(0.75) - daat_df[variable].quantile(0.25)

      # calculate the boundaries
      lower = data_df[variable].quantile(0.25) - (IQR * 1.5)
      upper = data_df[variable].quantile(0.75) - (IQR + 1.5)

      # replacing the outliers
      data_df[variable] = np.where(data_df[variable] > upper, upper. \
        np.where(data_df[variable] < lower, lower, data_df[variable]))
    ```

  + Gaussian approximation

    ```python
    import pandas as pd
    import numpy as np

    # load data
    data_df = pd.read_csv("dataset.csv")

    for wariable in data_df.columns:
      # calculate the boundaries
      lower = data_df[variable].mean() - 3 * data_df[variable].std()
      upper = data_df[variable].mean() + 3 * data_df[variable].std()

      # replacing the outliers
      data_df[variable] = np.where(dat_df[variable] > upper, upper, \
        np.where(data_df[variable]< lower, lower, data_df[variable]))
    ```

  + using quantiles

    ```python
    import pandas as pd
    import numpy as np

    # load data
    data_df = pd.read_csv("dataset.csv")

    for variable in data_df.columns:
      # calculate the boundaries
      lower = data_df[variable].quantile(0.10)
      upper = data_df[variable].quantile(0.90)

      # replacing the outliers
      data_df[variable] = np.where(data_df[variable] > upper, upper, \
        np.where(data_df[variable] < lower, lower, data_df[variable]))
    ```

+ Imputing outliers
  + treating outliers as missing data
  + refer to  [imputing variable techniques](#4-imputing-missing-values)

+ Transforming outliers
  + applying some mathematical transformations, such as log transformation
  + refer to [transforming variables](#6-transforming-variables)


## 9. Feature Scaling

+ Feature scaling
  + methods used to normalize the range if values of independent variables
  + ways to set the feature value range within a similar scale
  + concerns
    + the scale of the variable directly influencing the regression coefficient
    + variable w/ a more significant magnitude dominate over the ones w/ a smaller magnitude range
    + gradient descent converges faster when features are on the same scales
    + feature scaling helps decrease the time to find support vectore of SVMs
    + Euclidean distances are senstivie to feature magnitude
  + algorithms sensitive to feature mangitude
    + linear and logistic regression
    + Neural networks
    + support vectore machine
    + KNN
    + K-means clustering
    + linear discriminant analysis (LDA)
    + principle component analysis (PCA)
  + algorithm insentive to feature magnitude
    + classification and regression trees
    + random forest
    + gradient boosted trees
  + scaling methods
    + mean normalization
    + standardization
    + robust to maximum and minimum
    + scale to absolute maximum
    + scale ti unit norm

+ Mean normalization
  + centering the variable at 0 and rescaling the variable's value range to the range -1 and 1
  + formula:

    \[ \oveline{x} = \frac{X - text{mean}(X)}{\max(X) - \min(X)} \]

  + not normalizing the variable distribution
  + characteristics
    + centering the mean at 0
    + different resulting variance
    + modifying the shape of original distribution
    + normalizing the minimum and maximum values w/ the range [-1, 1]
    + preserving outliers if existed
  + Python snippet

    ```python
    import pandas as pd
    import numpy as np

    # load data
    data_df = pd.read_csv("dataset.csv")

    # claculate the means
    means = train_df.mean(axis=0)

    # calculate max - min
    max_min = train_df.max(axis=0) - train_df.min(axis=0)

    # apply the transformation to data
    train_scaled_df = (train_df - means) / max_min
    test_scaled_df = (test - means) / max_min
    ```

+ Standardization
  + centering the variable at 0 and standarizing the variance to 1
  + formula:

    \[ \overline{x} = \frac{X - \text{mean}(X)}{\text{std}(X)} \]

  + nor normalized the variable distribution
  + characteristics
    + scaling the variance at 1
    + centering the mean at 0
    + preserving the shape of the original distribution
    + preversing outliers if existed
    + minimum nad maximum values varying
  + Python snippet

    ```python
    import pandas as pd
    import sklearn.preprocessing import StandardScaler

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create the scaler object
    scaler = StandardScaler()

    # fit the scaler to the train data
    scaler.fit(train_df)

    # transform train and test
    train_scaled_df = scaler.transform(train_df)
    test_scaled_df = scaler.transform(test_df)
    ```

+ Robust scaling (scaling to median and IQR)
  + using median instead of mean
  + formula

    \[ \overline{x} = \frac{X - \text{median}(X)}{Q3(x) - Q1(x)} \]

  + characteristics
    + centering the median at 0
    + resulted variance varying across variables
    + not preserving the shape of the original distribution
    + minimum and maximum values varying
    + robust to outliers
  + Python snippet

    ```python
    import pandas as pd
    from sklearn.preprocessing import RobustScaler

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create the scaler object
    robust = RobustScaler()

    # fit the scaler tot he train data
    robust.fit(train_df)

    # transform train and test
    train_scaled_df = robust.transform(train_df)
    test_scaled_df = robust.transform(test_df)
    ```

+ Min-Max scaling
  + compressing the value between 0 and 1
  + formula:

    \[ \overline{x} = \frac{X - \min(X)}{\max(X) - \min{X}} \]

  + not normalizing the variable distribution
  + characteristics
    + not centering the mean at 0
    + making the variance vary across variables
    + not maintaining the shape of the original distribution
    + maximum and minimum values in the range of [0, 1]
    + sensitive to outliers
  + Python snippet

    ```python
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # load data
    data_df = pd.read_csv("dataset.csv")

    # create the scaler object
    min_max = MinMaxScaler()

    # fit the scaler to the train data
    min_max.fit(train_df)

    # transform train and test data
    train_scaled_df = min_max.tranbsform(train_df)
    test_scaled_df = min_max.transform(test_df)
    ```








  