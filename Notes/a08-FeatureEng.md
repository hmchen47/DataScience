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
        <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
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
  + usually not working w/ datetimme variables in their raw format
    + date variables containing a considerable number of different categories
    + able to extract much more infroamtion from datetime variables by pre-processing them correctly
  + date variables
    + containing dates not present in the dataset used to train the learning model
    + containing dates placed in the future, w.r.t. the dates in the training dataset
  
+ Mixed variables
  + containing both numbers and labels
  + occurring in a given dataset, especially when filling its values
  + example
    + a number of something, e.g., the income, or the number of children
    + number not able to be retrieved for a variety of reasons, e.g., survey of income of a person
    + returning a label to represent the reason behind the issue, e.g., ERROR_OMMIT for client omit to answer




