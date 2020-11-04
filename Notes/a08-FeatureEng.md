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
  




