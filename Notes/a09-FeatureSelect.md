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




