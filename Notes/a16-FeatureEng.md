# Beyon One-Hot: An Explanation of Categorical Variables

Author: Will McGinnis

Date: Nov. 29, 2015

[Original](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)


## Introduction

+ Categorical variables
  + data represented a fixed number of possible values
  + value assigned to one of the finite groups
  + ordinal variables: ordering
  + ML algorithms preferring numbers, not strings

+ Concept of dimensionality
  + simple definition: the number of columns in the dataset
  + significant downstream effects on the eventual models
  + curse of dimensionality: probably models stop working properly in high dimensions
  + dataset w/ more dimensions requiring more parameters of the model to understand $\implies$ more rows to reliably learn those parameters
  + fixed number of rows, additional of extra dimensions w/o adding more info for the models $\to$ detrimental effect on the eventual model accuracy

+ Categorical variables and dimensionality
  + conflict: coding categorical variables and dimensionality problem
  + solution
    + ordinal coding: assigning an integer to each category
    + not adding any dimensions
    + implying an order to the variable probably not existed





