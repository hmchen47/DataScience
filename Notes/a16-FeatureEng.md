# Beyond One-Hot

Title: Beyond One-Hot: An Explanation of Categorical Variables

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


## Methodology

+ Process overview
  + gathering a dataset for a classification problem w/ categorical variables
  + using some method of coding to convert the X dataset into numeric values
  + using scikit-learn's cross-validation-score and a BernoulliNB() classifier to generate scores for the dataset, repeated 10 times
  + storing the dimensionality of the dataset, mean score, and time to code the data and generate the score

+ UCI dataset repositories
  + [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
  + [Mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom)
  + [Splice Junctions](http://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/)

+ Encoding methods used
  + Ordinal: as described above
  + One-Hot: one column per category, with a 1 or 0 in each cell for if the row contained that column's category
  + Binary: first the categories are encoded as ordinal, then those integers are converted into binary code, then the digits from that binary string are split into separate columns.  This encodes the data in fewer dimensions that one-hot, but with some distortion of the distances.
  + Sum: compares the mean of the dependent variable for a given level to the overall mean of the dependent variable over all the levels. That is, it uses contrasts between each of the first k-1 levels and level k In this example, level 1 is compared to all the others, level 2 to all the others, and level 3 to all the others.
  + Polynomial: The coefficients taken on by polynomial coding for k=4 levels are the linear, quadratic, and cubic trends in the categorical variable. The categorical variable here is assumed to be represented by an underlying, equally spaced numeric variable. Therefore, this type of encoding is used only for ordered categorical variables with equal spacing.
  + Backward Difference: the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable.
  + Helmert: The mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels. Hence, the name ‘reverse’ being sometimes applied to differentiate from forward Helmert coding.



