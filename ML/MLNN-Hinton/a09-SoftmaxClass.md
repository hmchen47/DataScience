# Softmax Classifier

[URL](http://cs231n.github.io/linear-classify/#softmax)

## Introduction

+ Two common classifiers: SVM & Softmax classifier

+ Softmax classifier
  + a generalization of Logistic Regression classifier to multiple classes
  + providing a intuitive output (normalized class probabilities)
  + a probabilistic interpretation 
  + function mapping $f(x_i; W) - Wx_i$ stays unchanged
  + interpret these scores as the uncommonalized log probabilities for each class

+ Cross-entropy loss function

  \[L_i = -\log \left( \frac{e^{f_{y_i}}}{\sum_j e^{f_j}} \right) \qquad \text{or equivalently} \qquad L_i = -f_{y_i} + \log \sum_j e^{f_j}\]

  + $f_j$: the $i$-th element of the vector of class score $f$
  + the full loss for the database: the mean of $L-i$ over all training examples together with a regularization term $R(W)$
  + softmax function: $f_j(z) = \frac{e^{z_j}}{\sum_k e^{f_j}}$; taking a vector of arbitrary real-valued scores (in $z$) and squashing it to a vector of values between zero and one that sum to one


## Information Theory View




## Probabilistic Interpretation





## Practical Issues: Numeric stability




## Possible confusion naming conventions





