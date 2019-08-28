# Deep Learning Tutorial - Softmax Regression

Author: Chris McCormick

[URL](http://mccormickml.com/2014/06/13/deep-learning-tutorial-softmax-regression/)

Date: 13 Jun 2014


## Introduction

+ Softmax regression
  + a generalized form of logistic regression
  + used in multi-class classification problems where the classes are mutually exclusive
  + e.g., hand-written digit datasets
  + Formula

    \[h_\theta(x^{(i)}) = \begin{bmatrix} p(y^{(i)} = 1 | x^{(i)}; \theta) \\ p(y^{(i)} = 2 | x^{(i)}; \theta) \\ \vdots \\ p(y^{(i)} = k | x^{(i)}; \theta) \end{bmatrix} = \frac{1}{\sum_{j=1}^k e^{\theta_j^T} x{(i)}} \begin{bmatrix} e^{\theta_1^T x^{(i)}} \\ e^{\theta_2^T x^{(i)}} \\ \vdots \\ e^{\theta_k^T x^{(i)}}\end{bmatrix}\]

  + $x^{(i)}$: the input vector of the $i$th sampling case
  + $y^{(i)}$: the actual calculated output value of the $i$th sampling case
  + The output is a vector of the probability w/ actual output value of $y^{(k)} = i$ where $i = 1, 2, \dots, k$


## Over-Parameterization (<span style="color: red;">NOT GET THE POINT</span>)

+ Able to fix the vector of parameters for one of the $k$ classifiers to a vector of all zeros

+ Able to function by learning the parameters for the other $k-1$ classifiers

+ E.g., set $\theta_1$ to all zeros, the first component of the un-normalized output vector would always be equal to 1, no matter what input is.

+ The normalization allows this component to take on different values depending on the un-normalized outputs of the other 9 classifiers.


## Logistic Regression as a Specific Case

+ Logistic regression as a special case of Softmax regression with 2 classes


## Cost Function

The cost function with weight decay for Softmax Regression

\[J(\theta) = -\frac{1}{m} \left[ \sum_{i=1}^m \sum_{j=1}^k \mathbf{1}\{y^{(i)}=j\} \log\left( \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k} e^{\theta_i^T x{(i)}} \right) \right] + \frac{2}{\lambda} \sum_{i=1}^k \sum_{j=0}^n \theta_{ij}^2 \]

+ $\mathbf{1}\{y^{(i)} = j\}$: an indicator function; only the output of the classifier corresponding to the correct class label
+ $log(x) \in (-\infty, 0] \text{ when } x \in [0, 1]$
+ if the classifier outputs 1 for the training example, then the cos is zero.


## Gradients






