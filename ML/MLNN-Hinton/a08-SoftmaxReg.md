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


## Over-Parameterization




## Logistic Regression as a Specific Case




## Cost Function




## Gradients






