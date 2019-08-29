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

+ The cross-entropy btw a "true" distribution $p$ and an estimated distribution $q$

  \[H(p, q) = - \sum_x p(x) \log q(x)\]

  + $p = [0, \dots 1, \dots]$ with 1 at the $y_i$-th position
  + $q = e^{f_{y_i}} / \sum_j e^{f_j}$

+ Minimizing the cross-entropy btw the estimated class probability ($q$) and the "true" distribution

+ Cross-entropy can be written as entropy and the Kullback-Leibler divergence

  \[H(p, q) = H(p) + D_{KL}(p || q)\]

  + the entropy of the delta function $H(p) = 0$ 

+ Minimizing the cross-entropy equivalent to minimizing the KL divergence between two distributions (a measure of distance)

+ Objective: the predicted distribution to have all of its mass on the correct answer


## Probabilistic Interpretation

+ The (normalized) probability assigned to the correct label $y_i$ given the image $x_i$ and parameterized by $W$

  \[P(y_i | x_i; W) = \frac{e^{f_i}}{\sum_j e^{f_j}}\]

+ The softmax classifier interprets the scores inside the output vector $f$ as the unnormalized log probabilities.

+ Objective: Minimizing the negative log likelihood of the correct class, as performing __Maximum Likelihood Estimation__ (MLE)

+ Viewing $R(W)$, the regularization term, as coming from a Gaussian prior over the weight matrix $W$

+ Performing __Maximum a posteriori__ (MAP) estimation instead of performing MLE


## Practical Issues: Numeric stability

+ The intermediate terms $e^{f_{y_i}}$ and $\sum_j e^{f_j}$ might very large due to exponential when coding.

+ Solution: normalization

  \[\frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = \frac{C e^{f_{y_i}}}{C \sum_j e^{f_j}} = \frac{e^{f_{y_i}+\log C}}{\sum_j e^{f_j+\log C}}\]

  + not change any of the result
  + improve the numerical stability of the computation
  + Common choice $C$: $\log C = -\max_j f_j$

  ```python
  f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
  p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

  # instead: first shift the values of f so that the highest number is 0:
  f -= np.max(f) # f becomes [-666, -333, 0]
  p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
  ```


## Possible confusion naming conventions

+ SVM classifier
  + using the hinge loss, or max-margin loss

+ Softmax classifier
  + using the cross-entropy loss
  + using softmax function: used to squash the raw class scores into normalized positive values that sum to one



