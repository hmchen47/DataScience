# Machine Learning Theory Aspects

## General

### Model Representation

<a href="https://www.coursera.org/learn/machine-learning/supplement/cRa2m/model-representation">
    <img src="../ML/ML-Stanford/images/m02-17.png" style="display: block; margin: auto; background-color: black" alt="Flowchart" title="Modeling Process" width="300" >
</a>

+ $h\;$: hypothesis function
  + mapping from $x$ to predicted $y$
  + E.g., $h_\theta(x) = \theta_0 + \theta_1 \cdot x_1$

### Pipeline


## Supervised Learning

### Learn Regression

#### Objective: Learn Regression

+ [Simple Linear Regression](../ML-Stanford/02-ModelCost.md#cost-function-intuition-ii):

	+ Hypothesis: $h_\theta (x) = \theta_0 + \theta_1 \cdot x$

  + Parameters: $\theta_0$, $\theta_1$

  + Cost Function: $J(\theta_0, \theta_1) = \displaystyle \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$

  + Goal: $\displaystyle \min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$





#### Cost Function: Learn Regression

+ [Squared error function](../ML-Stanford/02-ModelCost.md#cost-function-intuition-ii): $J(\theta_0, \theta_1)$

	An average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

	$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$






#### Gradient Descent: Linear Regression





#### Vectorization: Linear Regression






### Logistic Regression



### Neural Network



### Support Vector Machine (SVM)





## Unsupervised Learning

### K-means



### Principal Component Analysis (PCA)



### Anomaly Detection



## Special Applications

### Recommender System




### Large Scale Machine Learning




## Advice on building a Machine Learning System

### Bias/Variance



### Regularization



### Evaluation




### Learning Curve




### Error Analysis



### Ceiling Analysis