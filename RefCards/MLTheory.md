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

#### Model: Learn Regression

+ [Simple Linear Regression](../ML/ML-Stanford/02-ModelCost.md#cost-function-intuition-ii):

	+ Hypothesis: $h_\theta (x) = \theta_0 + \theta_1 \cdot x$

  + Parameters: $\theta_0$, $\theta_1$

  + Cost Function: $J(\theta_0, \theta_1) = \displaystyle \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$

  + Goal: $\displaystyle \min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$


+ [Multivariate Linear Regression Model](../ML/ML-Stanford/04-LRegMVar.md#gradient-descent-for-multiple)

  + Hypothesis function

    $$h_\theta(x) =\theta^T x = \theta_0 x_0 + \theta_1 x_1 + \ldots + \theta_n x_n \Longrightarrow \theta^T \cdot X$$

  + Parameters: 
  
    $$\quad \theta_0, \theta_1, \ldots, \theta_n \Longrightarrow \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \ldots \\ \theta_n \end{bmatrix}\quad$$

    $\theta$: a $(n+1)$-dimensional vector

  + Cost function:

    $$J(\theta) = J(\theta_0, \theta_1, \ldots, \theta_n) = \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2$$



#### Cost Function: Learn Regression

+ [Squared error function](../ML/ML-Stanford/02-ModelCost.md#cost-function-intuition-ii): $J(\theta_0, \theta_1)$

	An average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

	$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$

+ [Multivariate Cost Function](../ML/ML-Stanford/04-LRegMVar.md#gradient-descent-for-multiple)

  $$J(\theta) = J(\theta_0, \theta_1, \ldots, \theta_n) = \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2$$



#### Gradient Descent: Linear Regression

+ [Simplest Gradient descent](../ML/ML-Stanford/02-ModelCost.md#gradient-descent)
  + Objective: Have some function $J(\theta_0, \theta_1)$ <br/>
    Want $\;\;\displaystyle \min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$
  + Outline
    + start with some $\theta_0, \theta_1$
    + keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up with at a minimum


+ [Simplest Gradient descent algorithm](../ML/ML-Stanford/02-ModelCost.md#gradient-descent):

   Repeat until convergence {

	<span style="padding-left: 2em;"/>$$\theta_j := \theta_j - \alpha \displaystyle \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$

	<span style="padding-left: calc(50vw - 10em);">(simultaneously update</span> $i = 0, 1$)<br/>
	}

	+ $\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)\;$: derivative; sign (+, -) as slope and value as steepness
	+ $:=\;$: assignment, take the right-hand side value asn assign to the symbol right-hand side
	+ $=\;$: truth association, comparison
	+ $\alpha\;$: learning rate, step size


+ [Multivariate gradient decent](../ML/ML-Stanford/02-ModelCost.md#gradient-descent)
  + Objective: Have some function $J(\theta)$ where $\theta = (\theta_0, \theta_1, \ldots, \theta_n)$ <br/>
    Want $\;\;\displaystyle \min_{\theta} J(\theta)$
  + Outline
    + start with some $\theta$
    + keep changing $\theta$ to reduce $J(\theta)$ until we hopefully end up with at a minimum

+ [Multivariate linear repression ($n \geq 1$) algorithm](../ML/ML-Stanford/04-LRegMVar.md#gradient-descent-for-multiple)

  Repeat {

    $$\theta_j := \theta_j -\alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})= \theta_j -\alpha \frac{\partial}{\partial \theta_j} J(\theta)$$
    <span style="padding-top: 0.5em; padding-left: calc(50vw - 5em);"> (simultaneously update </span> $\theta_j \;$ for $j=0, 1, \ldots, n$)<br/>
  }

  Extended version: with $x_0^{(i)} = 1$

    $$\begin{array}{ccc}
      \theta_0 &:=& \theta_0 -\alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_0^{(i)} \\\\
      \theta_1 &:=& \theta_1 -\alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \\\\
      \theta_2 &:=& \theta_2 -\alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \\\\
      & \cdots &
    \end{array}$$



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