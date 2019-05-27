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

### Linear Regression

#### Model: Linear Regression

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



#### Cost Function: Linear Regression

+ [Squared error function](../ML/ML-Stanford/02-ModelCost.md#cost-function-intuition-ii): $J(\theta_0, \theta_1)$

	An average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

	$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$

+ [Multivariate Cost Function](../ML/ML-Stanford/04-LRegMVar.md#gradient-descent-for-multiple)

  $$J(\theta) = J(\theta_0, \theta_1, \ldots, \theta_n) = \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2$$

+ [Convergence](../ML/ML-Stanford/06-Logistic.md#logistic-regression-model)

  $$J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \dfrac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2$$

  <br/>

  $$\text{Cost}(h_\theta(x^{(i)}, y^{(i)})) = \dfrac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2 \quad \Longrightarrow \quad \text{Cost}(h_\theta(x, y)) = \dfrac{1}{2} (h_\theta(x) - y)^2  \quad \Rightarrow \quad \text{Convex}$$


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


+ The Gradient Descent rule can be expressed as:

  $$\theta := \theta - \alpha \nabla J(\theta)$$

  Where $\nabla J(\theta)$ is a column vector of the form:

  $$\nabla J(\theta) = \begin{bmatrix} \dfrac{\partial J(\theta)}{\partial \theta_0} \\\\ \dfrac{\partial J(\theta)}{\partial \theta_1} \\ \vdots \\  \dfrac{\partial J(\theta)}{\partial \theta_n} \end{bmatrix}$$

  The $j$-th component of the gradient is the summation of the product of two terms:

  $$\begin{array}{ccc} \dfrac{\partial J(\theta)}{\partial \theta_j} & = &\dfrac{1}{m} \displaystyle \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \\\\ & = & \dfrac{1}{m} \displaystyle \sum_{i=1}^m x_j^{(i)} \cdot (h_\theta(x^{(i)}) - y^{(i)}) \end{array}$$

  Sometimes, the summation of the product of two terms can be expressed as the product of two vectors.

  Here, $x_j^{(i)}$, for $i = 1, \ldots, m$, represents the $m$ elements of the j-th column, $\vec{x_j}$, of the training set $X$.

  The other term $\left(h_\theta(x^{(i)}) - y^{(i)} \right)$ is the vector of the deviations between the predictions $h_\theta(x^{(i)})$ and the true values $y^{(i)}$. Re-writing $\frac{\partial J(\theta)}{\partial \theta_j}$, we have:

  $$\begin{array}{ccc} \dfrac{\partial J(\theta)}{\partial \theta_j} & = & \dfrac{1}{m} \vec{x_j}^T (X\theta - \vec{y}) \\\\ \nabla J(\theta) &=& \dfrac{1}{m} X^T (X\theta - \vec{y}) \end{array}$$

  Finally, the matrix notation (vectorized) of the Gradient Descent rule is:

  $$\theta := \theta - \dfrac{\alpha}{m} X^T (X\theta - \vec{y})$$



#### [Polynomial Regression](../ML/ML-Stanford/04-LRegMVar.md#features-and-polynomial-regression)

+ Polynomial regression

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="url">
      <img src="../ML/ML-Stanford/images/m04-04.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
    </a></div>
  </div>

  $$\theta_0 + \theta_1 x + \theta_2 x^2 \quad \text{or} \quad \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 \quad \text {or} \quad \ldots$$

  + If cubic model fits,

      $$\begin{array}{rcl}
          h_\theta(x) & = & \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 \\\\
          & = & \theta_0 + \theta(size) + \theta_2 (size)^2 + \theta_3 (size)^3
      \end{array}$$

      where $x_1 = (size), x_2 = (size)^2, x_3 = (size)^3$

+ Choose of features

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="url">
      <img src="../ML/ML-Stanford/images/m04-05.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
    </a></div>
  </div>

  $$\begin{array}{rcl}
    h_\theta(x) & = & \theta_0 + \theta_1 (size) + \theta_2 (size)^3 \\\\
    h_\theta(x) & = & \theta_0 + \theta_1 (size) + \theta_2 \sqrt{(size)}
  \end{array}$$


#### [Normal Equation](../ML/ML-Stanford/04-LRegMVar.md#normal-equation)

+ Normal equation: Method to solve for $\theta$ analytically.

+ Generalized: for $\; \theta \in \mathbb{R}^{n+1}$

  $$J(\theta_0, \theta_1, \ldots, \theta_m) = \dfrac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$
  <br/>

  $$\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_m \end{bmatrix} = (X^TX)^{-1} X^T y$$
  <br/>

  $$\dfrac{\partial}{\partial \theta_j} J(\theta) = \cdots = 0, \quad \forall j \Longrightarrow \text{solve for } \theta_0, \theta_1, \ldots, \theta_n$$


+ comparison of gradient descent and the normal equation:
  + Gradient descent
    + Need to choose $\alpha$
    + Need many iterations
    + $\mathcal{O}(kn^2)$
    + Works well when $n$ is large
  + Normal Equation
    + Not need to choose $\alpha$
    + No need to iterate
    + $\mathcal{O}(n^3)$, need to calculate inverse of $X^TX$ 
    + Slow if $n$ is very large

+ [What if $X^TX$ is non-invertible?](../ML/ML-Stanford/04-LRegMVar.md#normal-equation-noninvertibility)
  + Redundant features (linearly dependent)
  + Too many features (e.g. $m \leq n$): Delete some features, or use regularization


### Logistic Regression

#### Model: Logistic Regression

+ [Classification problem](../ML/ML-Stanford/06-Logistic.md#cost-function)
  + Training set: $\quad \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)})\}$
  + $m$ examples: $\quad x \in \begin{bmatrix} x_0 \\ x_1 \\ \cdots \\ x_n \end{bmatrix} \in \mathbb{R}^{n+1} \quad\quad x_0 = 1, y \in \{0, 1\}$
  + Hypothesis function: $\quad h_\theta(x) = \dfrac{1}{1 + e^{-\theta^Tx}}$
  + How to choose parameters $\theta$?

+ [Logistic Regression Model](../ML/ML-Stanford/06-Logistic.md#hypothesis-representation)
  + Want: $\quad 0 \leq h_\theta (x) \leq 1$

      $$h_\theta(x) = \theta^T x$$
  + Sigmoid/Logistic function
    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div style="background-color: white;"><a href="https://en.wikipedia.org/wiki/Logistic_function">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png" style="margin: 0.1em;" alt="Sigmoid function" title="Sigmoid curve" width="250">
      </a></div>
    </div><br/>

    $$h_\theta(x) = \dfrac{1}{1+e^{-x}} \quad \Longrightarrow \quad g(z) = h_\theta(x) = g(\theta^Tx) = \dfrac{1}{1 + e^{-\theta^T x}}$$

  + Parameter: $\theta \;\; \longleftarrow \;\;$ find the value to fit parameter

+ [Decision Boundary](../ML/ML-Stanford/06-Logistic.md#decision-boundary)

  $$h_\theta(x) = g(\theta^T x) = P(y=1 | x; \theta)$$

  + Logistic Function

    $$\begin{array}{rcl} g(z) =  \dfrac{1}{1 + e^{-z}} & & \\\\ z=0,\quad e^0=1 & \Rightarrow & g(z)=1/2 \\ z \rightarrow  \infty, \quad e^{−\infty} \rightarrow  0 & \Rightarrow & g(z)=1 \\ z \rightarrow −\infty, \quad e^{\infty} \rightarrow \infty & \Rightarrow & g(z)=0 \end{array}$$

  + Suppose predict "$y = 1$" if $\;\; h_\theta(x) \geq 0.5$:

    $$g(z) \geq 0.5 \text{ when } z \geq 0 \quad \Longrightarrow \quad h_\theta(x) = g(\theta^Tx) \geq 0.5 \;\;\text{  whenever  } \theta^Tx \geq 0$$
  + Suppose predict "$y = 0$" if $\;\; h_\theta(x) < 0.5$:

    $$g(z) < 0.5 \text{ when } z < 0 \quad \Longrightarrow \quad h_\theta(x) = g(\theta^Tx) < 0.5 \;\;\text{  whenever  } \theta^Tx < 0$$


#### Cost Function: Logistic Regression

+ [Cost function](../ML/ML-Stanford/06-Logistic.md#simplified-cost-function-and-gradient-descent)

  + cost function, __[Cross-Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#loss-cross-entropy)__, also known as __Log Loss__,

    $$\begin{array}{rcl} J(\theta) & = & \dfrac{1}{m} \text{Cost}(h_\theta(x^{(i)}), y^{(i)}) \\\\\\ \text{Cost}(h_\theta(x), y) & = & \left\{ \begin{array}{rl} -\log(h_\theta(x)) & \;\; \text{if } y = 1 \\ -\log(1 - h_\theta(x)) & \;\; \text{if } y = 0 \end{array} \right. \\\\ & & \Downarrow \\\\ \text{Cost}(h_\theta(x), y) & = & -y \cdot \log(h_\theta(x)) - (1-y) \cdot \log(1 - h_\theta(x)) \quad y \in \{0, 1\} \\\\ \text{If } y=1 & : & \text{Cost}(h_\theta(x), y) = -y \cdot \log(h_\theta(x)) \\ \text{If } y=0 & : & \text{Cost}(h_\theta(x), y) = (1-y) \cdot \log(1 - h_\theta(x)) \end{array}$$
  
    Therefore,

    $$\begin{array}{rcl} J(\theta) & = & \dfrac{1}{m} \displaystyle \sum_{i=1}^m \text{Cost}(h_\theta(x^{(i)}), y^{(i)}) \\\\ & = & - \dfrac{1}{m} \left[ \displaystyle \sum_{i=1}^m \log(h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]  \end{array}$$

  + fit parameter $\theta$:

    $$\min_{\theta} J(\theta)$$

  + prediction with new $x$:

    Output of $h_\theta(x) = \dfrac{1}{1 + e^{\theta^Tx}} \quad \Leftarrow P(y = 1 | x; \theta)$


+ [Convergence](../ML/ML-Stanford/06-Logistic.md#logistic-regression-model):

  $$J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \text{Cost}(h_\theta(x^{(i)}), y^{(i)})$$

  <br/>

  $$\text{Cost}(h_\theta(x^{(i)}, y^{(i)})) = \dfrac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2 \quad \Longrightarrow \quad \text{Cost}(h_\theta(x, y)) = \dfrac{1}{2} (\dfrac{1}{1+ e^{-\theta^Tx}} - y)^2  \quad \Rightarrow \quad \text{Non-Convex}$$


#### Gradient Descent: Logistic Regression

+ [Simple Gradient Descent](../ML/ML-Stanford/06-Logistic.md#simplified-cost-function-and-gradient-descent)

  $$J(\theta) = - \dfrac{1}{m} \left[ \displaystyle \sum_{i=1}^m \log(h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

  Objective: $\min_{\theta} J(\theta)$

  Repeat {

    $$\theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j} J(\theta)$$
    <span style="text-align: center; padding-top: 0.5em;padding-left: calc(50vw - 2em);"> (Simultaneously update all </span> $\theta_j$)
  
  }

  $$\dfrac{\partial}{\partial \theta_j} J(\theta) = \dfrac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$



#### Vectorization: Logistic Regression

+ [Derivation of Vectorized Cost and Hypothesis function](../ML/ML-Stanford/06-Logistic.md#simplified-cost-function-and-gradient-descent)

  $$X = \begin{bmatrix} x_0^{(1)} & x_1^{(1)} & \cdots & x_n^{(1)} \\ x_0^{(2)} & x_1^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ x_0^{(m)} & x_1^{(m)} & \cdots & x_n^{(m)} \\ \end{bmatrix} \quad\quad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} \quad\quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)}  \end{bmatrix}\quad\quad  h_\theta(x) = h = g(X\theta) = \begin{bmatrix} h_\theta(x^{(1)}) \\ h_\theta(x^{(2)}) \\ \vdots \\ h_\theta(x^{(m)}) \end{bmatrix}$$

  __Cost function:__

  $$\begin{array}{rcl} J(\theta) & = & - \dfrac{1}{m} \left[ \displaystyle \sum_{i=1}^m \log(h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \\\\ & = & \dfrac{-1}{m} \left[ \underbrace{\sum_{i=1}^m (y^{(i)} \cdot \log(h_\theta(x^{(i)})))}_{(A)} + \underbrace{\sum_{i=1}^m (1 - y^{(i)})\cdot \log(1 - h_\theta(x^{(i)}))}_{(B)}   \right]
  \end{array}$$

  Part (A): 

  $$\begin{array}{rcl} (A) & = & \sum_{i=1}^m (y^{(i)} \cdot \log(h_\theta(x^{(i)}))) = y^{(1)} \cdot \log(h_\theta(x^{(1)})) + y^{(2)} \cdot \log(h_\theta(x^{(2)})) + \ldots + y^{(m)} \cdot \log(h_\theta(x^{(m)})) \\\\ & = & \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} \begin{bmatrix} \log(h_\theta(x^{(1)})) \\ \log(h_\theta(x^{(2)})) \\ \vdots \\ \log(h_\theta(x^{(m)})) \end{bmatrix} = y^T \cdot \log \left(\begin{bmatrix} h_\theta(x^{(1)} \\ h_\theta(x^{(2)})) \\ \vdots \\ h_\theta(x^{(m)}) \end{bmatrix} \right) = y^T \cdot \log(h) \end{array}$$

  Part (B):

  $$(B) = \sum_{i=1}^m (1 - y^{(i)})\cdot \log(1 - h_\theta(x^{(i)})) = ( 1 - y)^T \cdot \log(1-h)$$

  Therefore,

  $$J(\theta) = \dfrac{1}{m} \left[ -y^T  \cdot \log(h) - (1-y)^T \cdot \log(1-h)  \right]$$

  __Gradient Descent:__

  $$\begin{array}{rcl} \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j} J(\theta) & \text{ and } & \dfrac{\partial}{\partial \theta_j} J(\theta) = \dfrac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \\\\ \theta_j := \theta_j & - \;\; \dfrac{\alpha}{m} & \left( \begin{bmatrix} x_j^{(1)} & x_j^{(2)} & \cdots & x_j^{(m)} \end{bmatrix} \begin{bmatrix} h_\theta(x^{(1)}) \\ h_\theta(x^{(2)}) \\ \vdots \\ h_\theta(x^{(m)}) \end{bmatrix} \right)\end{array}$$

  <br/>

  $$\begin{array}{rcl} 
    \theta & := & \theta - \alpha \dfrac{1}{m} \begin{bmatrix} \sum_{i=1}^m (h_\theta(x^{(i)})) \cdot x^{(i)}_0 - \sum_{i=1}^m y^{(i)} \cdot x^{(i)}_0 \\ \sum_{i=1}^m (h_\theta(x^{(i)})) \cdot x^{(i)}_1 - \sum_{i=1}^m y^{(i)} \cdot x^{(i)}_0 \\ \vdots \\ \sum_{i=1}^m (h_\theta(x^{(i)})) \cdot x^{(i)}_n - \sum_{i=1}^m y^{(i)} \cdot x^{(i)}_n  \end{bmatrix}  =  \theta -\alpha \dfrac{1}{m} \left( \begin{bmatrix} \sum_{i=1}^m (h_\theta(x^{(i)})) \cdot x^{(i)}_0 \\ \sum_{i=1}^m (h_\theta(x^{(i)})) \cdot x^{(i)}_1 \\ \vdots \\ \sum_{i=1}^m (h_\theta(x^{(i)})) \cdot x^{(i)}_n \end{bmatrix} - \begin{bmatrix} \sum_{i=1}^m y^{(i)} \cdot x^{(i)}_0 \\ \sum_{i=1}^m y^{(i)} \cdot x^{(i)}_1 \\ \vdots \\ \sum_{i=1}^m y^{(i)} \cdot x^{(i)}_n \end{bmatrix} \right) \\\\ & = & \theta - \dfrac{\alpha}{m} \left( \begin{bmatrix} x_0^{(1)} & x_0^{(2)} & \cdots & x_0^{(m)} \\ x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(m)} \\ \vdots & \vdots & \ddots & \vdots \\ x_n^{(1)} & x_n^{(2)} & \cdots & x_n^{(m)} \end{bmatrix} \begin{bmatrix} h_\theta(x_0^{(1)})  \\ h_\theta(x^{(2)}) \\ \vdots  \\ h_\theta(x^{(m)})  \end{bmatrix} - \begin{bmatrix} x_0^{(1)} & x_0^{(2)} & \cdots & x_0^{(m)} \\ x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(m)} \\ \vdots & \vdots & \ddots & \vdots \\ x_n^{(1)} & x_n^{(2)} & \cdots & x_n^{(m)} \end{bmatrix} \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix} \right) \\\\ & = &  \theta - \dfrac{\alpha}{m} \left( X^T \cdot g(X\theta) - X^T \cdot y \right) = \theta - \dfrac{\alpha}{m}\; X^T \left( g(X\theta) - y \right)
  \end{array}$$


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


### [Learning Rate $\alpha$](../ML/ML-Stanford/04-LRegMVar.md#gradient-descent-in-practice-ii-learning-rate)

+ Learning rate & Gradient Descent

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.coursera.org/learn/machine-learning/supplement/TnHvV/gradient-descent-in-practice-ii-learning-rate">
      <img src="../ML/ML-Stanford/images/m04-03.png" style="margin: 0.1em;" alt="Learning rate vs gradient descent" title="Learning rate vs gradient descent" width="350">
    </a></div>
  </div>

  + $J(\theta) \uparrow$ as number of iterations $\uparrow \quad \Longrightarrow \quad$ gradient not working, $\alpha$ too big
  + For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration.
  + If $\alpha$ is too small, gradient descent can be slow to converge.

+ Summary
  + If $\alpha$ too small: slow convergence.
  + If $\alpha$ too large: $J(\theta)$ may not decrease on every iteration; may not converge; slow converge also possible
  + Best practice: to choose $\alpha$, try

    $$\ldots, 0.001, 0.003, , 0.01, 0.03, 0.1, 0.3, 1, \ldots$$

### [Optimization](../ML/ML-Stanford/06-Logistic.md#advanced-optimization)

+ Cost function $J(\theta)$. Objective: $\;\; min_{\theta} J(\theta)$
+ Given $\theta$, we have code that can compute
  + $J(\theta)$
  + $\dfrac{\partial}{\partial \theta_j} J(\theta) \quad \forall \;j = 0, 1, \ldots, n$
+ Gradient descent:

  Repeat{

    $$\theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j} J(\theta)$$
  }
+ Optimization algorithms:
  + Gradient descent
  + Conjugate gradient
  + BFGS
  + L-BFGS
+ Advantages: 
  + No need to manually pick $\alpha$
  + Often faster than gradient descent
+ Disadvantages:
  + More complex


### [Multiclass Classification: One-vs-all](../ML/ML-Stanford/06-Logistic.md#multiclass-classification-one-vs-all)

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/supplement/HuE6M/multiclass-classification-one-vs-all">
    <img src="../ML/ML-Stanford/images/m06-09.png" style="margin: 0.1em;" alt="Ways to classify the classes" title="One-vs-All" width="350">
  </a></div>
</div>

+ Hypothesis: $h_\theta^{(i)} = P(y=i | x; \theta) \quad \forall \; i$
+ Train a logistic regression classifier $\;\; h_\theta^{(i)}(x)\;$ for each class $i$ to predict the probability that $y=i$.
+ On a new input $x$ to make a prediction, pick the class $i$ that maximizes $\;\max_i h_\theta^{(i)}(x)$.


### Bias/Variance



### Regularization



### Evaluation




### Learning Curve




### Error Analysis



### Ceiling Analysis