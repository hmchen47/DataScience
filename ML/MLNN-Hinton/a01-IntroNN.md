# Introduction to Neural Networks

Author: Matthew Stewart

[URL](https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c)


## The motivation for Neural Networks

+ regressions (and Ridge, LASSO, etc.): methods that are centered around modeling and prediction of a quantitative response variable

+ classification problem: the response variable is categorical

+ binary classification problem: the goal is to attempt to classify each observation into a category (such as a class or cluster) defined by Y, based on a set of predictor variables X.

+ Logistic regression
  + the problem of estimating a probability that someone has heart disease, P(y=1), given an input value X.
  + the logistic function, to model P(y=1):

    $$P(Y=1) = \frac{e^{\beta_0+\beta_1 X}}{1 + e^{\beta_0+\beta_1 X}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$

  + general shape: the model will predict P(y=1) with an S-shaped curve
  + $\beta_0$ shifts the curve right or left by $c = − \beta_0 / \beta_1$, whereas $\beta_1$ controls the steepness of the S-shaped curve.
  + change of the $beta_0$ value to move offset
  + change of the $beta_1$ value to distort gradient

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;margin: 0.5em;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*PD7DZlFWkYCxrg5Y1-OOXQ.png" style="margin: 0.1em;" alt="Diagram of logistic regression" title="Diagram of logistic regression" width=250>
      <img src="https://miro.medium.com/max/875/1*B0W_JthGRm6NFEvtD3ZxIA.png" style="margin: 0.1em;" alt="Diagram of logistic regression with beta_0" title="Diagram of logistic regression with beta_0=80" width=250>
      <img src="https://miro.medium.com/max/875/1*YpPeJSaOwD0Pv83KQgj3Iw.png" style="margin: 0.1em;" alt="Diagram of logistic regression with beta_1 = 1.0" title="Diagram of logistic regression with beta_1 = 1.0" width=250>
    </a>
  </div>

+ Optimization of Logistic Regression
  + using a loss function in order to quantify the level of error that belongs to our current parameters
  + find the coefficients that minimize this loss function
  + the parameters of the neural network have a relationship with the error the net produces
  + gradient descent:
    + changing the parameters using an optimization algorithm
    + useful for finding the minimum of a function

  + the loss function or the objective function

    $$\mathcal{L}(\beta_0, \beta_1) = - \sum_i \left[ y_i \log(p_i) + ( 1- y_i) \log(1 - p_i)\right]$$

+ Neural Network Algorithm

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*QIKMKejAH9cjXxe-PIIU7g.png" style="margin: 0.1em;" alt="Formulation of Neural Networks" title="Formulation of Neural Networks" width=600> <br/>
      <img src="https://miro.medium.com/max/1250/1*yaFDjDACzD1cDSERb7U1Sw.png" style="margin: 0.1em;" alt="Example of Neural Networks" title="Example of Neural Networks" width=600>
    </a>
  </div>

  + weights in neural networks: these regression parameters of our various incoming functions
  + passed to an activation function which decides whether the result is significant enough to 'fire' the node
  + start with some arbitrary formulation of values in order for us to start updating and optimizing the parameters
  + assessing the loss function after each update and performing gradient descent

+ Ways to minimize the loss function
  + Descent
    + The value of $w$ to minimize $\mathcal{L}(w)$
    + to find the optimal point of a function$\mathcal{L}(w)$: $\frac{d \mathcal{L}(W)}{dW} = 0$
    + find the $w$ that satisfies the equation
  + more flexible method
    + start from any point and then determine which direction to go to reduce the loss (left or right in this case)
    + calculate the slope of the function at this point
    + then shift to the right if the slope is negative or shift to the left if the slope is positive

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*qbvfebFdO7rxU4QVl6tQtw.png" style="margin: 0.1em;" alt="Diagram of the loss function" title="Diagram of the loss function" height=200>
      <img src="https://miro.medium.com/max/875/1*4l_ZpZRZ6mwKAXWo4Q20QA.png" style="margin: 0.1em;" alt="Diagram of the loss function with starting point" title="Diagram of the loss function with starting point" height=200>
    </a>
  </div>


## Gradient Descent

+ Gradient descent
  + an iterative method for finding the minimum of a function
  + a.k.a. delta rule
  + Making a step means: $w_{new} = w^{old} + step$
  + Opposite direction of the derivative means: $w^{new} = w^{old} - \lambda \frac{d\mathcal{L}}{dw}$
  + change to move conventional notation: $w^{(i+1)} = w^{(i)} - \lambda \frac{d\mathcal{L}}{dw}$
  + learning rate ($\lambda$):
    + large learning rate:
      + put more weight on the derivative
      + made larger step for each iteration of the algorithm
    + smaller learning rate
      + less weight is put on the derivative
      + smaller steps made for each iteration

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*MizSwb7-StSLiWlI2MKsxg.png" style="margin: 0.1em;" alt="Illustration of learning rate" title="Illustration of learning rate" width=250>
    </a>
  </div>

+ Considerations for gradient descent
  + derive the derivatives
  + know what the learning rate is or how to set it
  + avoid local minima
  + the full loss function includes summing up all individual 'errors'

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/625/1*tIqU7GK--aJ-SOdOBrh37Q.png" style="margin: 0.1em;" alt="Illustration of local & global optimal" title="Illustration of local & global optimal" width=250>
      <img src="https://miro.medium.com/max/875/1*MwnXifl-uLdTrjjxiNCDJw.png" style="margin: 0.1em;" alt="Network getting stuck in local minima" title="Network getting stuck in local minima" width=200>
      <img src="https://miro.medium.com/max/875/1*K7HNhO3Fsedvx94psTpBHA.png" style="margin: 0.1em;" alt="Network reach global minima" title="Network reach global minima" width=200>
    </a>
  </div>

+ Local minimum: problematic for neural networks

+ Batch and stochastic gradient descent
  + use a batch (a subset) of data as opposed to the whole set of data, such that the loss surface is partially morphed during each iteration
  + the loss (likelihood) function used to derive the derivatives for iteration $k$

    $$\mathcal{L}^k = - \sum_{i \in b^k} \left[ y_i \log(p_i) + (1 - p_i)\log(1 - p_i) \right]$$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*OLk3R-i5C_oVfRjPd8niPA.png" style="margin: 0.1em;" alt="start off with the full loss (likelihood) surface, and our randomly assigned network weights provide us an initial value" title="Randomly assigned network weights" width=300>
      <img src="https://miro.medium.com/max/875/1*EqJnB-tlbjCsR-bIIk_YPw.png" style="margin: 0.1em;" alt="select a batch of data, perhaps 10% of the full dataset, and construct a new loss surface" title="Batch and Full sets of data" width=300>
      <img src="https://miro.medium.com/max/875/1*JkzxQTDszPNnMNbZ7xjc_w.png" style="margin: 0.1em;" alt="perform gradient descent on this batch and perform our update" title="perform gradient descent and update" width=300><br/>
      <img src="https://miro.medium.com/max/875/1*pkB-a5CHXilyqiCgjOzv-g.png" style="margin: 0.1em;" alt="select a new random subset of the full data set and again construct our loss surface" title="new random subset" width=300>
      <img src="https://miro.medium.com/max/875/1*lFfeBDkyVUWtKrMf2S74HQ.png" style="margin: 0.1em;" alt="perform gradient descent on this batch and perform our update" title="perform gradient descent on new batch" width=300>
      <img src="https://miro.medium.com/max/875/1*3acc4wR7Vz-768LdAQQHiw.png" style="margin: 0.1em;" alt="continue this procedure again with a new subset" title="continue with a new subset" width=300><br/>
      <img src="https://miro.medium.com/max/875/1*nY08LihdetZjB9svFqi1Hw.png" style="margin: 0.1em;" alt="perform our update" title="perform update" width=300>
      <img src="https://miro.medium.com/max/875/1*xg-JYwr6K215lh0-y8R5PQ.png" style="margin: 0.1em;" alt="continues for multiple iterations" title="perform multiple iterations" width=300>
      <img src="https://miro.medium.com/max/875/1*jE7iqvFS6oOWyZ-7lvrTUg.png" style="margin: 0.1em;" alt="converge to the global minimum" title="converge to global minimum" width=300>
    </a>
  </div>




