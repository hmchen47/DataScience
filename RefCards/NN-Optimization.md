
# Optimization for Neural Networks

## Issues and Algorithms for Optimization

### Challenges with optimization

+ [Convex optimization](../ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + a function in which there is only one optimum, corresponding to the global optimum (maximum or minimum)
  + no concept of local optima for convex optimization problems, making them relatively easy to solve

+ [Non-convex optimization](/ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + a function which has multiple optima, only one of which is the global optima
  + Maybe very difficult to locate the global optima depending on the loss surface

+ [Neural network](../ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + loss surface: minimize the prediction error of the network
  + interested in finding the global minimum on this loss surface

+ [Multiple problems on neural network training](/ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + What is a reasonable learning rate to use?
    + small learning rate: long to converge
    + large learning rate: not converge
  + How do we avoid getting stuck in local optima?
    + one local optimum may be surrounded by a particularly steep loss function
    + difficult to escape this local optimum
  + What if the loss surface morphology changes?
    + no guarantee found global minimum remain indefinitely
    + trained dataset not representative of the actual data distribution
    + different dataset might with different loss surface
    + importance: make the training and test datasets representative of the total data distribution


### Local optima and Saddle Points

+ [Local optima](../ML/MLNN-Hinton/a03-Optimization.md#local-optima)
  + viewed as a major problem in neural network training
  + using insufficiently large neural networks, most local minima incur a low cost
  + not particularly important to find the true global minimum
  + a local minimum with reasonably low error is acceptable

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*fGx-IJkvPLuurfR9VWlu2g.png" style="margin: 0.1em;" alt="Curve of loss function with local and global minimum" title="Curve of loss function with local and global minimum" width=350>
    </a>
  </div>

+ [Saddle pints](../ML/MLNN-Hinton/a03-Optimization.md#saddle-points)
  + more likely than local minima in high dimensions
  + more problematic than local minima because close to a saddle point the gradient can be very small
  + gradient descent results in negligible updates to the network and network training will cease

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/0*GuknkQNZ8pQqDGsr.jpg" style="margin: 0.1em;" alt="Saddle point — simultaneously a local minimum and a local maximum." title="Saddle point — simultaneously a local minimum and a local maximum." width=350>
    </a>
  </div>

+ [Rosenbrook function](../ML/MLNN-Hinton/a03-Optimization.md#saddle-points)
  + often used for testing the performance of optimization algorithms on saddle points
  + formula $f(x, y) = (a - x)^2 + b(y - x^2)^2$ with global minimum at $(x, y) = (a, a^2)$
  + a non-convex function with a global minimum located within a long and narrow valley
  + difficult to converge to the global minimum due to the flat valley
  + flat valley with small gradients makes it difficult for gradient-based optimization procedures to converge

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/750/1*2K4R96WUViI8r5zosC9F5Q.png" style="margin: 0.1em;" alt="A plot of the Rosenbrock function of two variables. Here a=1,b=100, and the minimum value of zero is at (1,1)." title="A plot of the Rosenbrock function of two variables. Here a=1,b=100, and the minimum value of zero is at (1,1)." height=250>
      <img src="https://miro.medium.com/max/750/1*zUFSP2xTzbOyXg4_ewOf9Q.gif" style="margin: 0.1em;" alt="Animation of Rosenbrock’s function of three variables." title="Animation of Rosenbrock’s function of three variables." height=250>
    </a>
  </div>


### Poor Conditioning

+ [Derivative Issues](../ML/MLNN-Hinton/a03-Optimization.md#poor-conditioning)
  + ill-conditioned derivatives of the error function
  + reflected in error landscapes containing many saddle points and flat areas

+ [Hessian matrix](../ML/MLNN-Hinton/a03-Optimization.md#poor-conditioning)
  + a square matrix of second-order partial derivatives of a scalar-valued function
  + the Hessian describes the local curvature of a function of many variables

  \[H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\\\ \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}\]

  + used to determine whether a given stationary points is a saddle point or not
  + full Hessian matrix takes $\mathcal{O}(n^2)$ memory, infeasible for high dimensional functions such as the loss functions of neural networks
  + use [truncated-Newton](https://en.wikipedia.org/wiki/Truncated_Newton_method) and [quasi-Newton](https://en.wikipedia.org/wiki/Quasi-Newton_method) algorithms to optimize
  + the quasi-Newton family of algorithms using approximations to the Hessian
  + [Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm): the most popular quasi-Newton algorithms

+ Neural network:
  + the Hessian matrix is poorly conditioned - the output changes rapidly for a small change of input
  + undesirable property: the optimization process is not particularly stable
  + learning is slow despite the presence of strong gradients because oscillations slow the learning process down

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*hT12fHjYZJPXCCxjCWvM3w.png" style="margin: 0.1em;" alt="slow learning w.r.t poorly conditioned" title="slow learning w.r.t poorly conditioned" width=250>
    </a>
  </div>


### Vanishing/Exploding Gradients

+ [Deep neural network](../ML/MLNN-Hinton/a03-Optimization.md#vanishingexploding-gradients)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*Hae-goX40dfHmNFmgd5jZw.png" style="margin: 0.1em;" alt="text" title="caption" width=350>
    </a>
  </div>

  + affine transformation followed by an activation function (a single layer)

    \[\begin{array}{lll} \text{Linear} \qquad & h_i = W_x & \\ \text{activation} & h_i = W h_{i-1}, & i = 2, \dots, n\end{array}\]

  + output for an $n$-layer network

    \[\text{Suppose } \mathbf{W} = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}: \qquad \begin{bmatrix} h_1^1 \\ h_2^1 \end{bmatrix} = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \quad \cdots \begin{bmatrix} h_1^n \\ h_2^n \end{bmatrix} = \begin{bmatrix} a^n & 0 \\ 0 & b^n \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\]

+ [Vanishing and Exploding](../ML/MLNN-Hinton/a03-Optimization.md#vanishingexploding-gradients)
  + Two possible cases depending on the magnitude of $a$ and $b$

  \[\begin{array}{lll} \text{Suppose } x = \begin{bmatrix} 1 \\ 1 \end{bmatrix} & & \\ \text{Case 1: } a = 1, b =2: & y \rightarrow 1,\; \Delta_y \rightarrow \begin{bmatrix} n \\ n \cdot 2^{n-1} \end{bmatrix} & \quad \text{Explodes!} \\ \text{Case 2: } a = 0.5, b = 0.9: & y \rightarrow 0,\; \Delta_y \rightarrow \begin{bmatrix} 0 \\ 0 \end{bmatrix} & \quad \text{Vanishes!} \end{array}\]

    + Gradient clipping rule

      \[\text{if } \parallel g \parallel > u, \quad g \leftarrow \dfrac{gu}{\parallel g \parallel}\]

  + for $a$, $b$ less than 1
    + the gradients quickly tends to zero
    + gradient values smaller than the precision threshold recognized as zero

+ [Vanishing gradient issue](../ML/MLNN-Hinton/a10-CNNsGuide.md#relu-rectified-linear-units-layers)
  + the lower layers of the network training very slowly
  + the gradient decreasing exponentially through the layers
  + Wiki, [Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
  + Quora, [https://www.quora.com/What-is-the-vanishing-gradient-problem](https://www.quora.com/What-is-the-vanishing-gradient-problem)


### Fast Learning Algorithms

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
    <img src="../ML/MLNN-Hinton/img/a12-24.png" style="margin: 0.1em;" alt="Taxonomy of learning algorithms" title="Taxonomy of learning algorithms" width=450>
  </a>
</div>


## Momentum

+ [Momentum](../ML/MLNN-Hinton/a03-Optimization.md#momentum)
  + an inertia motion of object to move in the direction of motion
  + the general direction that the optimization algorithm is moving
  + optimization algorithm moving in a general direction, the momentum causes it to 'resist' changes in the direction
  + dampening of oscillations for high curvature surfaces
  + an added term in the objective function
  + a value in $[0, 1]$ increasing the size of the steps taken towards the minimum by trying to jump from a local minimum
  + large momentum & small learning rate: fast convergence
  + large momentum & large learning rate: skip the minimum with a huge step
  + small momentum: not reliably avoid local minima and slow down  the training of the system
  + help in smoothing out the variations, if the gradient keeps changing direction
  + right value of momentum: either learned by hit and trial or through cross-validation


## Adaptive Learning Rates

+ [Oscillations along vertical direction](../ML/MLNN-Hinton/a03-Optimization.md#adaptive-learning-rates)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*0v0zucWChoudFcJRPpB74Q.png" style="margin: 0.1em;" alt="Gradient descent oscillates along vertical direction" title="Gradient descent oscillates along vertical direction" width=300>
    </a>
  </div>

+ List of Proposals
  + AdaGrad
  + RMSProp
  + Adam


## Parameter Initialization

+ [Initialization of network weights](../ML/MLNN-Hinton/a03-Optimization.md#parameter-initialization)
  + overlooked characteristics of developing neural networks
  + poorly initialized networks determined to network performance
  + initialized with all values of zero
    + the network not learn anything at all
    + after a gradient update, all weights would be zero
  + initialized with all weights 0.5
    + actually learn something
    + prematurely prescribed some form of symmetry between neural units

+ [Randomizing weights](../ML/MLNN-Hinton/a03-Optimization.md#parameter-initialization)
  + avoid presupposing any form of a neural structure by randomizing weights according to a normal distribution
  + often done in Keras by specifying a random state

+ [Scale of initialization](/ML/MLNN-Hinton/a03-Optimization.md#parameter-initialization)
  + large values for the weights: lead to exploding gradients
  + small values for the weights: lead to vanishing gradients
  + sweet spot that provides the optimum tradeoff between these two
  + not a priori but inferred through trial and error


## Normalization

+ Purpose
  + ways to navigate the loss surface of then neural network using momentum and adaptive learning rates
  + methods of parameter initialization to minimize a prior biases within the network

+ Normalizing features before applying the learning algorithm

+ [Min-max normalization](../ML/MLNN-Hinton/a03-Optimization.md#feature-normalization)
  + simplest method to scale data
  + rescaling the range of features to scale the range in [0, 1] or [-1, 0]
  + subtracting each value by the minimum value and scaling by the range of values present in the dataset
  + Issue: highly skewed data results in many values clustered inn one location
  + Solution: taking the logarithm of the feature variable

  \[x^\prime = \frac{x - \min(x)}{\max(x) - \min(x)}\]

+ [Mean normalization](../ML/MLNN-Hinton/a03-Optimization.md#feature-normalization)
  + essentially the same as min-max normalization except the average value is subtracted from each value
  + the least common way

  \[x^\prime = \frac{x - \text{average}(x)}{\max(x) - \min(x)}\]

+ [Feature normalization](../ML/MLNN-Hinton/a03-Optimization.md#feature-normalization)
  + make each feature normalized with zero mean and unit variance
  + widely used for normalization in many machine learning algorithms
  + typically involving distance-based methods
  + general method
    + determine the distribution mean and standard variation for each feature
    + subtract the mean from each feature
    + divide the values of each feature by its standard deviation
  + Formula

    \[x^\prime = \frac{x - \mu}{\sigma}\]

    + $x$: feature vector
    + $\mu$: vector of mean feature values
    + $\sigma$: vector of SD of feature values

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
        <img src="https://miro.medium.com/max/875/1*bo1Utxme6zS2nr0IHtATRg.png" style="margin: 0.1em;" alt="Contour to represent feature normalization" title="Contour to represent feature normalization" width=350>
      </a>
    </div>


## Assessment with Beale's Function

+ [Beale's function](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
  + one of many test functions commonly used for studying the effectiveness of various optimization techniques
  + a test function accesses how well the optimization algorithms perform when in flat regions with very well shallow gradients

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/0*b6VbjuQQJVXxd_rE.jpg" style="margin: 0.1em;" alt="The Beale function." title="The Beale function." width=350>
    </a>
  </div>

  + Optimizing a function $f: A \rightarrow R$, from some set A to the real numbers is finding an element $x_0 \in A$ such that $f(x_0) \leq f(x)$ for all $x \in A$ (finding the minimum) or such that $f(x_0) \geq f(x)$ fro all $x \in A$ (finding the maximum).
  + Formula:

    $$f(x,, y) = (1.5 -x +xy)^2 + (2.25 -x + xy^2)^2 + (2.625 - x +xy^3)^2$$

    Answer: $(x, y) = (3, 0.5)$

+ [Artificial landscape](../ML/MLNN-Hinton/a04-Hyperparameter.md#beales-function)
  + find a way of comparing the performance of various algorithms
    + Convergence (how fast they reach the answer)
    + Precision (how close do they approximate the exact answer)
    + Robustness (so they perform well for all functions or just a small subset)
    + General performance (e.g., computational complexity)
  + analogous to the loss surface of a neural network
  + goal of NN training: find the global minimum on the loss surface by performing some form of optimization - typically stochastic gradient


## Implementation with Keras

+ [Keras](../ML/MLNN-Hinton/a04-Hyperparameter.md#a-keras-refresher)
  + a Python library for deep learning that can run on top of both Theano or TensorFlow, tow powerful Python libraries for fast numerical computing created and released by Facebook and Google, respective
  + developed to make developing deep learning models as fast and easy and easy as possible for research and practical applications
  + built on the idea of a model
  + Sequential model: a sequence of layers, a linear stack of layers

+ [Summarize the construction of deep learning models in Keras using the Sequential model](../ML/MLNN-Hinton/a04-Hyperparameter.md#a-keras-refresher)
  1. Define model: create a Sequential model and add layers
  2. Compile model: specify loss function and optimizers and call the `.compile()` function
  3. Fit model: train the model on data by calling the `.fit()` function
  4. Make prediction: use the model to generate predictions on new data by calling functionbs such as `.evaluate()` or `.predict()`

+ [Callbacks](../ML/MLNN-Hinton/a04-Hyperparameter.md#callbacks-taking-a-peek-into-our-model-while-its-training)
  + how to examine the performance of the model
  + what happening in various stages of the model
  + a set of functions to be applied at given stages of the training procedure
  + get a view on internal states and statistics of the model during training
  + pass a list of callbacks (as the keyword arguments callbacks) to the `.fit()` method of th eSequential or Model classes

+ [Relevant methods of the callbacks at each stage of the training](../ML/MLNN-Hinton/a04-Hyperparameter.md#callbacks-taking-a-peek-into-our-model-while-its-training)
  + `keras.callbacks.History()`: a callback function automatically included in `.fit()`
  + `keras.callbacks.ModelCheckPoint` saves the model with its weights at a certain point in the training; e.g., a good practice to save the model weights only when an improvement is observed as measured by the `acc`
  + `keras.callbacks.EarlySStopping`: stop the training when a monitored quantity has stopped improving
  + `keras.callbacks.LearningRateScheduler`: change the learning rate during training

+ Procedure by Example
  + Step 1 - [Deciding on the network topology](../ML/MLNN-Hinton/a04-Hyperparameter.md#step-1---deciding-on-the-network-topology)
    + Preprocess the data
  + Step 2 - [Adjusting the `learning rate`](../ML/MLNN-Hinton/a04-Hyperparameter.md#step-2---adjusting-the-learning-rate)
    + Stochastic Gradient Descent (SGD)
    + Typical values for hyperparameter: $lr = 0.01$, $decay = 1e^{-6}$, $momentum = 0.9$, and nesterov = True
    + Learning rate hyperparameter
    + Implement a learning rate adaption schedule in Keras
    + Apply a custom learning rate change using `LearningRateScheduler`
  + Step 3 - [Choosing an optimizer and a loss function](../ML/MLNN-Hinton/a04-Hyperparameter.md#step-3---choosing-an-optimizer-and-a-loss-function)
    + goal of optimization: efficiently calculate the parameters/weights that minimize the loss function
    + [types of loss functions in keras](https://github.com/keras-team/keras/blob/master/keras/losses.py)
    + Distance: the 'loss' function
    + types of loss functions: MSE (for regression); categorical cross-entropy (for classification); binary cross entropy (for classification)
  + Step 4 - [Deciding on the batch size and number of epochs](../ML/MLNN-Hinton/a04-Hyperparameter.md#step-4---deciding-on-the-batch-szie-and-number-of-epochs)
    + batch size: the number of samples propagated through the network
    + advantages od using a batch size < number of all samples
    + Disadvantages of using a batch size < number of all samples
    + epoch: a hyperparameter defines the number times that the learning algorithm will work through the entire training dataset
  + Step 5 - [Random restarts](/ML/MLNN-Hinton/a04-Hyperparameter.md#step-5---random-restarts)
    + Not implemented in Keras
    + Easily done by altering `keras.callbacks.LearningRateScheduler`
    + Resetting the learning rate after a specified number of epoch for a finite number of times

## Implementation for Cross-Validation

+ [Tuning Hyperparameters using Cross-Validation](../ML/MLNN-Hinton/a04-Hyperparameter.md#tuning-hyperparameters-using-cross-validation)
  + Use `GridSearchCV` from Scikit-Learn to try out several values for hyperparameters and compare the results
  + Cross-validation with `keras`
    + use the wrappers for the Scikit-Learn API
    + Provide a way to use Sequential Keras models (single-input only) as part of Sckikit-Learn workflow
    + wrappers
      + Scikit-Learn classifier interface: `keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`
      + Scikit-Learn regressor interface: `keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`

+ [Cross-Validation with more than one hyperparameters](../ML/MLNN-Hinton/a04-Hyperparameter.md#cross-validation-with-more-than-one-hyperparameters)
  + effectively trying out combinations of them.
  + Cross-validation in neural networks is computationally expensive.
    + each combination evaluated using the k-fold cross-validation (k is a parameter we choose)


## Momentum

### Classical Momentum

+ Overview
  + applied to both full batch or mini-batch learning
  + probably the commonest recipe for big neural nets: combining stochastic gradient descent with mini matches and momentum

+ [Momentum Method](https://trongr.github.io/neural-network-course/neuralnetworks.html)
  
  __Intuition: rolling ball.__ Modify the Delta Rule

    \[\Delta w(t) = -\varepsilon \frac{\partial E}{\partial w}(t)\]

  to include a "momentum" term

    \[\Delta w(t) = \alpha \Delta w(t-1) - \varepsilon \frac{\partial E}{\partial w} (t)\]

  + $\alpha$: a factor slightly less than 1
  + $\Delta w(t)$ remembers a little bit of its previous direction via $\alpha \Delta w(t-1)$
  + [Implementation of Stochastic Gradient Descent w/ Momentum](../ML/MLNN-Hinton/src/sgd_momentum.py)

+ [The intuition behind the momentum method](../ML/MLNN-Hinton/06-MiniBatch.md#the-momentum-method)
  + Analogy
    + a ball on the error surface
    + weight vector: the location of the ball in the horizontal plane
    + the ball starting stationary
    + initialized by following the direction of steepest descent, the gradient
    + once gaining velocity, the ball no longer in the same direction as the gradient
    + its momentum making it keep going in the previous direction
  + damping oscillations in directions of high curvature by combining gradients w/ opposite signs
    + eventually getting to a low point on the surface
    + viscosity: making the velocity die off gently on each or update
  + built up speed in directions w/ a gradient but consistent gradient

+ [Mathematical representation](../ML/MLNN-Hinton/06-MiniBatch.md#the-momentum-method)
  
  \[\begin{align*}
    \mathbf{v}(t) &= \alpha \, \mathbf{v}(t-1) - \varepsilon \frac{\partial E}{\partial \mathbf{w}}(t) \tag*{(1)} \\\\
    \Delta \mathbf{w}(t) &= \mathbf{v} (t) \tag*{(2)} \\
     &= \alpha \, \mathbf{v}(t-1) - \varepsilon \frac{\partial E}{\partial \mathbf{w}}(t) \\
     &= \alpha \, \Delta \mathbf{w} (t-1) - \varepsilon \frac{\partial E}{\partial \mathbf{w}}(t) \tag*{(3)}
  \end{align*}\]

  + Eq. (1):
    + the velocity at time $t$: the mini-batch w/ $t-1$ attenuated by a number, like $0.9$, and adding the effect of the current gradient
    + $t$: the updates of weights
    + (alpha) momentum = the viscosity (0.9)
    + effect of gradient: downhill by a given learning rate times the gradient at time $t$
    + The effect of the gradient is to increment the previous velocity.
    + The velocity also decays by $\alpha$ which is slightly less than 1.
  + Eq. (2): The weight change is equal to the current velocity.
  + Eq. (3): The weight change can be expressed in terms of the previous weight change and the current gradient.

+ [The behavior of the momentum method](../ML/MLNN-Hinton/06-MiniBatch.md#the-momentum-method)
  + error surface as a tilted plane
    + the gain of velocity from the gradient balanced by the multiplicative attenuation of the velocity due to the momentum term
    + the ball reaches a terminal velocity
    + if momentum $\rightarrow 1$, going down much faster than simple gradient descent
    + terminal velocity: as $t \rightarrow \infty$

    \[\mathbf{v}(\infty) = \frac{1}{1 - \alpha} \left( -\varepsilon \frac{\partial E}{\partial \mathbf{w}} \right)\]

    + Derivation

      \[\begin{align*}
        & v(\infty) = \alpha \, v(\infty) - \varepsilon \, \frac{\partial E}{\partial w} (t) \\
        & (1 - \alpha) \, v(\infty) = - \varepsilon \frac{\partial E}{\partial w} (t) \\
        & v(\infty) = \frac{1}{(1 - \alpha)} \, \left(-\varepsilon \frac{\partial E}{\partial w}(t)\right)
      \end{align*}\]

    + $\alpha = 0.99$: 100 times as fast as the learning rate alone
  + preventing big momentum to change quickly $\rightarrow$ difficult to find the right relative values of different weights
    + playing w/ a small momentum (e.g. 0.5) to average out sloshes in obvious ravines
    + once the large gradients disappeared and the weights stuck in a ravine the momentum
  + using a small learning rate with a big momentum to get rid of an overall learning rate
  + learning at a rate alone that would cause divergent oscillations without the momentum

+ [Formula for Momentum](../ML/MLNN-Hinton/a03-Optimization.md#classical-momentum)
  + using past gradients for updating values
  + $v$: velocity
  + more weight applied to more recent gradients, creating an exponentially decaying average of gradients

  \[\begin{array}{rcl} g &=& \frac{1}{m} \displaystyle \sum_i \Delta_\theta L(f(x^{(i)}; \theta), y^{(i)}) \\ v &=& \alpha v + (-\varepsilon g) \end{array}\]

  + $\alpha \in [0, 1)$ controls how quickly effect of past gradients decay
  + $\varepsilon$: current gradient update

+ [Compute gradient estimate](../ML/MLNN-Hinton/a03-Optimization.md#classical-momentum)

    \[g = \frac{1}{m} \sum_i \Delta_\theta L(f(x^{(i)}; \theta), y^{(i)})\]
  + Update velocity: $v = \alpha v - \varepsilon g$
  + Update parameters: $\theta = \theta + v$
  + Impacts
    + SGD w/ momentum updates no real advantage at the first few updates over vanilla SGD
    + As the number of updates increases the momentum kickstarts and allows faster convergence.


### Backpropagation for Classical Momentum

+ [Momentum method](../ML/MLNN-Hinton/a12-Learning.md#811-backpropagation-with-momentum)
  + minimizing the error function: wide oscillation of the search process w/ the gradient descent
  + traditional gradient descent: computed for each new combination of weights
  + momentum approach: compute the negative gradient direction a weighted average of the current gradient and the previous correction direction for each step
  + accelerating convergence: increasing the learning rate up to an optimal value
  + purpose: allowing the attenuation of oscillations in the iteration process

+ [Mathematical representation for momentum](../ML/MLNN-Hinton/a12-Learning.md#811-backpropagation-with-momentum)
  + A network with $n$ different weights $w_1, w_2, \dots, w_n$
  + Assumption and Notations
    + $E$: the error function
    + $\gamma$: the learning rate
    + $\alpha$: the momentum rate
  + The $i$-th correction for weight $w_k$

    \[\Delta w_k(i) = -\gamma \, \frac{\partial E}{\partial w_k} + \alpha \, \Delta w_k (i-1)\]

+ [Optimization](../ML/MLNN-Hinton/a12-Learning.md#811-backpropagation-with-momentum)
  + optimal parameters highly depends on the learning task
  + no general strategy to deal with the problem
  + tradeoffs: choosing the a specific learning and momentum rates
  + observing the oscillating behavior on backpropagation feedback rule and large momentum rates

+ [Linear associator](../ML/MLNN-Hinton/a12-Learning.md#the-linear-associator)
  + a single computing element with associated weights $w_1, w_2, \dots, w_n$
  + input: $x_1, x_2, \dots, x_n$
  + output: $w_1x_1 + w_2x_2 + \cdots + w_n x_n$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/a12-02.png" style="margin: 0.1em;" alt="Linear associator" title="Linear associator" width=200>
    </a>
  </div>

+ [Mathematical Representation for Linear Associator](../ML/MLNN-Hinton/a12-Learning.md#the-linear-associator)
  + Assumptions & Notations
    + $(\mathbf{x_1}, y_1), (\mathbf{x_2}, y_2), \dots, (\mathbf{x_p}, y_p)$: the input-output $p$ ordered pairs
    + $\mathbf{x}$: vector of input patterns w/ $n$-dimensional rows
    + $\mathbf{w}$: vector of the weights of the linear associator w/ $n$-dimensional columns
    + $\mathbf{X}$: a $p \times m$ matrix w/ $\mathbf{x_1}, \mathbf{x_2}, \dots \mathbf{x_p}$ as rows
    + $\mathbf{y}$: a column vector of the scalars $y_1, y_2, \dots, y_p$
  + the learning task objective: minimize the quadratic error

    \[\begin{align*}
    E &= \sum_{i=1}^{n} \| \mathbf{x_i} \cdot \mathbf{w} - y_i \|^2 \\
      &= \| \mathbf{X}\mathbf{w} - \mathbf{y} \|^2 = (\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y}) \\
      &= \mathbf{w}(\mathbf{X}^T \mathbf{X})\mathbf{w} -2 \: \mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{y}^T\mathbf{y}
    \end{align*}\]

  + the lengths of the principal axes: determined by the magnitude of the eigenvalues of the correlation of matrix $\mathbf{X}^T\mathbf{X}$
  + gradient descent: most effective w/ the same length when the principal axes of the quadratic form

+ [Eigenvlaues in the correlation matrix $\mathbf{X}^T\mathbf{X}$](../ML/MLNN-Hinton/a12-Learning.md#the-linear-associator)
  + the eigenvalues $\to$ the lengths of the principal axes of the error function
  + the range of possible values of $\gamma$ reduces as one of these eigenvalues much larger than the others

+ [Convergence and Divergence zones](../ML/MLNN-Hinton/a12-Learning.md#the-linear-associator)
  + parameters combinations in the boundary btw regions: stable oscillations
  + $\gamma > 4 \cdot 2/k$: not balanced with any value of $\alpha$
  + $\gamma > 1$: a geometric explosion of the iteration process
  + $1/k < \gamma < 2/k$: stable oscillation; the boundaries between regions
  + $\gamma < 1/2k$: optimal convergence speed w/ a unique $\alpha$
  + jagged line: the optimal combinations of $\gamma$ and $\alpha$

+ [Critical parameter combinations](../ML/MLNN-Hinton/a12-Learning.md#critical-parameter-combinations)
  + Backpropagation: choosing a learn rate $\gamma$ w/o any previous knowledge of the correlation matrix of the input
  + Conservative approach: choosing a very small learning rate
  + In case of a correlation matrix $\mathbf{X}^T\mathbf{X}$ with some very large eigenvalues, a given choice of $\gamma$ could led to divergence in the associated direction in weight space.

+ [Error function in weight space](../ML/MLNN-Hinton/a12-Learning.md#critical-parameter-combinations)
  + Paths for backpropagation learning w/ linear associator
  + Bounded nonlinear error function and the result of several iterations

+ [learning rate considerations](../ML/MLNN-Hinton/a12-Learning.md#critical-parameter-combinations)
  + too small: possible to get stuck in local minima
  + too big: possible oscillatory traps

+ [Remedy](../ML/MLNN-Hinton/a12-Learning.md#critical-parameter-combinations)
  + adaptive learning rates
  + statistical preprocessing of the learning set w/ decorrelation; ie. no excessively large eigenvalues of the correlation matrix


### Nesterov Momentum

+ [A better type of momentum](../ML/MLNN-Hinton/06-MiniBatch.md#the-momentum-method) (Nesterov 1983)
  + standard momentum method
    1. compute the gradient at the current location
    2. take a big jump in the direction of the updated accumulated gradient
  + Ilya Sutskever (2012)
    + a new form of momentum working better
    + inspired by the Nesterov method for optimizing convex functions
  + Nesterov approach
    1. make a big jump in the direction of the previous accumulated gradient
    2. measure the gradient where ending up and making a correction: better to correct a mistake after you have made it
  + standard vs Nesterov
    + standard: adding the current gradient and then gambling on the big jump
    + Nesterov: using the previous accumulated gradient to make the big  and then correct itself at the place

+ [Nesterov Method](https://trongr.github.io/neural-network-course/neuralnetworks.html)
  + old momentum method: calculate gradient, i.e., correct your previous mistake, at current point, then jump

    \[\Delta w(t) = \alpha \Delta w(t-1) - \varepsilon \frac{\partial E}{\partial w} (t-1)\]

  + new and improved momentum method: jump first, then correct your mistake at the description

    \[\Delta w(t) = \alpha \Delta w(t-1) - \varepsilon \frac{\partial E}{\partial w} (t)\]

+ [Nesterov momentum](../ML/MLNN-Hinton/a03-Optimization.md#nesterov-momentum)
  + Sutskever, Martens et al. "[On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf)" 2013
  + Classical vs Nesterov Momentum
    + Classical
      + first correct velocity
      + make a big step according to that velocity (and then repeat)
  + Nesterov
    + first make a step into velocity direction
    + make a correction to a velocity vector based on a new location (then repeat)

  + Hugh difference in practice
    + Apply an interim update: $\tilde{\theta} = \theta + v$
    + Perform a correction based on gradient at the interim point

      \[\begin{array}{rcl} g &=& \frac{1}{m} \sum_i \Delta_\theta L(f(x^{(i)}; \tilde{\theta}), y^{(i)}) \\ v &=& \alpha v - \varepsilon g \\ \theta & = & \theta + v \end{array}\]

    + momentum based on look-ahead slope
    + visual representation of the difference between the traditional momentum update and Nesterov momentum

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
        <img src="https://miro.medium.com/max/875/1*hJSLxZMjYVzgF5A_MoqeVQ.jpeg" style="margin: 0.1em;" alt="Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this 'looked-ahead' position." title="Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this 'looked-ahead' position." width=450>
      </a>
    </div>  



## Parameter Initialization

### Initialization Strategies

+ [Convergence w/ initialization](../ML/MLNN-Hinton/a14-Advanced.md#6-initialization-strategies)
  + convex problem w/ a small learning rate: convergence guaranteed, no matter what the initialization
  + non-convex
    + not well understood to have principled, mathematically nice initialization strategies
    + heuristics used to select a reasonable set of starting weights from which the network is trained

+ [Initialization strategies](../ML/MLNN-Hinton/a14-Advanced.md#6-initialization-strategies)
  + architecture: a fully connected layer with $m$ inputs and $n$ outputs
  + uniform distributed to obtain the initial weights

    \[ W_{ij} \sim U \left( 1\frac{1}{\sqrt{m}}, \frac{1}{sqrt{m}} \right) \]

  + popular Xavier initialization for its uniform distribution

    \[ w_{ij} \sim U \left( -\frac{6}{\sqrt{m+n}}, \frac{6}{\sqrt{m+n}} \right) \]

    + derived considering that the network consists of matrix multiplication with no nonlinearites
    + seems to perform well in practice

+ List of weight initialization: [initializers section of the Kreas documentation](https://keras.io/initializers/)


### Xavier Initialization

+ [Xavier initialization](../ML/MLNN-Hinton/a03-Optimization.md#xavier-initialization) is a simple heuristic for assigning network weights.

+ Objective: the variance to remain the same with each passing layer

+ Keep the signal from exploding to high values or vanishing to zero

+ To initialize the weights in such a way that the variance remains the same for both the input and the output

+ The weights drawn from a distribution with zero mean and a specific variance.

+ For a fully-connected layer with $m$ inputs:

  \[W_{ij} \sim N \left(0, \frac{1}{m} \right)\]

  + $m$: fan-in; the number of incoming neurons (input units in the weight tensor)
  + heuristic value: merely empirically observed to perform well


### HE Normal Initialization

+ [HE normal initialization](../ML/MLNN-Hinton/a03-Optimization.md#he-normal-initialization)
  + the same as Xavier Initialization, except that the variance multiplied by a factor of two
  + initialized the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently
  + random but differ in range depending on the size of the previous layer of neurons
  + controlled initialization hence the faster and more efficient gradient descent

+ For ReLU units

  \[W_{ij} \sim N \left(0, \frac{2}{m} \right)\]


### Bias Initialization

+ [Bias initialization](../ML/MLNN-Hinton/a03-Optimization.md#bias-initialization): how the biases of the neurons should be initialized

+ The simplest and a common way of initializing biases is to set them to zero.

+ Asymmetry breaking: provided by the small random numbers in th weights

+ ReLU non-linearity
  + using small constant values such as 0.01 for all biases
  + ensure that all ReLU units fire in the beginning and obtain and propagate some gradient

+ Main concern: avoid saturation at initialization within hidden units, ReLU by initializing biases to 0.1 instead of zero


### Pre-initialization

+ [Pre-initialization](../ML/MLNN-Hinton/a03-Optimization.md#pre-initialization):
  + common for convolutional networks used for examining images
  + involve importing the weights of an already trained network
  + used as the initial weights of the network to be trained
  + a tenable method to utilize for analyzing images with few data samples
  + underlying concept behind transfer learning


## Normalization

### Internal Covariate Shift

+ [Internal Covariate Shift](../ML/MLNN-Hinton/a03-Optimization.md#internal-covariate-shift):
  + the change in the distribution of network activation due to the change in network parameters during training
  + the parameters of a layer changed, the distribution of inputs to subsequent layers also changes
  + Issue: the shifts in input distributions tend to slow down learning, especially deep neural networks

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Dnxnj2STbo-42DfalLMi-g.png" style="margin: 0.1em;" alt="Deep neural network: multiple hidden layers" title="Deep neural network: multiple hidden layers" width=250>
    </a>
  </div>

+ Whitened inputs
  + converge faster and uncorrelated
  + internal covariate shift leads to just the opposite

+ Ref: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  + gradient descent converges much faster with feature scaling than without it


### Batch Normalization

+ [Batch normalization](../ML/MLNN-Hinton/a03-Optimization.md#batch-normalization)
  + a method intended to mitigate internal covariate shift for neral networks
  + an extension to the idea of feature standardization to other layers of the neural network

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*x3FtLuoYjWeNctiNPlTBjw.png" style="margin: 0.1em;" alt="Matrix representation of weights for hidden layers" title="Matrix representation of weights for hidden layers" width=250>
    </a>
  </div>

  + reducing overfit due to a slight regularization effect
  + similar to dropout, add some noise to each hidden layer's activations

+ [Batch normalization transformation](../ML/MLNN-Hinton/a03-Optimization.md#batch-normalization)
  + normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

    \[\begin{array}{lcl} H^\prime &=& \frac{H - \mu}{\sigma} \\ \mu &=& \frac{1}{m} \sum_i H_{i,:} \\ \sigma &=& \sqrt{\frac{1}{m} \sum_i (H - \mu)^2 + \delta}\end{array}\]

    + $\mu$: vector of mean activations across mini-batch
    + $\sigma$: vector of SD of each unit across mini-batch
  + allowing each layer of a network to learn by itself more independently of other layers
  + after shift/scale of activation outputs by some randomly initialized parameters, the weights in the next layer are no longer optimal.
  + Adding two trainable (learnable) parameters to each layer
    + normalized output multiplied by a "standard deviation" parameter ($\gamma$) and add a "mean" parameter ($\beta$)
    + let SGD do the denormalization by changing only these two wrights for each activation
    + not losing the stability of the network by changing all the weights

    \[\gamma H^\prime + \beta\]

  + For each of the N mini-batches, calculate the mean and standard deviation of the output

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
        <img src="https://miro.medium.com/max/875/1*a-B6jX8B-8lfz5ZrIFQyFw.png" style="margin: 0.1em;" alt="Add normalization operations for layer 1" title="Add normalization operations for layer 1" height=200>
        <img src="https://miro.medium.com/max/875/1*LdQ7HKaqDYtszL8R0bdb4Q.png" style="margin: 0.1em;" alt="Add normalization operations for layer 2" title="Add normalization operations for layer 2" height=200>
      </a>
    </div>

  + subsequently repeated for all subsequent hidden layers
  + differentiate the joint loss for the N mini-batches and then backpropagate through the normalization operations

+ Testing time
  + the mean and standard deviations replaced with running average collected during training time
  + same as using the population statistics instead of mini-batch statistics
  + the output deterministically depends on the input

+ Advantages
  1. reduces internal covariate shift
  2. reduces the dependence of gradients on the scale of the parameters or their initial values
  3. regularizes the model ad reduces the need for dropout, photometric distortions, local response normalization and other regularization techniques
  4. allows use of saturating nonlinearities and higher learning rates


