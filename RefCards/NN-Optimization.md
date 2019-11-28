
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



### Parameter Initialization

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


### Normalization

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


### Assessment with Beale's Function

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


### Implementation with Keras

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

### Implementation for Cross-Validation

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


## Second-order backpropagation

### Overview

+ [second-order backpropagation](../ML/MLNN-Hinton/a12-Learning.md#843-second-order-backpropagation)
  + a method to efficiently compute the Hessian of a linear network of 1-dim functions
  + used to get explicit symbolic expressions or numerical approximations of the Hessian
  + able to use in parallel computers to improve second-order learning algorithms for neural networks

+ [Methods for the determination of the Hessian matrix](../ML/MLNN-Hinton/a12-Learning.md#843-second-order-backpropagation)
  + using a graphical approach
  + reducing the whole problem to a computation by inspection
  + able to handle arbitrary topologies
  + restriction: no cycles on the network


### Second-order derivatives

+ [Second-order derivatives](../ML/MLNN-Hinton/a12-Learning.md#second-order-derivatives)
  + investigating the expressions of the form $\partial^2 F / \partial w_i \partial w_j$
  + Assumptions & Notations
    + $F$: network function
    + $w_i$ & $w_j$: network's weights

+ [Second-order computation](../ML/MLNN-Hinton/a12-Learning.md#second-order-derivatives)
  + Assumptions & Notations (left diagram)
    + $x$: the input to the network, 1-dim value
    + $F$: the network function computed at the output node with label $q$ for the given input value
    + $F_{l_1q}, F_{l_2q}, \dots, F_{l_mq}$: the network functions computed by subnetworks of the original network
    + $q$: the 1-dim function at the output node
  + the network function

    \[F(x) = g\left(F_{l_1q}(x) + F_{l_2q}(x) + \cdots + F_{l_mq}(x)\right)\]

  + objective: computing $\partial^2 F(x)/\partial w_i \partial w_j$ for two given network weights $w_i$ and $w_j$

    \[\frac{\partial^2F(x)}{\partial w_i \partial w_j} = g^{\prime\prime}(s) \frac{\partial s}{\partial w_i} \frac{\partial s}{\partial w_j} + g^{\prime}(s) \left(\frac{\partial^2 F_{l_1q}(x)}{\partial w_i \partial w_j} + \cdots + \frac{\partial^2 F_{l_mq}(x)}{\partial w_i \partial w_j}\right)\]

    + $s = F_{l_1q}(x) + F_{l_2q}(x) + \cdots + F_{l_mq}(x)$
    + the desired second-order partial derivative w/ two terms
      + the second derivative of $q$ evaluated at its input multiplied by the partial derivatives of the sum of the $m$ subnetwork functions $F_{l_1q}, \dots, F_{l_mq}$ w.r.t. $w_i$ and $w_j$
      + the second-order correction: the second derivative of $q$ multiplied the sum of the of the second-order partial derivatives of each subnetwork function w.r.t. $w_i$ and $w_j$
    + compute the first partial derivatives of any network function w.r.t. a weight

+ [The feed-forward labeling phase of the backpropagation algorithm](../ML/MLNN-Hinton/a12-Learning.md#second-order-derivatives)
  + computing a 1-dim function $f$ at each node
  + store 3 values at each node: $f(x)$, $f^{\prime}(x)$, and $f^{\prime\prime}(x)$ where $x$ represents the input to this node
  + Idea: (right diagram)
    + performing the feed-forward labeling step in the usual manner, but storing additionally at each node the second derivative of the node's function evaluated at the input
    + selecting $w_1$ and $w_2$ and deriving network function of an output node
      + the second-order partial derivative of the stored network function w.r.t. those weights: the product of the stored $g^{\prime\prime}$ value with the backpropagation path values from the output node up to weight $w_i$ and $w_j$
      + intersection of the backpropagation paths of $w_i$ and $w_j$: requiring a second-order correction
      + the second-order correction: the stored value of $g^{\prime}$ multiplied by the sum of the second-order derivative w.r.t $w_i$ and $w_j$ of all subnetwork function inputs to the node which belong to intersecting paths

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="img/a12-19.png" style="margin: 0.1em;" alt="Second-order computation" title="Second-order computation" height=150>
      <img src="img/a12-20.png" style="margin: 0.1em;" alt="Interescting paths to a node" title="Interescting paths to a node" height=150>
    </a>
  </div>

+ [Multilayer perceptron](../ML/MLNN-Hinton/a12-Learning.md#second-order-derivatives)
  + Assumption and Notations
    + model sees the left diagram
    + $w_{ih}$: a weight in the first layer of weights
    + $w_{jm}$: a weight in the second layer
    + $w_{ih}$ and $w_{jm}$: only intersect at node $m$
  + The second derivative of $F_m$ w.r.t. $w_{ih}$ and $w_{jm}$: the stored value of $f^{\prime\prime}$ multiplied by the stored output of the hidden unit $j$ and the backpropagation path up to $w_{ih}$, i.e., $w_{hm} h^{\prime} x_i$

+ [Adjustment for one weight lying in the backpropagation path of another](../ML/MLNN-Hinton/a12-Learning.md#second-order-derivatives)
  + Assumptions & Notations
    + $w_{ik}$: a weight lies in the backpropagation path of weight $w_j$
  + performing the second-order backpropagation algorithm as usual
  + the backward computation proceeds up to the point where $w_{ik}$ transports an input to a node $k$ for which a second-order correction required
  + the information transported through the edge with weight $w_{ik}$ is the subnetwork function $F_{ik}$
  + the second-order correction for the node w/ primitive function $g$

    \[g^{\prime} \frac{\partial^2 F_{ik}}{\partial w_{ik} \partial{w_j}} = g^{\prime} \frac{\partial^2 w_{ik} F_i}{\partial w_{ik} \partial w_j}\]
  
  + simplified the previous equation as

    \[g^{\prime} \frac{\partial F_i}{\partial w_j}\]
  
  + The subnetwork function $F_i$ does not depend on $w_{ik}$, thus, the second-order backpropagation method complemented by the following rule
    + the second-order correction to a node $k$ with activation function $q$ involves a weight $w_{ik}$ and a node $w_j$
    + the second-order correction is just $g^{\prime}$ multiplied by the backpropagation path value of the subnetwork function $F_i$ w.r.t. $w_j$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="img/a12-21.png" style="margin: 0.1em;" alt="Multilayer perceptron" title="Multilayer perceptron" height=200>
    </a>
  </div>


### Explicit calculation of the Hessian

+ considering the case of a single input pattern into the network

+ [Algorithm: second-order backpropagation](../ML/MLNN-Hinton/a12-Learning.md#explicit-calculation-of-the-hessian)
  1. extend the neural network
    + adding node which compute the squared difference of each component of the output and the expected target values
    + collecting all these differences at a single node whose output is the error function of the network
    + activation function of the node: the identity
  2. label all nodes in the feed-forward phase with the result of computing $f(x)$, $f^{\prime}(x)$, and $f^{\prime\prime}(x)$
    + $x$: the global input to each node
    + $f$: the associated activation function of the node
  3. starting from the error function node in the extended network, compute the second-order function $g$, compute the second-order derivative of $E$ w.r.t. two weights $w_i$ and $w_j$, by proceeding recursively in the following way
    1. the second-order derivative of the output of a node $G$ w/ activation function $g$ w.r.t. two weights $w_i$ and $w_j$
      + the product of the stored $g^{\prime\prime}$ value w/ the backpropagation path values between $w_i$ and the node $G$ and between $w_j$ and the node $G$
      + a second-order correction require if both propagation paths intersect
    2. the second-order correction equals to the product of
      + the stored $g^{\prime}$ value w/ the sum of the second-order derivative (w.r.t. $w_i$ & $w_j$) of each node whose output goes directly to $G$
      + which belongs to the intersection of the backpropagation path of $w_i$ and $w_j$
    3. special case: one of the weights connected to node $h$ directly to node $G$, the second-order corrections is just $g^{\prime}$ multiplied by the backpropagation path value of the subnetwork function $F_h$ w.r.t. $w_j$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="img/a12-22.png" style="margin: 0.1em;" alt="Multilayer perceptron - the special case" title="Multilayer perceptron - the special case" height=200>
    </a>
  </div>

### Some conclusions

+ [the Hessian matrix](../ML/MLNN-Hinton/a12-Learning.md#some-conclusions)
  + easily computing the matrix even for convoluted feed-forward topologies
  + done either symbolically or numerically
  + once the recursive strategy has been defined, it is easy to implement in a computer
  + backpropagation tries to organize the data in such a way that redundant computations are avoided
  + calculation of the Hessian matrix involves repeated computation of the same terms
  + the network providing a data structure where to store partial results  and organizing the computation
  + explaining why the standard and second-order backpropagation are also of interest for computer algebra systems
  + minimizing the number of arithmetic operations required
  + the backpropagation path values stored to be used repetitively
  + the node w/ the backpropagation paths of different weights intersect need to be calculated only once
  + possibly using graph traversing algorithms to optimize computation of the Hessian

+ [Diagonal of the Hessian matrix](../ML/MLNN-Hinton/a12-Learning.md#some-conclusions)
  + only involving a local communication in a neural network
  + the backpropagation path to a weight intersects itself in its whole length
  + computation of the second-order derivative of the associated network function of an output unit w.r.t. a given weight organized as a recursive backward computation over this path
  + able to apply Pseudo-Newton methods


