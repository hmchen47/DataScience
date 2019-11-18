# Neural Network Optimization

## Challenges with optimization

+ Convex optimization
  + a function in which there is only one optimum, corresponding to the global optimum (maximum or minimum)
  + no concept of local optima for convex optimization problems, making them relatively easy to solve

+ Non-convex optimization
  + a function which has multiple optima, only one of which is the global optima
  + Maybe very difficult to locate the global optima depending on the loss surface

+ Neural network
  + loss surface: minimize the prediction error of the network
  + interested in finding the global minimum on this loss surface

+ Multiple problems on neural network training
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


### Local optima

+ viewed as a major problem in neural network training
+ using insufficiently large neural networks, most local minima incur a low cost
+ not particularly important to find the true global minimum
+ a local minimum with reasonably low error is acceptable

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
    <img src="https://miro.medium.com/max/875/1*fGx-IJkvPLuurfR9VWlu2g.png" style="margin: 0.1em;" alt="Curve of loss function with local and global minimum" title="Curve of loss function with local and global minimum" width=350>
  </a>
</div>


### Saddle Points

+ Saddle pints
  + more likely than local minima in high dimensions
  + more problematic than local minima because close to a saddle point the gradient can be very small
  + gradient descent results in negligible updates to the network and network training will cease

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/0*GuknkQNZ8pQqDGsr.jpg" style="margin: 0.1em;" alt="Saddle point — simultaneously a local minimum and a local maximum." title="Saddle point — simultaneously a local minimum and a local maximum." width=350>
    </a>
  </div>

+ Rosenbrook function
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

+ the particular form of the error function represents the learning problem
+ ill-conditioned derivatives of the error function
+ reflected in error landscapes containing many saddle points and flat areas
+ Hessian matrix
  + a square matrix of second-order partial derivatives of a scalar-valued function
  + the Hessian describes the local curvature of a function of many variables

  \[H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\\\ \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}\]

  + used to determine whether a given stationary points is a saddle point or not
  + a similar way by looking at the eigenvalues

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*QkQ29ciExhaeguzIIBD6nw.png" style="margin: 0.1em;" alt="Eigenvalues of minimum, maximum and saddle point" title="Eigenvalues of minimum, maximum and saddle point" width=550>
    </a>
  </div>

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

+ additional issues associated with the architecure of the neural network with deep learning

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*Hae-goX40dfHmNFmgd5jZw.png" style="margin: 0.1em;" alt="text" title="caption" width=350>
    </a>
  </div>

+ deep neural network:
  + $n$ hidden layers
  + features at the 1st layer propagated through the network
  + affine transformation followed by an activation function (a single layer)

    \[\begin{array}{lll} \text{Linear} \qquad & h_i = W_x & \\ \text{activation} & h_i = W h_{i-1}, & i = 2, \dots, n\end{array}\]

  + output for an $n$-layer network

    \[\text{Suppose } \mathbf{W} = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}: \qquad \begin{bmatrix} h_1^1 \\ h_2^1 \end{bmatrix} = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \quad \cdots \begin{bmatrix} h_1^n \\ h_2^n \end{bmatrix} = \begin{bmatrix} a^n & 0 \\ 0 & b^n \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\]

+ Two possible cases depending on the magnitude of $a$ and $b$

  \[\begin{array}{lll} \text{Suppose } x = \begin{bmatrix} 1 \\ 1 \end{bmatrix} & & \\ \text{Case 1: } a = 1, b =2: & y \rightarrow 1, \Delta_y \rightarrow \begin{bmatrix} n \\ n \cdot 2^{n-1} \end{bmatrix} & \quad \text{Explodes!} \\ \text{Case 2: } a = 0.5, b = 0.9: & y \rightarrow 0, \Delta_y \rightarrow \begin{bmatrix} 0 \\ 0 \end{bmatrix} & \quad \text{Vanishes!} \end{array}\]

  + for $a$, $b$ greater than 1
    + for a large value of $n$ ( a deep neural network), the gradient values will quickly explode as they propagate through the network
    + exploding gradients lead to "cliffs" unless gradient clipping implemented
    + the gradient clipped if it exceeds a certain threshold value

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
          <img src="https://miro.medium.com/max/875/1*LVV54iSMBfO0BC0B5g5okQ.png" style="margin: 0.1em;" alt="An example of clipped vs. unclipped gradients." title="An example of clipped vs. unclipped gradients." width=350>
        </a>
      </div>

    + Gradient clipping rule

      \[\text{if } \parallel g \parallel > u, \quad g \leftarrow \dfrac{gu}{\parallel g \parallel}\]

  + for $a$, $b$ less than 1
    + the gradients quickly tends to zero
    + gradient values smaller than the precision threshold recognized as zero

+ Neural Networks optimization
  + neural networks doomed to have large numbers of local optima
  + often containing both sharp and flat valleys which result in the stagnation of learning and unstable learning
  + regarding NN optimization, starting with momentum



## Momentum

+ Stochastic gradient descent (SGD)
  + the presence of oscillations which result from updates not exploiting curvature information
  + slow SDG with high curvature

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/0*sdxfb4SSSxqwON8W.png" style="margin: 0.1em;" alt="(Left) Vanilla SGD, (right) SGD with momentum. Goodfellow et al. (2016)" title="(Left) Vanilla SGD, (right) SGD with momentum. Goodfellow et al. (2016)" width=350>
    </a>
  </div>

  + taking the average gradient, obtain a faster path to optimization
  + dampen oscillations: gradients in opposite directions get canceled out

+ Momentum
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

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*95U8-BTfu-U_YXte2M6TFQ.png" style="margin: 0.1em;" alt="Momentum is what results in the dampening of oscillations for high curvature surfaces" title="Momentum is what results in the dampening of oscillations for high curvature surfaces" width=300>
    </a>
  </div>


### Classical Momentum

+ Formula for Momentum
  + using past gradients for updating values
  + $v$: velocity
  + more weight applied to more recent gradients, creating an exponentially decaying average of gradients

  \[\begin{array}{rcl} g &=& \frac{1}{m} \displaystyle \sum_i \Delta_\theta L(f(x^{(i)}; \theta), y^{(i)}) \\ v &=& \alpha v + (-\varepsilon g) \end{array}\]

  + $\alpha \in [0, 1)$ controls how quickly effect of past gradients decay
  + $\varepsilon$: current gradient update

+ Compute gradient estimate:

    \[g = \frac{1}{m} \sum_i \Delta_\theta L(f(x^{(i)}; \theta), y^{(i)})\]
  + Update velocity: $v = \alpha v - \varepsilon g$
  + Update parameters: $\theta = \theta + v$

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
        <img src="https://miro.medium.com/max/518/0*TTwIwHORCcMARMRB.png" style="margin: 0.1em;" alt="SGD without momentum (black) compared with SGD with momentum (red)" title="SGD without momentum (black) compared with SGD with momentum (red)" width=200>
      </a>
    </div>

    + the effects of adding momentum on an optimization algorithm
    + SGD w/ momentum updates no real advantage at the first few updates over vanilla SGD
    + As the number of updates increases the momentum kickstarts and allows faster convergence.

  + Matlab code

    ```matlab
    vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) )
    W(t+1) = W(t) + vW(t+1)
    ```


### Nesterov Momentum

+ Ref: Sutskever, Martens et al. "[On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf)" 2013

+ Classical vs Nesterov Momentum
  + Classical
    + first correct velocity
    + make a big step according to that velocity (and then repeat)
  + Nesterov
    + first make a step into velocity direction
    + make a correction to a velocity vector based on a new location (then repeat)

+ Matlab code

  ```matlab
  vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) + momentum.*vW(t) )
  W(t+1) = W(t) + vW(t+1)
  ```

+ Hugh difference in practice
  + Apply an interim update: $\tilde{\theta} = \theta + v$
  + Perform a correction based on gradient at the interim point

    \[\begin{array}{rcl} g &=& \frac{1}{m} \sum_i \Delta_\theta L(f(x^{(i)}; \tilde{\theta}), y^{(i)}) \\ v &=& \alpha v - \varepsilon g \\ \theta & = & \theta + v \end{array}\]

  + momentum based on look-ahead slope
  + visual representation of the difference between the traditional momentum update and Nesterov momentum

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*hJSLxZMjYVzgF5A_MoqeVQ.jpeg" style="margin: 0.1em;" alt="Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this 'looked-ahead' position." title="Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this 'looked-ahead' position." width=350>
    </a>
  </div>  



## Adaptive Learning Rates

+ Oscillations along vertical direction

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*0v0zucWChoudFcJRPpB74Q.png" style="margin: 0.1em;" alt="Gradient descent oscillates along vertical direction" title="Gradient descent oscillates along vertical direction" width=300>
    </a>
  </div>


### AdaGrad

+ Momentum adds updates to the slope of error function and speeds up SGD in turn.

+ AdaGrad adapts updates to each individual parameter to perform larger or smaller updates depending on their importance.

+ Accumulate squared gradients: $r_i = r_i + g_i^2$

+ Update each parameter:

  \[\theta_i = \theta_1 - \frac{\varepsilon}{\delta + \sqrt{r_i}} g_i\]

  + inversely proportional to cumulative squared gradient

+ Benefits:
  + eliminate the need to manually tune the learning rate
  + result in greater progress along gently sloped directions

+ Disadvantages:
  + accumulation of the squared gradients in the denominator
  + positive added term:
    + the accumulated sum keeps growing during training
    + the learning rate shrink and eventually become infinitesimally small

### RMSProp

+ For non-convex problems, AdaGrad can prematurely decrease the learning rate.

+ Use an exponentially weighted average for gradient accumulation.

  \[\begin{array}{rcl} r_i &=& \rho r_i + (1 - \rho) g_i^2 \\ \theta_i &=& \theta_i - \frac{\varepsilon}{\delta + \sqrt{r_i}} g_i \end{array}\]


### Adam

+ Adaptive moment estimation (Adam)
  + a combination of RMSprop and momentum
  + the most popular optimizer used for neural networks

+ Nadam: a combination of MRSprop and Nesterov momentum

+ Adam computes adaptive learning rates for each parameters.

+ Adam keeps an exponentially decaying average of past gradients, similar to momentum.
  + Estimate first moment: 

    \[v_i = \rho_1 v_i + (1 - \rho_1) g_i\]
  
  + Estimate second moment:

    \[r_i = \rho_2 r_i + 91 - \rho_2) g_i^2\]

    + applies bias correction to $v$ and $r$

  + Update parameters:

    \[\theta_i = \theta_i - \frac{\varepsilon}{\delta + \sqrt{r_i}} v_i\]

    + works well in practice, is fairly robust to hyper-parameters



## Parameter Initialization

+ Initialization of network weights
  + overlooked characteristics of developing neural networks
  + poorly initialized networks determined to network performance
  + initialized with all values of zero
    + the network not learn anything at all
    + after a gradient update, all weights would be zero
  + initialized with all weights 0.5
    + actually learn something
    + prematurely prescribed some form of symmetry between neural units

+ Randomizing weights
  + avoid presupposing any form of a neural structure by randomizing weights according to a normal distribution
  + often done in Keras by specifying a random state

+ Scale of initialization
  + large values for the weights: lead to exploding gradients
  + small values for the weights: lead to vanishing gradients
  + sweet spot that provides the optimum tradeoff between these two
  + not a priori but inferred through trial and error


### Xavier Initialization

+ Xavier initialization is a simple heuristic for assigning network weights.

+ Objective: the variance to remain the same with each passing layer

+ Keep the signal from exploding to high values or vanishing to zero

+ To initialize the weights in such a way that the variance remains the same for both the input and the output

+ The weights drawn from a distribution with zero mean and a specific variance.

+ For a fully-connected layer with $m$ inputs:

  \[W_{ij} \sim N \left(0, \frac{1}{m} \right)\]

  + $m$: fan-in; the number of incoming neurons (input units in the weight tensor)
  + heuristic value: merely empirically observed to perform well


### He Normal Initialization

+ HE normal initialization
  + the same as Xavier Initialization, except that the variance multiplied by a factor of two
  + initialized the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently
  + random but differ in range depending on the size of the previous layer of neurons
  + controlled initialization hence the faster and more efficient gradient descent

+ For ReLU units

  \[W_{ij} \sim N \left(0, \frac{2}{m} \right)\]


### Bias Initialization

+ Bias initialization: how the biases of the neurons should be initialized

+ The simplest and a common way of initializing biases is to set them to zero.

+ Asymmetry breaking: provided by the small random numbers in th weights

+ ReLU non-linearity
  + using small constant values such as 0.01 for all biases
  + ensure that all ReLU units fire in the beginning and obtain and propagate some gradient

+ Main concern: avoid saturation at initialization within hidden units, ReLU by initializing biases to 0.1 instead of zero


### Pre-initialization

+ Pre-initialization:
  + common for convolutional networks used for examining images
  + involve importing the weights of an already trained network
  + used as the initial weights of the network to be trained
  + a tenable method to utilize for analyzing images with few data samples
  + underlying concept behind transfer learning



## Normalization

+ Topics covered so far
  + ways to navigate the loss surface of then neural network using momentum and adaptive learning rates
  + methods of parameter initialization to minimize a prior biases within the network


### Future Normalization

+ Normalizing features before applying the learning algorithm

+ Ref: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  + gradient descent converges much faster with feature scaling than without it

+ Min-max normalization
  + simplest method to scale data
  + rescaling the range of features to scale the range in [0, 1] or [-1, 0]
  + subtracting each value by the minimum value and scaling by the range of values present in the dataset
  + Issue: highly skewed data results in many values clustered inn one location
  + Solution: taking the logarithm of the feature variable

  \[x^\prime = \frac{x - \min(x)}{\max(x) - \min(x)}\]

+ Mean normalization
  + essentially the same as min-max normalization except the average value is subtracted from each value
  + the least common way

  \[x^\prime = \frac{x - \text{average}(x)}{\max(x) - \min(x)}\]

+ Feature normalization
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


### Internal Covariate Shift

+ Internal Covariate Shift:
  + the change in the distribution of network activation due to the change in network parameters during training
  + the parameters of a layer changed, the distribution of inputs to subsequent layers also changes
  + Issue: the shifts in input distributions tend to slow down learning, especially deep neural networks

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Dnxnj2STbo-42DfalLMi-g.png" style="margin: 0.1em;" alt="Deep neural network: multiple hidden layers" title="Deep neural network: multiple hidden layers" width=350>
    </a>
  </div>

+ Whitened inputs
  + converge faster and uncorrelated
  + internal covariate shift leads to just the opposite


### Batch Normalization

+ Batch normalization
  + a method intended to mitigate internal covariate shift for neral networks
  + an extension to the idea of feature standardization to other layers of the neural network

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*x3FtLuoYjWeNctiNPlTBjw.png" style="margin: 0.1em;" alt="Matrix representation of weights for hidden layers" title="Matrix representation of weights for hidden layers" width=350>
    </a>
  </div>

  + reducing overfit due to a slight regularization effect
  + similar to dropout, add some noise to each hidden layer's activations

+ Batch normalization transformation
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
        <img src="https://miro.medium.com/max/875/1*a-B6jX8B-8lfz5ZrIFQyFw.png" style="margin: 0.1em;" alt="Add normalization operations for layer 1" title="Add normalization operations for layer 1" width=400>
        <img src="https://miro.medium.com/max/875/1*LdQ7HKaqDYtszL8R0bdb4Q.png" style="margin: 0.1em;" alt="Add normalization operations for layer 2" title="Add normalization operations for layer 2" width=400>
      </a>
    </div>

  + subsequently repeated for all subsequent hidden layers
  + differentiate the joint loss for the N mini-batches and then backpropagate through the normalization operations

+ Testing time
  + the mean and standard deviations replaced with running average collected during training time
  + same as using the population statistics instead of mini-batch statistics
  + the output deterministically depends on the input

+ Advantages
  1. reduces internal coariate shift
  2. reduces the dependence of gradients on the scale of the parameters or their initial values
  3. regularizes the model ad reduces the need for dropout, photometric distortions, local response normalization and other regularization techniques
  4. allows use of saturating nonlinearities and higher learning rates



