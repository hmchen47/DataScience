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

+ Multiple problem on neural network training
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

  $$H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\\\ \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

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



## Momentum





## Adaptive Learning Rates





## Parameter Initialization





## Batch Normalization




