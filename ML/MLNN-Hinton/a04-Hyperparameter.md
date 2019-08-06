# Simple Guide to Hyperparameter Tuning in Neural Networks

Matthew Stewart

URL: https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594


## Beale's Function

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

+ Artificial landscape
  + find a way of comparing the performance of various algorithms
    + Convergence (how fast they reach the answer)
    + Precision (how close do they approximate the exact answer)
    + Robustness (so they perform well for all functions or just a small subset)
    + General performance (e.g., computational complexity)
  + analogous to the loss surface of a neural network
  + goal of NN training: find the global minimum on the loss surface by performing some form of optimization - typically stochastic gradient

+ Code for Beale's function

  ```python
  # define Beale's function which we want to minimize
  def objective(X):
      x = X[0]; y = X[1]
      return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

  # function boundaries
  xmin, xmax, xstep = -4.5, 4.5, .9
  ymin, ymax, ystep = -4.5, 4.5, .9

  # Let's create some points
  x1, y1 = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))

  # initial guess
  x0 = [4., 4.]  
  f0 = objective(x0)
  print (f0)
  # 68891.203125

  bnds = ((xmin, xmax), (ymin, ymax))
  minimum = minimize(objective, x0, bounds=bnds)

  print(minimum)
  #      fun: 2.0680256388656271e-12
  # hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
  #      jac: array([ -1.55969780e-06,   9.89837957e-06])
  #  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
  #     nfev: 60
  #      nit: 14
  #   status: 0
  #  success: True
  #        x: array([ 3.00000257,  0.50000085])

  real_min = [3.0, 0.5]
  print (f'The answer, {minimum.x}, is very close to the optimum as we know it, which is {real_min}') 
  print (f'The value of the objective for {real_min} is {objective(real_min)}')
  # The answer, [ 3.00000257  0.50000085], is very close to the optimum as we know it, which is [3.0, 0.5]
  # The value of the objective for [3.0, 0.5] is 0.0
  ```


## Optimization in Neural Networks



### A Keras Refresher




### Callbacks: taking a peek into our model while it's training




### Step 1 - Deciding on the network topology




#### Preprocessing the data




### Step 2 - Adjusting the `learning rate`




### Step 3 - Choosing an optimizer and a loss function




### Step 4 - Deciding on the batch szie and number of epochs




### Step 5 - Random restarts




## Tuning Hyperparameters using Cross-Validation



### Trying Different Weight Initializations




### Save Neural Network Model to JSON




### Cross-Validation with more than one hyperparameters





