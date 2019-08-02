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



## Momentum





## Adaptive Learning Rates





## Parameter Initialization





## Batch Normalization




