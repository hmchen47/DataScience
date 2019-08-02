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




## Momentum





## Adaptive Learning Rates





## Parameter Initialization





## Batch Normalization




