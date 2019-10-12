# A Disciplined Approach to Neural Network Hyper-Parameters: Part-I - Learning Rate, Batch Size, Momentum, and Weight Decay

Author: Leslie N. Smith

[Original Article](https://arxiv.org/pdf/1803.09820)

[Original Article HTML Version](https://www.arxiv-vanity.com/papers/1803.09820/)


## Abstract

+ how to examine the training validation/test loss function for subtle clues of underfitting and overfitting

+ suggesting guidelines for moving toward the optimal balancing point

+ experiments show that it is crucial to balance every manner of regularization for each dataset and architecture

+ weight decay used as a sample regularizer to show how its optimal value is tightly coupled with the learning rate and momentum

+ Files to replicate the results](https://github.com/lnsmith54/hyperParam1)


## 1. Introduction

+ the process of setting the hyper-parameters
  + including the designing the network architecture
  + requiring expertise and extensive trial and error and time consuming

+ no simple and easy ways to set hyper-parameters
  + learning rate, batch size, momentum and weight decay
  + grid search or random search: computationally expensive and time consuming
  + James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281–305, 2012.
  + training time and final performance: highly dependent on good choice
  + choosing the standard architecture and the hyper-parameter files freely available in model zoo or from gitgub.com

+ proposed methodologies for finding optimal settings for several hyper-parameters
  
+ goal: providing practical advice that saves time and effort, yet improves performance

+ basis of the approach
  + well-known concept of the balance between underfitting and overfitting
  + examining the training's test-/validation loss for clues of underfitting and overfitting to strive for optimal set of hyper-parameters
  + paying close attention while using cyclical learning rates and cyclical momentum
  + Leslie N Smith. Cyclical learning rates for training neural networks. In Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on, pp. 464–472. IEEE, 2017.


## 2. Related Work




## 3. The Unreasonable Effectiveness of Validation/Test Loss




### 3.1 A Review of the Underfitting and Overfitting Trade-off




### 3.2 Underfitting





### 3.3 Overfitting




## 4. Cyclical Learning Rates, Batch Sizes, Cyclical Momentum, and Weight Decay




### 4.1 Cyclical Learning Rates and Super-convergence Revisited





### 4.2 Batch Size




### 4.3 Cyclical Momentum




### 4.4 Weight Decay




## 5. Experiments with Other Architectures and Datasets





### 5.1 Wide Resents on Cifar-10




### 5.2 Densenets on Cifar-10





### 5.3 MNIST





### 5.4 Cifar-100




### 5.5 Imagnet





## 6. Discussion





## 7. References




## A. Appendix




### A.1 Experimental Methods: Detailed Information about the Experiments to Enable Replication





### A.2 Implementation of Cyclical Momentum in Caffe





