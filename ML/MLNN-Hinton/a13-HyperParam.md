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





