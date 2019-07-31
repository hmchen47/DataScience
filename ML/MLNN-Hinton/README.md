# Neural Networks for Machine Learning

## Introduction to Machine Learning

+ [Why do we need machine learning](01-IntroML.md#)
+ [What are neural networks](01-IntroML.md#)
+ [Some simple models of neurons](01-IntroML.md#)
+ [A simple example of learning](01-IntroML.md#)
+ [Three types of learning](01-IntroML.md#)

## Introduction to Neural Networks

+ [An overview of the main types of network architecture](02-IntroNN.md#)
+ [Perceptrons](02-IntroNN.md#)
+ [A geometrical view of perceptrons](02-IntroNN.md#)
+ [Why the learning works](02-IntroNN.md#)
+ [What perceptrons can not do](02-IntroNN.md#)

## Neuron Weight
  
+ [Learning the weights of a linear neuron](03-Weighting.md#)
+ [The error surface for a linear neuron](03-Weighting.md#)
+ [Learning the weights of a logistic output neuron](03-Weighting.md#)
+ [The backpropagation algorithm](03-Weighting.md#)
+ [How to use the derivatives computed by the backpropagation algorithm](03-Weighting.md#)


## Cognitive Learning

+ [Learning to predict the next word](04-Cognitive.md#)
+ [A brief diversion into cognitive science](04-Cognitive.md#)
+ [Another diversion_The softmax output function](04-Cognitive.md#)
+ [Neuro-probabilistic language models](04-Cognitive.md#)
+ [ways to deal with large number of possible outputs](04-Cognitive.md#)


## Convolution Neural Networks

+ [Why object recognition is difficult](05-Convolution.md#)
+ [Ways to achieve viewpoint invariance](05-Convolution.md#)
+ [Convolutional neural networks for hand-written digit recognition](05-Convolution.md#)
+ [Convolutional neural networks for object recognition](05-Convolution.md#)

## Mini-batch Gradient Descent
  
+ [Overview of mini-batch gradient descent](06-MiniBatch.md#)
+ [A bag of tricks for mini-batch descent](06-MiniBatch.md#)
+ [The momentum method](06-MiniBatch.md#)
+ [A separate, adaptive learning rate for each connection](06-MiniBatch.md#)
+ [rmsprop_divide the gradient](06-MiniBatch.md#)


## Recurrent Neural Networks

+ [Modeling sequences: A brief overview](07-RNN.md#)
+ [Training RNNs with backpropagation](07-RNN.md#)
+ [A toy example of training an RNN](07-RNN.md#)
+ [Why it is difficult to train an RNN](07-RNN.md#)
+ [Long term short term memory](07-RNN.md#)


## Hessian-free optimization
  
+ [A brief overview of Hessian-free optimization](08-HFOptima.md#)
+ [Modeling character strings with multiplicative connections](08-HFOptima.md#)
+ [Learning to predict the next character using HF](08-HFOptima.md#)
+ [Echo state networks](08-HFOptima.md#)


## Weight Cost
  
+ [Overview of ways to improve generalization](09-WeightCost.md#)
+ [Limiting size of the weights](09-WeightCost.md#)
+ [Using noise as a regularizer](09-WeightCost.md#)
+ [Introduction to the bayesian approach](09-WeightCost.md#)
+ [The bayesian interpretation of weight decay](09-WeightCost.md#)
+ [MacKays quick and dirty method of fixing weight costs](09-WeightCost.md#)


## Model Combination
  
+ [Why it helps to combine models](10-Combine.md#)
+ [Mixtures of experts](10-Combine.md#)
+ [The idea of full bayesian learning](10-Combine.md#)
+ [Making full bayesian learning practical](10-Combine.md#)
+ [Dropout an efficient way to combine neural nets](10-Combine.md#)

## Hopfield Neural Networks

+ [Hopfield Nets](11-Hopfield.md#)
+ [Dealing with spurious minima in hopfield nets](11-Hopfield.md#)
+ [Hopfields Nets with hidden units](11-Hopfield.md#)
+ [Using stochastic units to improve search](11-Hopfield.md#)
+ [How a boltzmann machine models data](11-Hopfield.md#)


## Boltzmann Machine Learning

+ [The boltzmann machine learning algorithm](12-Boltzmann.md#)
+ [More efficient ways to get the statistics](12-Boltzmann.md#)
+ [Restricted boltzmann machines](12-Boltzmann.md#)
+ [An example of contrastive divergence learning](12-Boltzmann.md#)
+ [RBMs for collaborative filtering](12-Boltzmann.md#)


## Backpropagation Algorithms

+ [The ups and downs of backpropagation](13-Backpropagate.md#)
+ [Belief nets](13-Backpropagate.md#)
+ [The wake-sleep algorithm](13-Backpropagate.md#)


## Restricted Boltzmann Machines
  
+ [Learning layers of features by stacking RBMs](14-RBM.md#)
+ [Discriminative fine-tuning for DBNs](14-RBM.md#)
+ [What happens during discriminative fine-tuning](14-RBM.md#)
+ [Modeling real-valued data with an RBM](14-RBM.md#)
+ [RBMs are infinite sigmoid belief nets](14-RBM.md#)

## Autoencoders

+ [From principal components analysis to autoencoders](15-Autoencoder.md#)
+ [Deep Autoencoders](15-Autoencoder.md#)
+ [Deep autoencoders for document retrieval and visualization](15-Autoencoder.md#)
+ [Semantic hashing](15-Autoencoder.md#)
+ [Learning binary codes for image retrieval](15-Autoencoder.md#)
+ [Shallow autoencoders for pre-training](15-Autoencoder.md#)


## Joint Model for Images and Caption
  
+ [Learning a joint model of images and captions](16-JointModel.md#)
+ [Hierarchical coordinate frames](16-JointModel.md#)
+ [Bayesian optimization of neural network hyperparameters](16-JointModel.md#)


## Related Articles

+ Matthew Stewart, [Introduction to Neural Networks](a01-IntroNN.md)
+ Matthew Stewart, [Intermediate Topics in Neural networks](a02-IntermediateNN.md)


## Reference:

+ [Introduction to Neural Networks and Machine Learning - CSC321 Winter 2014](http://www.cs.toronto.edu/~tijmen/csc321/)
  + [Lecture Notes](http://www.cs.toronto.edu/~tijmen/csc321/lecture_notes.shtml)
  + [Assignmenets](http://www.cs.toronto.edu/~tijmen/csc321/assignments.shtml)
  + [Optional Readings](http://www.cs.toronto.edu/~tijmen/csc321/texts.shtml)
  + [Computing](http://www.cs.toronto.edu/~tijmen/csc321/computing.shtml)
  + [Exam](http://www.cs.toronto.edu/~tijmen/csc321/tests.shtml)
+ [2012 COURSERA COURSE LECTURES: Neural Networks for Machine Learning](http://www.cs.toronto.edu/~hinton/nntut.html)
  + [Lecture Videos](http://www.cs.toronto.edu/~hinton/coursera_lectures.html)
  + [Lecture Slides](http://www.cs.toronto.edu/~hinton/coursera_slides.html)

