# Neural Networks for Machine Learning

## 01. Introduction to Machine Learning

+ [Why do we need machine learning](01-IntroML.md#)
+ [What are neural networks](01-IntroML.md#)
+ [Some simple models of neurons](01-IntroML.md#)
+ [A simple example of learning](01-IntroML.md#)
+ [Three types of learning](01-IntroML.md#)

## 02. Perceptron Learning Procedure

+ [An overview of the main types of network architecture](02-Perceprtons.md#)
+ [Perceptrons](02-Perceprtons.md#)
+ [A geometrical view of perceptrons](02-Perceprtons.md#)
+ [Why the learning works](02-Perceprtons.md#)
+ [What perceptrons can not do](02-Perceprtons.md#)


## 03. Backpropagation Learning Procedure
  
+ [Learning the weights of a linear neuron](03-Backpropagation.md#learning-the-weights-of-a-linear-neuron)
+ [The error surface for a linear neuron](03-Backpropagation.md#the-error-surface-for-a-linear-neuron)
+ [Learning the weights of a logistic output neuron](03-Backpropagation.md#learning-the-weights-of-a-logistic-output-neuron)
+ [The backpropagation algorithm](03-Backpropagation.md#the-backpropagation-algorithm)
+ [How to use the derivatives computed by the backpropagation algorithm](03-Backpropagation.md#how-to-use-the-derivatives-computed-by-the-backpropagation-algorithm)


## 04. Multiclasses Machine Learning

+ [Learning to predict the next word](04-Multiclasses.md#)
+ [A brief diversion into cognitive science](04-Multiclasses.md#)
+ [Another diversion_The softmax output function](04-Multiclasses.md#)
+ [Neuro-probabilistic language models](04-Multiclasses.md#)
+ [ways to deal with large number of possible outputs](04-Multiclasses.md#)


## 05. Convolution Neural Networks

+ [Why object recognition is difficult](05-CNN.md#)
+ [Ways to achieve viewpoint invariance](05-CNN.md#)
+ [Convolutional neural networks for hand-written digit recognition](05-CNN.md#)
+ [Convolutional neural networks for object recognition](05-CNN.md#)


## 06. Mini-batch Gradient Descent
  
+ [Overview of mini-batch gradient descent](06-MiniBatch.md#)
+ [A bag of tricks for mini-batch descent](06-MiniBatch.md#)
+ [The momentum method](06-MiniBatch.md#)
+ [A separate, adaptive learning rate for each connection](06-MiniBatch.md#)
+ [rmsprop_divide the gradient](06-MiniBatch.md#)


## 07. Recurrent Neural Networks

+ [Modeling sequences: A brief overview](07-RNN.md#)
+ [Training RNNs with backpropagation](07-RNN.md#)
+ [A toy example of training an RNN](07-RNN.md#)
+ [Why it is difficult to train an RNN](07-RNN.md#)
+ [Long term short term memory](07-RNN.md#)


## 08. Multicaptive Connections
  
+ [A brief overview of Hessian-free optimization](08-Multicaptive.md#)
+ [Modeling character strings with multiplicative connections](08-Multicaptive.md#)
+ [Learning to predict the next character using HF](08-Multicaptive.md#)
+ [Echo state networks](08-Multicaptive.md#)


## 09. Overfitting
  
+ [Overview of ways to improve generalization](09-Overfitting.md#)
+ [Limiting size of the weights](09-Overfitting.md#)
+ [Using noise as a regularizer](09-Overfitting.md#)
+ [Introduction to the bayesian approach](09-Overfitting.md#)
+ [The bayesian interpretation of weight decay](09-Overfitting.md#)
+ [MacKays quick and dirty method of fixing weight costs](09-Overfitting.md#)


## 10. Model Combination and Dropouts
  
+ [Why it helps to combine models](10-CombineDropout.md#)
+ [Mixtures of experts](10-CombineDropout.md#)
+ [The idea of full bayesian learning](10-CombineDropout.md#)
+ [Making full bayesian learning practical](10-CombineDropout.md#)
+ [Dropout an efficient way to combine neural nets](10-CombineDropout.md#)


## 11. Hopfield Nets and Boltzmann Machines

+ [Hopfield Nets](11-Hopfield.md#)
+ [Dealing with spurious minima in hopfield nets](11-Hopfield.md#)
+ [Hopfields Nets with hidden units](11-Hopfield.md#)
+ [Using stochastic units to improve search](11-Hopfield.md#)
+ [How a boltzmann machine models data](11-Hopfield.md#)


## 12. Restricted Boltzmann Machine (RBMs)

+ [The boltzmann machine learning algorithm](12-Boltzmann.md#)
+ [More efficient ways to get the statistics](12-Boltzmann.md#)
+ [Restricted boltzmann machines](12-Boltzmann.md#)
+ [An example of contrastive divergence learning](12-Boltzmann.md#)
+ [RBMs for collaborative filtering](12-Boltzmann.md#)


## 13. Deep Belief Nets

+ [The ups and downs of backpropagation](13-BeliefNets.md#)
+ [Belief nets](13-BeliefNets.md#)
+ [The wake-sleep algorithm](13-BeliefNets.md#)


## 14. Generative Adversarial Networks (GANs)
  
+ [Learning layers of features by stacking RBMs](14-GANs.md#)https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9
+ [Discriminative fine-tuning for DBNs](14-GANs.md#)
+ [What happens during discriminative fine-tuning](14-GANs.md#)
+ [Modeling real-valued data with an RBM](14-GANs.md#)
+ [RBMs are infinite sigmoid belief nets](14-GANs.md#)


## 15. Hierarchical Structure with Neural Networks

+ [From principal components analysis to autoencoders](15-Hierarchy.md#)
+ [Deep Autoencoders](15-Hierarchy.md#)
+ [Deep autoencoders for document retrieval and visualization](15-Hierarchy.md#)
+ [Semantic hashing](15-Hierarchy.md#)
+ [Learning binary codes for image retrieval](15-Hierarchy.md#)
+ [Shallow autoencoders for pre-training](15-Hierarchy.md#)


## 16. Deep Neural Networks
  
+ [Learning a joint model of images and captions](16-DeepNN.md#)
+ [Hierarchical coordinate frames](16-DeepNN.md#)
+ [Bayesian optimization of neural network hyperparameters](16-DeepNN.md#)


## Related Articles

+ Matthew Stewart, [Introduction to Neural Networks](a01-IntroNN.md)
+ Matthew Stewart, [Intermediate Topics in Neural networks](a02-IntermediateNN.md)
+ Matthew Stewart, [Neural Network Optimization](a03-Optimization.md)
+ Matthew Stewart, [Simple Guide to Hyperparameter Tuning in Neural Networks](a04-Hyperparameter.md)
+ Matthew Stewart, [Neural Style Transfer and Visualization of Convolutional Networks](a05-VisualCNN.md)
+ Random Nerd, [Delta Learning Rule & Gradient Descent | Neural Networks](a06-DeltaRule.md)


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
  + [Youtube Videos](https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9)
+ Github Related Links
  + [Fuyang Liu](https://github.com/liufuyang/course-Neural-Networks-for-Machine-Learning)
  + [Chinmay Das](https://github.com/chinmaydas96/Neural-Networks-for-Machine-Learning)
  + [Chouffe](https://github.com/Chouffe/hinton-coursera)
  + [khanhnamle1994](https://github.com/khanhnamle1994/neural-nets)
