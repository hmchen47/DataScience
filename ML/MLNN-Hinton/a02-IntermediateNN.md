# Intermediate Topics in Neural Networks

Author: Matthew Stewart

[URL](https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98)


## Anatomy of a neural network

+ Artificial neural networks are one of the main tools used in machine learning.

+ Neural networks:
  + simplest case: consisting of input, output layers, and a hidden layer consisting of units that transform the input into something that the output layer can use.
  + excellent tools for finding patterns which are far too complex or numerous for a human programmer to extract and teach the machine to recognize
  + backpropagation: allowing networks to adjust their neuron weights in situations where the outcome doesn't match what the creator is hoping for
  + make use of affine __transformations__ to concatenate input features that converge at a specific node in the network
  + concatenated input passed through an activation function

+ Activation function
  + evaluate the signal response and determine whether the neuron should be activated given the current inputs
  + how to extended to multilayer and multi feature networks in order to increase the explanatory power of the network by increasing
    + the number of degrees of freedom (weights and biases) of the network
    + the number of features available which the network can use to make predictions

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*L_lfAEddxxAg2EqJfB5i6Q.png" style="margin: 0.1em;" alt="Activation function with affine transformation" title="Activation function with affine transformation" height=200>
      <img src="https://miro.medium.com/max/1250/1*l57B0pjXoO-1H1xZYV7QBA.png" style="margin: 0.1em;" alt="A neural network with one hidden layer and two features (the simplest possible multi-layer multi-feature network)." title="A neural network with one hidden layer and two features (the simplest possible multi-layer multi-feature network)." height=200>
    </a>
  </div>

+ Network parameters (weights and biases)
  + updated by assigning the error of the network
  + using backpropagation through the network to obtain the derivatives for each of the parameters w.r.t. the loss function
  + gradient descent used to update these parameters in an informed manner such that the predictive power of the network likely to improve

+ Training network
  + the process of accessing the error and updating the parameters
  + a training set to generate a functional network
  + performance of the network
  + test set: unseen data accessed by testing

+ Degrees of freedom
  + neural network having a large number of degrees
  + required a large amount of data for training to be able to make adequate predictions
  + useful for high dimensionality of the data

+ Generalized numtilayer and multi-feature network

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*GApLZ60775yXfUzO65PfEA.png" style="margin: 0.1em;" alt="Generalized multilayer perceptron with n hidden layers, m nodes, and d input features." title="Generalized multilayer perceptron with n hidden layers, m nodes, and d input features." width=450>
    </a>
  </div>

  + $m$ nodes:
    + $m$ as the width of a layer within the network
    + no relation between the number of features and the width of a network layer
  + $n$ hidden layers
    + the depth of the network
    + in general, deep learning with more than one hidden layer
    + image analysis: commonly having the state-of-the-art convolutional architectures with hundreds of hidden layers
  + $d$ inputs
    + pre-specified by the available data
    + image: the number of pixels in the image after the image is flattened into a one-dimensional array
    + normal Pandas data frame: equal to the number of feature columns
  + the hidden layers of the network have the same width (number of nodes)
  + the number of nodes may vary across the hidden layers
  + the output layer may also be of an arbitrary dimension depending on the required output
  + if trying to classify images into one of ten classes, the output layer will consist of ten nodes

+ Rule-based systems
  + prior to neural networks
  + gradually evolved into more modern machine learning
  + more and more abstract features learned
  + much more complex selection criteria
  + rule system breaks down in some cases due to the oversimplified features chosen

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*llphHy6X7at6-jHWNN_2jw.png" style="margin: 0.1em;" alt="Comparisons of rule-based system simple NN & CNN" title="Comparisons of rule-based system simple NN & CNN" width=450>
    </a>
  </div>


+ Convolutional neural networks
  + Neural network
    + an abstract representation of the data at each stage of the network
    + designed to detect specific features of the network
  + CNN:
    + commonly used to study images
    + hidden layers closer to the output of a deep network, the highly interpretable representations, such as faces, clothing, etc.
    + the first layers of the network: detecting very basic features such as corners, curves, and so on
    + abstract representations quickly become too complex to comprehend
    + the workings of neural networks to produce highly complex abstractions seen as somewhat magical

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*9AC8C_ybj-yeAS8OqpK1Eg.png" style="margin: 0.1em;" alt="An example of a neural network with multiple hidden layers classifying an image of a human face." title="An example of a neural network with multiple hidden layers classifying an image of a human face." width=400>
      <img src="https://miro.medium.com/max/1250/1*qpdTBdx8D-Z2WAMT24onLQ.png" style="margin: 0.1em;" alt="An example of a neural network with multiple hidden layers classifying written digits from the MNIST dataset." title="An example of a neural network with multiple hidden layers classifying written digits from the MNIST dataset." width=400>
    </a>
  </div>



## Activation functions






## Loss functions






## Output units






## Architecture









