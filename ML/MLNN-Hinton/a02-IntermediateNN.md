# Intermediate Topics in Neural Networks

Author: Matthew Stewart

[Original Article](https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98)


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

+ Generalized multilayer and multi-feature network

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

+ Activation functions
  + a very important part of the neural network
  + analogous to the build-up of electrical potential in biological neurons
  + activation potential: mimicked in artificial neural networks using a probability
  + Characteristics:
    + Ensures not linearity - non-linearity
    + Ensure gradients remain large through the hidden unit - differentiable
  + The general form of an activation function

    \[h = f(W^T X + b)\]
    + $h$: the neural output
    + $f(.)$: the activation function acting on the weights and bases

+ Non-linearity
  + linear function
    + a polynomial of one degree
    + linear equation easy to solve
    + limited in complexity and less power to learn complex functional mappings from data
  + Neural network w/o activation function
    + a linear regression model
    + limited in the set of functions able to approximate
  + Universal approximation theorem: generalized non-linear function approximators
  + non-linear activation able to generate non-linear mappings from inputs to outputs

+ Differentiable
  + required to perform backpropagation in the network
  + required to compute gradients of errors (loss) w.r.t. to the weights updated using gradient descent
  + linear activation function
    + an easily differentiable function
    + optimized using convex optimization
    + limited model capacity

+ Vanishing gradient problem
  + small gradients and several hidden layers results in multiplied gradient during backpropagation
  + computer limitation on precision when multiply many small numbers
  + the value of the gradient quickly vanished
  + important challenge generated in deep neural networks
  + the derivative is zero for half of the values of the input $x$
  + result in a large proportion of dead neurons (as high as 40%) in the neural network

+ Common choices of activation function
  + Sigmoid
  + ReLU (rectified linear unit)
  + Leaky ReLU
  + Generalized ReLU
  + MaxOut
  + Softplus
  + Tanh
  + Swish

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*22g-mJEUfAWBT7lzgiyIiw.png" style="margin: 0.1em;" alt="Summary of activation functions for neural networks." title="Summary of activation functions for neural networks." width=650>
    </a>
  </div>

+ Sigmoid and softmax functions

  \[\phi(z) = \frac{1}{1 + e^{-z}}\]

  + reasons to be active function
    + sigmoids suffer from the vanishing gradient problem
    + sigmoids are not zero centered; gradient updates go too far in different directions, making optimization more difficult
    + sigmoids saturate and kill gradients
    + sigmoids have slow convergence
  + used as output functions for binary classification
  + generally not used within hidden layers
  + softmax function
    + multidimensional version of the sigmoid
    + used for multiclass classification
  + issue: zero centeredness

+ Hyperbolic tangent function (Tanh) function

  \[\phi(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\]

  + resolving the zero centeredness issue of the sigmoid function
  + always preferred to the sigmoid function within hidden layers
  + suffer from the other problems plaguing the sigmoid function, including the vanishing gradient problem

+ Rectified Linear Unit (ReLU) and Softplus functions
  + Rectified linear function

    \[\phi(z) = max(o, x)\]

    + one of the simplest possible activation functions
    + simplest non-linear activation function
    + avoid and rectify the vanishing gradient problem
    + used by almost all deep learning models
    + only used within hidden layers of a neural network
    + not for output layer
    + issue: maybe unstable during training and die
    + the most successful and widely-used activation function

  + Softplus function

    \[\phi(z) = \ln(1 + e^z)\]

    + a slight variation of ReLU where the transition at zero is somewhat smooth
    + benefit: no discontinuities in the activation function

  + sigmoid for binary classification
  + softmax for multiclass classification
  + linear for a regression problem

+ Leaky ReLU and Generalized ReLU
  + dead neurons: ReLU unstable causes network never activated on any data point
  + Leaky ReLU

    \[g(x_i, \alpha) = \max{a, x_i} + \alpha \min{0, x_i}\]

    + contain a small slope
    + purpose of slope: keep the updates alive and prevent the production of dead neurons
    + still discontinuity at zero
    + no longer flat below zero
    + merely having a reduced gradient
    + a subset of generalized ReLU

  + Leaky ReLU & Generalized ReLU
    + slight variations on the basic ReLU function
    + difference: merely depend on the chosen value of $\alpha$

+ Maxout function

  \[g(x) = \max_{i \in \{ 1, \dots, k\}} \alpha_i x_i + \beta\]

  + simply the maximum of $k$ linear functions
  + a hybrid approach consisting of linear combinations of ReLU and leaky ReLU units

+ Swish: A Self-Gated Activation Function

  \[f(x) = x \cdot sigmoid(x)\]

  + tend to work better than ReLU on deeper models across a number of challenging datasets
  + developed by Google in 2017
  + a smooth non-monotonic function that does not suffer from the problem of zero derivatives
  + seen as a somewhat magical improvement to neural networks
  + a clear improvement for deep networks


<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
    <img src="https://miro.medium.com/max/1400/0*WYB0K0zk1MiIB6xp.png" style="margin: 0.1em;" alt="Curve of Sigmoid function" title="Curve of Sigmoid function" height=200>
    <img src="https://miro.medium.com/max/1400/0*VHhGS4NwibecRjIa.png" style="margin: 0.1em;" alt="Curve of hyperbolic tangent function" title="Curve of hyperbolic tangent function" height=200>
    <img src="https://miro.medium.com/max/875/0*TsH2CNeu5Qlt32Oj.png" style="margin: 0.1em;" alt="Curves of the ReLU & Softplus function" title="Curve of the ReLU & Softplus function" height=200><br/>
    <img src="https://miro.medium.com/max/875/1*pTuWvoEIiHQFBvosVjmW5A.png" style="margin: 0.1em;" alt="Curves of Leaky ReLU & Generalized ReLU functions" title="Curves of Leaky ReLU & Generalized ReLU functions" height=200>
    <img src="https://miro.medium.com/max/875/1*XZQ-Op5RiB2gwXQqOlCvkA.png" style="margin: 0.1em;" alt="Curves of Maxout function" title="Curves of Maxout function" height=200>
    <img src="https://miro.medium.com/max/1250/1*2c9kIQBN0gV-fk4cmr2sAQ.png" style="margin: 0.1em;" alt="Curves of swish functions" title="Curves of swish functions" height=200>
  </a>
</div>



## Loss functions

+ Loss function
  + a.k.a cost function
  + an important aspect of neural networks
  + NN trained using an optimization process that requires a loss function to calculate the model error
  + many functions used to estimate the error of a set of weights in a neural network
  + prefer a function where the space of candidate solutions maps onto a smooth (but high-dimensional) landscape that the optimization algorithm can reasonably navigate via iterative updates to the model wights
  + maximum likelihood: a framework for choosing a loss function when training neural networks and machine models in general
  + what loss function to use depends on the output data distribution and is closely coupled to the output unit
  + main types of loss functions: cross-entropy and mean squared error
  + Neural networks for classification that use a sigmoid or softmax activation function in the output layer learn faster and more robustly using a cross-entropy loss function than using mean squared error
  + The use of cross-entropy looses greatly improved the performance of models with sigmoid and softmax outputs, which had previously suffered from saturatoin and slow learning when using the mean squared error loss. - Deep Learning, 2016

+ Cross-entropy vs. Mean Squared Error
  + form for training data and model distribution (i.e., negative log-likelihood)

    \[J(W) = - \mathbb{E}_{x, y \sim \hat{p}_{data}} \log(p_{model}(y|x))\]

  + examples of cost functions

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
        <img src="https://miro.medium.com/max/1250/1*ERuk0wZZw7sejFI9zwd7zQ.png" style="margin: 0.1em;" alt="an example of a sigmoid output coupled with a mean squared error loss" title="an example of a sigmoid output coupled with a mean squared error loss" height=200>
        <img src="https://miro.medium.com/max/1250/1*mJRBxNfU_mjhmi2lvZLxBg.png" style="margin: 0.1em;" alt="example using a sigmoid output and cross-entropy loss" title="example using a sigmoid output and cross-entropy loss" height=200>
      </a>
    </div>



## Output units

+ Binary classification
  + Example: determining whether a hospital patient has cancer (y=1) or does not have cancer (y=0), the sigmoid function is used as the output

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Vvu7gX6gClWrxTRx5eAkWQ.png" style="margin: 0.1em;" alt="Sigmoid output function used for binary classification." title="Sigmoid output function used for binary classification." width=400>
    </a>
  </div>

+ Multiclass classification
  + Example: dataset trying to filter images into the categories of dogs, cats, and humans

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*-bXQOXPFX03nqSNe2IeoiQ.png" style="margin: 0.1em;" alt="Softmax function for multiclass classification output units." title="Softmax function for multiclass classification output units." width=450>
    </a>
  </div>

+ Specific loss functions
  + MSE on binary data makes very little sense
  + Binary data uses the binary cross entropy loss function
  + more complex deep learning problems: generative adversarial networks (GANs) or autoencoders

+ Summary of data types, distributions, output layers and cost functions

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*s83dd-WhOgE6ZckGST-C8Q.png" style="margin: 0.1em;" alt="Summary of data types, distributions, output layers and cost functions" title="Summary of data types, distributions, output layers and cost functions" width=550>
    </a>
  </div>


## Architecture

+ Model the function $y = x \cdot sin(x)$ using a neural network (NN)
  + Assume NN using ReLU activation function (Fig.1)
  + NN with a two-node single hidden layer as one degree of freedom (Fig.2)
  + NN with a three-node single hidden layer as two degree of freedom (Fig.3)
  + NN with a multi-node hidden layer (Fig.4)
  + NN with 2 one-node hidden layers (Fig.5) approximates the function as a single hidden layer (Fig.1)
  + NN with 3 hidden layers and 3 nodes in each layer (Fig.7) gives a pretty good approximation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Ghcl9sY_91pf7EY4H35SvQ.png" style="margin: 0.1em;" alt="neural network using ReLU activation functions" title="Fig.1 neural network using ReLU activation functions" height=250>
      <img src="https://miro.medium.com/max/875/1*xelnKarSd6ueNrROyOuCrA.png" style="margin: 0.1em;" alt="neural network with a single hidden layer" title="Fig.2 neural network with a single hidden layer" height=250>
      <img src="https://miro.medium.com/max/875/1*3JeEUpze45bJFMvKpE8_MQ.png" style="margin: 0.1em;" alt="neural network with a hidden layer adding another hidden node" title="Fig.3 neural network with a hidden layer adding another hidden node" height=250>
      <img src="https://miro.medium.com/max/875/1*qCxqhOgQeE_7fonbIMTIOw.png" style="margin: 0.1em;" alt="neural network with a multi-node hidden layer" title="Fig.4 neural network with a multi-node hidden layer" height=250>
    </a>
  </div>

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Hg3nySTNWb9N9sMYXRJ32g.png" style="margin: 0.1em;" alt="neural network with 2 one-node hidden layers" title="Fig.5 neural network with 2 one-node hidden layers" height=250>
      <img src="https://miro.medium.com/max/875/1*RkQ1s4RXBaz909vXnPBzJg.png" style="margin: 0.1em;" alt="neural network with 2 2-node hidden layers" title="Fig.6 neural network with 2 2-node hidden layers" height=250>
      <img src="https://miro.medium.com/max/875/1*Jx9Pol3A-ofo8Xl1oalpVw.png" style="margin: 0.1em;" alt="neural network with 3 3-node hidden layers" title="Fig.7 neural network with 3 3-node hidden layers" height=250>
    </a>
  </div>

+ Architectures for neural networks
  + tradeoff by selecting a network architecture 
    + large enough to approximate the function of interest
    + not too large taken an excessive amount of time to train
  + large network requiring large amounts of data to train

+ Good practice
  + using multiple hidden layers as well as multiple nodes within the hidden layers
  + Goodfellow shown that
    + increasing the number of layers of neural networks tends to improve overall test set accuracy
    + large, shallow networks tend to overfit more

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*LtzuTUh0kkCfvlwKxx2zvw.png" style="margin: 0.1em;" alt="increasing the number of layers of neural networks tends to improve overall test set accuracy" title="increasing the number of layers of neural networks tends to improve overall test set accuracy" width=350>
      <img src="https://miro.medium.com/max/875/1*nUHyEEaHEMy3Kl72xTu9Pg.png" style="margin: 0.1em;" alt="large, shallow networks tend to overfit more" title="large, shallow networks tend to overfit more" width=405>
    </a>
  </div>


## Further Reading

+ Deep learning courses:
  + [Andrew Ng’s course on machine learning](https://www.coursera.org/course/ml) has a nice introductory section on neural networks.
  + [Geoffrey Hinton’s course: [Coursera Neural Networks for Machine Learning (fall 2012)](https://www.coursera.org/course/neuralnets)
  + [Michael Nielsen’s free book Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
  + [Yoshua Bengio, Ian Goodfellow and Aaron Courville wrote a [book on deep learning](http://www.iro.umontreal.ca/~bengioy/dlbook/) (2016)
  + [Hugo Larochelle’s course (videos + slides) at Université de Sherbrooke](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html)
  + [Stanford’s tutorial (Andrew Ng et al.) on Unsupervised Feature Learning and Deep Learning](http://ufldl.stanford.edu/wiki/index.php/Main_Page)
  + [Oxford’s ML 2014–2015 course](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
  + [NVIDIA Deep learning course (summer 2015)](https://developer.nvidia.com/deep-learning-courses)
  + [Google’s Deep Learning course on Udacity (January 2016)](https://www.udacity.com/course/deep-learning--ud730)

+ NLP-oriented:
  + [Stanford CS224d: Deep Learning for Natural Language Processing (spring 2015) by Richard Socher](http://cs224d.stanford.edu/syllabus.html)
  + [Tutorial given at NAACL HLT 2013: Deep Learning for Natural Language Processing (without Magic) (videos + slides)](http://nlp.stanford.edu/courses/NAACL2013/)

+ Vision-oriented:
  + CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/) by Andrej Karpathy (a previous version, shorter and less polished: [Hacker’s guide to Neural Networks](http://karpathy.github.io/neuralnets/)).

+ Important neural network articles:
  + [Deep learning in neural networks: An overview](https://www.sciencedirect.com/science/article/pii/S0893608014002135)
  + [Continual lifelong learning with neural networks: A review — Open access](https://www.sciencedirect.com/science/article/pii/S0893608019300231)
  + [Recent advances in physical reservoir computing: A review — Open access](https://www.sciencedirect.com/science/article/pii/S0893608019300784)
  + [Deep learning in spiking neural networks](https://www.sciencedirect.com/science/article/pii/S0893608018303332)
  + [Ensemble Neural Networks (ENN): A gradient-free stochastic method — Open access](https://www.sciencedirect.com/science/article/pii/S0893608018303319)
  + [Multilayer feedforward networks are universal approximators](https://www.sciencedirect.com/science/article/pii/0893608089900208)
  + [A comparison of deep networks with ReLU activation function and linear spline-type methods — Open access](https://www.sciencedirect.com/science/article/pii/S0893608018303277)
  + [Networks of spiking neurons: The third generation of neural network models](https://www.sciencedirect.com/science/article/pii/S0893608097000117)
  + [Approximation capabilities of multilayer feedforward networks](https://www.sciencedirect.com/science/article/pii/089360809190009T)
  + [On the momentum term in gradient descent learning algorithms](https://www.sciencedirect.com/science/article/pii/S0893608098001166)



