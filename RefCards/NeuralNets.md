# Neural Networks

## Fundamentals

### Motivations

+ [Machine learning problems](../ML/MLNN-Hinton/a01-IntroNN.md#the-motivation-for-neural-networks)
  + regressions (and Ridge, LASSO, etc.)
  + classification problem
  + binary classification problem


### Gradient Descent

+ [Gradeint descent/Delta rule](../ML/MLNN-Hinton/a01-IntroNN.md#gradient-descent)
  + an iterative method for finding the minimum of a function
  + Making a step means: $w^{new} = w^{old} + \text{step}$
  + Opposite direction of the derivative means: $w^{new} = w^{old} - \lambda \frac{d\mathcal{L}}{dw}$
  + change to move conventional notation: $w^{(i+1)} = w^{(i)} - \lambda \frac{d\mathcal{L}}{dw}$
  + learning rate ($\lambda$):
    + large learning rate:
      + put more weight on the derivative
      + made larger step for each iteration of the algorithm
    + smaller learning rate
      + less weight is put on the derivative
      + smaller steps made for each iteration

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*MizSwb7-StSLiWlI2MKsxg.png" style="margin: 0.1em;" alt="Illustration of learning rate" title="Illustration of learning rate" width=200>
    </a>
  </div>

+ [Considerations for gradient descent](../ML/MLNN-Hinton/a01-IntroNN.md#gradient-descent)
  + derive the derivatives
  + know what the learning rate is or how to set it
  + avoid local minima
  + the full loss function includes summing up all individual 'errors'

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/625/1*tIqU7GK--aJ-SOdOBrh37Q.png" style="margin: 0.1em;" alt="Illustration of local & global optimal" title="Illustration of local & global optimal" height=150>
      <img src="https://miro.medium.com/max/875/1*MwnXifl-uLdTrjjxiNCDJw.png" style="margin: 0.1em;" alt="Network getting stuck in local minima" title="Network getting stuck in local minima" height=150>
      <img src="https://miro.medium.com/max/875/1*K7HNhO3Fsedvx94psTpBHA.png" style="margin: 0.1em;" alt="Network reach global minima" title="Network reach global minima" height=150>
    </a>
  </div>

+ [Batch and stochastic gradient descent](../ML/MLNN-Hinton/a01-IntroNN.md#gradient-descent)
  + use a batch (a subset) of data as opposed to the whole set of data, such that the loss surface is partially morphed during each iteration
  + the loss (likelihood) function used to derive the derivatives for iteration $k$

    \[\mathcal{L}^k = - \sum_{i \in b^k} \left[ y_i \log(p_i) + (1 - p_i)\log(1 - p_i) \right]\]


### Anatomy

+ [Neural networks:](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-network)
  + excellent tools for finding patterns
  + backpropagation
  + make use of affine __transformations__ to concatenate input features
  + concatenated input passed through an activation function
  + Neural network
    + an abstract representation of the data at each stage of the network
    + designed to detect specific features of the network

+ [Activation function](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-network)
  + evaluate the signal response and determine whether the neuron should be activated given the current inputs
  + extended to multilayer and multi feature networks
    + the number of degrees of freedom (weights and biases) of the network
    + the number of features available which the network can use to make predictions

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*L_lfAEddxxAg2EqJfB5i6Q.png" style="margin: 0.1em;" alt="Activation function with affine transformation" title="Activation function with affine transformation" height=150>
      <img src="https://miro.medium.com/max/1250/1*l57B0pjXoO-1H1xZYV7QBA.png" style="margin: 0.1em;" alt="A neural network with one hidden layer and two features (the simplest possible multi-layer multi-feature network)." title="A neural network with one hidden layer and two features (the simplest possible multi-layer multi-feature network)." height=150>
    </a>
  </div>

+ [Network parameters (weights and biases)](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-network)
  + updated by assigning the error of the network
  + using backpropagation through the network to obtain the derivatives for each of the parameters w.r.t. the loss function
  + gradient descent used to update these parameters

+ [Training network](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-network)
  + the process of accessing the error and updating the parameters
  + a training set to generate a functional network
  + performance of the network
  + test set: unseen data accessed by testing

+ [Degrees of freedom](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-network)
  + neural network having a large number of degrees
  + required a large amount of data for training to be able to make adequate predictions
  + useful for high dimensionality of the data

+ [Generalized multilayer and multi-feature network](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*GApLZ60775yXfUzO65PfEA.png" style="margin: 0.1em;" alt="Generalized multilayer perceptron with n hidden layers, m nodes, and d input features." title="Generalized multilayer perceptron with n hidden layers, m nodes, and d input features." width=350>
    </a>
  </div>

  + $m$ nodes: the width of a layer within the network
  + $n$ hidden layers: the depth of the network
  + $d$ inputs: pre-specified by the available data
    + normal Pandas data frame: equal to the number of feature columns
  + the hidden layers of the network have the same width (number of nodes)
  + the number of nodes may vary across the hidden layers
  + the output layer may also be of an arbitrary dimension depending on the required output

+ [Convolutional neural networks](../ML/MLNN-Hinton/a02-IntermediateNN.md#anatomy-of-a-neural-network)
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




### Types of Learning

+ [Problem Modeling](../ML/MLNN-Hinton/01-IntroML.md#three-types-of-learning)
  + Supervised learning: Regression & Classification
  + Reinforcement learning
  + Unsupervised learning

+ [Typical Supervised learning procedure](../ML/MLNN-Hinton/01-IntroML.md#three-types-of-learning)
  1. Choosing a model class: $y = f(\mathbf{x}; \mathbf{W})$
    + $\mathbf{x}$: input vector
    + $\mathbf{W}$: weight vector
    + $f$: activation function to transform input $\mathbf{x}$ with weight vector $\mathbf{W}$ to the output $y$
  2. Learning by adjust $\mathbf{W}$ with cost function
    + reduce the difference between target value $t$ and actual output $y$
    + Regression measurement: usually $\frac{1}{2} (t - y)^2$
    + Classification measurement: other sensible measures

+ [Reinforcement learning](../ML/MLNN-Hinton/01-IntroML.md#three-types-of-learning)
  + Output: an action or sequence of actions
  + The only supervisory signal: an occasional scalar reward
  + Decision of action(s) selected: maximize the expected sum of the future reward
  + typically delayed reward makes model hard

+ [Unsupervised learning](../ML/MLNN-Hinton/01-IntroML.md#three-types-of-learning)
  + no clear goal
  + typically find sensible clusters


### Learning Methodologies

+ [Learning by perturbing weights](../ML/MLNN-Hinton/03-Backpropagation.md#the-backpropagation-algorithm)
  + randomly perturb one weight and see if it improves performance: very inefficient
  + Alternative: randomly perturb all the weights in parallel and correlate the performance gain with the weight changes
  + Better: randomly perturb the activities of the hidden units

+ [Randomly perturb the activities of the hidden units](../ML/MLNN-Hinton/03-Backpropagation.md#the-backpropagation-algorithm)
  + adding a layer of hand-coded features
    + more powerful but hard to design the features
    + finding good features w/o requiring insights into the task or repeated trial and error
    + guess features and see how well they work
  + automate the loop of designing features for a particular task and seeing ho well they work


### Considerations of Learning Procedures

+ [Main decisions about how to use error derivatives](../ML/MLNN-Hinton/03-Backpropagation.md#how-to-use-the-derivatives-computed-by-the-backpropagation-algorithm)
  + Optimization issue: how to discover a good set of weights with the error derivatives on individual cases?
  + Generalization issue: how to ensure non-seen cases during training work well with trained weights?

+ [Optimization Concerns](../ML/MLNN-Hinton/03-Backpropagation.md#how-to-use-the-derivatives-computed-by-the-backpropagation-algorithm)
  + How often to update the weights
    + Online
    + Full batch
    + Mini-batch
  + How much to update the weights
    + fixed learning rate
    + adaptive learning rate globally
    + adaptive learning rate on each connection separately

+ [Generalization Concern - Overfitting](../ML/MLNN-Hinton/03-Backpropagation.md#how-to-use-the-derivatives-computed-by-the-backpropagation-algorithm)
  + Unable to identify which regularities causing errors
  + Possible solutions:
    + Weight-decay
    + Weight-sharing
    + Early stopping
    + Model averaging
    + Bayesian fitting on neural nets
    + Dropout
    + Generative pre-training


### Concepts and Neural Networks

+ [Concepts in cognition science](../ML/MLNN-Hinton/04-Multiclasses.md#a-brief-diversion-into-cognitive-science)
  + The feature theory: a concept is a set of semantic features
  + The structuralist theory: the meaning of a concept lies in its relationships to other concepts
  + Minsky (1970s): in favor of relational graph representations with structuralist theory
  + Hinton - both applicable
    + able to use vectors of semantic features to implement a relational graph
    + no intervening conscious steps but many computation in interactions of neurons
    + explicit rules for conscious, deliberate, reasoning
    + commonsense, analogical reasoning: seeing the answer w/o conscious intervening steps

+ [Localist and distributed representations of concepts](../ML/MLNN-Hinton/04-Multiclasses.md#a-brief-diversion-into-cognitive-science)
  + Localist representation
    + implementation of relational graph in a neural net
    + neuron = node in the graph
    + connection = a binary relationship
    + "localist" method not working: many different types of relationship and the connections in neural nets w/o discrete labels
  + Distributed representations
    + open issue: how to implement relational knowledge in a neural net
    + many-to-many mapping btw concepts and neurons

+ [Neural Network Algorithm](../ML/MLNN-Hinton/a01-IntroNN.md#the-motivation-for-neural-networks)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*QIKMKejAH9cjXxe-PIIU7g.png" style="margin: 0.1em;" alt="Formulation of Neural Networks" title="Formulation of Neural Networks" width=400>
    </a>
  </div>

  + weights in neural networks: these regression parameters of our various incoming functions
  + passed to an activation function which decides whether the result is significant enough to 'fire' the node
  + start with some arbitrary formulation of values in order for us to start updating and optimizing the parameters
  + assessing the loss function after each update and performing gradient descent

+ [Ways to minimize the loss function](../ML/MLNN-Hinton/a01-IntroNN.md#the-motivation-for-neural-networks)
  + Descent
    + The value of $w$ to minimize $\mathcal{L}(w)$
    + to find the optimal point of a function$\mathcal{L}(w)$: $\frac{d \mathcal{L}(W)}{dW} = 0$
    + find the $w$ that satisfies the equation
  + more flexible method
    + start from any point and then determine which direction to go to reduce the loss (left or right in this case)
    + calculate the slope of the function at this point
    + then shift to the right if the slope is negative or shift to the left if the slope is positive

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*qbvfebFdO7rxU4QVl6tQtw.png" style="margin: 0.1em;" alt="Diagram of the loss function" title="Diagram of the loss function" height=150>
      <img src="https://miro.medium.com/max/875/1*4l_ZpZRZ6mwKAXWo4Q20QA.png" style="margin: 0.1em;" alt="Diagram of the loss function with starting point" title="Diagram of the loss function with starting point" height=150>
    </a>
  </div>


### Activation Functions

+ [Activation functions](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)
  + analogous to the build-up of electrical potential in biological neurons
  + activation potential: mimicked in artificial neural networks using a probability
  + Characteristics:
    + non-linearity: ensures not linearity
    + differentiable: ensure gradients remain large through the hidden unit
  + The general form of an activation function

    \[h = f(W^T X + b)\]

    + $h$: the neural output
    + $f(.)$: the activation function acting on the weights and bases

+ [Non-linearity](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)
  + linear function
    + a polynomial of one degree
    + linear equation easy to solve
    + limited in complexity and less power to learn complex functional mappings from data
  + Neural network w/o activation function
    + a linear regression model
    + limited in the set of functions able to approximate
  + Universal approximation theorem: generalized non-linear function approximations
  + non-linear activation able to generate non-linear mappings from inputs to outputs

+ [Differentiable](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)
  + required to perform backpropagation in the network
  + required to compute gradients of errors (loss) w.r.t. to the weights updated using gradient descent
  + linear activation function
    + an easily differentiable function
    + optimized using convex optimization
    + limited model capacity

+ [Vanishing gradient problem](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)
  + small gradients and several hidden layers results in multiplied gradient during backpropagation
  + computer limitation on precision when multiply many small numbers
  + the value of the gradient quickly vanished
  + important challenge generated in deep neural networks

+ [Common choices of activation function](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*22g-mJEUfAWBT7lzgiyIiw.png" style="margin: 0.1em;" alt="Summary of activation functions for neural networks." title="Summary of activation functions for neural networks." width=550>
    </a>
  </div>

+ [Sigmoid and softmax functions](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  \[\phi(z) = \frac{1}{1 + e^{-z}}\]

  + used as output functions for binary classification
  + generally not used within hidden layers
  + softmax function
    + multidimensional version of the sigmoid
    + used for multiclass classification
  + issue: zero centeredness

+ [Hyperbolic tangent function (Tanh) function](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  \[\phi(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\]

  + resolving the zero centeredness issue of the sigmoid function
  + always preferred to the sigmoid function within hidden layers
  + suffer from the other problems plaguing the sigmoid function, including the vanishing gradient problem

+ [Rectified Linear Unit (ReLU)](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  \[\phi(z) = max(o, x)\]

  + simplest non-linear activation function
  + avoid and rectify the vanishing gradient problem
  + used by almost all deep learning models
  + only used within hidden layers of a neural network
  + issue: maybe unstable during training and die
  + the most successful and widely-used activation function

+ [Softplus functions](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  \[\phi(z) = \ln(1 + e^z)\]

  + a slight variation of ReLU where the transition at zero is somewhat smooth
  + benefit: no discontinuities in the activation function

  + sigmoid for binary classification
  + softmax for multiclass classification
  + linear for a regression problem

+ [Leaky ReLU and Generalized ReLU](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)
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

+ [Maxout function](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  \[g(x) = \max_{i \in \{ 1, \dots, k\}} \alpha_i x_i + \beta\]

  + simply the maximum of $k$ linear functions
  + a hybrid approach consisting of linear combinations of ReLU and leaky ReLU units

+ [Swish: A Self-Gated Activation Function](../ML/MLNN-Hinton/a02-IntermediateNN.md#activation-functions)

  \[f(x) = x \cdot sigmoid(x)\]

  + tend to work better than ReLU on deeper models across a number of challenging datasets
  + a smooth non-monotonic function that does not suffer from the problem of zero derivatives
  + seen as a somewhat magical improvement to neural networks
  + a clear improvement for deep networks


<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
    <img src="https://miro.medium.com/max/1400/0*WYB0K0zk1MiIB6xp.png" style="margin: 0.1em;" alt="Curve of Sigmoid function" title="Curve of Sigmoid function" height=140>
    <img src="https://miro.medium.com/max/1400/0*VHhGS4NwibecRjIa.png" style="margin: 0.1em;" alt="Curve of hyperbolic tangent function" title="Curve of hyperbolic tangent function" height=140>
    <img src="https://miro.medium.com/max/875/0*TsH2CNeu5Qlt32Oj.png" style="margin: 0.1em;" alt="Curves of the ReLU & Softplus function" title="Curve of the ReLU & Softplus function" height=140>
    <img src="https://miro.medium.com/max/875/1*pTuWvoEIiHQFBvosVjmW5A.png" style="margin: 0.1em;" alt="Curves of Leaky ReLU & Generalized ReLU functions" title="Curves of Leaky ReLU & Generalized ReLU functions" height=140><br/>
    <img src="https://miro.medium.com/max/875/1*XZQ-Op5RiB2gwXQqOlCvkA.png" style="margin: 0.1em;" alt="Curves of Maxout function" title="Curves of Maxout function" height=150>
    <img src="https://miro.medium.com/max/1250/1*2c9kIQBN0gV-fk4cmr2sAQ.png" style="margin: 0.1em;" alt="Curves of swish functions" title="Curves of swish functions" height=150>
  </a>
</div>


### Lost/Cost Function

+ [Loss function/cost function](../ML/MLNN-Hinton/a02-IntermediateNN.md#loss-functions)
  + NN trained using an optimization process that requires a loss function to calculate the model error
  + many functions used to estimate the error of a set of weights in a neural network
  + prefer a function where the space of candidate solutions maps onto a smooth (but high-dimensional) landscape that the optimization algorithm can reasonably navigate via iterative updates to the model wights
  + maximum likelihood: a framework for choosing a loss function when training neural networks and machine models in general
  + what loss function to use depends on the output data distribution and is closely coupled to the output unit
  + main types of loss functions: cross-entropy and mean squared error
  + cross-entropy loss function > mean squared error: classification that use a sigmoid or softmax activation function in the output layer learn faster and more robustly
  + The use of cross-entropy looses greatly improved the performance of models with sigmoid and softmax outputs, which had previously suffered from saturation and slow learning when using the mean squared error loss. - Deep Learning, 2016

+ [Cross-entropy vs. Mean Squared Error](../ML/MLNN-Hinton/a02-IntermediateNN.md#loss-functions)
  + form for training data and model distribution (i.e., negative log-likelihood)

    \[J(W) = - \displaystyle \mathbb{E}_{x, y \sim \hat{p}_{data}} \log(p_{\text{model}}(y|x))\]

  + example of cost functions

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
        <img src="https://miro.medium.com/max/1250/1*ERuk0wZZw7sejFI9zwd7zQ.png" style="margin: 0.1em;" alt="an example of a sigmoid output coupled with a mean squared error loss" title="an example of a sigmoid output coupled with a mean squared error loss" height=180>
        <img src="https://miro.medium.com/max/1250/1*mJRBxNfU_mjhmi2lvZLxBg.png" style="margin: 0.1em;" alt="example using a sigmoid output and cross-entropy loss" title="example using a sigmoid output and cross-entropy loss" height=180>
      </a>
    </div>


### Output Units

+ [Summary of data types, distributions, output layers and cost functions](../ML/MLNN-Hinton/a02-IntermediateNN.md#output-units)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*s83dd-WhOgE6ZckGST-C8Q.png" style="margin: 0.1em;" alt="Summary of data types, distributions, output layers and cost functions" title="Summary of data types, distributions, output layers and cost functions" width=500>
    </a>
  </div>


## Architectures

### Types of Architectures

+ [A mostly complete chart of Neural Networks](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464" ismap target="_blank">
      <img src="https://miro.medium.com/max/2500/1*cuTSPlTq0a_327iTPJyD-Q.png" style="margin: 0.1em;" alt="Mostly complete neural network architecture" title="Mostly complete neural network architecture" width=100%>
    </a>
  </div>

  + Perceptron (P)
    + simplest and oldest model
    + takes some inputs, sums them up, applies activation function and pass them to output layer

  + Feed-forward neural networks
    + all nodes fully connected
    + activation flows from input layer to output, w/o back loops
    + one hidden layer between input and output layers
    + training using backpropagation method

  + Radical Basis Neural (RBF) Networks
    + FF (feed-forward) NNs
    + activation function: radial basis function
    + perfect for function approximation, and machine control

  + Deep Feed Forward (DFF) Neural Network
    + FF NN w/ more than one hidden layer
    + stacking errors with more layers resulted in exponential growth of training times
    + approaches developed in 00s allowed to train DFFs effectively

  + Recurrent Neural network (RNN)
    + a.k.a Jordan network
    + each of hidden cell received its own output with fixed delay
    + mainly used =when context is important

+ [Feed-forward neural Networks](../ML/MLNN-Hinton/02-Perceprtons.md#an-overview-of-the-main-types-of-network-architecture)
  + Input layer: the first layer
  + Output layer: the last layer
  + Hidden layer(s): layer(s) between the Input & Output layers
  + Deep Neural network: more than one hidden layer

+ [Recurrent neural network](../ML/MLNN-Hinton/02-Perceprtons.md#an-overview-of-the-main-types-of-network-architecture) (RNN)
  + the previous network state influencing the output
  + a function with inputs $x_t$ (input vector) and previous state $h_{t-1}$
  + complicated dynamics and difficult to train
  + a very natural way to model sequential data
  + able to remember information in their hidden state for a long time

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781788397872/1/ch01lvl1sec21/feed-forward-and-feedback-networks" ismap target="_blank">
    <img src="https://static.packt-cdn.com/products/9781788397872/graphics/1ebc2a0a-2123-4351-b7e1-eb57f098bafa.png" style="margin: 0.1em;" alt="Feed-forward network" title="Feed-forward network" height=150>
  </a>
  <a href="https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/recurrent_neural_networks.html" ismap target="_blank">
    <img src="https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_6/recurrent.jpg" style="margin: 0.1em;" alt="Recurrent Neural Network" title="Recurrent Neural Network" height=150>
  </a>
</div>

+ [Symmetrically connected neural networks](../ML/MLNN-Hinton/02-Perceprtons.md#an-overview-of-the-main-types-of-network-architecture)
  + Hopfield neural networks
    + an example of recurrent network
    + output of neurons connected to input of every neuron by means of appropriate weights
    + much easier to analyze than recurrent networks
    + the same weight in both direction
  + Boltzman machines
    + symmetrically connected networks with hidden units
    + more powerful than Hopfield networks but less powerful than recurrent networks
    + fully connected within and between layers
    + the stochastic, generative counterpart of Hopfield networks
    + Restricted Boltzmann Machine (RBM): the lateral connections in the visible and hidden layers are removed

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="http://galaxy.agh.edu.pl/~vlsi/AI/hopf/hopfield_eng.html" ismap target="_blank">
    <img src="http://galaxy.agh.edu.pl/~vlsi/AI/hopf/hopfield_eng_pliki/image002.jpg" style="margin: 0.1em;" alt="Hopfield Neural Network" title="Hopfield Neural Network" height=150>
  </a>
  <a href="https://www.researchgate.net/figure/Boltzmann-and-Restricted-Boltzmann-Machines-A-Boltzmann-machine-is-fully-connected_fig8_257649811" ismap target="_blank">
    <img src="https://www.researchgate.net/profile/Dan_Neil/publication/257649811/figure/fig8/AS:272067278929927@1441877302138/Boltzmann-and-Restricted-Boltzmann-Machines-A-Boltzmann-machine-is-fully-connected.png" style="margin: 0.1em;" alt="Boltzmann and Restricted Boltzmann Machines" title="Boltzmann and Restricted Boltzmann Machines" height=150>
  </a>
</div>

+ [Model the function $y = x \cdot sin(x)$ using a neural network (NN)](../ML/MLNN-Hinton/a02-IntermediateNN.md#architecture)
  + Assume NN using ReLU activation function (Fig.1)
  + NN with a two-node single hidden layer as one degree of freedom (Fig.2)
  + NN with a three-node single hidden layer as two degree of freedom (Fig.3)
  + NN with a multi-node hidden layer (Fig.4)
  + NN with 2 one-node hidden layers (Fig.5) approximates the function as a single hidden layer (Fig.1)
  + NN with 3 hidden layers and 3 nodes in each layer (Fig.7) gives a pretty good approximation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Ghcl9sY_91pf7EY4H35SvQ.png" style="margin: 0.1em;" alt="neural network using ReLU activation functions" title="Fig.1 neural network using ReLU activation functions" height=200>
      <img src="https://miro.medium.com/max/875/1*xelnKarSd6ueNrROyOuCrA.png" style="margin: 0.1em;" alt="neural network with a single hidden layer" title="Fig.2 neural network with a single hidden layer" height=200>
      <img src="https://miro.medium.com/max/875/1*3JeEUpze45bJFMvKpE8_MQ.png" style="margin: 0.1em;" alt="neural network with a hidden layer adding another hidden node" title="Fig.3 neural network with a hidden layer adding another hidden node" height=200>
      <img src="https://miro.medium.com/max/875/1*qCxqhOgQeE_7fonbIMTIOw.png" style="margin: 0.1em;" alt="neural network with a multi-node hidden layer" title="Fig.4 neural network with a multi-node hidden layer" height=200>
    </a>
  </div>

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*Hg3nySTNWb9N9sMYXRJ32g.png" style="margin: 0.1em;" alt="neural network with 2 one-node hidden layers" title="Fig.5 neural network with 2 one-node hidden layers" height=200>
      <img src="https://miro.medium.com/max/875/1*RkQ1s4RXBaz909vXnPBzJg.png" style="margin: 0.1em;" alt="neural network with 2 2-node hidden layers" title="Fig.6 neural network with 2 2-node hidden layers" height=200>
      <img src="https://miro.medium.com/max/875/1*Jx9Pol3A-ofo8Xl1oalpVw.png" style="margin: 0.1em;" alt="neural network with 3 3-node hidden layers" title="Fig.7 neural network with 3 3-node hidden layers" height=200>
    </a>
  </div>

+ [Architectures for neural networks](../ML/MLNN-Hinton/a02-IntermediateNN.md#architecture)
  + tradeoff by selecting a network architecture 
    + large enough to approximate the function of interest
    + not too large taken an excessive amount of time to train
  + large network requiring large amounts of data to train

+ [Good practice](../ML/MLNN-Hinton/a02-IntermediateNN.md#architecture)
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



### Simple Neuron Model

+ A biological neuron with a basic mathematical mode

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.embedded-vision.com/platinum-members/cadence/embedded-vision-training/documents/pages/neuralnetworksimagerecognition" ismap target="_blank">
      <img src="https://www.embedded-vision.com/sites/default/files/technical-articles/CadenceCNN/Figure3a.jpg" style="margin: 0.1em;" alt="Illustration of a biological neuron" title="Illustration of a biological neuron" width=350>
      <img src="https://www.embedded-vision.com/sites/default/files/technical-articles/CadenceCNN/Figure3b.jpg" style="margin: 0.1em;" alt="Illustration of a biological neuron's mathematical model" title="Illustration of a biological neuron's mathematical model" width=350>
    </a>
  </div>

+ [Linear neuron](../ML/MLNN-Hinton/01-IntroML.md#some-simple-models-of-neurons)

  \[z = b + \sum_i w_i x_i\]

  + $y$: the output
  + $b$: the bias
  + $w_i$: the weight on the $i$-th input
  + $x_i$: the $i$-th input

+ [Typical Activation functions $f(\cdot)$](../ML/MLNN-Hinton/01-IntroML.md#some-simple-models-of-neurons)
  + Binary threshold

    \[z = b + \sum_i w_i x_i \implies y = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}\]

  + Rectified Linear Neurons

      \[z = b + \sum_i x_i w_i \implies y = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}\]

  + Sigmoid neurons

    \[z = b + \sum_i x_i w_i \implies y = \frac{1}{1 + e^{-z}}\]

  + Stochastic binary neurons

    \[z = b + \displaystyle \sum_i x_i w_i \implies p(s = 1) = \frac{1}{1 + e^{-z}}\]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://blog.zaletskyy.com/some-simple-models-of-neurons" ismap target="_blank">
      <img src="https://blog.zaletskyy.com/Media/Default/NeuralNetworks/binaryNeuron.png" style="margin: 0.1em;" alt="Binary threshold neuron" title="Binary threshold neuron" height=120>
    </a>
    <a href="https://www.bo-song.com/coursera-neural-networks-for-machine-learning/" ismap target="_blank">
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-2.png" style="margin: 0.1em;" alt="Rectified Linear Neurons" title="Rectified Linear Neurons  (ReLU)" height=120>
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-5.png" style="margin: 0.1em;" alt="Sigmoid neurons" title="Sigmoid neurons" height=120>
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-6.png" style="margin: 0.1em;" alt="Stochastic binary neurons" title="Stochastic binary neurons" height=120>
    </a>
  </div>


### Perceptrons

+ [The standard Perceptron architectures](../ML/MLNN-Hinton/02-Perceprtons.md#perceptrons-the-first-generation-of-neural-networks)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html" ismap target="_blank">
      <img src="https://sebastianraschka.com/images/blog/2015/singlelayer_neural_networks_files/perceptron_schematic.png" style="margin: 0.1em;" alt="Rosenblatt's Perceptron architecture" title="Rosenblatt's Perceptron architecture" height=150>
    </a>
    <a href="https://towardsdatascience.com/perceptron-the-artificial-neuron-4d8c70d5cc8d" ismap target="_blank">
      <img src="https://miro.medium.com/max/806/1*-JtN9TWuoZMz7z9QKbT85A.png" style="margin: 0.1em;" alt="Minsky-Papert Perceptron architecture" title="Minsky-Papert  Perceptron architecture" height=150>
    </a>
    <a href="https://www.researchgate.net/figure/The-McCulloch-Pitts-Neuron_fig1_265486784" ismap target="_blank">
      <img src="https://www.researchgate.net/profile/Sean_Doherty2/publication/265486784/figure/fig1/AS:669465553432601@1536624434844/The-McCulloch-Pitts-Neuron.png" style="margin: 0.1em; background-color: white;" alt="McCulloch-Pitts Perceptron architecture" title="McCulloch-Pitts Perceptron architecture" height=150>
    </a>
  </div>

+ Frank Rosenblatt (1960's)
  + a very powerful learning algorithm
  + clams on what they can learn to do

+ Minsky & Papert, "Perceptrons" (1969)
  + analyze what they could do and their limitations
  + people think the limitations applied to all neural network models

+ McCulloch-Pitts (1943): Binary threshold neurons

  \[z = b + \sum_i x_i w_i \implies y = \begin{cases}1 & \text{if } z > 0 \\ 0 & \text{otherwise}\end{cases}\]


+ [Structure of neurons & model](../ML/MLNN-Hinton/a01-IntroNN.md#artificial-neural-network-ann)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/0*6CQ5E2qYm1kOwEW2.png" style="margin: 0.1em;" alt="The structure of a neuron looks a lot more complicated than a neural network, but the functioning is similar." title="Structure of neuron" height=150>
      <img src="https://miro.medium.com/max/875/1*TiQJRO4b3--hIBmEccukUg.png" style="margin: 0.1em;" alt="a neural diagram that makes the analogy between the neuron structure and the artificial neurons in a neural network." title="artificial neurons in a neural network" height=150>
    </a>
  </div>

+ [Affine transformation](../ML/MLNN-Hinton/a01-IntroNN.md#artificial-neural-network-ann)
  + basically an addition (or subtraction) and/or multiplication
  + resembling a regression equation
  + becomes important with multiple nodes converging at a node in a multilayer perceptron
  + abstract the affine and activation blocks into a single block

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*t5H6ohP8hC2bMr680XX9xw.png" style="margin: 0.1em;" alt="Analogous of single neural network with perceptron" title="Analogous of single neural network with perceptron" width=450>
    </a>
  </div>

  + the amalgamation of the outputs from upstream nodes and the summed output is then passed to an activation function, which assesses the probability to determine whether it’s the quantitative value (the probability) sufficient to make the neuron fire

+ [Perceptron convergence procedure:](../ML/MLNN-Hinton/02-Perceprtons.md#perceptrons-the-first-generation-of-neural-networks)
  + training binary output as classifier
  + bias
    + adding extra component with value 1 to each input vector
    + minus the threshold
  + using policy to ensure the correct cases should be picked
  + find a set of weights to pick all correct ones

+ [Weight space](../ML/MLNN-Hinton/02-Perceprtons.md#a-geometrical-view-of-perceptrons)
  + 1-dim per weight
  + point: a particular setting of all the weights
  + a training case as a hyperplane though the origin
  + cone of feasible solutions
    + find a point on the right side of all planes
    + any weight vectors for all training cases correct

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture2/lec2.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m02-05.png" style="margin: 0.1em;" alt="Weight space: input vector with correct answer=1" title="Weight space: input vector with correct answer=1" height=200>
      <img src="../ML/MLNN-Hinton/img/m02-06.png" style="margin: 0.1em;" alt="Weight space: input vector with correct answer=0" title="Weight space: input vector with correct answer=0" height=200>
    </a>  $\implies$
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture2/lec2.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m02-07.png" style="margin: 0.1em;" alt="Feasible solutions" title="Feasible solutions" height=200>
    </a>
  </div>

+ [Learning procedure](../ML/MLNN-Hinton/02-Perceprtons.md#why-the-learning-works)
  + using margin instead of squared distance
  + provide a feasible region by a margin at least as large as the length of the input vector

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m02-08.png" style="margin: 0.1em;" alt="Distance btw the current and feasible vectors" title="Distance btw the current & feasible vectors" height=100>
      <img src="../ML/MLNN-Hinton/img/m02-09.png" style="margin: 0.1em;" alt="margin: the squared length btw hyperplan and feasible weight vectors" title="margin: the squared length btw hyperplan and feasible weight vectors" height=100>
    </a>
  </div>

+ [Limitations of Perceptrons](../ML/MLNN-Hinton/02-Perceprtons.md#what-perceptrons-can-not-do)
  + hard-coded features restrict what a perceptron do
    + Solution: adding extra feature(s) to separate
  + Minsky & Papert, "Group Invariance Theorem": unable to discriminating simple patterns under translation w/ wrap-around
    + Solution: adding multiple layers of adaptive, non-linear hidden units

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.youtube.com/watch?v=mI6jTc-8sUY&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9&index=11&t=0s" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m02-10.png" style="margin: 0.1em;" alt="A geometric view of what binary threshold neurons cannot do" title="A geometric view of what binary threshold neurons cannot do" height=150>
      <img src="../ML/MLNN-Hinton/img/m02-11.png" style="margin: 0.1em;" alt="Discriminating simple patterns under translation with wrap-around" title="Discriminating simple patterns under translation with wrap-around" height=150>
    </a>
  </div>


## Linear Neurons

### Model of Linear Neurons

+ Comparisons
  + Perceptron: the weights getting closer to a good set of weights
  + Linear neurons: the output getting closer to target outputs
  + perceptron unable to generalize to hidden layers

+ [Linear neurons](../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-linear-neuron)
  + linear filter in EE
  + real-valued output: weighted sum of outputs

    \[y = \sum_i x_i w_i = \mathbf{W}^T \mathbf{x}\]

    + $y$: neuron's estimate the desired output
    + $\mathbf{W}$: weight vector
    + $\mathbf{x}$: input vector
  + aim of learning (objective): to minimize the error summed over all training cases
  + error (measure): the squared difference btw the desired output and the actual output


### Cost Function for Linear Neurons

+ [Definition](../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-linear-neuron):
  
  \[E = \frac{1}{2} \sum_{n \in training} (t^n - y^n)^2\]

  + $E$: total error
  + $t^n$: the target value of $n$-th sampling case
  + $y^n$: the actual value of $n$-th sampling case
  + $1/2$: factor to cancel the derivative constant

+ [Derivative of Error function for weights](../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-linear-neuron)

  \[\dfrac{\partial E}{\partial w_i} = \frac{1}{2} \sum_n \dfrac{\partial y^n}{\partial w_i} \dfrac{dE^n}{dy^n} = - \sum_n x_i^n (t^n - y^n)\]

  + applying chain rule
  + explain how the output changes as we change the weights times how the error changes as we change the output

+ [Batch delta rule](../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-linear-neuron)

  \[\Delta w_i = -\varepsilon \dfrac{\partial E}{\partial w_i} = \sum_n \varepsilon x_i^n (t^n - y^n)\]

+ [online delta-rule vs learning rule for perceptrons]((../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-linear-neuron))
  + perceptron learning
    + increment or decrement the weight vector by the input vector
    + only change the weights when making an error
  + online version of the delta-rule
    + increment or decrement the weight vector by the input vector but scaled by the residual error and the learning rate
    + choose a learning rate $\rightarrow$ annoying
      + too big $\rightarrow$ unstable
      + too small $\rightarrow$ slow


### Error Surface for Linear Neuron

+ [Error surface in extended weight space](../ML/MLNN-Hinton/03-Backpropagation.md#the-error-surface-for-a-linear-neuron)
  + Linear neuron with a squared error
    + quadratic bowl: linear neuron with a squared error
    + parabolas: vertical cross-sections
    + ellipses: horizontal cross-sections
  + multi-layer, non-linear nets: much more complicated
    + smooth curves
    + local minima
  + pictorial view of gradient descent learning using Delta rule

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m03-03.png" style="margin: 0.1em;" alt="error surface" title="error surface" height=150>
    </a>
    <a href="https://math.stackexchange.com/questions/1249308/what-is-the-difference-between-an-elliptical-and-circular-paraboloid-3d/1249309#1249309" ismap target="_blank">
      <img src="https://i.stack.imgur.com/goYnm.gif" style="margin: 0.1em;" alt="An elliptical paraboloid" title="An elliptical paraboloid" height=150>
    </a>
  </div>

+ [Online vs batch learning](../ML/MLNN-Hinton/03-Backpropagation.md#the-error-surface-for-a-linear-neuron)
  + Simplest kind of batch learning (left diagram)
    + elliptical contour lines
    + steepest descent on the error surface
    + travel perpendicular to the contour lines
    + batch learning: the gradient descent summed over all training cases
  + simplest kind of online learning (right diagram)
    + online learning: update the weights in proportion to the gradient after each training case
    + zig-zag around the direction of steepest descent
  + elongated ellipse: the direction of steepest descent almost perpendicular to the direction towards the minimum

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m03-04.png" style="margin: 0.1em;" alt="Contour for batch learning" title="Contour for batch learning" height=150>
      <img src="../ML/MLNN-Hinton/img/m03-05.png" style="margin: 0.1em;" alt="Contour for online learning" title="Contour for online learning" height=150>
      <img src="../ML/MLNN-Hinton/img/m03-06.png" style="margin: 0.1em;" alt="enlongated ellipse with slow learning" title="enlongated ellipse with slow learning" height=150>
    </a>
  </div>


### Backpropagation

+ [Backpropagation](../ML/MLNN-Hinton/a01-IntroNN.md#backpropagation)
  + the central mechanism by which neural networks learn
  + During prediction, a neural network propagates signal forward through the nodes of the network until it reaches the output layer where a decision is made.
  + A neural network propagates signal forward through the nodes of the network until it reaches the output layer where a decision is made.
  + The network then backpropagates information about this error backward through the network such that it can alter each of the parameters.
  + Backpropagation performed first in order to gain the information necessary to perform gradient descent.

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*q1M7LGiDTirwU-4LcFq7_Q.png" style="margin: 0.1em;" alt="The forward pass on the left calculates z as a function f(x,y) using the input variables x and y. The right side of the figures shows the backward pass. Receiving dL/dz, the gradient of the loss function with respect to z from above, the gradients of x and y on the loss function can be calculate by applying the chain rule, as shown in the figure." title="Forward propagation and Back propagation" width=550>
    </a>
  </div>


+ [Automatic differentiation](../ML/MLNN-Hinton/a01-IntroNN.md#backpropagation)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*fdYrXF6IGhS0nitxkoHJcA.png" style="margin: 0.1em;" alt="Derivative of Loss function with differentiation" title="Derivative of Loss function with differentiation" width=450>
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*fdYrXF6IGhS0nitxkoHJcA.png" style="margin: 0.1em;" alt="function library to the architecture such that the procedure is abstracted and update automatically as the network architecture" title="Function library" width=450>
    </a>
    </a>
  </div>

  + a function library that is inherently linked to the architecture such that the procedure is abstracted and updates automatically as the network architecture is updated

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/1250/1*fdYrXF6IGhS0nitxkoHJcA.png" style="margin: 0.1em;" alt="function library to the architecture such that the procedure is abstracted and update automatically as the network architecture" title="Function library" width=650>
    </a>
  </div>


## Logistic Neurons

### Model for Logistic Neurons

+ [Definition](../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-logistic-output-neuron)

  \[z = b + \sum_i x_i w_i \qquad y = \frac{1}{1 + e^{-z}}\]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.bo-song.com/coursera-neural-networks-for-machine-learning/" ismap target="_blank">
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-5.png" style="margin: 0.1em;" alt="Logistic function" title="Logistic function" width=200>
    </a>
  </div>

+ [Derivative of the output w.r.t. the logit](../ML/MLNN-Hinton/03-Backpropagation.md#learning-the-weights-of-a-logistic-output-neuron)

  \[y = \frac{1}{1 + e^{-z}} \quad \implies \quad \frac{dy}{dz} = y(1-y)\]

+ [Logistic regression](../ML/MLNN-Hinton/a01-IntroNN.md#the-motivation-for-neural-networks)
  + the problem of estimating a probability that someone has heart disease, P(y=1), given an input value X.
  + the logistic function, to model P(y=1):

    \[P(Y=1) = \frac{e^{\beta_0+\beta_1 X}}{1 + e^{\beta_0+\beta_1 X}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}\]

  + general shape: the model will predict P(y=1) with an S-shaped curve
  + $\beta_0$ shifts the curve right or left by $c = − \beta_0 / \beta_1$, whereas $\beta_1$ controls the steepness of the S-shaped curve.
  + change of the $beta_0$ value to move offset
  + change of the $beta_1$ value to distort gradient

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;margin: 0.5em;">
    <a href="https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*PD7DZlFWkYCxrg5Y1-OOXQ.png" style="margin: 0.1em;" alt="Diagram of logistic regression" title="Diagram of logistic regression" width=250>
      <img src="https://miro.medium.com/max/875/1*B0W_JthGRm6NFEvtD3ZxIA.png" style="margin: 0.1em;" alt="Diagram of logistic regression with beta_0" title="Diagram of logistic regression with beta_0=80" width=250>
      <img src="https://miro.medium.com/max/875/1*YpPeJSaOwD0Pv83KQgj3Iw.png" style="margin: 0.1em;" alt="Diagram of logistic regression with beta_1 = 1.0" title="Diagram of logistic regression with beta_1 = 1.0" width=250>
    </a>
  </div>


### Backpropagation for Logistic Neurons

+ [Idea Behind](../ML/MLNN-Hinton/03-Backpropagation.md#the-backpropagation-algorithm)
  + knowing what actions in the hidden units
  + efficiently computing error derivatives

+ Error derivatives w.r.t activities to get error derivatives w.r.t. the incoming weights on a sampling case

  \[E = \frac{1}{2} \sum_{j \in output} (t_j - y_j)^2 \quad \implies \quad \frac{\partial E}{\partial y_j} = - (t_j - y_j)\]

+ [Total error derivatives w.r.t. various factors](../ML/MLNN-Hinton/03-Backpropagation.md#the-backpropagation-algorithm)

  \[\begin{array}{rcl} \dfrac{\partial E}{\partial z_j} & = & \dfrac{dy_j}{dz_j} \dfrac{\partial E}{\partial y_j} = y_j(1- y_j)\dfrac{\partial E}{\partial y_j} \\\\ \dfrac{\partial E}{\partial y_j} &=& \displaystyle \sum_j \dfrac{dz_j}{dy_i} \dfrac{\partial E}{\partial z_j} = \sum_j w_{ij} \dfrac{\partial E}{\partial z_j} \\\\ \dfrac{\partial E}{\partial w_{ij}} &=& \dfrac{\partial z_j}{\partial w_{ij}} \dfrac{\partial E}{\partial z_j} = y_i \dfrac{\partial E}{\partial z_j} \end{array}\]

+ [Optimization of Logistic Regression](../ML/MLNN-Hinton/a01-IntroNN.md#the-motivation-for-neural-networks)
  + using a loss function in order to quantify the level of error that belongs to our current parameters
  + find the coefficients that minimize this loss function
  + the parameters of the neural network have a relationship with the error the net produces
  + gradient descent:
    + changing the parameters using an optimization algorithm
    + useful for finding the minimum of a function

  + the loss function or the objective function

    \]\mathcal{L}(\beta_0, \beta_1) = - \sum_i \left[ y_i \log(p_i) + ( 1- y_i) \log(1 - p_i)\right]\]



### The Softmax Function

+ [The architecture](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture4/lec4.pptx" ismap target="_blank">
        <img src="../ML/MLNN-Hinton/img/m04-13.png" style="margin: 0.1em;" alt="Representation of Softmax group" title="Representation of Softmax group" width=200>
      </a>
      <a href="https://www.ritchieng.com/machine-learning/deep-learning/neural-nets/" ismap target="_blank">
        <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-nanodegree/master/deep_learning/introduction/lr2.png" style="margin: 0.1em;" alt=" multinomial logistic regression or softmax logistic regression" title=" multinomial logistic regression or softmax logistic regression" width=300>
      </a>
    </div>

+ [Definition](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  A softmax group $G$ is a group of output neurons whose outputs use the softmax activation defined by

  \[y_i = \frac{e^{z_i}}{\displaystyle \sum_{j \in G} e^{z_j}}\]

  so that the outputs sum to 1. The cost function is given by

  \[C = - \sum_j t_j \ln(y_j)\]

+ [Proposition](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  By the Quotient Rule, the derivatives are

  \[\frac{\partial y_i}{\partial z_i} = \frac{\partial}{\partial z_i} \left(\frac{e^{z_i}}{\sum_{j \in G} e^{z_j}}\right) = y_i(1 - y_i) \qquad\qquad \frac{\partial y_i}{\partial z_j} = \frac{\partial}{\partial z_j} \frac{1}{2} (t_j - y_j)^2 = - y_i y_j\]

  or more fancy-like using the Kronecker Delta:

  \[\frac{\partial y_i}{\partial z_j} = y_i (\delta_{ij} - y_j)\]

+ [Proposition](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  The derivatives of the cost function are

  \[\frac{\partial C}{\partial z_i} = y_i - t_i.\]

+ [Cross-entropy](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  the suggested cost function to use with softmax

  \[C = - \sum_j t_j \ln(y_j) = -\ln(y_i)\]

  + $t_j$: target values
  + $t_j = \begin{cases} 1 & j \in I \subset G \\ 0 & j \in G-I \end{cases}$
  + $y_i$: the probability of the input belonging to class $I$
  + simply put 0 on the wrong answers and 1 for the right answer ($t_i$)
  + Cross-entropy cost function

+ [Property](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  $C$ w/ very big gradient descent if target value = 1 and actual value approx. 0.

+ better than the gradient descent w/ squared error


## Issues and Algorithms for Optimization

### Challenges with optimization

+ [Convex optimization](../ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + a function in which there is only one optimum, corresponding to the global optimum (maximum or minimum)
  + no concept of local optima for convex optimization problems, making them relatively easy to solve

+ [Non-convex optimization](/ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + a function which has multiple optima, only one of which is the global optima
  + Maybe very difficult to locate the global optima depending on the loss surface

+ [Neural network](../ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
  + loss surface: minimize the prediction error of the network
  + interested in finding the global minimum on this loss surface

+ [Multiple problems on neural network training](/ML/MLNN-Hinton/a03-Optimization.md#challenges-with-optimization)
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


### Local optima and Saddle Points

+ [Local optima](../ML/MLNN-Hinton/a03-Optimization.md#local-optima)
  + viewed as a major problem in neural network training
  + using insufficiently large neural networks, most local minima incur a low cost
  + not particularly important to find the true global minimum
  + a local minimum with reasonably low error is acceptable

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/1*fGx-IJkvPLuurfR9VWlu2g.png" style="margin: 0.1em;" alt="Curve of loss function with local and global minimum" title="Curve of loss function with local and global minimum" width=350>
    </a>
  </div>

+ [Saddle pints](../ML/MLNN-Hinton/a03-Optimization.md#saddle-points)
  + more likely than local minima in high dimensions
  + more problematic than local minima because close to a saddle point the gradient can be very small
  + gradient descent results in negligible updates to the network and network training will cease

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0" ismap target="_blank">
      <img src="https://miro.medium.com/max/875/0*GuknkQNZ8pQqDGsr.jpg" style="margin: 0.1em;" alt="Saddle point — simultaneously a local minimum and a local maximum." title="Saddle point — simultaneously a local minimum and a local maximum." width=350>
    </a>
  </div>

+ [Rosenbrook function](../ML/MLNN-Hinton/a03-Optimization.md#saddle-points)
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


## Applications

### Family Tree - Multiclass Learning

+ [Family tree](../ML/MLNN-Hinton/04-Multiclasses.md#learning-to-predict-the-next-word)
  + Q: Figuring out the regularities from given family trees
  + Block - local encoding of person 1: 24 people: 12 British & 12 Italian
  + Block - local encoding of relationship: 12 relationships
  + Block - Distributed encoding of person 1: 6 big gray boxes
  + Observe the patterns from the right diagram
    + top right unit (big grey block): nationality
    + 2nd right block: generation
    + left bottom block: branches of family tree
  + features: only useful if the other bottlenecks use similar representations
  + Generalization: able to complete those triples correctly?
    + trained with 108 triples instead of 112 triples
    + Validate on the 4 held-out cases
  + (A r B): A has a relationship r with B
    + predict 3rd term (B) from the first two terms (A & r)
    + using the trained net to find very unlikely triples

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture4/lec4.pptx" ismap target="_blank">
    <img src="../ML/MLNN-Hinton/img/m04-01.png" style="margin: 0.1em;" alt="Example of family trees" title="Example of family trees" height=150>
    <img src="../ML/MLNN-Hinton/img/m04-02.png" style="margin: 0.1em;" alt="The structure of neural network to search symbolic rules" title="The structure of neural network to search symbolic rules" height=150>
    <img src="../ML/MLNN-Hinton/img/m04-03.png" style="margin: 0.1em;" alt="The example to search symbolic rules" title="The example to search symbolic rules" height=150>
  </a>
</div>


### Speech Recognition

+ A basic problem in speech recognition
  + Not able to identify phonemes perfectly in noisy speech
  + Ambiguous acoustic input: several different words fitting the acoustic signal equally well
  + Human using their understanding of the meaning of the utterance to hear the right words
  + knowing which words are likely to come next and which are not in speech recognition

+ [The standard Trigram method](../ML/MLNN-Hinton/04-Multiclasses.md#neuro-probabilistic-language-models)
  + Gather a huge amount of text and count the frequencies of all triples or words
  + Use the formula to bet the relative probabilities of words with the two previous words

    \[\frac{p(w_3 = c | w_2 = b, w_1 = a)}{p(w_3 = d | w_2 =b, w_1 = a)} = \frac{\text{count}(abc)}{\text{count}(abd)}\]

  + The state-of-the-art methodology recently
  + drawback: not understand similarity btw words

+ [Bengio's neural net](../ML/MLNN-Hinton/04-Multiclasses.md#neuro-probabilistic-language-models)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m04-04.png" style="margin: 0.1em;" alt="Bengio's neural net for predicting the next word" title="Bengio's neural net for predicting the next word" width=350>
    </a>
  </div>

  + similar to family tree problem but larger scale
  + Typical 5 previous words used but shown 2 in the diagram
  + Using distributed representations via hidden layers to predict via huge sofmax to get probabilities for all various words might coming next
  + refinement:
    + skip layer connection to skip from input to output
    + input words individually informative about what the word might be
  + A problem w/ a very large vector of weights
    + unnecessary duplicates: plural of a word and tenses of verbs
    + each unit in the last hidden layer w/ 100,000 outgoing weights

+ [A serial architecture](../ML/MLNN-Hinton/04-Multiclasses.md#dealing-with-large-number-of-possible-outputs)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture4/lec4.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m04-05.png" style="margin: 0.1em;" alt="A serial architecture for speech recognition" title="A serial architecture for speech recognition" width=350>
    </a>
  </div>

  + adding an extra input as candidate for the next word same as the context word
  + output: score for how good the candidate in the context
  + execute the net many times but most of them only one required

+ [Structure words as a tree](../ML/MLNN-Hinton/04-Multiclasses.md#dealing-with-large-number-of-possible-outputs) (Minih and Hinton, 2009)
  + predicting a path through a binary tree
  + arranging all the words in a binary tree with words as the leaves
  + using the previous context to generate a __prediction vector__, $v$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m04-07.png" style="margin: 0.1em;" alt="Neural network architecture for speech recognition" title="Neural network architecture for speech recognition" height=150>
      <img src="../ML/MLNN-Hinton/img/m04-08.png" style="margin: 0.1em;" alt="The path for word searching with computed probabilities" title="The path for word searching with computed probabilities" height=150>
    </a>
  </div>

  + $\sigma$: the logistic function
  + using contexts to learn a prediction vector with the neural net
  + the prediction vector compared with the vectors learned for all the nodes on the path to the correct next word
  + take the path with high sum of their log probabilities: take the higher probability on each node

  + A convenient decomposition
    + maximizing the log probability of picking the target word: $\mathcal{O}(\log(N))$
    + Still slow at test time though a few hundred times faster


### A Unified Architecture for Natural Language Processing

+ Collobert and Weston, [A unified architecture for natural language processing: deep neural networks with multitask learning](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf), ICML'08, 2008
4-Multiclasses.md#dealing-with-large-number-of-possible-outputs)
  + learned feature vectors for words
  + applied to many different natural language processing tasks well
  + not try to predict the next word but good feature vectors for words
  + use both the past and future contexts
  + observe a window with 11 words, 5 in the past and 5 in the future
  + the middle word either the correct word actually occurred in the text or a random word
  + train the neural net to produce the output
    + high probability: correct word
    + low probability: random word
  + map the individual words to feature vectors
  + use the feature vectors in the neural net (possible many hidden layers) to predict whether the word correct or not

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture4/lec4.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m04-09.png" style="margin: 0.1em;" alt="Neural network architecture for feature vectors learning (Collobert & Weston, 2008)" title="Neural network architecture for feature vectors learning (Collobert & Weston, 2008)" height=150>
    </a>
  </div>

+ [2D map to display the learned feature vectors](../ML/MLNN-Hinton/04-Multiclasses.md#dealing-with-large-number-of-possible-outputs)
  + get idea of the quality of the learned feature vectors
  + display similar vectors close to each other
  + T-SNE: a multi-scale method to display similarity at different scale

+ [Checking strings of words](../ML/MLNN-Hinton/04-Multiclasses.md#dealing-with-large-number-of-possible-outputs)
  + learned feature vectors capturing lots of subtle semantic distinctions
  + no extra supervision required
  + information of all words in the context

