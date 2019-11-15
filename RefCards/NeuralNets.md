# Neural Networks

## Modeling

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
      <img src="https://blog.zaletskyy.com/Media/Default/NeuralNetworks/binaryNeuron.png" style="margin: 0.1em;" alt="Binary threshold neuron" title="Binary threshold neuron" height=150>
    </a>
    <a href="https://www.bo-song.com/coursera-neural-networks-for-machine-learning/" ismap target="_blank">
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-2.png" style="margin: 0.1em;" alt="Rectified Linear Neurons" title="Rectified Linear Neurons  (ReLU)" height=150>
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-5.png" style="margin: 0.1em;" alt="Sigmoid neurons" title="Sigmoid neurons" height=150>
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-6.png" style="margin: 0.1em;" alt="Stochastic binary neurons" title="Stochastic binary neurons" height=150>
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

+ Perceptron convergence procedure:
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


### Backpropagation for Logistic Neurons

+ [Idea Behind](../ML/MLNN-Hinton/03-Backpropagation.md#the-backpropagation-algorithm)
  + knowing what actions in the hidden units
  + efficiently computing error derivatives

+ Error derivatives w.r.t activities to get error derivatives w.r.t. the incoming weights on a sampling case

  \[E = \frac{1}{2} \sum_{j \in output} (t_j - y_j)^2 \quad \implies \quad \frac{\partial E}{\partial y_j} = - (t_j - y_j)\]

+ [Total error derivatives w.r.t. various factors](../ML/MLNN-Hinton/03-Backpropagation.md#the-backpropagation-algorithm)

  \[\begin{array}{rcl} \dfrac{\partial E}{\partial z_j} & = & \dfrac{dy_j}{dz_j} \dfrac{\partial E}{\partial y_j} = y_j(1- y_j)\dfrac{\partial E}{\partial y_j} \\\\ \dfrac{\partial E}{\partial y_j} &=& \displaystyle \sum_j \dfrac{dz_j}{dy_i} \dfrac{\partial E}{\partial z_j} = \sum_j w_{ij} \dfrac{\partial E}{\partial z_j} \\\\ \dfrac{\partial E}{\partial w_{ij}} &=& \dfrac{\partial z_j}{\partial w_{ij}} \dfrac{\partial E}{\partial z_j} = y_i \dfrac{\partial E}{\partial z_j} \end{array}\]


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

  $$y_i = \frac{e^{z_i}}{\displaystyle \sum_{j \in G} e^{z_j}}$$

  so that the outputs sum to 1. The cost function is given by

  $$C = - \sum_j t_j \ln(y_j)$$

+ [Proposition](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  By the Quotient Rule, the derivatives are

  $$\frac{\partial y_i}{\partial z_i} = \frac{\partial}{\partial z_i} \left(\frac{e^{z_i}}{\sum_{j \in G} e^{z_j}}\right) = y_i(1 - y_i) \qquad\qquad \frac{\partial y_i}{\partial z_j} = \frac{\partial}{\partial z_j} \frac{1}{2} (t_j - y_j)^2 = - y_i y_j$$

  or more fancy-like using the Kronecker Delta:

  $$\frac{\partial y_i}{\partial z_j} = y_i (\delta_{ij} - y_j)$$

+ [Proposition](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  The derivatives of the cost function are

  $$\frac{\partial C}{\partial z_i} = y_i - t_i.$$

+ [Cross-entropy](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  the suggested cost function to use with softmax

  $$C = - \sum_j t_j \ln(y_j) = -\ln(y_i)$$

  + $t_j$: target values
  + $t_j = \begin{cases} 1 & j \in I \subset G \\ 0 & j \in G-I \end{cases}$
  + $y_i$: the probability of the input belonging to class $I$
  + simply put 0 on the wrong answers and 1 for the right answer ($t_i$)
  + Cross-entropy cost function

+ [Property](../ML/MLNN-Hinton/04-Multiclasses.md#another-diversion-the-softmax-output-function)

  $C$ w/ very big gradient descent if target value = 1 and actual value approx. 0.

+ better than the gradient descent w/ squared error


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

    $$\frac{p(w_3 = c | w_2 = b, w_1 = a)}{p(w_3 = d | w_2 =b, w_1 = a)} = \frac{\text{count}(abc)}{\text{count}(abd)}$$

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

## Convolutional Neural Network

### Object Recognition and Classification

+ [Issues about object recognition](../ML/MLNN-Hinton/05-CNN.md#lecture-notes)
  + Segmentation: real scenes cluttered with other objects
  + Lighting: intensities of pixels determined by the nature of the object
  + Deformation: deforming objects in various non-affine ways
  + Affordances: the quality or property of an object that defines its possible uses or makes clear how it can or should be used
  + Viewpoint: 3-D object w/ variety of viewpoints

+ [Dimension-hopping phenomenon](../ML/MLNN-Hinton/05-CNN.md#lecture-notes):
  + info jumping from one input dimension to another
  + e.g., age and weight of a patient are inputs


### Solutions for Viewpoint Invariance

+ [Viewpoint invariance](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-1)
  + one of the main difficulties in making computers perceive
  + still no accepted solutions
  + Approaches
    + redundant invariant features
    + a box around the object w/ normalized pixels
    + convolutional neural networks by replicating features with pooling
    + hierarchy of parts that have explicit poses relative to the camera

+ Redundant invariant feature approach
  + extract a large, redundant set of invariant features under transformations
  + with enough invariant features, only one way to assemble them into an object
  + avoid forming features from parts of different objects

+ Judicious normalization approach
  + putting a box around the object
  + using the box as a coordinate frame for a set of normalized pixels
  + solving the dimension-hopping problem
    + correctly choosing the box results in the same normalized pixels for the same part of an object
    + any box providing invariant to many degrees of freedom: translation, rotation, scale, shear, stretch, ...
  + Issues
    + segmentation errors
    + occlusion
    + unusual orientations
  + chicken-egg problem: getting the box right $\leftrightarrow$ recognizing the shape
  + Human recognizes the letter before doing mental rotation to decide if it's a mirror image

+ Brute force normalization approach
  + Using well-segmented, upright images to fit the correct box for training
  + Designing period - try all possible boxes in a range of positions and scales




