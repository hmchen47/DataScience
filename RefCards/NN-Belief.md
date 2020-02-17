# Belief/Bayesian Networks

## Overview

+ [Incorporating probability into AI](../ML/MLNN-Hinton/13-BeliefNets.md#132-belief-networks)
  + combination of graph theory and probability theory
  + AI works in the 1980's
    + using bags of rules for tasks such as medical diagnosis and exploration for minerals
    + dealing w/ uncertainty for practical problems
    + made up ways of doing uncertainty that did not involve probabilities $\to$ bad bet
  + Graphical models
    + Pearl, Heckeman, Lauritzen, and others shown that probabilities worked better than the ad-hoc methods of expert systems
    + discrete graph good for representing what depended on what other variables
    + computing for nodes of the graph, given the states of other nodes
  + Belief nets:
    + a particular subset of graph
    + sparsely connected, directly acyclic graphs
    + clever inference algorithms to compute the probabilities of unobserved node efficiently for sparsely connected graph
    + working exponentially in the number of nodes that influence each node $\implies$ not for densely connected networks

+ [Belief Networks](../ML/MLNN-Hinton/13-BeliefNets.md#132-belief-networks)
  + a directed acyclic graph composed of stochastic variables (see diagram)
  + observe some of the variables
  + Problems to solve:
    + __the inference problem__: infer the states of the unobserved variables
    + __the learning problem__: adjust the interactions btw variables to make the network more likely to generate the training data

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture13/lec13.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m13-01.png" style="margin: 0.1em;" alt="Architecture of Belief nets" title="Architecture of Belief nets" width=350>
    </a>
  </div>

+ [Graphical models vs. Neural networks](../ML/MLNN-Hinton/13-BeliefNets.md#132-belief-networks)
  + early graphical models
    + using experts to define the graph structure and the conditional probabilities; example: medical experts
    + sparsely connected
    + initially focused on doing correct inference, not on learning
  + Neural networks
    + main task: learning
    + hand-wiring knowledge $\to$ not cool: wiring in some basic properties as in convolutional nets was a very sensible thing to do
    + knowledge from learning the training data not from experts
    + not aiming for interpretability or sparse connectivity

+ [Types of generative neural network composed of stochastic binary neurons](../ML/MLNN-Hinton/13-BeliefNets.md#132-belief-networks)
  + Energy-based models
    + connected binary stochastic neurons using symmetric connections to get a Boltzmann Machine
    + ways to learn a Boltzmann machine but difficult $\implies$ restrict the connectivity in a special way (RBM)
  + Causal models
    + connecting binary stochastic neurons in a directed acyclic graph
    + Sigmoid Belief Networks (Neal 1992) $\to$ easier to learn than BM
    + causal sequence from layer to layer to get unbiased sample of the kinds of vectors of visible values that the NN believes in
  + causal model easier to generate data than Boltzmann machine

## Sigmoid Belief Networks

+ [Sigmoid Belief Networks](../ML/MLNN-Hinton/13-BeliefNets.md#133-learning-sigmoid-belief-nets)
  + only one positive phase required
  + locally normalized models, no partition function or derivatives required
  + easily to generate an unbiased example at the leaf nodes
  + following the gradient specified by maximum likelihood learning in a mini-batch stochastic kinds of ways $\to$ understanding what kinds of data the network believes in
  + hard to infer the posterior distribution over all possible configurations of hidden causes when observing the visible effects
  + hard to even get a sample from the posterior $\gets$ stochastic gradient descent required
  + how to learn sigmoid belief networks w/ millions of parameters?
    + very different regime from the normally used graphical models
    + graphical models: interpretable models and trying to learn dozens or hundreds of parameters


## Explaining Away Effects

+ [Explaining away](../ML/MLNN-Hinton/13-BeliefNets.md#133-learning-sigmoid-belief-nets)
  + two independent hidden causes in the prior
  + becoming dependent when observing an effect that they can both influence
  + example: earthquake and truck (see diagram)

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture13/lec13.pdf" ismap target="_blank">
        <img src="../ML/MLNN-Hinton/img/m13-03.png" style="margin: 0.1em;" alt="Example of explaining away" title="Example of explaining away" width=300>
      </a>
    </div>

## Factorial Distribution

+ [Factorial distributions](../ML/MLNN-Hinton/13-BeliefNets.md#134-the-wake-sleep-algorithm)
  + defintion: the probability of a whole vector is just the product of the probabilities of its individual terms
  + example:
    + individual probabilities of three hidden units in a layer: 0.3, 0.6. 0.8
    + probability of the hidden units w/ states $(1, 0, 1)$ if the distribution is factorial: $p(1, 0, 1) = 0.3 \times (1 -0.6) \times 0.8$
  + degrees of freedom
    + a general distribution over binary vectors of length $N: (2^N - 1)$ degrees of freedom
    + factorial distribution: only $N$ degrees of freedom


## Learning for Belief Networks

+ [Learning rule](../ML/MLNN-Hinton/13-BeliefNets.md#133-learning-sigmoid-belief-nets)
  + learning is easy $\gets$ getting an unbiased sample from the posterior distribution over hidden states given the observed data

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture13/lec13.pdf" ismap target="_blank">
        <img src="../ML/MLNN-Hinton/img/m13-02.png" style="margin: 0.1em;" alt="Architecture of Sigmoid Belief nets" title="Architecture of Sigmoid Belief nets" width=250>
      </a>
    </div>
  
  + maximizing the log probability for each unit
    + $p_i$: the probability to turn on node $i$ involves the binary states of the parents
    + $\Delta w_{ji}$: the maximum likelihood learning rule

    \[\begin{align*}
      p_i \equiv p(s_i = 1) &= \frac{1}{1 + \exp \left( -b_i - \sum_j s_j w_{ji} \right)} \\
      \Delta w_{ji} &= \varepsilon \, s_j (s_i - p_i)
    \end{align*}\]

  + given an assignment of binary states to all the hidden nodes $\implies$ easily to do maximum likelihood learning in typical stochastic way

+ [Issue for learning](../ML/MLNN-Hinton/13-BeliefNets.md#133-learning-sigmoid-belief-nets)
  + multiple layers of hidden variables to give rise to some data in the causal model (see diagram)
  + hard to learn sigmoid belief nets one layer at a time
  + learning $W$: reqiring sample from the posterio distribution in the first hidden layer
  + problem 1: the posterior not factorial because of "explaining away"
  + problem 2: posterior depending on the prior as well as the likelihood
  + problem 3: required to integrate over all possible configurations in the higher layers to get the prior for the first hidden layer $\to$ hopeless

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture13/lec13.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m13-04.png" style="margin: 0.1em;" alt="Architecure of Belief nets" title="Architecure of Belief nets" width=150>
    </a>
  </div>

+ [Learning methods](../ML/MLNN-Hinton/13-BeliefNets.md#133-learning-sigmoid-belief-nets)
  + Monte Carlo methods
  + Variational mehods: only get approximated samples from the posterior
  + Learning from wrong distribution: maximum likelihood learning requiring unbiased samples from the posterior
  + sampling from wrong distribution + the maximum likelihood learning rule:
    + no guarantee on improvement
    + guaranteed to improve:
      + the log probability related to the generation of data
      + providing a lower bound on that mode probability
      + pushing lower bound to push up log probability


## The Wake-Sleep Algorithm

+ [Crazy idea](../ML/MLNN-Hinton/13-BeliefNets.md#134-the-wake-sleep-algorithm)
  + problem: hard to infer to the posterior distribution over hidden configurations when given a data vector
  + Crazy idea: doing the inference wrong
    + observe what's driving the weights during the learning when using an approximate posterior
    + two terms driving the weights
      + driving weights to get a better model of the data that makes the sigmoid belief net more likely to generate the observed data in the training set
      + driving weights towards sets of weights for which the proximate posterior used is a good fit to the real posterior by manipulating the real posterior to make it for the approximate posterior
  + each hidden layer
    + using distribution that ignores explaining away
    + assumption (__wrongly__): the posterior over hidden configurations factorizes into a product of distributions for each separate hidden unit

+ [The wake-sleep algorithm](../ML/MLNN-Hinton/13-BeliefNets.md#134-the-wake-sleep-algorithm)
  + lead new area of machine learning $\to$ variational learning in the late 1990s for learing complicated graphical models
  + used for directly graphical models like sigmoid belief nets
  + make uses of the idea of using the wrong distribution
  + architecture: (see diagram)
    + a neural network w/ two different sets of weights
    + a generative model
      + generative weights: weight defining the probability distribution over data vectors
      + recognition weights: using the weights to get factorial distribution in each hidden layer
  + Wake phase
    + use recognition weights to perform a bottom-up pass
    + train the generative weights to reconstruct activities in each layer from the layer above
  + Sleep phase
    + use generative weights to generate samples from the model
    + train the recognition weights to reconstruct activities in each layer from the layer below

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture13/lec13.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m13-05.png" style="margin: 0.1em;" alt="Architecure of the wake-sleep algorithm" title="Architecure of the wake-sleep algorithm" width=150>
    </a>
  </div>

+ [Limitations of the wake-sleep algorithm](../ML/MLNN-Hinton/13-BeliefNets.md#134-the-wake-sleep-algorithm)
  + the recognition weights
    + initial phase: weights not very good at the beginning of learing $\to$ waste not not big issue
    + progressive phase: only approximately follow the gradient of the variational bound on the probability $\to$ incorrect mode-averaging
  + explaning away effects $\implies$ the posterior over the top hidden layer is very far from independent
  + Karl Friston: similar to how the brian works

+ [Mode averaging](../ML/MLNN-Hinton/13-BeliefNets.md#134-the-wake-sleep-algorithm)
  + learning recognition weights (left diagram)
    + learning to produce $(0.5, 0.5)$ for the recognition weights
    + representing a distribution that put half its mass on $(1, 0)$ or $(0, 1)$: very improbabe hidden configurations
  + much better picking one mode (right diagram)
    + the best solution to pick one of its states to give it all the probability mass (green curve)
    + variational learning manipulating the true posterior to make it fit the approximation
    + normal learing manipulating an approximation to fit the true posterior

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture13/lec13.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m13-06.png" style="margin: 0.1em;" alt="Example and distribution of the mode averaging" title="Example and distribution of the mode averaging" width=450>
    </a>
  </div>



