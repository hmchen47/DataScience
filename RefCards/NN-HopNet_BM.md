# Hopfield Networks & Boltzmann Machine

## Hopfield Networks

### Overview for Hopfield Networks

+ [Hopfield Networks](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + energy-based model: properties derive from a global energy function
  + combposed of binary threshold units w/ recurrent connections between them
  + recurrent network of non-linear units
    + generally very hard to analyze
    + behave in many different ways
      + settle to a stable state
      + oscillate
      + follow chaotic trajectories that cannot be predicted far into the future unless knowing the starting state w/ infinite precision
  + John Hopfield's proposal
    + existing a global energy function w/ __symmetric__ connections
    + each binary "configuration" of the whole network w/ an energy
    + binary threshold decision rule causing the network to settle to a minimum of the energy function

+ [Hopfield combines two ideas](../ML/MLNN-Hinton/11-Hopfield.md#113-hopfields-nets-with-hidden-units)
  + find local energy minimum by using as network of symmetrically connected binary threshold units
  + local energy minimum might correspond to memories


### Energy Function

+ [The energy function](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + global energy: the sum of many local contributions
  + the main contributions: the form of the product of one connection weight w/ the binary of two neurons

    \[ E = - \sum_i s_i \cdot b_i - \sum_{i < j} s_i s_j \cdot w_{ij} \]

    + energy is bad $\implies$ low energy is good $\implies$ minus sign (-) for the equation
    + $s_i$: binary variable w/ values of $1$ or $0$ or in another kind of Hopfield net w/ values of $1$ or $-1$
    + $w_{ij}$: weight for the symmetric connection strength btw two neurons
    + $s_i b_i$: bias term involves the state of individual units
    + $s_i s_j$: the activities of the two connected neurons

  + simple _quadratic_ energy function makes it possible for each unit to compute locally how it's state affects the global energy:

    \[ \text{Energy gap} = \Delta E_i = E(s_i = 0) - E(s_i = 1) = b_i + \sum_j s_j \cdot w_{ij} \]

+ [Settling to an energy minimum](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + finding the minimum energy
    + start from a random state
    + sequential update: update units _one at a time_ in random order
  + commputing the goodness
    + all pairs of units w/ on and add in the weight between them
    + minimum energy


### Memories

+ [Sequential decisions](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + if units make __simultaneous__ decisions the energy could go up
  + simultaneous parallel updating $\implies$ getting oscillations (always w/ period 2)
  + the updates occur in parallel but w/ random timing $\implies$ the oscillations usually destroyed

+ [Neat way to compute sequential decisions](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + Hopfield proposal
    + memories could be energy minima of a neural net w/ symmetric weights
    + binary threshold decision rule used to "clean up" incomplete or corrupted memories
  + Principles of Literary Criticism
    + idea of memories as energy minima
    + memories are like a large crystal that can sit on different phases
  + energy minima
    + represent memories w/ a content-addressable memory
    + access an item by just knowing part of its content
    + biological property: robust against hardware damage
    + psychological point of view: like reconstructing a dinosaur from a few bones

+ [Storing memories in a Hopefield net](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + with activities of $1$ and $-1$
    + stored as a binary state vector by incrementing the weight btw any two units by the product of their activities
    + treating biases as weights from a permanently on unit
    + very simple rule: not error-driven
    + both its strength and weakness
      + not error correction rule
      + able to be online but not a very efficient way to store things

    \[ \Delta w_{ij} = s_i \cdot s_j \]

  + with state of $0$ and $1$, the rule is slightly more complicated

    \[ \Delta w_{ij} = 4 (s_i - \frac{1}{2}) (s_j - \frac{1}{2}) \]


### Spurious Minima & Capacity

+ [Storage capacity](../ML/MLNN-Hinton/11-Hopfield.md#112-dealing-with-spurious-minima-in-hopfield-nets)
  + Hopfield nets w/ memory storage devices
  + obsessed by the storage capacity of the Hopfield net
  + the capacity of a totally connected net w/ $N$ units by Hopfield storage rule
    + only about $0.15N$ memories
    + only $0.15N^2$ bits at $N$ bits per memory
    + not making efficient use of the bits required to store the weights
  + the net w/ $N^2$ weights and biases
  + after storing $M$ memories, the integer value range of each connection weight is $[-M, M]$
  + number of bits to store the weights and biases:

    \[ N^2 \log(2M+1) \]

+ [Spurious minima limit capacity](../ML/MLNN-Hinton/11-Hopfield.md#112-dealing-with-spurious-minima-in-hopfield-nets)
  + create a new energy minimum for each configuration memorizing (top fig)
  + limitation of the capacity of Hopfield net (bottom fig)
    + two nearby patterns $\implies$ unable to separate two minimum
    + unable to distinguish these two separate memories

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
        <img src="../ML/MLNN-Hinton/img/m11-06.png" style="margin: 0.1em;" alt="State space for spurious minima" title="State space for spurious minima" width=200>
      </a>
    </div>

+ [Avoiding spurious minima](../ML/MLNN-Hinton/11-Hopfield.md#112-dealing-with-spurious-minima-in-hopfield-nets)
  + unlearing
    + strategy
      + let the net settle from a random initial state and then do __unlearning__
      + whatever binary state it settles, apply the opposite of the storage rule
      + get rid of deep, spurious minima and increase memory capacity
    + shown that the strategy works but no good analysis to explain
  + analogy w/ Reverse Learning
    + proposed by F. Crick and G. Mitchison
    + a model of what dreams are for
    + that's what why you don't remember them
  + how much unlearning should apply?
    + unlearning as part of the process of fitting a model to data
    + apply maximum likelihood fitting of the model
    + unlearning automatically come out of fitting the model
  + derive unlearning as the right way to minimize some cost function

+ [Increasing the cpacity](../ML/MLNN-Hinton/11-Hopfield.md#112-dealing-with-spurious-minima-in-hopfield-nets)
  + physics:
    + the math already known might explain how the brain works
    + many papers published in physics journals
    + related to Hopfield nets and their storage capacity
  + Elizabeth Gardiner proposal
    + using the full capacity of the weights
    + a much better storage rule
  + updating the memories
    + cycle through the training set many times than storing vectors in on shot
    + using the perceptron convergence procedure
  + statistics: pseudo-likelihood
    + get one thing right given all the other things
    + w/ high dimensional data, trying to get the value on 1-dim right given the values on all the other dimensions
    + main difference btw perceptron convergence procedure
  + the way to use the full capacity of Hopfield net
    + to use the perceptron convergence procedure
    + go through the data several times


### Issues with Hidden Units

+ [Introduction](../ML/MLNN-Hinton/11-Hopfield.md#113-hopfields-nets-with-hidden-units)
  + the weights btw units represent constraints on good interpretations
  + finding a low energy state $\to$ finding a good interpretation of the input data

+ [Different computational role](../ML/MLNN-Hinton/11-Hopfield.md#113-hopfields-nets-with-hidden-units)
  + using Hopfield net to construct interpretations of sensory input instead of using net to store memories
  + Notations:
    + input: visible units
    + interpretation: the states of the hidden units
    + badness of the interpretation: the energy
  + construct an interpretation of input in the set of hidden units
  + the interpretation or explanation of the input as a binary configuration over the hidden units
  + the energy of the whole system represents the badness of that interpretation
  + to get good interpretations according to current model is the energy function
  + to find the low energy states of the hidden units given by the input represented by the visible units

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m11-07.png" style="margin: 0.1em;" alt="Hopfield net about hidden and visible units" title="Hopfield net about hidden and visible units" width=150>
    </a>
  </div>

+ [Infer about 3-D edges from 2-D lines](../ML/MLNN-Hinton/11-Hopfield.md#113-hopfields-nets-with-hidden-units)
  + 2-D line in an image might be caused by many different 3-D edges
  + a straight 3-D edge
  + pictorial explanation of 2-D and 3-D mapping (left diagram)
  + find low energy states of a network binary unit to find interpretations of sensory input
  + example: Interpreting a line drawing to 3-D image (right diagram)
    + 2-D line: using one "2-D line" for each possible line in the picture
    + 3-D line: using one "3-D line" for each for each possible 3-D line in the scene
    + make 3-D lines support each other if they join in 3-D: join the two 3-D edge units w/ same or different depth
    + make 3-D lines __strongly support__ each other if they join at right angles
      + <span style="color: green; weight: bold">thicker green line</span>: join two 3-D edge unit w/ same depth and right angle

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m11-08.png" style="margin: 0.1em;" alt="Image of 2-D lines to represent 3-D edges" title="Image of 2-D lines to represent 3-D edges" height=80>
      <img src="../ML/MLNN-Hinton/img/m11-09.png" style="margin: 0.1em;" alt="Interpretation of 3-D edges with 2-D lines" title="Interpretation of 3-D edges with 2-D lines" height=200>
    </a>
  </div>

+ [Two difficult computational issues](../ML/MLNN-Hinton/11-Hopfield.md#113-hopfields-nets-with-hidden-units)
  + using the states of the hidden units to represent an interpretation of input raises two difficult issues
  + <span style="color: blue;">Search</span>: how do we avoid getting trapped in poor local minima of the energy function?
  + <span style="color: blue;">Learning</span>: how do we learn the weights on the connections to the hidden units and btw the hidden units?


### Searching w/ Noise

+ [Noisy networks](../ML/MLNN-Hinton/11-Hopfield.md#114-using-stochastic-units-to-improve-search)
  + Hopfield always making decisions to reduce the energy $\implies$ impossible to escape from local minima
  + using random noise to escape from poor minima
    + start w/ a lot of noise to cross energy barriers
    + simulated annealing: slowly reduce the noise so that the systems ends up in a deep minimum

+ [Temperature in physical system](../ML/MLNN-Hinton/11-Hopfield.md#114-using-stochastic-units-to-improve-search)
  + energy function in simulated system
  + high temperature transition probabilities (top diagram)
  + low temperature transition probabilities (bottom diagram)
  + better solution: start w/ high temperature and then gradually reduce the temperature

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m11-12.png" style="margin: 0.1em;" alt="Transition probabilities" title="Transition probabilities" width=250>
    </a>
  </div>

+ [Stochastic binary units](../ML/MLNN-Hinton/11-Hopfield.md#114-using-stochastic-units-to-improve-search)
  + biased random decisions w/ binary stochastic units
    + temperature controls the amount of noise
    + raising the noise level equivalent to decreasing all the energy gaps btw configuration
  + normal logistic equation

    \[ p(s_i = 1) = \frac{1}{1 + \exp(-\Delta E_i / T)} \]

    + $T$: temperature
    + energy gap scaled by a temperature
    + $T \to \infty, \exp(-\Delta E_i / T) \approx 0 \to p(s_i=1) = 1/(1+1) \implies$ knees on and off states
    + $T \to 0, \exp(-\Delta E_i/T) \approx -\infty \text{ or } \infty \text{ (depending on value of } \Delta E_i) \to p(s_i =1) = 0 \text{ or } 1 \implies$ firmly off or on $\implies$ behave deterministically and a binary threshold unit
  + energy gap

    \[ \text{Energy gap} = \Delta E_i = E(s_i = 0) - E(s_i = 1) = b_i + \sum_j s_j \cdot w_{ij} \]


### Simulated Annealing

+ [Simulated annealing](../ML/MLNN-Hinton/11-Hopfield.md#114-using-stochastic-units-to-improve-search)
  + a powerful method for improving searches that get stuck in local optima
  + one of the idea leading to Boltzmann machines
  + a big distraction from the main ideas behind Boltzmann machines
  + using binary stochastic units having a temperature of 1 for standard logistic function in the energy gap

+ [Thermal equilibrium](../ML/MLNN-Hinton/11-Hopfield.md#114-using-stochastic-units-to-improve-search)
  setting temperature as 1 $\implies$ fixed temperature
  + a difficult concept
    + reaching thermal equilibrium not meant that the system has settled down into the lowest energy configuration
    + state units still rattling around at thermal equilibrium unless the temperature  is zero
    + settle down means the <span style="color: red;">probability distribution</span> over configurations
    + settle to the stationary distribution determined by the energy function of the system
    + stationary distribution: the probability of any configuration proportional to $\exp(-E)$
  + intuitive view
    + imagine a huge ensemble of systems w/ exactly same energy function
    + the probability of a configuration: just the fraction of the systems w/ that configuration

+ [Approaching thermal equilibrium](../ML/MLNN-Hinton/11-Hopfield.md#114-using-stochastic-units-to-improve-search)
  + start w/ any distribution for all the identical systems
  + then keep applying stochastic update rule to pick the next configuration for each individual system
  + running the systems stochastically in the right way
  + eventually reach a situation where the fraction of systems in each configuration remains constant
    + thermal equilibrium: the stationary distribution in physics
    + any given system keeps changing its configuration


## Boltzmann Machines

### Overview of Boltzmann Machine

+ [Boltzmann machine](../ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data)
  + a stochastic Hopfield networks w/ hidden units
  + good at modeling binary data

+ [Modeling binary data]/ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data
  + building a model: given a training set of binary vectors
  + useful for deciding if other binary vectors come from the same distribution
  + used for monitoring complex systems to detect unusual behavior
  + models of several different distributions $\implies$ used to computer the posterior probability that a paticular distribution produced the observed data by using Bayes theorem

    \[ p(Model \; i | data) = \frac{p(data | Model \; i)}{\displaystyle \sum_j p(data | Model \; j)} \]


### Causal Generative Model

+ [Causal generative model](../ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data)
  + generate the states of some latent variables
  + using latent variables to generate binary vector
  + procedures
    + pick the hidden states from their prior distribution (hidden layer)
    + pick the visible states from their conditional distribution given by the hidden states (visible layer)
  + probability of generating a visible vector, $\mathbf{v}$
    + the probability of generating that hidden state times the probability of generating $\mathbf{v}$ given that already generated that hidden state
    + a causal model factor analysis; e.g., a causal model using continuous variables
    + probably the most natural way to generate data
    + generative model: a causal model alike
    + each hidden state is an "explanation" of $\mathbf{v}$

    \[ p(\mathbf{v}) = \sum_{\mathbf{h}} p(\mathbf{h}) p(\mathbf{v}|\mathbf{h}) \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m11-13.png" style="margin: 0.1em;" alt="Causal model for generating data" title="Causal model for generating data" width=250>
    </a>
  </div>


### Modeling for Boltzmann Machine

+ [Generating data in Boltzmann machine](../ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data)
  + not a causal generative model but an energy based model
  + everything defined in terms of the energies of joint configurations of the visible and hidden units
  + energies of join configuration related to their probabilities in two ways
    + simply define the probability to be $p(\mathbf{v}, \mathbf{h}) \propto \exp(-E(\mathbf{v}, \mathbf{h}))$
    + define the probability to be the probability of finding the network in that joint configuration after updating all of the stochastic binary units many times
    + two definitions agree
  + The energy of a joint configuration

    \[ -E(\mathbf{v}, \mathbf{h}) = \underbrace{\sum_{i \in vis} v_i b_i + \sum_{k \in hid} h_k b_k}_{\text{bias terms}} + \sum_{i < k} v_i v_j \cdot w_{ij} + \sum_{i, k} v_i h_k \cdot w_{ik} + \sum_{k < l} h_k h_l \cdot w_{kl} \]

    + $E(\mathbf{v}, \mathbf{h})$: energy w/ configuration $\mathbf{v}$ on the visible units and $\mathbf{h}$ on the hidden units
    + $v_i$: binary state of unit $i$ in $\mathbf{v}$
    + $b_k$: bias of unit $k$
    + $w_{ik}$: weight between visible unit $i$ and hidden unit $k$
    + $i < k$: indexes every non-identical pair of $i$ and $j$ once

+ [Using energies to define probabilities](../ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data)
  + the probability of a joint configuration over both visible and hidden units
    + the energy of that joint configuration compared w/ the energy of all other joint configurations

    \[ p(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{\displaystyle \sum_{\mathbf{u}, \mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}} \]

    + $\sum_{\mathbf{u}, \mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}$: normalized term or partition function in physics
  + the probability of a configuration of the visible units

    \[ p(\mathbf{v}) = \frac{\sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}}{\sum_{\mathbf{u}, \mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}} \]

+ [Getting a sample from the model](../ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data)
  + the normalizing term (the partition function): growing exponentially as hidden units increase
  + Markov Chain Monte Carlo (MCMC)
    + get samples from the model starting from a random global configuration
    + keep picking units randomly and allowing them to stochastically update their states based on their energy gaps
    + energy gaps deptermined by the states of all the other units in the network
  + run the Markov chain
    + reaching the stationary distribution (thermal equilibrium at a temperature of 1)
    + the probability of a global configuration related to its energy by the Boltzmann distribution

    \[ p(\mathbf{v}, \mathbf{h}) \propto e^{-E(\mathbf{v}, \mathbf{h})} \]

+ [Sample from the posterior distribution](../ML/MLNN-Hinton/11-Hopfield.md#115-how-a-boltzmann-machine-models-data)
  + get sample from the posterior distribution over hidden configurations for a given data vector
  + the number of possible hidden configurations is exponential
  + purpose:
    + getting samples from the posterior given a data vector required for learning the weights
    + knowing a good explanation for the observed data
    + better explanations having lower energy


### Learning for BM

+ [Boltzmann machine learning algorithm](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + an unsupervised learning algorithm
  + no backpropagation required

+ [Goal of learning](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + maxizing the product of the probabilities that the Boltzmann machine assigns to a set of binary vectors in the training set
  + equivalent to maxizing the sum of the log probabilities that the Boltzmann machine assigns to the training vectors
  + equivalent to maximizing the probabilities to obtain exactly the $N$ training cases if
    + settle the network to its stationary $N$ different times w/o external input
    + sample the visible vector once each time
    + repeat the process

+ [Issue for learning](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + consider a chain of hidden units w/ visible units at the ends (see diagram)
  + Goal: the training set consisting of $(1, 0)$ and $(0, 1)$; i.e., two visible units to be in opposite states
  + Solution: by making the product of all the weights to be negative

+ [Surprising fact](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + one weight needs to know about other weights in order to be able to change even in the right direction
  + the learning algorithm only requires local information
  + everything that one weight needs to know about the other weights: the data contained in the difference of two correlations
  + the probability of Boltzmann machine assigns to a visible vector $\mathbf{v}$

    \[ \frac{\partial \log p(\mathbf{v})}{\partial w_{ij}} = \langle s_i s_j \rangle_{\mathbf{v}} - \langle s_i s_j \rangle_{model} \]

  + changing weight

    \[ \Delta w_{ij} \propto \langle s_i s_j \rangle_{data} - \langle s_i s_j \rangle_{model} \]

    + $\langle s_i s_j \rangle_{data}$: changing in the weight proportional to the expected product of the activities averaged over all visible vectors in the training set (data)
    + $\langle s_i s_j \rangle_{model}$: the product of the same two activities when not clamped anything and the network reaches thermal equilibrium w/o external interference


### Phases of Energy Function for Learning

+ [Simple derivative](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + the probability of a global configuration <span style="color: red;">at thermal equilibrium</span>
    + exponential function of its energy ($e^{-E}$)
    + a linear function of the energy achieved by the log probability when settling to equilibrium
  + the energy: a linear function of the weights and states

    \[ - \frac{\partial E}{\partial w_{ij}} = s_i s_j \]
  
  + the process of settling to thermal equilibrium propagates information about the weight $\implies$ no backpropagation required
  + two stages required:
    + settle w/ the data
    + settle w/o the data
  + the units deep in the network doing the same thing just w/ different boundary conditions

+ [Phases of energy function](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + similar to unlearning in Hopfield nets to get rid of spurious minima

  \[ p(\mathbf{v}) = \frac{\sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}}{\sum_{\mathbf{u}} \sum_{\mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}} \]

  + $\sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$:
    + the positive phase finds hidden configurations that work well w/ $v$ and lowers their energies
    + find these terms by settling to thermal equilibrium w/ vector $\mathbf{v}$ clamped
    + sampled those vectors $\mathbf{h}$ then changing the weights to make that energy even lower
  + $\sum_{\mathbf{u}} \sum_{\mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}$:
    + the negative phase finds the joint configurations that are the best competitors and raises their energies
    + finding the global configurations w/ combinations of visible and hidden states to have low energy
    + finding the global configurations raising their energy to contribute less


### Collecting Statistics

+ [Collecting statistics for learning](../ML/MLNN-Hinton/12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
  + Positive phase
    + clamp a data vector on the visible units
    + set the hidden units to random binary states
    + update the hidden units one at a time until the network reaches thermal equilibrium at a temperature of 1
    + sample $\langle s_i s_j \rangle$ for every connected pair of units
    + repeat for all data vectors in the training set and then average hem
  + Negative phase
    + prevent from clamping a data vector on the visible units $\implies$ unlearning
    + set all the units to random binary states
    + update all the units one at a time until the network reaches thermal equilibrium at a temperature of 1
    + sample $\langle s_j s_j \rangle$ for every connected pair of units
    + repeat many times (how many?) and average to get good estimates
    + expect the energy landscape to have many different seperately minima w/ about the same energy
    + expect a small fraction of the space to be these low energy states and a very large fraction of the space to be bad high energy states

+ [Better way to collect statistics](../ML/MLNN-Hinton/12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)
  + initial w/ a random state
  + initial w/ the last state - warm start
  + Advantages of warm start
    + small changed on weights at some thermal equilibrium
    + only a few updates to get back to equilibrium
    + able to use particles for both

+ [Radford Neal's method for collecting statistics](../ML/MLNN-Hinton/12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)
  + Positive phase
    + keep a set of "data-specific particles" per training case
    + each particle w/ a current value, ie., a configuration of the hidden units which data vector goes with
    + sequentially update all the hidden units a few times in each particle w/ the relevant data vector clamped
    + connected pair of units: average the probability of the two units $s_i, s_j$ on over all the data-specific particles
  + Negative phase
    + keep a set of "fantasy particles"
    + each particle w/ a value that is a global configuration
    + sequentially update all the units in each fantasy particle a few times $\implies$ updating the visible units
    + connected pair of units: average $s_i s_j$ over all the fantasy particles
  + weight changing

    \[ \Delta w_{ij} \propto \langle s_i s_j \rangle_{data} - \langle s_i s_j \rangle_{model} \]

  + mini-batches
    + not working well
    + no idea how long to reach theremal equilibrium again
  + strong assumption about how people understand the world ( kind of epistemological assumption)
    + assume clamped data vector: the set of good explanations (i.e., hidden unit states as interpretations of the data vector): uni-modal
    + restricting learning model: one sensory input vector w/o multiple very different explannations
    + w/ this assumption able to use a very efficient method for approaching thermal equilibrium or an approximation thermal equilibrium w/ the data


### Mean Filed approximation

+ [Simple mean field approximation](../ML/MLNN-Hinton/12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)

  \[\begin{align*}
    prob(s_i) &= \sigma \left( b_i + \sum_j s_j w_{ij} \right)  \tag{1}\\
    p_i^{t+1} &= \sigma \left( b_i + \sum_j p_j^t w_{ij} \right) \tag{2}\\
      &= \lambda p_i^t + (1 - \lambda) \sigma \left( b_i + \sum_j p_j^t w_{ij} \right) \tag{3}
  \end{align*}\]

  + right statistics $\implies$ updating the units stochastically and sequentially (Eq.(1))
  + using probabilities instead of binary states and update the unit in parallel to accelerate (Eq.(2))
  + using damped mean field to avoid bi-phasic oscillations
    + bi-phasic oscillations result in updating everything in parallel
    + damped mean field (Eq.(3)) resolves the oscillations

+ [Efficient mini-batch learning procedure](../ML/MLNN-Hinton/12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)
  + Positive phase
    + initialize all the hidden probabilities at $0.5$
    + clamp a data vector on the visible units
    + update all the hidden units in parallel until convergence using mean field updates
    + after net converged, record $p_i p_j$ for every connected pair of units and average this over all data in the mini-batch
  + Negative phase
    + keep a set of "fantasy particles"
    + each particle w/ a value  that is a global configuration
    + sequentially update all the units in each fantasy particle a few times
    + connected pair of units: average $s_i s_j$ (stochastic binary values) over all the fantasy particles
  + the difference btw these averages:
    + the learning rule
    + change the weights by amount proportional to that difference

+ [Parallel updates](../ML/MLNN-Hinton/12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)
  + general Boltzmann machine: the stochastic updates of units need to be sequential
  + Deep Boltzmann Machine (DBM) (left diagram)
    + special architecture: allowing alternative parallel updates for fantasy particles $\implies$ more efficient
    + no connections within a layer
    + no skip-layer connections
    + general Boltzmann machine w/ many missing connections
  + Update for states (right diagram)
    + update the states of the 1st hidden layer (top) and 3rd hidden layer (3 units) w/ current states of the visible units and 2nd hidden layer (2 units)
    + update the states if the visible units in the 2nd hidden layer
    + repeat the previous two processes
    + update half of the states of all units in parallel

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture12/lec12.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m12-02.png" style="margin: 0.1em;" alt="Example architecture of DBM" title="Example architecture of DBM" height=150>
      <img src="../ML/MLNN-Hinton/img/m12-03.png" style="margin: 0.1em;" alt="Example architecture of DBM" title="Example architecture of DBM" height=150>
    </a>
  </div>

+ [Effective mixing rate](../ML/MLNN-Hinton/12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)
  + learning
    + interact w/ the Markov chain used to gather the "negative statistics"; i.e., the data-independent statistics
    + not able to analyze the learning by viewing it as an outer loop and the gathering of statistics as an inner loop
  + fantasy particles outnumber the positive data
    + raising energy surface $\to$ an effect on the mixing rate of the Markov chain
    + the fantasies rush around hyperactively
    + moving around MUCH faster than the mixing rate of the Markov chain defined by the static current weights
  + Moving fantasy particles btw model's modes
    + more fantasy particles than data
      + energy surface raised until the fantasy particles escape
      + overcome energy barrier: too high for the Markov chain to jump in a reasonable time
    + changing energy surface: mixing in addition to defining the model
      + energy surface represent the model
      + energy surface manipulated by the learning algorithm to make the Markov chain mix faster or rather have the affect of a faster mixing Markov chain
    + fantasy particles filled in a hole
      + rush off somewhere else to deal w/ the next problem
      + like investigative journalists

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture12/lec12.pptx" ismap target="_blank">
        <img src="../ML/MLNN-Hinton/img/m12-05.png" style="margin: 0.1em;" alt="Examples of moving fantasy particles" title="Examples of moving fantasy particles" width=200>
      </a>
    </div>


