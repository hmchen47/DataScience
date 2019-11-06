# 7. Recurrent Neural Networks

## 7.1 Modeling sequences: A brief overview

### Lecture Notes

+ Getting targets when modeling sequences
  + when applying machine learning to sequences, often turn an input sequence into an output sequence that lives in a different domain
    + e.g., turn a sequence of sound pressures into a sequence of word identities
  + when no separate target sequence, get a teaching signal by trying to predict the next term in the input sequence
    + target output sequence: the input sequence with an advance of 1 step
    + seeming much more natural than trying to predict one pixel in an image from other pixels, or one patch of an image from the rest of the image
    + temporal sequences: a natural order for the predictions
  + predicting the next terms in a sequence blurs the distinction between supervised and unsupervised learning
    + using methods designed for supervised learning but not require a separate teaching signal (sounds like unsupervised)

+ Memoryless models for sequences
  + autoregressive models: predict the next term in a sequence from a fixed number of previous terms using "delay taps"
  + feed-forward neural nets
    + these generalize autoregressive models by using one or more layers of non-linear hidden units
    + e.g. Bengio's first language model

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="img/m07-01.png" style="margin: 0.1em;" alt="Autoregressive models" title="Autoregressive models" height=80>
      <img src="img/m07-02.png" style="margin: 0.1em;" alt="Feed-forward neural nets" title="Feed-forward neural nets" height=80>
    </a>
  </div>

+ Beyond memoryless models
  + generative model w/ hidden state that has its own internal dynamics
    + providing a much more interesting kind of model
    + storing information in its hidden state for a long time
    + dynamics is noisy $\to$ outputs generated from its hidden states is noisy $\to$ the exact hidden state unknown
    + best practice: infer a probability distribution over the space of hidden state vectors
  + inference: only tractable for two types of hidden state model
    + two types of hidden state model: linear dynamic systems & hidden Markov models
    + showing how RNNs differ

+ Linear dynamical systems (engineers perspective)
  + generative models
    + a real-value hidden state not able to observed directly
    + hidden state having linear dynamics w/ Gaussian noise
    + producing the observations using a linear model w/ Gaussian noise
    + there may also be <span style="color: red;">driving inputs</span>
  + to predict the next output
    + required to infer the hidden state
    + a linear transformed Gaussian is a Gaussian
    + distribution over the hidden state given the data so far is Gaussian
    + computed using "Kalman filtering", an efficient recursive way of updating the representation of the hidden state given a new observation
  + Summary: 
    + given the observations of the output system, not sure what hidden state in but able to estimate a Gaussian distribution over the possible hidden state it might have been in
    + always assuming the model is the correct one

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="img/m07-03.png" style="margin: 0.1em;" alt="Linear dynamical systems" title="Linear dynamical systems" width=200>
    </a>
  </div>

+ Hidden Markov Models (computer scientists perspective)
  + a discrete one-of-N hidden state
    + transition btw states are stochastic and controlled by a transition matrix
    + stochastic outputs w/ a state
    + not sure which state produced a given output $\to$ the state is "hidden"
    + easy to represent a probability distribution across $N$ states w/ $N$ numbers
  + to predict the next output
    + required to infer probability distribution over hidden states
    + HMMs w/ efficient algorithms for inference and learning
  + an easy method based on dynamic programming to observer the output and compute the probability distribution across the hidden states

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="img/m07-04.png" style="margin: 0.1em;" alt="Hidden Markov models" title="Hidden Markov models" width=200>
    </a>
  </div>

+ A fundamental limitation of HMMs
  + considering what happens when a hidden Markov model generates data
    + select one of its hidden states at each time step
    + with $N$ hidden states, only remember $\log(N)$ bits about what it generated
  + considering the first half of an utterance contains about the second half
    + syntax to fit, e.g., number and tense agreement
    + semantic to fit; intonation to fit
    + accent, rate, volume and vocal tract characteristics must all fit
  + all aspects combined could be 100 bits of information that the fist half of an utterance needs to convey to the second half. $2^{100}$ is big!

+ Recurrent neural networks
  + efficient way to remember the information
  + very powerful
  + Properties of RNNs
    + distributed hidden state: to store a lot of information about the past efficiently
    + non-linear dynamics: to update their hidden state in complicated ways
  + with enough neurons and time RNNs able to compute anything that can be computed by your computer

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="img/m07-05.png" style="margin: 0.1em;" alt="Recurrent neural networks" title="Recurrent neural networks" width=150>
    </a>
  </div>

+ Do generative models need to be stochastic?
  + stochastic models: both linear dynamical systems and hidden Markov models
    + the posterior probability distribution over their hidden states given the observed data so far is a deterministic function of the data
    + probability distribution: a bunch of numbers that are a deterministic function of the data
  + recurrent neural networks are deterministic
    + think of the hidden state of an RNN as the equivalent of the deterministic probability distribution over hidden states in a linear dynamical system or hidden Markov model
    + the numbers constitute the hidden state of a recurrent neural network

+ Recurrent neural networks
  + what kinds of behavior can RNNs exhibit?
    + oscillate: good for motor control?
    + settle to point attractors: good for retrieving memories?
    + chaotic behavior: bad for information processing? randomness?
    + RNNs potentially learn to implement lots small programs using different subsets of its hidden state
    + each program able to capture a nugget of knowledge
    + the programs able to run in parallel
    + interacting each other to produce very complicated effects
  + the computational power of RNNs makes them very hard to train
    + unable to exploit the computational power of RNNs
    + implementing on a parallel computer


### Lecture Video

<video src="https://youtu.be/enI1YMvCJ34?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 7.2 Training RNNs with backpropagation

### Lecture Notes

+ The equivalence between feed-forward nets and recurrent nets
  + assumption: a time delay of 1 in using each connection
  + the recurrent net is just a layered net that keeps reusing the same weights

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="img/m07-06.png" style="margin: 0.1em;" alt="Feedforward nets" title="Feedforward networks" width=250>
      <img src="img/m07-07.png" style="margin: 0.1em;" alt="Recurrent neural networks" title="Recurrent neural networks" width=250>
    </a>
  </div>

+ Recall: backpropagation w/ weight constraints
  + easy to modify the backpropagation algorithm to incorporate linear constraints between the weights
  + computing the gradients as usual, and then modify the gradients to satisfy then constraints
  + continuing process to satisfy the constraints if the weights not satisfied<br/><br/>

    To constraint: $w_1 = w_2$

    we need: $\Delta w_1 = \Delta w_2$

    compute: $\frac{\partial E}{\partial w_1}$ and $\frac{\partial E}{\partial w_2}$

    use $\frac{\partial E}{\partial w_1} + \frac{\partial E}{\partial w_2}$ for $w_1$ and w_2$

+ backpropagation through time
  + recurrent net: a layered, feed-forward net with shared weights
  + training the feed-forward net w/ weight constraints
  + training algorithm in the time domain
    + forward pass: a stack of the activities of all the units at each time step
    + backward pass: peeling activities off the stack to compute the error derivatives at each time step
    + adding the derivatives at all the different times for each weight after backward pass

+ An irritating extra issue
  + specifying the initial activity state of all the hidden and output units
  + just fixing these initial states to have some default value like 0.5
  + better to treat the initial states as learned parameters
  + training them in the same way as we learn the weights
    + starting off w/ an initial random guess for the initial states
    + at the end of each training sequence, backpropagate through time all the way to the initial states to get the gradient of the error function w.r.t. each initial state
    + adjusting the initial states by following the negative gradient

+ Input and Output of recurrent networks
  + specifying inputs in several ways
    + the initial states of all the units
    + the initial states of a subset of the units
    + the states of the same subset of the units at every time step
      + the natural way to model most sequential data
  + specifying targets in several ways
    + desired final activities of all the units
    + desired activities of all units for the last few steps
      + good for learning attractors
      + easy to add in extra error derivatives as backpropagating
    + the desired activity of a subset of the units
      + other units: input or hidden units

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="img/m07-08.png" style="margin: 0.1em;" alt="Inputs of recurrent nets" title="Inputs of recurrent nets" height=250>
      <img src="img/m07-09.png" style="margin: 0.1em;" alt="Targets of recurrent nets" title="Targets of recurrent nets" height=250>
    </a>
  </div>


### Lecture Video

<video src="https://youtu.be/hTcm8AJjvfE?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 7.3 A toy example of training an RNN

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 7.4 Why it is difficult to train an RNN

### Lecture Notes

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>





### Lecture Video

## 7.5 Long term short term memory

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


