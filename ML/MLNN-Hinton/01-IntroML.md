# Introduction to Machine Learning

## Why do we need machine learning

### Lecture Notes

+ What is Machine Learning?
  + difficult to write programs to solve problems like recognizing a 3D object
    + don't know how to be done in brian
    + horrendously complicated program
  + hard to write program to compute the probability of fraudulent transaction
    + no simpler and reliable rules
    + moving target

+ Machine Learning Approach
  + Collect lots of examples that specify the correct output for a given input
  + Develop a ML algorithm and then take these examples and produce a program to do the job
    + ML program different from typical hand-written program
    + Right solution works both on prediction and training datasets
    + Data changes result in parameter changes in program
  + Cheaper massive amounts of computation than salary to write a task-specific program

+ Examples for Machine Learning
  + Recognition patterns
    + Objects in real scenes
    + Facial identities or facial expressions
    + Spoken words
  + Recognizing anomalies
    + Unusual sequences of credit card transactions
    + Unusual patterns of sensor readings in a nuclear power plant
  + Prediction
    + Future stock prices or currency exchange rates
    + Which movies will a person like?

+ A Standard Example of Machine Learning in the course
  + lot of genetics done on fruit flies
    + convenient due to breed fast
    + well known
  + MNIST database of hand-written digits is the ML equivalent of fruit flies
    + publicly available and lean quite fast in a moderate-sized neural net
    + know a huge amount about how well various ML methods do on MNIST

+ Hard to say what makes a 2

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="img/m01-01.png" style="margin: 0.1em;" alt="Example of digits" title="Example of digits" width=350>
    </a>
  </div>

+ The ImageNet Task
  + 1000 different object classes in 1.3 million high-resolution training images
    + best system in 2010 competition got 47% error for its first choice and 25% error for its top 5 choices
  + Jitendra Malik: this competition is a good test of whether deep neural networks work well for object recognition
    + very deep neural net (Krizhevsky et. al. 2012) gets less that 40% error for its first choice and less than 20% for its top 5 choices

+ The speech Recognition Task
  + Several stages
    + Pre-processing:
      + convert the sound wave into a vector of acoustic coefficients
      + extract a new vector about every 10 mille seconds
    + The acoustic model
      + use a few adjacent vectors of acoustic coefficients to place bets on which part of which phoneme is being spoken
    + Decoding
      + find the sequence of bets that does the best job of fitting the acoustic data
      + fit a model of the kinds of things people say
  + Deep neural networks
    + pioneered by George Dahl and Abdel-rahman Mohamed
    + replacing the previous ML method for the acoustic model

+ Phone recognition on the TIMIT benchmark
  + Procedures: 15 frames pf 40 filterbank output + their temporal derivatives --> 2000 logistic hidden units --> 2000 logistic hidden units -- (5 more layers of pre-trained weights) --> 2000 logistic hidden units -- (not pre-trained) --> 183 HMM-state labels
  + After standard post-processing using a bi-phone model, a deep net with 8 layers gets `20.7%` error rate
  + best previous speaker-independent result in TIMIT: `24.4%` and required averaging several models
  + Li Deng (at MSR) realized that this result could change the way speech recognition was done

+ Word error rates from MSR, IBM, & Google (Hinton et. al. IEEE Signal Processing Magazine, Nov. 2012)

  <table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=90%>
    <thead>
    <tr>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">The task</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Hours of training data</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Deep neural network</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Gaussian Mixture Model</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">GMM with more data</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>Switchboard (Microsoft Research)</td><td>309</td><td>18.5%</td><td>27.4%</td><td>18.6% (2000 hrs)</td>
    </tr>
    <tr>
      <td>English broadcast news (IBM)</td><td>50</td><td>17.5%</td><td>18.8%</td><td></td>
    </tr>
    <tr>
      <td>Google voice search (Android 4.1)</td><td>5,870</td><td>12.3% (and falling)</td><td></td><td>16.0% (>>5,870 hrs)</td>
    </tr>
    </tbody>
  </table>




### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture1/lec1a.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## What are neural networks

### Lecture Notes

+ Reasons to study neural computation
  + to understand how the brain actually works
    + big, complicated, and then die when poking --> computer simulation used
  + to understand a style of parallel computation inspired by neurons and their adaptive connections
    + different from sequential computation
    + good for things that brands are good at, e.g., vision
    + bad for things that brains are bad at, e.g., 23 x 71
  + to solve practical problems by using novel learning algorithms inspired by the brain
    + learning algorithms very useful even if they are not how the brain actually works

+ A typical cortical neuron
  + gross physical structure
    + axon: branches, path to transmit
    + dendritic tree: collect input from other neurons
  + Axons typically contact dendritic trees at synapses
    + a spike of activity in the axon causes charge to be injected into the post-synaptic neuron
  + Spike generation
    + axon hillock: generate outgoing spikes whenever enough charge has flowed in at synapses to depolarize the cell membrane

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://en.wikipedia.org/wiki/Multipolar_neuron" ismap target="_blank">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Blausen_0657_MultipolarNeuron.png/500px-Blausen_0657_MultipolarNeuron.png" style="margin: 0.1em;" alt="A multipolar neuron (or multipolar neurone) is a type of neuron that possesses a single axon and many dendrites (and dendritic branches), allowing for the integration of a great deal of information from other neurons. These processes are projections from the nerve cell body. " title="Anatomy of multipolar neuron" width=450>
    <a href="https://www.researchgate.net/figure/Biological-neuron-and-synapse_fig1_324229756" ismap target="_blank">
      <img src="https://www.researchgate.net/publication/324229756/figure/fig1/AS:612287031308289@1522992012648/Biological-neuron-and-synapse.png" style="margin: 0.1em;" alt="In human brain, excitatory and inhibitory postsynaptic potentials are delivered from pre-synaptic neuron to postsynaptic neuron through chemical and electrical signal at synapses, driving the change of synaptic weight." title="Biological neuron and synapse" width=450>
    </a>
    </a>
  </div>

+ Synapses
  + a small gap separating neurons
  + when a spike of activity travels along an axon and arrives at a synapses it causes vesicles of transmitter chemical to be released
    + several kinds of transmitter
  + the transmitter molecules diffuse across the synaptic cleft and bind too receptor molecules in the membrane of the post-synaptic neuron thus changing their shape
    + open up holes to allow specific ions in or our

+ How synapses adapt
  + effectiveness of the synapse changes
    + vary the number of vesicles of transmitter
    + vary the number of receptor molecules
  + synapses slow, but having advantages over RAM
    + very small and very low-power
    + adapt using locally available signals
      + what rules do they use to decide how to change?

+ How the brain works on one slide
  + each neuron receives inputs from other neurons
    + a few neurons also connect to receptors
    + cortical neurons use spikes to communicate
  + the effect of each input line on the neuron controlled by a synaptic weight
    + weights either positive or negative
  + synaptic weights adapt so that the whole network learns to perform useful computations
    + recognizing objects
    + understanding language
    + making plans
    + controlling the body
  + about $10^{11}$ neurons each with about $10^4$ weights
    + huge number of weights effect the computation in a very short time
    + better bandwidth than a workstation

+ Modularity and the brain
  + different bits of the cortex do different things
    + local damage to the brain has specific effects
    + specific tasks increase the blood flow to specific regions
  + cortex looks pretty much the same all over
    + early brain damage makes functions relocate
  + cortex made of general purpose stuff that has the ability to turn into special purpose hardware in response to experience
    + rapid parallel computation plus flexibility
    + conventional computers get flexibility by having store sequential programs
    + require very fast central processors to perform long sequential computations


### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture1/lec1b.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Some simple models of neurons

### Lecture Notes

+ Idealized neurons
  + to model things we have to idealize them (e.g., atoms)
    + idealization removes complicated details not essential for understanding the main principles
    + applying mathematics and analogies to other, familiar systems
    + understanding the basic principles, add complexity to make the model more faithful
  + Worth understanding models known to be wrong
    + E.g., neurons that communicate real values rather than discrete spikes of activity

+ Linear neurons
  + Simple but computationally limited
  + make them learn we __may__ get insight into more complicated neurons

    $$y = b + \sum_i x_i w_i$$

    + $y$: output
    + $b$: bias
    + $i$: index over input connections
    + $x_i$: $i^{th}$ input
    + $w_i$: wight on $i^{th}$ input

+ Binary threshold neurons
  + McCulloch-Pitts (1943): influenced Von Neumann
    + first compute a weighted sum of the inputs
    + send out a fixed size spike of activity if the weighted sum exceeds a threshold
    + each spike is like the true value of a proposition
    + each neuron combines true values to compute the true value of another proposition
  + two equivalent ways to write the equations for a binary threshold neuron

    $$\begin{array}{rcl}z = \displaystyle \sum_i x_i w_i & \quad & y = \begin{cases} 1 & \text{if } z \geq \theta \\ 0 & \text{otherwise} \end{cases} \\ & \theta = -b & \\ & \Downarrow & \\ z = b + \displaystyle \sum_i x_i w_i & \quad & y = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases} \end{array}$$

+ Rectified Linear Neurons
  + a.k.a, Linear threshold neurons
  + compute a linear weighted sum of their inputs
  + output: a non-linear function of the total input

    $$z = b + \sum_i x_i w_i \quad y = \begin{cases} z & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

+ Sigmoid neurons
  + a real-valued output that is a smooth and bounded function of their total input
  + use the logistic function
  + nice derivatives to make learning easy

  $$z = b + \sum_i x_i w_i \quad y = \frac{1}{1 + e^{-z}}$$

+ Stochastic binary neurons
  + same equations as logistic units
    + treat the output of the logistic as the probability of producing a spike in a short time window
  + similar trick for rectified linear units
    + treat the output as the Poisson rate for spikes

  $$z = b + \displaystyle \sum_i x_i w_i \quad p(s = 1) = \frac{1}{1 + e^{-z}}$$


<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://blog.zaletskyy.com/some-simple-models-of-neurons" ismap target="_blank">
    <img src="https://blog.zaletskyy.com/Media/Default/NeuralNetworks/binaryNeuron.png" style="margin: 0.1em;" alt="graphical representation with case if threshold = 1" title="graphical representation with case if threshold = 1" width=110>
  </a>
  <a href="https://www.bo-song.com/coursera-neural-networks-for-machine-learning/" ismap target="_blank">
    <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-2.png" style="margin: 0.1em;" alt="a non-linear function of the total input" title="a non-linear function of the total input" width=225>
  </a>
  <a href="https://www.bo-song.com/coursera-neural-networks-for-machine-learning/" ismap target="_blank">
    <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-5.png" style="margin: 0.1em;" alt="The sigmoid function consists of 2 functions, logistic and tangential. The values of logistic function range from 0 and 1 and -1 to +1 for tangential function." title="Sigmoid function" width=200>
  </a>
</div>



### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture1/lec1c.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## A simple example of learning

### Lecture Notes





### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture1/lec1d.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Three types of learning

### Lecture Notes





### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture1/lec1e.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>

