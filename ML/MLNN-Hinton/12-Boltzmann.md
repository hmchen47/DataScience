# 12. Restricted Boltzmann Machine (RBMs)

## 12.1 The Boltzmann machine learning algorithm

### Lecture Notes

+ Goal of learning
  + maxizing the product of the probabilities that the Boltzmann machine assigns to the binary vectors in the training set
  + equivalent to maxizing the sum of the log probabilities that the Boltzmann machine assigns to the training vectors
  + equivalent to maximizing the probability to obtain exactly the $N$ training cases if
    + settle the network to its stationary $N$ different times w/o external input
    + sample the visible vector once each time

+ Why difficult
  + consider a chain of units w/ visible units aat the ends (see diagram)
  + the training set consists of $(1, 0)$ and $(0, 1) \to$ the product of all the weights to be negative
  + knowing $w3 \to$ change $w1$ or $w5$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture12/lec12.pptx" ismap target="_blank">
      <img src="img/m12-01.png" style="margin: 0.1em;" alt="Example of Hopfield nets" title="Example of Hopfield nets" width=350>
    </a>
  </div>

+ Surprising fact
  + everything that one weight needs to know about the other weights and the data is contained in the difference of two correlations

    \[ \frac{\partial \log p(\mathbf{v})}{\partial w_{ij}} = \langle s_i s_j \rangle_{\mathbf{v}} - \langle s_i s_j \rangle_{model} \]

    + $\frac{\partial \log p(\mathbf{v})}{\partial w_{ij}}$: derivative of log probability of one training vector, $v$ under the model
    + $\langle s_i s_j \rangle_{\mathbf{v}}$: expected value of product of states at thermal equilibrium when $v$ is clamped on the visible units
    + $\langle s_i s_j \rangle_{model}$: expected value of product of states at thermal equilibrium w/o clamping
  + weight

    \[ \Delta w_{ij} \propto \langle s_i s_j \rangle_{data} - \langle s_i s_j \rangle_{model} \]

+ Derivative
  + the probability of a global configuration at thermal equilibrium
    + exponential function of its energy
    + settling to equilibrium makes the log probability a linear function of the energy
  + the energy: a linear function of the weights and states

    \[ - \frac{\partial E}{\partial w_{ij}} = s_i s_j \]
  
  + the process of settling to thermal equilibrium propagates information about the weight $\implies$ no backpropagation required

+ Negative phases

  \[ p(\mathbf{v}) = \frac{\sum_{\mathbf{h}} w^{-E(\mathbf{v}, \mathbf{h})}}{\sum_{\mathbf{u}} \sum_{\mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}} \]

  + $\sum_{\mathbf{h}} w^{-E(\mathbf{v}, \mathbf{h})}$: the positive phase finds hidden configurations that work well w/ $v$ and lowers their energies
  + $\sum_{\mathbf{u}} \sum_{\mathbf{g}} e^{-E(\mathbf{u}, \mathbf{g})}$: the negative phase finds the joint configurations that are the best competitors and raises their energies

+ Collecting statistics for learning
  + Positive phase
    + clamp a data vector on the visible units
    + set the hidden units to random binary states
    + update the hidden units one at a time until the network reaches thermal equilibrium at a temperature of 1
    + sample $\langle s_i s_j \rangle$ for every connected pair of units
    + repeat for all data vectors in th training set and average
  + Negative phase
    + set all the units to random binary states
    + update all he units one at a time until the network reaches thermal equilibrium at a temperature of 1
    + sample $\langle s_j s_j \rangle$ for every connected pair of units
    + repeat many times (how many?) and average to get good estimates


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 12.2 More efficient ways to get the statistics

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 12.3 Restricted Boltzmann machines

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 12.4 An example of contrastive divergence learning

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 12.5 RBMs for collaborative filtering

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


