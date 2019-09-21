# Mini-batch Gradient Descent
  
## Overview of mini-batch gradient descent

### Lecture Notes

+ Reminder: the error surface for a linear neuron
  + error surface
    + lying in a space with a horizontal axis for each weight and one vertical axis for the error
    + quadratic bowl (top figure): a linear neuron with a squared error
    + parabolas: vertical cross-sections
    + ellipses (bottom figure): horizontal cross-sections
  + multi-layer, non-linear nets
    + more complicated
    + a piece of a quadratic bowl: a very good approximation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pptx" ismap target="_blank">
      <img src="img/m06-01.png" style="margin: 0.1em;" alt="Error surface for a linear neuron" title="Error surface for a linear neuron" width=200>
    </a>
  </div>

+ Convergence speed of full batch learning
  + going downhill reducing the error
  + the direction of steepest descent not point at the minimum unless the ellipse is circle (see figure)
  + big gradient in the direction traveling a small distance
  + small gradient in the direction traveling a large distance  
  + locally quadratic error surface applying the same speed issues even for non-linear multi-layer nets

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pptx" ismap target="_blank">
      <img src="img/m06-02.png" style="margin: 0.1em;" alt="Quadractic error surface" title="Quadractic error surface" width=200>
    </a>
  </div>

+ How the learning goes wrong
  + big learning rate
    + the weights slosh to and fro across the ravine
    + too big causing oscillation diverges
  + what to achieve
    + quickly in directions with small but consistent gradients
    + slowly in directions with big but inconsistent gradients

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pptx" ismap target="_blank">
      <img src="img/m06-03.png" style="margin: 0.1em;" alt="Illustration of learning rate" title="llustration of learning rate" width=200>
    </a>
  </div>

+ Stochastic gradient descent
  + highly redundant dataset
    + the first half gradient $\simeq$ the second half gradient
    + update the weights using the first half gradient then get a gradient for the new weights on the second half gradient
  + online learning: update weights after each case
  + mini-batches usually better than online
    + typically 10, 100, even 1000 examples
    + advantages:
      + less computation to update the weights
      + using matrix-matrix multiplies to compute the gradient for many cases simultaneously
    + efficient matrix multiplications, especially on GPUs
  + mini-batches required to be balanced for classes
    + allocating the same class in a batch causing sloshing weights
    + random permutation for mini batches and randomly select the mini  batches for training

+ Two types of learning algorithm
  + full gradient computed from all the training cases
    + ways to speed up learning, eg. non-linear conjugate gradient
    + optimization community: the general problem of optimizing smooth non-linear functions
    + multilayer neural nets: not typical of the problems; required a lot of modification to make them work
  + mini-batch learning for large neural networks w/ very large and highly redundant training sets
    + mini-batches may be quite big when adapting fancy methods
    + big mini-batches: more computationally efficient

+ A basic mini-batch gradient descent algorithm
  + guess an initial learning rate
    + measured on a validation set
    + each mini-batch just a rough estimate of the overall gradient
    + reducing learning rate: error getting worse or oscillated
    + increasing learning rate: error falling fairly consistent but slowly
  + write a simple program to automate this way to adjusting the learning rate
  + toward end of mini-batch learning
    + nearly always help to turn down the learning rate
    + remove fluctuations in the final weights caused by the variations btw mini-batches
  + cease the learning (criteria)
    + the error stops decreasing
    + using the error on a separate validation set



### Lecture Video

<video src="https://youtu.be/4BZBog1Zx6c?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## A bag of tricks for mini-batch descent

### Lecture Notes

+ Criteria to stop the learning
  + error fluctuations caused by the different gradients on different mini-batches
  + turing down the learning rate reduces the random fluctuations in the error
    + a quicker win
    + a slower learning
  + Don't turn down the learning rate too soon!

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pptx" ismap target="_blank">
      <img src="img/m06-04.png" style="margin: 0.1em;" alt="Illustration of learning rate" title="llustration of learning rate" width=350>
    </a>
  </div>

+ Initializing the weights
  + two different units w/ exactly the same bias and exactly the same incoming and outgoing weights
    + exactly the same gradients
    + never learn to be different features
    + break symmetry by initializing the weights to have small random values
  + overshooting learning
    + a hidden unit w/ a big fan-in, small changes on many of its incoming weights
    + generally smaller incoming weights when the fan-in is big
    + initialize the weights to be proportional to sqrt(fan-in)
  + scale the learning rate the same way

+ Shifting and scaling the inputs
  + Shifting
    + when using steepest descent, shifting the input values makes a big difference
    + help to transform each component of the input vector so that it has zero mean over the whole training set
    + the hyperbolic tangent (2*logistic - 1) produces hidden activations roughly zero mean
      + better than the logistic
  + Scaling
    + when using steepest descent, scaling the input values makes a big difference
    + help to transform each component of the input vector so that it has unit variance over the whole training set

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pptx" ismap target="_blank">
      <img src="img/m06-05.png" style="margin: 0.1em;" alt="Illustration for shifting the inputs" title="Illustration for shifting the inputs" height=350>
      <img src="img/m06-06.png" style="margin: 0.1em;" alt="Illustration for scaling the inputs" title="Illustration for scaling the inputs" height=350>
    </a>
  </div>

+ Decorrelating the input components - a thorough method
  + linear neuron: a big win by decorrelating each component of the input from the other input components
  + ways to decorrelate inputs
    + reasonable method: Principal Components Analysis
    + drop the principal components with the smallest eigenvalues
    + achieving some dimensionality reduction
    + divide the remaining principal components by the square roots of their eigenvalues
    + linear neuron: convert an axis aligned elliptical error surface into a circular one
  + circular error surface: the gradient points straight towards the minimum

+ Common problems occurring in multilayer networks
  + Starting w/ a very big learning rate
    + the weights of each hidden unit will all become very big and positive or very big and negative
    + tiny error derivatives for the hidden units and not decreasing
    + a plateau: mistaken for a local minimum
  + Strategy for learning
    + classification networks: using a squared error or a cross-entropy error
    + the best guessing strategy: to make each output unit always produce an output equal to the proportion of time it should be a 1
    + take a time to improve on it by making use of the input
    + another plateau like a local minimum

+ Methods to speed up mini-batch learning
  + use "momentum"
    + instead of using the gradient to change the position of the weight "particle", use it to change the velocity
  + use separate adaptive learning rates for each parameter
    + slowly adjust the rate using the consistency of the gradient for that parameter
  + rmsprop
    + divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight
    + the mini-batch version of just using the sign of the gradient
  + take a fancy method from the optimization literature that makes use of curvature information
    + adapt it to work for neural nets
    + adapt it to work for mini-batches


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## The momentum method

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## A separate, adaptive learning rate for each connection

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## rmsprop_divide the gradient

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>



