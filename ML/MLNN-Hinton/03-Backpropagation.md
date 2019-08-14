# Backpropagation Learning Procedure
  
## Learning the weights of a linear neuron

### Lectue Notes

+ Perceptron: the weights are always getting closer to a good set of weights

+ Linear neuron: the output always getting closer to target outputs

+ Perceptron learning procedure unable to generalize to hidden layers
  + perceptron convergence procedure
    + every time the weights change, get closer to every "generously feasible" set of weights
    + No guarantee to be extended to more complex networks
    + the average of two good solutions may be a bad solution
  + multi-layer neural networks not using the perceptron learning procedure
    + Never called multi-layer perceptrons

+ A different way to show that a learning procedure makes progress
  + Weight vs. target values
    + Instead of showing the weights get closer to a good set of weights, show that the actual output values get closer the target values
    + true even for non-convex problems in which there are many quite different sets of weights that work well and averaging two good sets of weights may give a bad set of weights
    + Not true for perceptron learning
      + the outputs as a whole can get further away from the target output even the weights are getting closer to good sets of weights
  + simplest example: a linear neuron with a squared error measure

+ Linear neurons
  + a.k.a linear filters in EE
  + real-valued output = weighted sum of its inputs

    $$y = \sum_i w_i x_i = \mathbf{W}^T \mathbf{x}$$

    + $y$: neuron's estimate the desired output
    + $\mathbf{W}$: weight vector
    + $\mathbf{x}$: input vector
  + aim of learning (objective): to minimize the error summed over all training cases
  + error (measure): the squared difference btw the desired output and the actual output

+ Why not solved analytically?
  + Standard engineering approach: straight-forward to write down
    + a set of equations
    + one per training case
    + to solve for the best set of weights
    + but not used, why?
  + Scientific answer
    + understand what real neurons might be doing
    + probably not solving equations symbolically
    + find a method that real neurons could use
  + Engineering answer
    + find a method that can be generalized to multi-layer, non-linear neural networks
    + analytic solution: based on linearity and squared error measure
    + iterative methods: less efficient but much easier to generalize

+ Example: illustrate the iterative method
  + Get lunch at the cafeteria every day
    + diet: fish, chips, and ketchup
    + several portions of each
  + Cashier only told the total price of the meal
  + Question: figure out the price of each portion after several days
  + iterative approach
    + start with random guesses for the prices
    + adjust them to get a better fir to the observed prices of whole meals

+ Solving the equations iteratively
  + Each meal price gives a linear constraint on the prices of the portions

    $$price = x_{fish} w_{fish} + x_{chips} w_{chips} + x_{ketchup} w_{ketchup}$$
  
  + The prices of the portions are like the weights in of a linear neuron

    $$\mathbf{W} = (x_{fish}, w_{chips}, w_{ketchup})$$

  + Start with guesses for the weights and then adjust the guesses slightly to give a better fit to the prices given by the cashier

+ A model of the cashier with arbitrary initial weights
  + Residual error = 350
  + the "delta-rule" for learning is: $\Delta w_i = \varepsilon x_i(t - y)$
  + with a learning rate $\varepsilon = 1/35$, the weight changes $ = (+20, +50, +30)$
  + The new weights: $(70, 100, 80)$
  + The weight for chips got worse
  + No guarantee the way of learning theta individual weights will keep getting better
  + Only the difference btw cashier & estimation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3.pptx" ismap target="_blank">
      <img src="img/m03-01.png" style="margin: 0.1em;" alt="The true wights used by the cashier" title="The true wights used by the cashier" height=200>
      <img src="img/m03-02.png" style="margin: 0.1em;" alt="A model of the cashier with arbitrary initial weights" title="A model of the cashier with arbitrary initial weights" height=200>
    </a>
  </div>

+ Deriving the delta rule
  + Define the error as the squared residuals summed over all training cases

    $$E = \displaystyle \frac{1}{2} \sum_{n \in training} (t^n - y ^n)^2$$

    + $1/2$: cancel 2 when differentiating

  + Now differentiate to get error derivatives for weights

    $$\dfrac{\partial E}{\partial w_i} = \frac{1}{2} \sum_n \dfrac{\partial y^n}{\partial w_i} \dfrac{dE^n}{dy^n} = - \sum_n x_i^n (t^n - y^n)$$

    + applying chain rule
    + explain how the output changes as we change the weights times how the error changes as we change the output
    + $\partial w_i$: many ways to change the output but just considering the change of the weight $w_i$
    + $\frac{\partial y}{\partial w_i} = x_i$: $y = w_i \times x_i$
    + $\frac{d E}{dy} = (t-y)$: $1/2 \times 2 \cdot (t - y)$

  + The batch delta rule changes the weights in portion to their derivatives <span style="color: green;">summed over all training cases</span>

    $$\Delta w_i = -\varepsilon \dfrac{\partial E}{\partial w_i} = \sum_n \varepsilon x_i^n (t^n - y^n)$$

+ Behavior of the iterative learning procedure
  + Does the learning procedure eventually get the right answer?
    + There may be no perfect answer
      + after providing the linear neuron a bunch of training cases with desired answers but no set of weights giving the desired answer
      + still some set of weights that gets the best approximation on all training cases minimizes the error measure summed over all training cases
    + by making the learning rate small enough we can get as close as we desire to the best answer
  + How quickly do the weights converge to their correct values?
    + very slow if two input dimensions are highly correlated
    + almost always have the same number of portions of ketchup and chips
    + hard to decide how to divide the price btw ketchup and chips

+ online delta-rule vs learning rule for perceptrons
  + perceptron learning
    + increment or decrement the weight vector by the input vector
    + only change the wights when making an error
  + online version of the delta-rule
    + increment or decrement the weight vector by the input vector scaled by the residual error and the learning rate
    + choose a learning rate --> annoying


### Lecture Video

<video src="https://youtu.be/Q0mTl9dQ4_I?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## The error surface for a linear neuron

### Lectue Notes




### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3b.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Learning the weights of a logistic output neuron

### Lectue Notes




### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3c.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## The backpropagation algorithm

### Lectue Notes




### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3d.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## How to use the derivatives computed by the backpropagation algorithm

### Lectue Notes




### Lecture Video

<video src="http://www.cs.toronto.edu/~hinton/coursera/lecture3/lec3e.mp4" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>

