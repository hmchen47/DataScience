# Overfitting
  
## 9.1 Overview of ways to improve generalization

### Lecture Notes

+ Reminder: Overfitting
  + training data in the mapping from input to output
    + containing information about the regularities
    + containing sampling error
    + accidental regularities existed because of the particular training cases that were chosen
  + unable to identify the regularities real or caused by sampling error
    + fit both kinds of regularity
    + flexible model able to model the sampleing error really well
  + variance:
    + fit the model w/ another training set drawn from the same distribution over cases
    + making different predictions on the test data

+ Preventing overfitting
  + Approach 1: get more data!
    + almost always the best bet if the data is cheap
    + require more computing power to train on more data
  + Approach 2: using a model w/ the right capacity
    + enough to fit the true regularities
    + not enough to fit spurious regularities (if they are weaker)
  + Approach 3: average many different models
    + use models w/ different forms
    + bagging: train the model on different subsets of the training data
  + Approach 4: Bayesian
    + use a single neural network architecture
    + average the predictions made by many different weight vectors

+ Ways to limit the cpacity of a neural network
  + controlling capacity w/
    + <span style="color: blue;">architecture</span>: limit the number of hidden layers and the number of units per layer
    + <span style="color: blue;">early stopping</span>: start w/ small weights and stop the learning before it overfits
    + <span style="color: blue;">weight-decay</span>:
      + given model a number of hiddent layers or units per layer which is a little too large
      + penalize large weights using penalties or constrains on their squared values (L2 penalty) or absolute values (L1 penalty)
    + <span style="color: blue;">noise</span>: add to the weights of the activities
  + typically, using a combinaition of several of these methods

+ Meta parameters to control capacity
  + meta parameters: the number of hidden units, the number of layers, or the size of the weight penalty
  + trying lots of alternatives and choosing the one w/ the best performance on the test set
    + wrong method
    + easy to do but giving false impression of how well the methods works
    + settings w/ best performance on a test set unlikely to work on new data set drawn from the same distribution
    + model tuned for a particular test set
  + extreme example
    + assume that the test set has random answers that does not depend on the input
    + the best architecture perform better than chance on the test set
    + unable to be expected to do better than chance on a new test set

+ Cross-validation
  + a better way to choose meta parameters
  + divide the total dataset into 3 subsets
    + <span style="color: red;">training data</span>: used for learning the parameters of the model
    + <span style="color: red;">validation data</span>: not for learning but for deciding what settings of the meta parameters work best
    + <span style="color: red;">test data</span>: 
      + used to get a final, unbiased estimate of how well the network works
      + expect the estimate to be worse than on the validation data
  + competitions:
    + organizations held back the true test data and asked participationers to send in predictions
    + then validate the predictions can really predict on true test data or they're just overfitting to validation data by selecting meta parameters particularly well on the validation data but won't generalize to new test sets
  + N-fold cross-validation
    + divide the total dataset into one final test set and $N$ other subsets
    + train on all but one of those subsets to get $N$ different estimates of the validation error rate
    + the $N$ estimates not independent

+ Early stopping
  + goal: preventing overfitting
  + expensive to keep re-training the model
    + assumption: lots of data and a big model
    + training w/ different sized penalities on the weights of different architectures
  + Solution:
    + start w/ very small weights
    + grow the weights until the performance on the validation set starts getting worse
  + issues:
    + hard to decide when the performance getting worse
      + performance on the validation set might fluctuate
      + particularly, measure on error rate rather than a squared error or cross-emtropy error
    + limited capacity of the model
      + smaller weights give the network less capacity
      + weights not had time to grow big
  + Why small weights lower the cpacity?

+ Why early stopping works
  + very small weights
    + every hidden unit in its linear range
    + even w/ a large layer of hidden units
    + no more capacity than a linear net (inputs directly connected to outputs)
  + as weights grow
    + hidden units start using the non-linear ranges
    + capacity grows
  + example: (see diaggram)
    + hidden units w/ logistic units
    + small weights $\implies$ total inputs close to zero
    + the inputs for hidden units in the middle of their linear range $\implies$ very likely linear unit
    + multiply the inputs with weight matrice $w_1$ and $w_2$ to connect to the outputs
    + hidden units behave much like linear net
    + therefore, no more capacity than the linear net
    + $(3 \times 6) + (6 \times 2)$ wights $\implies$ no more capacity than a network w/ $(3 \times 2)$ weights
    + weight grow $\implies$ using the nonlinear region of the sigmoid
    + start making use of all those parameters $\implies$ from 6 to 30 weights increasing smoothly
    + early stopping: stop the learning w/ the right number of parameters $\implies$ optimize the trade-off between fitting the true regularities in the data and fitting the spurious regularities

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture9/lec9.pptx" ismap target="_blank">
      <img src="img/m09-01.png" style="margin: 0.1em;" alt="Modeling early stopping" title="Modeling early stopping" width=200>
    </a>
  </div>


### Lecture Video

<video src="https://youtu.be/W0SP8FTmGW0?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 9.2 Limiting size of the weights

### Lecture Notes

+ Limiting the size of the weights
  + the standard L2 weight penalty
    + adding an extra term to the cost function that penalizes the squared weights
    + keep the weights small unless they have big error derivaties
  + mathematical representation

    \[\begin{align*} 
      C &= E + \frac{\lambda}{2} \sum_i w_i^2 \\
      \frac{\partial C}{\partial w_i} &= \frac{\partial E}{\partial w_i} + \lambda w_i
    \end{align*}\]

    + $\frac{\partial C}{\partial w_i} = 0$
    + $w_i = \frac{1}{\lambda} \frac{\partial E}{\partial w_i}$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture9/lec9.pptx" ismap target="_blank">
      <img src="img/m09-02.png" style="margin: 0.1em;" alt="Limiting size of the weights" title="Limiting size of the weights" width=200>
    </a>
  </div>

+ The effect of L2 weight cost
  + prevent the network from using weights not needed
    + improve generalization significantly $\impliedby$ stop the network from fitting the sampling error
    + a soother model: the output changes more slowly as the inputs changes
  + prefer to equally divide the weight on each branch
    + a network w/ two very similar inputs
    + put half the weight on each tather than all the weight on one

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture9/lec9.pptx" ismap target="_blank">
      <img src="img/m09-03.png" style="margin: 0.1em;" alt="Examples of distributing weights" title="Examples of distributing weights" width=200>
    </a>
  </div>

+ Other kinds of weight penalty
  + penalize the absolute values of the weights
    + make many weights exactly equal to zero which helps interpretation a lot
  + weight penalty w/ negligible effect on large weights
    + allowing a few large weights

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture9/lec9.pptx" ismap target="_blank">
      <img src="img/m09-04.png" style="margin: 0.1em;" alt="Examples of weight penalty" title="Examples of weight penalty" width=200>
    </a>
  </div>

+ Weight penalities vs weight constraints
  + usually penalize the squared value of each weight separately
  + constraint on the maximum squared length of the incoming weight vector of each unit
    + an update violates constraint $\implies$ scale down the vector of incoming weights to the allowed length
  + advantages of weight constraints over weight penalities
    + easier to set a sensible value
    + preventing hidden units getting stuck near zero
    + prevent weights exploding
  + the effectiveness of weight penalty determined by the big gradients
    + when a unit hits its limit
    + more effective than a fixed penalty at pushing irrelevant weights towards zero


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 9.3 Using noise as a regularizer

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 9.4 Introduction to the bayesian approach

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 9.5 The bayesian interpretation of weight decay

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 9.6 MacKays quick and dirty method of fixing weight costs

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


