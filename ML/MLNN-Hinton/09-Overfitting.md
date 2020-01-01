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
    + architecture: limit the number of hidden layers and the number of units per layer
    + early stopping: start w/ small weights and stop the learning before it overfits
    + weight-decay: penalize large weights using penalties or constrains on their squared values (L2 penalty) or absolute values (L1 penalty)
    + noise: add to the weights of the activities
  + typically, using a combinaition of several of these methods

+ Meta parameters to control capacity
  + trying lots of alternatives and choosing the one w/ the best performance on the test set
    + wrong method
    + easy to do but giving false impression of how well the methods works
    + settings w/ best performance on a test set unlikely to work on new data set drawn from the same distribution
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
  + N-fold cross-validation
    + divide the total dataset into one final test set and $N$ other subsets and
    + train on all but one of those subsets to get $N$ different estimates of the validation error rate
    + the $N$ estimates not independent

+ Early stopping
  + goal: preventing overfitting
  + expensive to keep re-training the model
    + assumption: lots of data and a big model
    + training w/ different sized penalities on the weights of different architectures
  + Solution:
    + start e/ very small weights
    + grow the weights until the performance on the validation set starts getting worse
  + issues:
    + hard to decide when the performance getting worse
    + limited capacity of the model
      + weights not had time to grow big
      + smaller weights give the network less capacity

+ Why early stopping works
  + very small weights
    + every hidden unit in its linear range
    + even w/ a large layer of hidden units
    + no more capacity than a linear net (inputs directly connected to outputs)
  + as weights grow
    + hidden units start using the non-linear ranges
    + capacity grows

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture9/lec9.pptx" ismap target="_blank">
      <img src="img/m09-01.png" style="margin: 0.1em;" alt="Modeling early stopping" title="Modeling early stopping" width=200>
    </a>
  </div>


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 9.2 Limiting size of the weights

### Lecture Notes





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


