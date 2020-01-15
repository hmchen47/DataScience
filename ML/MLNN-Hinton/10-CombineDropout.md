# 10. Model Combination and Dropouts
  
## 10.1 Why it helps to combine models

### Lecture Notes

+ Combining networks: the bias-variance trade-off
  + limited amount of training data $\implies$ overfitting
    + reducing overfitting by averaging the predictions of many different models
    + most useful w/ models making very different predictions
  + regression: squared error = "bias" term + "variance" term
    + large bias: too little capacity to fit the data
    + high variance: so much capacity that it is good at fitting the sampling error in each particular training set
  + using high variance and low bias models w/ high capacity to average out the variance

+ Combined predictor
  + on any one test case, some individual predictors may be better than the combined predictor
    + different individual predictors will be better on different cases
  + individual predictors disagree significantly $\implies$ the combined predictor typically better than all of the individual predictors when averaging over test cases
    + try to make the individual predictors disagree (w/o making them much worse individually)

+ Combining networks reduces variance
  + compare two expected squared errors
    + randomly pick a predictor
    + using the average of all the predictors: $i$ as an index over the $N$ models

    \[ \overline{y} = \;<y_i>_i \;=\; \frac{1}{N} \sum_{i=1}^{N} y_i \]

  + expected squared errors

    \[\begin{align*}
      <(t-y_i)^2>_i &= <\left( (t - \overline{y}) - (y_i - \overline{y}) \right)^2>_i \\
       &= <(t-\overline{y})^2 + (y_i - \overline{y})^2 - 2 (t-\overline{y})(y_i - \overline{y})>_i \\
       &= (t - \overline{y})^2 + <(y_i - \overline{y})^2>_i \underbrace{\;-2 \; (t - \overline{y})<(y_i - \overline{y})_i>_i}_{\text{vanished}}
    \end{align*}\]

  + Pictorial explanation (see diagram)
    + bad guy: the predictors that are further than average from $t$ make bigger than average squared errors
    + good guy: the predictors that are nearer than average to $t$ make smaller then average squared errors
    + the bad guy dominates because squares work like that
    + Don't try averaging: the nose is not Gaussian

      \[ \frac{(\overline{y} - \varepsilon)^2 + (\overline{y} + \varepsilon)^2}{2} = \overline{y}^2 + \varepsilon^2 \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
        <img src="img/m10-01.png" style="margin: 0.1em;" alt="Example of combined models" title="Example of combined models" width=200>
      </a>
    </div>

+ Discrete distributions over class labels
  + Assumption:
    + one model gives the correct label probability $p_i$
    + the other model gives the correct probability $p_j$
  + better way: randomly pick one model or averaging two probabilities

    \[ \log \left( \frac{p_i + p_j}{2} \right) \geq \frac{\log p_i + \log p_j}{2} \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
      <img src="img/m10-02.png" style="margin: 0.1em;" alt="Example of average models" title="Example of average models" width=150>
    </a>
  </div>

+ Overview of ways to make predictors differ
  + rely on the learning algorithm getting stuck in different local optima $\implies$ a dubious hack (but worth to try)
  + using different non-neural network models
    + decision trees
    + gaussian process models
    + support vector machines
    + others
  + neural network models
    + different numbers of hidden layers
    + different numbers of units per layer
    + different types of unit
    +_different types or strengths of weight penalty
    + different learning algorithms

+ Making models different  by changing the training data
  + Bagging
    + train different models on different subsets of the data
    + get different training sets by using sampling w/ replacement: $a, b, c, d, e \to a \, c \, c \, d \, d$
    + random forest: using lots of different decision trees trained using bagging
  + able to use w/ neural networks but very expensive
  + Boosting
    + train a sequence of low capacity models
    + weight the training cases differently for each model in the sequence
    + boosting up-weights cases that previous models got wrong
    + an early use of boosting was w/ neural networks for MNIST
    + focused the computational resources on modeling the tricky cases


### Lecture Video

<video src="https://youtu.be/yIIFnTkvhrQ?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 10.2 Mixtures of experts

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 10.3 The idea of full bayesian learning

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 10.4 Making full bayesian learning practical

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 10.4 Dropout an efficient way to combine neural nets

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


