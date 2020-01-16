# 10. Model Combination and Dropouts
  
## 10.1 Why it helps to combine models

### Lecture Notes

+ Combining networks: the bias-variance trade-off
  + limited amount of training data $\implies$ overfitting
    + reducing overfitting by averaging the predictions of many different models
    + most useful w/ models making very different predictions
  + regression: squared error = "bias" term + "variance" term
    + large bias:
      + model w/ too little capacity to fit the data
      + measuring how poorly the model approximates the true function
    + high variance:
      + so much capacity that it is good at fitting the sampling error in each particular training set
      + another training set w/ same size from the same distribution $\to$ fit differently due to different sampling error $\implies$ variance in the model fitting different training set
  + using high variance and high capacity (typically w/ low bias) models to average out the variance

+ Combined predictor vs individual predictors
  + on any one test case, some individual predictors may be better than the combined predictor
    + different individual predictors will be better on different cases
  + individual predictors <span style="color: red;">disagree</span> significantly
    + combined predictor typically better than all of the individual predictors when averaging over test cases
    + try to make the individual predictors disagree (w/o making them much worse individually)

+ Combining networks reduces variance
  + compare two expected squared errors
    + randomly pick a predictor to make prediction
    + average all the predictors: $i$ as an index over the $N$ models, $<\;>$ as expection

      \[ \overline{y} = \;<y_i>_i \;=\; \frac{1}{N} \sum_{i=1}^{N} y_i \]

  + expected squared errors

    \[\begin{align*}
      <(t-y_i)^2>_i &= <\left( (t - \overline{y}) - (y_i - \overline{y}) \right)^2>_i \\
       &= <(t-\overline{y})^2 + (y_i - \overline{y})^2 - 2 (t-\overline{y})(y_i - \overline{y})>_i \\
       &= (t - \overline{y})^2 + \underbrace{<(y_i - \overline{y})^2>_i}_{\text{variance of }y_i} \underbrace{\;-2 \; (t - \overline{y})<(y_i - \overline{y})_i>_i}_{\text{vanished}}
    \end{align*}\]

  + Pictorial explanation (see diagram)
    + horizontal line: the possible values of the output
    + all of the different models predict a value too high
    + bad guy: the predictors that are further than average from $t$ make bigger than average squared errors
    + good guy: the predictors that are nearer than average to $t$ make smaller then average squared errors
    + the bad guy dominates because its square error contributes more

      \[ \frac{(\overline{y} - \varepsilon)^2 + (\overline{y} + \varepsilon)^2}{2} = \overline{y}^2 + \varepsilon^2 \]

    + Don't try averaging: the nose is not Gaussian

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
        <img src="img/m10-01.png" style="margin: 0.1em;" alt="Example of combined models" title="Example of combined models" width=200>
      </a>
    </div>

+ Discrete distributions over class labels
  + Assumption:
    + one model gives the correct label probability $p_i$
    + the other model gives the correct probability $p_j$
  + better way: randomly pick one model or averaging two probabilities?

    \[ \log \left( \frac{p_i + p_j}{2} \right) \geq \frac{\log p_i + \log p_j}{2} \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
      <img src="img/m10-02.png" style="margin: 0.1em;" alt="Example of average models" title="Example of average models" width=150>
    </a>
  </div>

  + the average of $p_i$ and $p_j$ (middle point of gold line) below the blue dot due to log probability

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
    + different types of unit, e.g., rectified linear units and logistic units
    +_different types or strengths of weight penalty; e.g., early stopping, L2 penalty, and L1 penalty
    + different learning algorithms; e.g., full bach and mini-batch

+ Making models different  by changing the training data
  + Bagging
    + train different models on different subsets of the data
    + get different training sets by using sampling w/ replacement; e.g., $a, b, c, d, e \to a \, c \, c \, d \, d$
    + random forest: using lots of different decision trees trained using bagging (better result)
  + able to use w/ neural networks w/ bagging but very expensive; e.g., 20 neural nets $\implies$ 20 training and 20 testing
  + Boosting
    + train a sequence of low capacity models w/ the whole training set
    + weight the training cases differently for each model in the sequence
    + boosting up-weights cases w/ previous models got wrong
    + boosting down-weight cases w/ previous cases got right
    + use resources to try and deal w/ wrong models
    + example
      + an early use of boosting was w/ neural networks for MNIST due to low computational power
      + focused the computational resources on modeling the tricky cases


### Lecture Video

<video src="https://youtu.be/yIIFnTkvhrQ?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 10.2 Mixtures of experts

### Lecture Notes

+ Purpose of mixtures of experts
  + Better way than just averaging models
    + possible: looking at the input data for a particular case to help decide which model to rely on
    + allowing particular models to specialize in a subset of the training cases
    + not learn on cases for which they are not picked $\implies$ ignore stuff not good at modeling
    + individual model might be very good at something and very bad at other things
  + key idea
    + make each model or expert focus on predicting the right answer
    + the cases w/ right answer where it is already doing better than the other experts
    + causing specialization

+ A spectrum of models
  + Very local model (left diagram)
    + e.g., nearest neighbors
    + very fast to fit: just store training cases
    + predict $y$ from $x$ $\implies$ simply find the stored value of $x$ closest to the test value of $x$ to predict the $y$
    + local smoothing would obviously improve things
  + Fully global models (right diagram)
    + e.g., a polynomial
    + may be slow to fit and also unstable
    + small changes to data can cause big changes to the fit
    + each parameter depends on all the data

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
      <img src="img/m10-03.png" style="margin: 0.1em;" alt="Local and global models" title="Local and global models" width=450>
    </a>
  </div>

+ Multiple local models
  + in between the very local & fully global models
  + using several models of intermediate complexity than using a single global model or lots of very local models
    + good if the dataset contains several different regimes which have different relationships btw input and output
    + e.g., the state of the economy has a big effect on determining the mapping between inputs and outputs 
      + using different models for different states of the economy
      + unknown in dadvance how to decide what constitutes different states of the economy $\implies$ required to learn
  + how to partition the dataset into different regimes?

+ Datset partitioning
  + ways: based on input vs. based on the input-output relationship
  + cluster the training cases into subsets
  + one for each local model
  + aim of the clustering:
    + Not to find clusters of similar input vectors
    + each cluster to have a relationship btw input and output that can be well-modeled by one local model
  + example (see diagram)
    + four data points nicely fitted by the red parabola
    + another four data points nicely fitted by the green parabola
    + partition the data based on the input-output mapping 
      + based on the idea that a parabola will fit the data nicely
      + the brown line partitions the data
    + partition the data by just clustering the input
      + the blue line partitioning accordingly
      + the left side of the blue line $\to$ stuck w/ a sunset of data
      + unable to model nicely by a simple model

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
      <img src="img/m10-04.png" style="margin: 0.1em;" alt="Dataset partitioning" title="Dataset partitioning" width=200>
    </a>
  </div>

+ Cooperation vs. Specialization
  + error function encouraging cooperation
    + compare the average to all the predictors w/ the target
    + train all the predictors together to reduce the discrepancy btw the target and the average
    + overfit badly
      + making the model much more powerful than training each predictor separately
      + the models learn to fix up the errors that other models make

      \[ E = (t - \underbrace{<y_i>_i}_{\text{average of all}\\ \text{the predictor}})^2 \]

  + pictorial explanation: averaging models during training causes cooperation not specialization

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
        <img src="img/m10-05.png" style="margin: 0.1em;" alt="Example of predictors, target, and model output" title="Example of predictors, target, and model output" width=300>
      </a>
    </div>

    + average all models except for model $i$ on the right side
    + model $i$ alone on the left side
    + overall average to be close to the target $\implies$ move the output of model $i$ away from the target value
    + model $i$ learning to compensate for the errors made by all other models
    + really want to move the output of model $i$ in the wrong direction?
    + intitutively, it is better to move model $i$ towards the target (green arrow)

  + error function encouraging specialization
    + compare each predictor separately w/ the target
    + use a "manager" to determine the probability of picking each expert
    + most experts end up ignoring most targets
      + each expert only deal w/ a small subset of the training cases
      + good at learning w/ the small subset of data

      \[ E = <p_i(t-y_i)^2> \]

    + $p_i$: probability of the manager picking expert $i$ for this case
  
+ The mixture of experts architecture (almost)
  + a simple cost function:
    + an intuition for explanation
    + a better cost function based on a mixture model introduced later

    \[ E = \sum_i p_i (t - y_i)^2 \]

  + architecture
    + different experts (the right hand side) making their own predictions based on the input
    + the manager (the left hand side)
      + multiple layer(s)
      + the last layer: softmax
      + output: probabilities for the experts
    + using output of manager and experts to compute the value of the error function

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
        <img src="img/m10-07.png" style="margin: 0.1em;" alt="Architecture for mixture of experts" title="Architecture for mixture of experts" width=350>
      </a>
    </div>

+ The derivatives of the simple cost function
  + differentiate w.r.t. the outputs of the experts
    + a signal for training each expert
    + the gradient as the probability of picking that expert times the difference btw that expert and the target
  + differentiate w.r.t. the outputs of the gating network
    + a signal for training the gating network
    + as differentiate w.r.t. the quantity entering the softmax
    + the derivative w.r.t. $x_i$ as product of 
      + the probability of the expert picked and the difference btw the squared error made by the expert
      + the average over all experts when using the weighting provided by the manager of the squared error
    + raise $p$ for all experts that give less than the average squared error of all the experts (weighted by $p$)
      + expert $i$ makes a lower square error than the average of the other experts $\to$ raise the probability of expert $i$
      + expert $i$ makes a higher squared error than the average of the other experts $\to$ lower its probability
      + causing specialization
  + math representation

    \[ p_i = \frac{e^{x_i}}{\sum_j e^{e^{x_j}}}, \qquad\qquad E = \sum_i p_i (t-y_i)^2 \]

    \[ \frac{\partial E}{\partial y_i} = p_i (t-y_i) \qquad\qquad \frac{\partial E}{\partial x_i} = p_i \left( (t-y_i)^2 - E \right) \]

+ A better cost function for mixtures of experts
  + Jacobs, Robert & Jordan, Michael & Nowlan, Steven & Hinton, Geoffrey. (1991). [Adaptive Mixture of Local Expert](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf). Neural Computation. 3. 78-88. 10.1162/neco.1991.3.1.79.
  + each expert as making a prediction w/ a Gaussian distribution around its output (w/ variance 1)
    + assumption (see the left diagram)
      + $y_1$ as the output of a particular value w/ a unit variance Gaussian prediction (red expert)
      + $y_2$ as the prediction of another expert makes a Gaussian prediction (green expert)
  + the manager:
    + deciding on a scale for each of these Gaussian
    + the scale called a "mixing proportion"; e.g., $\{ 0.4 \; 0.6 \}$ for red and green experts respectively (see right diagram)
    + predictive distribution of mixture of expert: no longer Gaussian after summing of scaled down read Gaussian and scaled down green Gaussian
  + maximize the log probability of the target value under this mixture of Gaussian model; i.e., the sum of the two scaled Gaussian
    + max the log probability under the black curve
    + black curve as the sum of red and green curves

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture10/lec10.pptx" ismap target="_blank">
      <img src="img/m10-08.png" style="margin: 0.1em;" alt="Gaussian distributions of two models" title="Gaussian distributions of two models" width=350>
    </a>
  </div>

  + the probability of the target under a mixture of Gaussian

    \[ p(t^c | MoE) = \sum_i p_i^c \frac{1}{\sqrt{2\pi}} \exp \left(-\frac{1}{2} (t^c - y_i^c)^2 \right) \]

    + $p(t^c | MoE)$: prob. of target value on case $c$ given the mixture
    + $p_i^c$: mixing proportion assigned to expert $i$ for case $c$ by the gating network
    + $y_i^c$: output of expert $i$
    + $1/\sqrt{2 \pi}$: normoralization term for a Gaussian w/ $\sigma^2 = 1$


### Lecture Video

<video src="https://youtu.be/d_GVvIBlWtI?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 10.3 The idea of full bayesian learning

### Lecture Notes





### Lecture Video

<video src="https://youtu.be/mAoSCUZQEMY?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
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


