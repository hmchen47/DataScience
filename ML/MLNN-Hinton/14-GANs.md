# 14. Generative Adversarial Networks (GANs)
  
## 14.1 Learning layers of features by stacking RBMs

### Lecture Notes

+ Stacking RBMs as deep belief net
  + procedure
    + training a layer of features w/ input directly from the pixels
    + treating the activations of the trained features as if they were pixels and learn features from features
    + repeat the steps
  + each new layer of features by modeling the correlated activity in the feature in the layer below
  + adding another layer of features
    + improving a variatonal lower bound on the log probability of generating the training data
    + complicated proof and only applied to unreal cases
    + proof based on a neat equivalence btw an RBM and an infinitely deep belief net
  + example: combining two RBMs to make a DBN
    + left diagram
      + stacked RBMs $\to$ deep belief net
      + learning RBM ($v \stackrel{w_1}{\longleftrightarrow} h_1$) w/ its own weight
      + once trained, hidden activity patterns as data for the 2nd RBM ($h_1 \stackrel{w_2}{\longleftrightarrow} h_2$)
      + starting the 2nd RBM off w/ $w_2$ being transpose of $w_1$ and same number of hidden units as $v$
      + the 2nd RBM as good model of $h_1 \to$ the 1st RBM upside down $\implies$ bipartite graph (no need to distinguish layers for visible and hidden units)
    + compose both RBMs (left diagram) to form a single model (right diagram)
    + right diagram
      + top two layers = 2nd RBM $\implies$ undirected model w/ symmetric actions
      + bottom two layers = directed model as a sigmoid belief net $\implies$ remove symmetry connection btw $v$ and $h_1 \to$ remove upgoing part
      + not a RBM $\to$ deep belief net
      + lower layers as a sigmoid belief nets while the top two layers forming a RBM $\to$ a hybrid model

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://bit.ly/2JpLNti" ismap target="_blank">
        <img src="img/m14-01.png" style="margin: 0.1em;" alt="Basic RBM to DBN" title="Basic RBM to DBN" width=450>
      </a>
    </div>

+ Generative model w/ 3 layers
  + stacked up 3 RBMs (diagram)
    + $h_2 \stackrel{w_2}{\longleftrightarrow} h_3$ as a RBM
    + $data$ and $h_1$ forming directed layers as in a sigmoid belief nets
  + procedure to generate data
    + get an equilibrium sample from the top-level RBM by performing alternating Gibbs sampling for a long time
      + back and forth btw $h_2$ and $h_3$ to reach equilibrium w/ top-level RBM
      + involving alternating Gibbs sampling
      + update all units in $h_3$ in parallel then update all units in $h_2$ in parallel then ...
    + perform a top-down pass to get states for all the other layers
      + top-level RBM = the prior distribution over $h_2$
      + $h_2 \xrightarrow[\text{generativ connecction}]{w_2} h_1 \xrightarrow{w_1} data \implies$ performing as a belief net
  + the lower bottom-up connections (transposes of corresponding weights) not part of the generative model $\to$ used for inference
  + Gibbs sampling
    + generating posterior samples by sweeping through each variables (or block of variables) to sample from its conditional distribution w/ the remaining variables fixed to their current values
    + a MCMC algorithm for obtaining a sequence of observations, approximately from a specified multivariate probability distribution

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/2JpLNti" ismap target="_blank">
      <img src="img/m14-02.png" style="margin: 0.1em;" alt="DBN w/ 3 stacked RBM" title="DBN w/ 3 stacked RBM" width=100>
    </a>
  </div>

+ Averaging factorial distributions
  + average some factorial distributions
    + not a factorial distribution
    + in an RBM, the posterior over 4 hidden units is factorial <span style="color: red;">for each visible vector</span>
  + example
    + 4 units w/ the probabilities to turn on each unit
    + posterior for v1: (0.9, 0.9, 0.1, 0.1)
    + posterior for v2: (0.1, 0.1, 0.9, 0.9)
    + averaged $\neq$ (0.5, 0.5, 0.5, 0.5)
  + consider the binary vector (1, 1, 0, 0)
    + posterior for v1: $p(1, 1, 0, 0) = 0.9 \times 0.9 \times (1-0.1) \times (1-0.1) = 0.9^4 = 0.43$
    + posterior for v2: $p(1, 1, 0, 0) = 0.1^4 = 0.0001$
    + aggregated posterior: $p(1, 1, 0, 0) = 0.215$
  + aggregated posterior: factorial w/ $p=0.5^4 \implies$ aggregated posterior not factorial distribution

+ Mechanism in greedy learning
  + weights, $w$, in the bottom level RBM define many different distribution
    + $p(v|h)$: the probability of visible  units given the hidden units
    + $p(h|v)$: the probability of hidden units given the visible units
    + $p(h)$: the probability of hidden units
    + $p(v)$$: the probability of visible units
  + process and restriction
    + $p(v|h)$ and $p(h|v)$ used for running alternative Markov Chain updates
    + a long sequence of updates $\to$ a sample from the joint distribution of $v$ and $h$
    + the weights also clearly defined the joint distribution
    + the weights also define the joint distribution more directly in terms of $e^{-E}$
    + not for nets w/ a large number of units
    + w/ the joint distribution of $v$ and $h$
      + ignoring $v \to$ distribution for $h$ = the prior distribution defined by this RBM
      + ignoring $h \to$ the prior distribution over $v$ defined by the RBM
  + the RBM models: $p(v) = \sum_h p(h) p(v|h)$
  + $p(v|h)$ fixed, improve $p(h) \implies p(v)$ improved
    + learn new parameters giving a better model of $p(h)$
    + substitute $p(h)$ in stead of the old model out of $p(h) \to$ improve model over  $v$
  + to improve $p(h)$, a better model than $p(h; w)$ of the <span style="color: blue;">aggregated posterior</span> distribution over hidden vectors produced by applying $W$ transpose to the data
    + better model $p(h)$: fitting the aggregated posterior better
    + the aggregated posterior: the average over all vectors in the training set of the posterior distribution over $h$
    + using the 1st RBM to get aggregated posterior
    + using the 2nd RBM to build a better model of this aggregated posterior $\to$ 1st RBM reaching a better model
    + start w/ 2nd RBM as the previous but upside down
    + start w/ the same model of the aggregated posterior as the 1st RBM has
    + changing the weights $\implies$ better fit 

+ Contrastive version of the wake-sleep algorithm <br/> after learning many layers of features, fine-tune the features to improve generation
  1. do a stochastic bottom-up pass
    + adjust the top-bottom generative weights of lower layer to be good at reconstructing the feature activities in the layer below
    + a standard wake-sleep algorithm
  2. do a few iterations of sampling in the top level RBM
    + adjust the weights in the top-level RBM using contrasted divergence (CD)
    + working as a regular algorithm for RBM
    + update the weight of the RBM using the difference btw 
      + the correlations when activity first got to that RBM
      + the correlations after a few iterations in that RBM
  3. do a stochastic top-down pass
    + adjust the bottom-up weights to be good at reconstructing the feature activities in the layer above
    + using the directed lower connections to generate some data from that sigmoid belief net
    + the sleep phase of the wake-sleep algorithm
  + differences
    + top-level RBM acts as a much better prior over the top layers than just a layer of units assumed to be independent
    + rather than generating data by sampling from the prior, looking at a training case going up to the top-level RBM and just running a few iterations before generating data

+ Example: Modeling with the DBN on MNIST digits
  + first two hidden layers learned w/o labels
    + bottom two layers involved
    + task: modeling all 10 digit classes in images of 28 by 28 pixels (right bottom input box)
    + learning RBM w/o knowing labels $\to$ unsupervised learning
    + taking the patterns of activities in the hidden units
    + treating the patterns of activity as data
    + learn another RBM (2nd hidden layer - 500 units) w/o knowing the labels
  + top layer learned as an RBM for modeling the labels concatenated w/ the features in the second hidden layer
    + adding a big top layer w/ 2000 hidden units (top layer)
    + providing 10 labels (left 2nd top boxes)
    + concatenating those 10 labels w/ the 500 units representing features $\to$ really w/ one softmax unit
    + training top-level RBM to model the concatenation of the softmax unit for the 10 labels w/ the 500 feature activities reproduced by the two layers below
  + fine-tuning weights as a better generative model using contrastive wake-sleep
    + once top layer trained, find-tune the whole system by using a trusty wake-sleep algorithm
    + retain a good generative model

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/2JpLNti" ismap target="_blank">
      <img src="img/m14-03.png" style="margin: 0.1em;" alt="DBN used for modeling the joint distribution of MNIST digits and their labels" title="DBN used for modeling the joint distribution of MNIST digits and their labels" width=200>
    </a>
  </div>

### Lecture Video

<a href="https://bit.ly/2JvxsLP" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 14.2 Discriminative fine-tuning for DBNs


### Lecture Notes





### Lecture Video

<a href="https://bit.ly/33VvMEQ" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 14.3 What happens during discriminative fine-tuning


### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 14.4 Modeling real-valued data with an RBM


### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 14.5 RBMs are infinite sigmoid belief nets


### Lecture Notes




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


