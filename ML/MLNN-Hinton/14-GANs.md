# 14. Generative Adversarial Networks (GANs)
  
## 14.1 Learning layers of features by stacking RBMs

### Lecture Notes

+ Deep network w/ stacking RBMs
  + procedure
    + training a layer of features w/ input directly from the pixels
    + treating the activations of the trained features as if they were pixels and learn features from features
    + repeat the steps
  + each new layer of features by modeling the correlated activity in the feature in the layer below
  + adding another layer of features
    + improving a variatonal lower bound on the log probability of generating the training data
    + complicated proof and only applied to unreal cases
    + based on a neat equivalence btw an RBM and an infinitely deep belief net

+ Combining two RBMs to make a DBM

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/2JpLNti" ismap target="_blank">
      <img src="img/m14-01.png" style="margin: 0.1em;" alt="text" title="caption" width=350>
    </a>
  </div>

+ Generative model w/ 3 layers
  + to generate data
    + get an equilibrium sample from the top-level RBM by performing alternating Gibbs sampling for a long time
    + perform a top-down pass to get states for all the other layers
  + the lower bottom-up connections are not part of the generative model $\to$ used for inference

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/2JpLNti" ismap target="_blank">
      <img src="img/m14-02.png" style="margin: 0.1em;" alt="text" title="caption" width=100>
    </a>
  </div>

+ Averaging factorial distributions
  + average some factorial distributions
    + not a factorial distribution
    + in an RBM, the posterior over 4 hidden units is factorial for each visible vector
  + example
    + posterior for v1: (0.9, 0.9, 0.1, 0.1)
    + posterior for v2: (0.1, 0.1, 0.9, 0.9)
    + averageed   /=    (0.5, 0.5, 0.5, 0.5)
  + consider the binary vector (1, 1, 0, 0)
    + posterior for v1: $p(1, 1, 0, 0) = 0.9^4 = 0.43$
    + posterior for v2: $p(1, 1, 0, 0) = 0.1^4 = 0.0001$
    + aggregated posterior: p(1, 1, 0, 0) = 0.215$
  + aggregated posterior: factorial w/ $p=0.5^4$

+ Mechanism in greedy learning
  + weights, $W$, in the bottom level RBM define many different distribution: $p(v|h); p(h|v), p(h); p(v)$
  + the RBM models: $p(v) = \sum_h p(h) p(v|h)$
  + $p(v|h)$ fixed, improve $p(h) \implies p(v)$ improved
  + to improve $p(h)$, a better model than $h(h; W)$ of the <span style="color: blue;">aggregated posterior</span> distribution over hidden vectors produced by applying $W$ transpose to the data

+ Contrastive version of the wake-sleep algorithm <br/> after learning many layers of features,fine-tune the features to improve generation
  1. do a stochastic bottom-up pass
    + adjust the top-bottom weights of lower layer to be good at reconstructing the feature activities in the layer below
  2. do a few iterations of sampling in the top level RBM
    + adjust the weights in the top-level RBM using CD
  3. do a stochastic top-down pass
    + adjust the bottom-up weights to be good at reconstructing the feature activities in the layer above

+ Modeling with the DBN
  + first two hidden layers learned w/o using labels
  + top layer learned as an RBM for modeling the labels concatenated w/ the features in the second hidden layer
  + fine-tuning weights as a better generative model using contrastive wake-sleep


### Lecture Video

<a href="https://bit.ly/2JvxsLP" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 14.2 Discriminative fine-tuning for DBNs


### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
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


