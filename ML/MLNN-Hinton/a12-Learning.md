# Fast Learning Algorithms

P. Rojas, [Chapter 8](http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf) in [Neural Networks - A Systematic Introduction](http://page.mi.fu-berlin.de/rojas/neural/), 1996


<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
    <img src="img/a12-24.png" style="margin: 0.1em;" alt="Taxonomy of learning algorithms" title="Taxonomy of learning algorithms" width=350>
  </a>
</div>

## 8.1 Introduction - Classical backpropagation

+ The reasons to explore the combinations of new algorithm in learning algorithms
  + backpropagation algorithms
  + artificial neural networks

+ Backpropagation algorithm
  + a rather slow learning algorithm
  + malicious selection of parameters made even slower
  + non-linear optimization: accelerate the training method with practically no effort

+ Artificial neural networks
  + NP-complete in the worst cases
  + network parameters grow exponentially w/ the number of unknown

+ Standard online backpropagation performs better than many fast learning algorithms when
  + a realistic level of complexity in the learning task
  + the size of the training set beyond a critical threshold



### 8.1.1 Backpropagation with momentum



### 8.1.2 The fractal geometry of backpropagation



## 8.2 Some simple improvements to backpropagation



### 8.2.1 Initial weight selection



### 8.2.2 Clipped derivatives and offset term



### 8.2.3 Reducing the number of floating-point operations



### 8.2.4 Data decorrelation



## 8.3 Adaptive step algorithms



### 8.3.1 Silva and Almeida´s algorithm



### 8.3.2 Delta-bar-delta



### 8.3.3 RPROP



### 8.3.4 The Dynamic Adaption Algorithm



## 8.4 Second-order algorithms



### 8.4.1 Quickprop



### 8.4.2 Second-order backpropagation



## 8.5 Relaxation methods



### 8.5.1 Weight and node perturbation



### 8.5.2 Symmetric and asymmetric relaxation



### 8.5.3 A final thought on taxonomy



### 8.6 Historical and bibliographical remarks






