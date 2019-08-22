# Delta Learning Rule & Gradient Descent | Neural Networks

Author: Random Nerd

[URL](https://medium.com/@neuralnets/delta-learning-rule-gradient-descent-neural-networks-f880c168a804)

Date: Apr. 19, 2018

## Introduction

+ Delta Rule
  + a mathematically-derived (and thus potentially more flexible and powerful) rule for learning
  + a.k.a. Widrow & Hoff Learning rule or the Least Mean Square (LMS) rule
  + developed early 1960’s by Widrow and Hoff
  + similar to the perceptron learning rule by McClelland & Rumelhart, 1988.
  + using the difference between target activation (i.e., target output values) and obtained activation to drive learning
  + Linear Activation function: the output node’s activation is simply equal to the sum of the network’s respective input/weight products


## Model for Neural Networks

+ Model used
  + two-layer network capable of deploying the Delta Rule

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://medium.com/@neuralnets/delta-learning-rule-gradient-descent-neural-networks-f880c168a804" ismap target="_blank">
      <img src="https://miro.medium.com/max/1156/1*mKJG6c8gHPOpc6oCf7DlHw.jpeg" style="margin: 0.1em;" alt="A graphical depiction of a simple two-layer network capable of deploying the Delta Rule" title="A graphical depiction of a simple two-layer network capable of deploying the Delta Rule" width=350>
    </a>
  </div>

+ Forward propagation:
  + the output (activation) of a given node is a function of its inputs
  
    $$S_j = \sum_i w_{ij}a_i \quad \text{and} \quad a_j = f(S_i)$$

  + $S_j$: the sum of all relevant products of weights
  + $w_{ij}$: the relevant weights connecting layer $i$ with layer $j$
  + $a_i$: the activation of node in the previous layer $i$
  + $a_j$: the activation of the node at hand
  + $f(.)$: the activation function

+ Gradient Descent Learning  
  + modification of weights along the most direct path in weight-space to minimize error
  + applied to a given weight is proportional to the negative of the derivative of the error with respect to that weight





