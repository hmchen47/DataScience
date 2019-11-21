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
      <img src="https://miro.medium.com/max/1156/1*mKJG6c8gHPOpc6oCf7DlHw.jpeg" style="margin: 0.1em;" alt="A graphical depiction of a simple two-layer network capable of deploying the Delta Rule" title="A graphical depiction of a simple two-layer network capable of deploying the Delta Rule" width=250>
    </a>
  </div>

+ Similar Model representation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.ebc.cat/2017/01/08/understanding-neural-networks-part-2-vectorized-forward-propagation/" ismap target="_blank">
      <img src="http://www.ebc.cat/wp-content/uploads/2017/01/notation_forw_prop_vect.png" style="margin: 0.1em;" alt="Notation for Vectorization of the Forward Propagation Algorithm (single unit)" title="Notation for Vectorization of the Forward Propagation Algorithm (single unit)" width=300>
      <img src="http://www.ebc.cat/wp-content/uploads/2017/01/forw_prop_with_notation.png" style="margin: 0.1em; margin-left: 2em;" alt="Forward Propagation with formal notation" title="Forward Propagation with formal notation" width=450>
      <img src="http://www.ebc.cat/wp-content/uploads/2017/01/forw_prop_vectorization_matrices.png" style="margin: 0.1em;" alt="Forward Propagation Vectorization Matrices" title="Forward Propagation Vectorization Matrices" width=850>
    </a>
  </div>

  + $a_i^{(n)}$: value if the unit $i$ in layer $n$
  + $z_i^{(n)}$: internal value in each unit that sums all (weighted) inputs
  + $g(Z)$: activation function (usually sigmoid function)
  + $\theta_{ab}^{(n)}$: weight of the link that connects unit $b$ in layer $n$ to unit $a$ in layer $n+1$ (note $a$ and $b$ are in the opposite order)
  + $X_m$: value of input sample $m$
  + $h_k$: value of output prediction $k$


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


## Error Function and Derivatives

+ Error function/cost function

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://medium.com/@neuralnets/delta-learning-rule-gradient-descent-neural-networks-f880c168a804" ismap target="_blank">
      <img src="https://miro.medium.com/max/1100/1*nAvwz7r9xfYzYyQhAEwNog.jpeg" style="margin: 0.1em;" alt="Error function with just 2 weights w1 and w2" title="Error function with just 2 weights w1 and w2" width=350>
    </a>
  </div>

  + commonly given as the sum of the squares of the differences between all target and actual node activation for the output layer
  + a particular training pattern

    $$E_p = \frac{1}{2} \sum_n (t_{j_n} - a_{j_n})^2$$

    + $E_p$: total error over the training pattern
    + $1/2$: value apllied to simplify the function derivative
    + $n$: all output nodes for a given training pattern
    + $t_{j_n}$: the target value for node $n$ in output layer $j$
    + $a_{j_n}$: the actual activation for node $n$ in output layer $j$

  + Error over an entire set of training patterns (i.e., over one iteration, or epoch)

    $$E = \sum_p E_p = \frac{1}{2} \sum_p \sum_n (t_{j_n} - a_{j_n})^2$$

    + $E$: total error
    + $p$: all training pattern

  + Normalized error function: Mean Squared Error (MSE)

    $$MSE = \frac{1}{2 P N} \sum_p \sum_n (t_{j_n} - a_{j_n})^2$$

    + $P$: total number of training patterns
    + $N$: total number of output nodes
  
  + Total Sum of Squares (tss) error: the sum of all squared errors over all output nodes and all training patterns
  + The negative of the derivative of the error function is required in order to perform Gradient Descent Learning.

+ Derivative of $E_p$ w.r.t. $w_{ij_x}$
  + Applied chain rule

    $$\frac{\partial  E_p}{\partial {w_{ij_x}}} = \frac{\partial E_p}{\partial a_{j_z}} \frac{\partial a_{j_z}}{\partial w_{ij_x}}$$

    + $a_{j_z}$: activation of the node in the output layer 
    + $w_{ij_x}$: weight with
      + $i$ and $j$ refer to a particular layers of nodes or weights
      + $x$ refers to individual weights and node within these layers

  + extend derivative

    $$\frac{\partial E_p}{\partial a_{j_z}} = (2)(\frac{1}{2})(t_{j_z} - a _{j_z})(-1) = -(t_{j_z} - a_{j_z})$$

    and

    $$\frac{\partial a_{j_z}}{\partial w_{ij_x}} = \frac{\partial}{\partial w_{ij_x}} \sum_n (w_{ij_n} a_{i_n}) = \frac{\partial }{\partial  w_{ij_x}}(w_{ij_0} a_{i_0} + w_{ij_1}a_{i_1}+ \cdots + w_{ij_n} a_{i_n}) = a_{i_x}$$

  + Therefore,

    $$\frac{\partial E_p}{\partial w_{ij_x}} = -(t_{j_z} - a _{j_z})(a_{i_x})$$


## Delta Rule

+ Delta rule for perceptrons
  + Gradient descent learning requires that any change in a particular weight be proportional to the negative of the derivative of the error
  + The change in a given weight must be proportional to the negative of the prior equation

  $$\Delta w_{ij_x} = - \varepsilon \frac{\partial E}{\partial w_{ij}} = \varepsilon \partial a_{i_x}$$

  + $\Delta$: the difference btw the target and actual activation of the relevant output node
  + $\varepsilon$: learning rate

+ Linear Activation function instead of a Threshold Activation function
  + Threshold Activation function:
    + characterize both the McColloch and Pitts network and the perceptron is not differentiable at the transition between the activations of $0$ and $1$ (slope = $\infty$) 
    + derivative = 0 over the remainder of the function
  + Linear Activation function or other differentiable functions
    + allow the derivative of the error to be calculable

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://medium.com/@neuralnets/delta-learning-rule-gradient-descent-neural-networks-f880c168a804" ismap target="_blank">
      <img src="https://miro.medium.com/max/1191/1*t9qCtZ_hMGTYiEGt7kwT6w.jpeg" style="margin: 0.1em;" alt="Three-dimensional depiction of an Actual error surface (Leverington, 2001)" title="Three-dimensional depiction of an Actual error surface (Leverington, 2001)" height=240>
      <img src="https://miro.medium.com/max/1139/1*jnG5--4AcQSNHaFqquiu7A.jpeg" style="margin: 0.1em;" alt="Two-dimensional depiction of the error surface" title="Two-dimensional depiction of the error surface" height=240>
    </a>
  </div>



