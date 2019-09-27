# Fast Learning Algorithms

P. Rojas, [Chapter 8](http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf) in [Neural Networks - A Systematic Introduction](http://page.mi.fu-berlin.de/rojas/neural/), 1996


<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
    <img src="img/a12-24.png" style="margin: 0.1em;" alt="Taxonomy of learning algorithms" title="Taxonomy of learning algorithms" width=450>
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

+ Momentum method
  + minimizing the error function: wide oscillation of the search process w/ the gradient descent
  + traditional gradient descent: computed for each new combination of weights
  + momentum approach: compute the negative gradient direction a weighted average of the current gradient and the previous correction direction for each step
  + Example: a network w/ two weights $w_1$ and $w_2$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="img/a12-01.png" style="margin: 0.1em;" alt="Backpropagation without (a) or with (b) momentum" title="Backpropagation without (a) or with (b) momentum" width=350>
    </a>
  </div>
  + accelerating convergence: increasing the learning rate up to an optimal value
  + purpose: allowing the attenuation of oscillations in the iteration process

+ Mathematical representation
  + A network with $n$ different weights $w_1, w_2, \dots, w_n$
  + Assumption and Notations
    + $E$: the error function
    + $\gamma$: the learning rate
    + $\alpha$: the momentum rate
    + 
  + The $i$-th correction for weight $w_k$

    \[\Delta w_k(i) = -\gamma \, \frac{\partial E}{\partial w_k} + \alpha \, \Delta w_k (i-1)\]

+ Optimization
  + optimal parameters highly depends on the learning task
  + no general strategy to deal with the problem
  + tradeoffs: choosing the a specific learning and momentum rates
  + observing the oscillating behavior on backpropagation feedback rule and large momentum rates


#### The linear associator

+ Linear associator
  + a single computing element with associated weights $w_1, w_2, \dots, w_n$
  + input: $x_1, x_2, \dots, x_n$
  + output: $w_1x_1 + w_2x_2 + \cdots + w_n x_n$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="img/a12-02.png" style="margin: 0.1em;" alt="Linear associator" title="Linear associator" width=200>
    </a>
  </div>

+ Mathematical Representation
  + Assumptions & Notations
    + $(\mathbf{x_1}, y_1), (\mathbf{x_2}, y_2), \dots, (\mathbf{x_p}, y_p)$: the input-output $p$ ordered pairs
    + $\mathbf{x}$: vector of input patterns w/ $n$-dimensional rows
    + $\mathbf{w}$: vector of the weights of the linear associator w/ $n$-dimensional columns
    + $\mathbf{X}$: a $p \times m$ matrix w/ $\mathbf{x_1}, \mathbf{x_2}, \dots \mathbf{x_p}$ as rows
    + $\mathbf{y}$: a column vector of the scalars $y_1, y_2, \dots, y_p$
  + the learning task objective: minimize the quadratic error

    \[\begin{align*}
    E &= \sum_{i=1}^{n} \| \mathbf{x_i} \cdot \mathbf{w} - y_i \|^2 \\
      &= \| \mathbf{X}\mathbf{w} - \mathbf{y} \|^2 = (\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y}) \\
      &= \mathbf{w}(\mathbf{X}^T \mathbf{X})\mathbf{w} -2 \: \mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{y}^T\mathbf{y}
    \end{align*}\]

  + the lengths of the principal axes: determined by the magnitude of the eigenvalues of the correlation of matrix $\math{X}^T\mathbf{X}
  + gradient descent: most effective w/ the same length when the principal axes of the quadratic form

+ Example: quadratic function $ax^2 + b y^2$
  + iteration rule of the gradient descent w/ the $x$ and $y$ direction, respectively
  
    \[\begin{align*}
      \Delta x(i) &= -2 \, \gamma \, a x + \alpha \, \alpha x(i-1) \\
      \Delta y(i) &= -2 \, \gamma \, b x + \alpha \, \alpha y(i-1)
    \end{align*}\]

  + an optimal parameter combination
    + in the $x$ direction: $\gamma = 1/2a$ and $\alpha = 0$
    + in the $y$ direction: $\gamma = 1/2b$ and $\alpha = 0$
  + Example
    + The number of iterations needed to find the minimum of the error function to a given precision as a function of $\gamma$, when $a = 0.9$ nd $b = 0.5$.
    + optimal value: $\gamma = 0.7$ at the intersection of the two curves.
    + oscillations in $y$ direction $\rightarrow$ slow convergence in $x$ direction

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
        <img src="img/a12-03.png" style="margin: 0.1em;" alt="Optimal gamma in the two-dimensional case" title="Optimal gamma in the two-dimensional case" width=300>
      </a>
    </div>


#### Minimizing oscillations

+ Eigenvlaues in the correlation matrix $\mathbf{X}^T\mathbf{X}$
  + the eigenvalues $\rightarrow$ the lengths of the principal axes of the error function
  + the range of possible values of $\gamma$ reduces as one of these eigenvalues much larger than the others

+ Example: $kx^2$
  + optimal learning rate: $\gamma = 1/2k$
  + the rate $\gamma = 1/k \rightarrow$ oscillation btw the initial point $x_0$ and $-x_0$
  + $\gamma > 2/k \rightarrow$ "explosion" of iteration process

+ Convergence and Divergence zones
  + parameters combinations in the boundary btw regions: stable oscillations
  + $\gamma > 4 \cdot 2/k$: not balanced with any value of $\alpha$
  + $\gamma > 1$: a geometric explosion of the iteration process
  + $1/k < \gamma < 2/k$: stable oscillation; the boundaries between regions
  + $\gamma < 1/2k$: optimal convergence speed w/ a unique $\alpha$
  + jagged line: the optimal combinations of $\gamma$ and $\alpha$
  + issue: With certain direction in weight space, the principle axis of the error function is much smaller compared to another axes.
  + solution: a compromise by adjusting the momentum rate such that the direction w/ less oscillating and improving convergence speed
  + the compromise could be dominated by a single direction w/ $n$ axes in weight space

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="img/a12-04.png" style="margin: 0.1em;" alt="Convergence zone for combinations of gamma and alpha" title="Convergence zone for combinations of gamma and alpha" width=300>
    </a>
  </div>




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






