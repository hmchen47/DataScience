
# Adaptive Learning Rates

## Overview

+ [The intuition behind separate adaptive learning rates](../ML/MLNN-Hinton/06-MiniBatch.md#adaptive-learning-rate-for-each-connection)
  + appropriate learning rates vary widely between weights
    + different layers w/ different magnitudes of the gradients, in particular, small initial weights
    + overshoot effect: simultaneously changing many of the incoming weights of a unit to correct the same error
  + using a global learning rate (set by hand) multiplied by an appropriate local gain determined empirically for each weight

+ [Approach to determine the individual learning rates](../ML/MLNN-Hinton/06-MiniBatch.md#adaptive-learning-rate-for-each-connection)
  + starting with a local gain of 1 for every weight
    + changing the weight $w_{ij}$ by the learning rate times the gain of $g_{ij}$ times the error derivative for that weight
  + increasing the local gain $g_{ij}$ if the gradient not changing sign for that weight
  + using small additive increases and multiplicative decreases (for mini-batch)
    + ensuring big gains decay rapidly when oscillations start
    + the gain hovering around 1

  \[\begin{align*}
    \Delta w_{ij} &= -\varepsilon \, g_{ij} \; \frac{\partial E}{\partial w_{ij}} \\\\
    \text{if } & \; \left( \frac{\partial E}{\partial{w_{ij}}}(t) \frac{\partial E}{\partial w_{ij}} (t-1) \right) > 0 \\
    \text{then } & \; g_{ij}(t) = g_{ij}(t-1) + .05 \\
    \text{else } & \; g_{ij}(t) = g_{ij}(t-1) \times .95
  \end{align*}\]

+ [Tricks for making adaptive learning rates work better](../ML/MLNN-Hinton/06-MiniBatch.md#adaptive-learning-rate-for-each-connection)
  + limit the gains to lie in some reasonable range
  + designed for full batch learning or big min-batches
  + Adaptive learning rates combined with momentum
  + Adaptive learning rates only dealt with axis-aligned effects

+ [Per Connection Adaptive Learning Rate](https://trongr.github.io/neural-network-course/neuralnetworks.html)

  __Definition__. Define a gain parameter $g_{ij}$ for each connection

  \[\Delta w_{ij} = - \varepsilon g_{ij} \frac{\partial E}{\partial w_{ij}}\]

  and increase $g_{ij}$ if the gradient does not change signs, otherwise decrease it:

  \[g_{ij} = \begin{cases}
    g_{ij}(t-1) + \alpha & \text{if } \frac{\partial E}{\partial w_{ij}}(t) \frac{\partial E}{\partial w_{ij}} (t-1) > 0 \\
    g_{ij}(t-1) \times \beta & \text{otw}
  \end{cases}\]

  where $\alpha$ is close to 0 and $\beta$ is close to 1, e.g., 0.05 and 0.95 resp.

+ [Basic strategy](../ML/MLNN-Hinton/a12-Learning.md#83-adaptive-step-algorithms)
  + increasing step size: the algorithm proceeds down the error function over several iterations
  + decreasing step size: the algorithm jumps over a valley of the error function

+ [Learning rates](../ML/MLNN-Hinton/a12-Learning.md#83-adaptive-step-algorithms)
  + global learning rate
  + local learning rate

+ [Model for algorithms](../ML/MLNN-Hinton/a12-Learning.md#83-adaptive-step-algorithms)
  + each weight $w_i$ w/ an associated learning constant $\gamma_i$

    \[\Delta w_i = -\gamma_i \frac{\partial E}{\partial w_i}\]

  + motivation: use of different learning constants for each weight to correct the direction of the negative gradient to make it point directly to the minimum of the error function
  + with degenerate function, the gradient direction leads to many oscillations
  + adequate scaling of the gradient components: reaching the minimum in fewer steps


## Second-order algorithms

+ [Idea](../ML/MLNN-Hinton/a12-Learning.md#84-second-order-algorithms)
  + considering more information about the shape of the error function than the mere value of the gradient
  + considering the curvature of the error function at each step for better performance

+ [Quadratic approximation of the error function](../ML/MLNN-Hinton/a12-Learning.md#84-second-order-algorithms)
  + Assumptions & Notations
    + Quadratic error function
    + $\mathbf{w}$: the vector denoting all weights of a network
    + $E(\mathbf{w})$: the error function
    + $\Delta^2 E(\mathbf{w})$: the $n \times n$ Hessian matrix of second order partial derivatives
    + $\mathbf{w}^{(k)}$: the wright vector at the $k$-th iteration
  + Truncated Taylor series to approximate the error function $E$

    \[E(\mathbf{w} + \mathbf{h}) \approx E(\mathbf{w)} + \Delta E(\mathbf{w})^T \mathbf{h} + \frac{1}{2} \mathbf{h}^T \Delta^2 E(\mathbf{w}) \mathbf{h} \tag{6}\]
  
  + the $n \times n$ Hessian matrix of second order partial derivatives, $\Delta^2 E(\mathbf{w})$

    \[\Delta^2 E(\mathbf{w}) = \begin{bmatrix}
      \frac{\partial^2 E(\mathbf{w})}{\partial w_1^2} & \frac{\partial^2 E(\mathbf{w})}{\partial w_1 w_2} & \cdots & \frac{\partial^2 E(\mathbf{w})}{\partial w_1 w_n} \\
      \frac{\partial^2 E(\mathbf{w})}{\partial w_2 w_1} & \frac{\partial^2 E(\mathbf{w})}{\partial w_2^2} & \cdots & \frac{\partial^2 E(\mathbf{w})}{\partial w_2 w_n} \\
      \vdots & \vdots & \ddots & \vdots \\
      \frac{\partial^2 E(\mathbf{w})}{\partial w_n w_1} & \frac{\partial^2 E(\mathbf{w})}{\partial w_n w_2} & \cdots & \frac{\partial^2 E(\mathbf{w})}{\partial w_n^2}
    \end{bmatrix}\]

  + [The gradient of the error function](../ML/MLNN-Hinton/a12-Learning.md#84-second-order-algorithms)

    \[\Delta E(\mathbf{w} + \mathbf{h})^T \approx \Delta E(\mathbf{w}^T + \mathbf{h}^2 \Delta^2 E(\mathbf{w})\]

  + Solving Eq.(6) = 0

    \[\mathbf{h} = -(\Delta^2 E(\mathbf{w}))^{-1} \Delta E(\mathbf{w}) \tag{7}\]
  
  + The solution can be done in a single step if the Hessian matrix and the gradient have computed
  + Applying Newton's method to get the $(k+1)$-th iteration

    \[\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - (\Delta^2 E(\mathbf{w}))^{-1} \Delta E(\mathbf{w}) \tag{8}\]

  + Eq. (8) as a position where the gradient w/ reduced magnitude
  + Retain the minimum of the error function with several iterations

+ [Computing Hessian matrix, $\Delta^2 E(\mathbf{w})$](../ML/MLNN-Hinton/a12-Learning.md#84-second-order-algorithms)
  + Becker, S., and Y. le Cun (1989), “Improving the Convergence of BackPropagation Learning with Second Order Methods”, in: [Touretzky et al. 1989], pp. 29–37.
  + a difficult task
  + requiring the inverse of the Hessian matrix
  + many proposals to approximate the second-order information using certain heuristic
  + Pseudo-Newton's methods
    + variants of Newton's method working w/ a simplified form of the Hessian matrix
    + non-diagonal elements set all zeros and only diagonal elements computed
    + therefor, only computing $\partial^2 E(\mathbf{w})/\partial w_i^2$
    + Simplifying Eq. (8) to 

      \[w_i^{(k+1)} = w_i^{(k)} - \frac{\Delta_i E(\mathbf{w})}{\partial^2 E(\mathbf{w})/\partial w_i^2} \tag{9}\]

    + No inverse of the Hessian matrix required
    + limited effort in finding the second order partial derivatives
    + working well w/ nice quadratic error function
    + extremely large corrections required if a small second-order partial derivatives


## Cyclical Learning Rates

+ [Cyclical learning rates](../ML/MLNN-Hinton/a14-Advanced.md#31-cyclical-learning-rates-for-neural-networks)
  + main use: escape local extreme points, especially sharp local minima (overfitting)
  + saddle points:
    + abundant in high dimensions
    + convergence very slow if not impossible
  + increasing the learning rate periodically
    + a short term negative effects but help to achieve a long-term beneficial effect
  + decreasing the learning rate
    + reduce error towards the end

+ [Examples of cyclical learning rates](../ML/MLNN-Hinton/a14-Advanced.md#31-cyclical-learning-rates-for-neural-networks)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/780/1*kk4fq-pLk95ZWN6KyXM3RA.png" style="margin: 0.1em;" alt="Examples of cyclical learning rates" title="Examples of cyclical learning rates" width=250>
    </a>
  </div>

+ Limitations: what learning rate scheme set and the magnitude of these learning rates


## Estimating the Learning Rate

+ [Learning rate estimation](../ML/MLNN-Hinton/a14-Advanced.md#32-estimating-the-learning-rate)
  + starting w/ a small learning rate and increasing it on every batch exponentially
  + computing the loss function on a validation set
  + working for finding bounds for cyclic learning rates

+ [Learning rates and loss function](../ML/MLNN-Hinton/a14-Advanced.md#32-estimating-the-learning-rate)
  + the cliff region in between the two extremes
  + steadily decreasing and stable learning occurring

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/933/1*J-PZ-RanI1Ve-_kbuYzghQ.png" style="margin: 0.1em;" alt="Exponentially increasing learning rate across epochs" title="Exponentially increasing learning rate across epochs" height=150>
      <img src="https://miro.medium.com/max/1220/1*Pep5Xicj_1C1WhFQAgL-nA.png" style="margin: 0.1em;" alt="Loss function as a function of learning rate" title="Loss function as a function of learning rate" height=150>
    </a>
  </div>


## SGD with Warm Restarts

+ [Warm restarts](../ML/MLNN-Hinton/a14-Advanced.md#33-sgd-with-warm-restarts)
  + restart the learning after a specified number of epochs
  + record the best estimates each time before resetting the learning rate
  + restarts not from scratch but from the last estimate
  + providing most of the benefits as cyclical learning rates
  + able to escape extreme local minima

+ [Warm restarts with cosine annealing](../ML/MLNN-Hinton/a14-Advanced.md#33-sgd-with-warm-restarts)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/805/1*Iymc7F6RF_PKije9dhZA2g.png" style="margin: 0.1em;" alt="Warm restarts with cosine annealing done every 50 iterations of Cifar10 dataset" title="Warm restarts with cosine annealing done every 50 iterations of Cifar10 dataset" width=250>
    </a>
  </div>


## Snapshot ensembles

+ [Ensemble networks](..e/ML/MLNN-Hinton/a14-Advanced.md#34-snapshot-ensembles)
  + training a single neural network with $M$ different models
  + much more robust and accurate than individual networks
  + another type of regularization technique
  + converge to $M$ different local optima and save network parameters
  + training w/ many different neural networks and then optimizing w/ major vote, or averaging of the prediction output

+ [Example of snapshot ensembles](..e/ML/MLNN-Hinton/a14-Advanced.md#34-snapshot-ensembles)
  + Left diagram: Illustration of SGD optimization with a typical learning rate schedule. The model converges to a minimum at the end of training.
  + Right diagram: Illustration of Snapshot Ensembling. The model undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima.

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/2118/1*Lp8rhR6C_TWuSIcF_QXXfA.png" style="margin: 0.1em;" alt="Left: Illustration of SGD optimization with a typical learning rate schedule. The model converges to a minimum at the end of training. Right: Illustration of Snapshot Ensembling. The model undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima." title="Snapshot ensembles" width=450>
    </a>
  </div>

+ [Classification](..e/ML/MLNN-Hinton/a14-Advanced.md#34-snapshot-ensembles)
  + often developing ensemble or blended models
  + providing superior results to any single model
  + constraint: high correlation btw models

+ [Procedure](..e/ML/MLNN-Hinton/a14-Advanced.md#34-snapshot-ensembles)
  + Model training
    + training w/ each model to reach local minimum w.r.t the training loss
    + take a snapshot of the model weights before raising the training rate
    + after $M$ cycles, $M$ model snapshots $f_1, f_2, \dots, f_M$ obtained
  + model ensemble
    + taking average of snapshots
    + used to obtain result

+ [Advantages](..e/ML/MLNN-Hinton/a14-Advanced.md#34-snapshot-ensembles)
  + achieving a neural network w/ smoothened parameters
  + reducing the total noise and the total error
  + w/o any additional training cost: total training time of $M$ snapshots same as training a model w/ a standard schedule

+ Not perfect: different initialization points or hyperparameter choices converging to different local minimum


## Polyak-Ruppert averaging

+ [Polyak averaging](../ML/MLNN-Hinton/a14-Advanced.md#35-polyak-ruppert-averaging)
  + motivation: gradient descent w/ a large learning rate unable to converge effectively to the global minimum
  + another approach to address the unstable learning issue that simulated the use of snapshot ensembles
  + using an average of the weights from multiple models seen towards the end of the training run
  + taking the time average of these parameters to obtain a smoother estimator for the true parameter, $t$ iterations

    \[ \hat{\theta}(t) = \frac{1}{t} \sum_i \hat{\theta}^{(i)} \]

  + leveraged in several ways
    + time averaging: using hte weights from the same model at several different epochs towards the end of the training run
    + ensemble averaging: using the weights from multiple models towards their individual training runs
    + hybrid approach: using the weights from snapshots and then averaging these weights for an ensemble prediction
  
+ [Convergence](../ML/MLNN-Hinton/a14-Advanced.md#35-polyak-ruppert-averaging)
  + guarantee strong converge in a convex setting
  + non-convex surfaces: the parameter space differed greatly in different regions; averaging less useful
  + Considering the exponentially decaying average

    \[ \hat{\theta}^{(t)} = \alpha \hat{\theta}^{(t-1)} + (1 - \alpha) \hat{\theta}^{(t)} \quad \text{with} \quad \alpha \in [0, 1] \]

  + depending on the chosen value of $\alpha$ additional weight either placed on the newest parameter values or the older parameter values
  + the importance of the older parameters exponentially decays over time

+ Polyak averaging & snapshot ensembles: different ways of smoothing the random error manifestly present in the unstable learning process of neural networks


## Silva and Almeida´s algorithm

+ Proposal: different learning rates for each weight in a network

+ [Assumptions & Notations](../ML/MLNN-Hinton/a12-Learning.md#831-silva-and-almeida´s-algorithm)
  + a network consists of $n$ weights $w_1, w_2, \dots, w_n$
  + $\gamma_1, \gamma_2, \dots, \gamma_n$: individual learning rates associated to the weights
  + $c_1, c_2, \dots, c_n$, $d_{ij}$ & $C$: constants
  + $k_1$ & $k_2$: constants depending on the values of the "frozen" variables at the current iteration point
  + $u > 1$: a constant for up manually
  + $d < 1$: a constant for down manually

+ [Algorithm](../ML/MLNN-Hinton/a12-Learning.md#831-silva-and-almeida´s-algorithm)
  + performing several optimization steps in the horizontal direction
  + horizontal cuts to a quadratic function: quadratic
  + minimizing at each step: parabola
  + general form of a quadratic function w/ $n$ variable weights

    \[c_1^2 w_1^2 + c_2^2 w_2^2 + \cdots + c_n^2 w_n^2 + \sum_{i \neq j} \, d_{ij}w_iw_j + C\]

  + Minimize the form of the $i$-th direction for a 1-dim minimization step

    \[c_i^2 w_i^2 + k_1 w_i + k_2\]

  + the learning rate for each weight at $k$-th iteration for the next step

    \[\gamma_i^{(k+1)} = \begin{cases}
      \gamma_i^{(k)} \, u & \text{if } \Delta_i E^{(k-1)} \geq 0 \\
      \gamma_i^{(k)} \, d & \text{if } \Delta_i E^{(k-1)} < 0
    \end{cases}\]
  
  + Weight update

    \[\Delta^{(k)} w_i = -\gamma_i^{(k)} \Delta_i E^{(k)}\]

  + With constants $u$ and $d$, the learning rate grow and decease exponentially.
  + not follow the gradient direction directly
  + With perfect circles for the level curves of the quadratic approximation, successive 1-dim optimizations reached after $n$ steps.

+ [Slow convergency](../ML/MLNN-Hinton/a12-Learning.md#831-silva-and-almeida´s-algorithm)
  + if the quadratic approximation w/ semi-axes of very different lengths, the iteration process can be arbitrary slowed.
  + Solution: adding momentum $\alpha$
  + Contradictory: 
    + the individual learning rates optimized if updates are strictly 1-dimensional
    + tuning the constant $\alpha$ quite problem-specific
  + Solution: preprocessing the original data to achieve a more regular error function
  + dramatically effecting the convergence speed of algorithms


## Delta-bar-Delta

+ Jacob's proposal: acceleration of the learning rate made more caution than deceleration

+ [Modeling algorithm](../ML/MLNN-Hinton/a12-Learning.md#832-delta-bar-delta)
  + Algorithms & Notations
    + $U$ & $d$: constants
    + $\delta_i^{(k)}$: an exponential average partial derivative in the direction of weight $w_i$
    + $\phi$: a constant determining what weight given to the last averaged term
  + starting w/ individual learning rates $\gamma_1, \dots, \gamma_n$ set all to a small value
  + the learning rate ar the $k$-th iteration

    \[\gamma_i^{(k+1)} = \begin{cases}
      \gamma_i^{(k)} + u & \text{if } \Delta_i E^{(k)} \cdot \delta_i^{(k-1)} > 0 \\
      \gamma_i^{(k)} \cdot d & \text{if } \Delta_i E^{(k)} \cdot \delta_i^{(k-1)} < 0 \\
      \gamma_i^{(k)} & \text{otherwise}
    \end{cases}\]

  + exponential averaged partial derivative w.r.t. $w_i$

    \[\delta_i^{(k)} = (1 - \phi) \Delta_i^{(k)} + \phi \delta_i^{(k-1)}\]

  + the weight updates performed w/o momentum

    \[\Delta^{(k)} w_i = - \gamma_i^{(k)} \Delta_i E^{(k)}\]

+ [Motivation](../ML/MLNN-Hinton/a12-Learning.md#832-delta-bar-delta)
  + avoid excessive oscillations of the basic algorithm
  + Issue: set a new constant and its value
  + error function: regions allowing a good quadratic approximation optimized at $\phi = 0 \to$ Silva and Almeida's algorithm



## AdaGrad

+ Momentum adds updates to the slope of error function and speeds up SGD in turn.

+ [AdaGrad](../ML/MLNN-Hinton/a03-Optimization.md#adagrad) adapts updates to each individual parameter to perform larger or smaller updates depending on their importance.

+ Accumulate squared gradients: $r_i = r_i + g_i^2$

+ Update each parameter:

  \[\theta_i = \theta_1 - \frac{\varepsilon}{\delta + \sqrt{r_i}} g_i\]

  + inversely proportional to cumulative squared gradient

+ Benefits:
  + eliminate the need to manually tune the learning rate
  + result in greater progress along gently sloped directions

+ Disadvantages:
  + accumulation of the squared gradients in the denominator
  + positive added term:
    + the accumulated sum keeps growing during training
    + the learning rate shrink and eventually become infinitesimally small


## Rprop

+ [rprop: using only the sign of the gradient](../ML/MLNN-Hinton/06-MiniBatch.md#rmsprop-normalized-the-gradient)
  + the magnitude of the gradient: different widely for different weights
  + full batch learning
  + approach
    + combining the idea of only using the sign of the gradient w/ the idea of adapting the step size separately for each weight
    + the sign of the last two gradients, not the magnitude of the gradients
      + agree: increasing the step size for a weight <span style="color: red;">multiplicatively</span> (e.g. times 1.2)
      + disagree: decreasing the step size multiplicatively (e.g. times 0.5)
    + Mike Shuster's advice: limiting the step sizes to be less than 50 and more than a millionth ($10^{-6}$)
    + the step size depends on the problem dealing with
      + tiny inputs: big weights required for the inputs

+ [rprop not working with mini-batches](../ML/MLNN-Hinton/06-MiniBatch.md#rmsprop-normalized-the-gradient)
  + rprop works with very big mini-batches where much more conservative changes to the step sizes
  + violate central idea of stochastic gradient descent
    + for small learning rate, the gradient gets effectively averaged over successive mini-batches
  + rprop:
    + not following the idea of stochastic gradient descent
    + assumption: any adaptation of the step sizes is smaller on the time scale of the mini-batches
    + increasing the weight nine times by whatever it's current step size is
    + decreasing the weight only once
    + making the weight much bigger
    + weight vector grows
  + criteria to judge the combination
    + the robustness of rprop: just using the sign of the gradient
    + the efficiency of mini-batches
    + the effective averaging of the gradients over mini-batches

+ [RPROP: resilient backpropagation](https://trongr.github.io/neural-network-course/neuralnetworks.html)

  __Definition__. Instead of relying on the gradient and a learning rate as in the Delta Rule, RPROP keeps track of a step size $\Delta_{ij}$ per weight:

  \[\Delta_{ij}(t) = \begin{cases}
    \eta^+ \Delta_{ij}(t-1) & \text{if } \frac{\partial E}{\partial w_{ij}}(t) \frac{\partial E}{\partial w_{ij}}(t-1) > 0 \\
    \eta^- \Delta_{ij}(t-1) & \text{if } \frac{\partial E}{\partial w_{ij}}(t) \frac{\partial E}{\partial w_{ij}}(t-1) < 0 \\
    \Delta_{ij}(t-1) & \text{otw.}
  \end{cases}\]

  where $0 < \eta^- < 1 < \eta^+$, usually $0.5% and $1.2$. Next, the weight update is

  \[\Delta w_{ij} (t) = \begin{cases}
    - \Delta_{ij} (t) & \text{if } \frac{\partial E}{\partial w_{ij}} > 0 \\
    + \Delta_{ij} (t) & \text{if } \frac{\partial E}{\partial w_{ij}} < 0 \\
    0 & \text{otw.}
  \end{cases}\]

  In other words, if the gradient changes sign, then our last step size $\Delta_{ij}(t-1)$ was too big, so wee need to scale it back by $\eta^-$.  On the other hand, if the gradient keeps its sign, then we're going in the right direction, and we scale up the step size by $\eta^+$ to go faster.  Otherwise, the gradient is zero and there's no need to change the weight.

  __Note.__ Even when $\frac{\partial E}{\partial w_{ij}}(t) = 0$, that doesn't we've reached a minimal $w_{ij}$, because at time $t+1$ the weights might move somewhere else and $\frac{\partial E}{\partial w_{ij}} (t+1)$ might be nonzero.

+ [Riedmiller and Braun proposal](../ML/MLNN-Hinton/a12-Learning.md#832-delta-bar-delta)
  + main idea: update the network weight using just the learning rate and the sign of the partial derivative of the error function w.r.t. each weight
  + accelerating learning mainly in the flat regions of the error function and approaching a local minimum
  + set $\gamma_{min}$ and $\gamma_{max}$ to avoid accelerating and decelerating too much

+ [Modeling algorithm](../ML/MLNN-Hinton/a12-Learning.md#832-delta-bar-delta)
  + covering the weight space between $\gamma_{min}$ and $\gamma_{max}$
  + $\gamma_{min}$, $\gamma_{max}$: $n$-dim grid of side length
  + individual 1-dim optimization steps: moving all possible intermediate grids
  + the learning rates updated at $k$-th iteration

    \[\gamma_i^{(k+1)} = \begin{cases}
      \min(\gamma_i^{(k)} \, u, \gamma_{max}) & \text{if } \Delta_i E^{(k-1)} \cdot \Delta_i^{(k-1)} > 0 \\
      \max(\gamma_i^{(k)} \, d, \gamma_{min}) & \text{if } \Delta_i E^{(k-1)} \cdot \Delta_i^{(k-1)} < 0 \\
      \gamma_i^{(k)} & \text{otherwise}
    \end{cases}\]

    + $u > 1$ and $d < 1$
  + the weight updated

    \[\Delta^{(k)} w_i = \begin{cases} -\gamma_i^{(k)} sgn(\Delta_i E^{(k)}) & \text{if } \Delta_i E^{(k-1)} \cdot \Delta_i^{(k-1)} \geq 0 \\ 0 & \text{otherwise} \end{cases}\]

+ [One-dimensional approximation of the error function](../ML/MLNN-Hinton/a12-Learning.md#832-delta-bar-delta)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/a12-17.png" style="margin: 0.1em;" alt="Local approximation of RPROP" title="Local approximation of RPROP" width=350>
    </a>
  </div>


## RMSProp

+ [rmsprop: A mini-batch version of rprop](../ML/MLNN-Hinton/06-MiniBatch.md#rmsprop-normalized-the-gradient)
  + problem w/ mini-batch rprop
    + using the gradient but also divided by the different magnitude of the gradient for each mini-batch
    + solution: force the number divided by to be very similar for adjacent mini-batches
  + rmsprop: keep a moving average of the squared gradient for each weight (e.g. $\alpha = 0.9$)

    \[MeanSquare(w, t) = \alpha \cdot MeanSquare(w, t-1) + (1 - \alpha) \, \left(\frac{\partial E}{\partial w}(t)\right)^2\]
  
  + dividing the gradient by $\sqrt{MeanSquare(w, t)}$ makes the learning work much better (Tijmen Tieleman, unpublished)
  + not adapting the learning rate separately for each connection
  + simple solution: for each connection keep a running average of the root mean squared gradient and the gradient divides by that RMS

+ [RMSPROP: root mean square backpropagation/mini-batch RPROP](https://trongr.github.io/neural-network-course/neuralnetworks.html)

  __Definition.__ Note that RPROP is equivalent to regular per-weight adaptive learning rate with the gradient magnitude normalized out by dividing by itself. So in order to make the magnitude relevant  again, we divide successive gradients not by themselves individually but by their root mean square: first define the running mean square

  \[\mu_{ij}(t) = \gamma \mu_{ij}(t-1) + (1-\gamma) \left(\frac{\partial E}{\partial w_{ij}}\right)^2\]

  where $\gamma$ is the forgetting factor: bigger $\gamma \in (0, 1)$ makes $\mu_{ij}$ remember more of its previous values.  Then the weight updates is

  \[\Delta w_{ij} = - \frac{\alpha}{\sqrt{\mu_{ij}}} \frac{\partial E}{\partial w}\]

  In vector form,

  \[\begin{align*}
    \mu &\gets \gamma \mu + (1 - \gamma)\left(\frac{\partial E}{\partial w}\right)^2 \\
    \Delta w &= - \frac{\alpha}{\sqrt{\mu}} \frac{\partial E}{\partial w} \\
    w &\gets w + \Delta w
  \end{align*}\]

  where all operations are done components wise with broadcasting if necessary.

  [RMSPROP implementation](../ML/MLNN-Hinton/src/rmsprop.py)

+ For non-convex problems, AdaGrad can prematurely decrease the learning rate.

+ Use an exponentially weighted average for gradient accumulation.

  \[\begin{array}{rcl} r_i &=& \rho r_i + (1 - \rho) g_i^2 \\ \theta_i &=& \theta_i - \frac{\varepsilon}{\delta + \sqrt{r_i}} g_i \end{array}\]


## Adam

+ [Adaptive moment estimation (Adam)](..](../ML/MLNN-Hinton/a03-Optimization.md#adam))
  + a combination of RMSprop and momentum
  + the most popular optimizer used for neural networks

+ Nadam: a combination of MRSprop and Nesterov momentum

+ Adam computes adaptive learning rates for each parameters.

+ Adam keeps an exponentially decaying average of past gradients, similar to momentum.
  + Estimate first moment: 

    \[v_i = \rho_1 v_i + (1 - \rho_1) g_i\]
  
  + Estimate second moment:

    \[r_i = \rho_2 r_i + 91 - \rho_2) g_i^2\]

    + applies bias correction to $v$ and $r$

  + Update parameters:

    \[\theta_i = \theta_i - \frac{\varepsilon}{\delta + \sqrt{r_i}} v_i\]

    + works well in practice, is fairly robust to hyper-parameters


## The Dynamic Adaption Algorithm

+ [Salomon's Proposal](../ML/MLNN-Hinton/a12-Learning.md#834-the-dynamic-adaption-algorithm)
  + Salomon, R. (1992), Verbesserung konnektionistischer Lernverfahren, die nachder Gradientenmethode arbeiten, PhD Thesis, Technical University of Berlin
  + Idea: using the negative gradient direction to generate tow new points instead of one, the one w/ lowest error used for the next iteration

+ [Modeling the algorithm](../ML/MLNN-Hinton/a12-Learning.md#834-the-dynamic-adaption-algorithm)
  + Assumptions & Notations
    + $\eta$: a small constant; e.g., $\xi = 1.7$
  + the $k$-th iteration

    \[\begin{align*}
      \mathbf{w}^{(k_1)} &= \mathbf{w}^{(k)} - \Delta E(\mathbf{w}^{(k)}) \gamma^{(k-1)} \cdot \xi \\
      \mathbf{w}^{(k_2)} &= \mathbf{w}^{(k)} - \Delta E(\mathbf{w}^{(k)}) \gamma^{(k-1)} / \xi
    \end{align*}\]

  + Update e learning rate

    \[\gamma = \begin{cases} \gamma^{(k-1)} \cdot \xi & \text{if } E(\mathbf{w}^{(k_1)}) \leq E(\mathbf{w}^{(k_2)}) \\ \gamma^{(k-1)} / \xi & \text{otherwise} \end{cases}\]

  + Update the weight

    \[\mathbf{w}^{(k+1)} = \begin{cases} \mathbf{w}^{(k_1)} & \text{if } E(\mathbf{w}^{(k_1)}) \leq E(\mathbf{w}^{(k_2)}) \\ \mathbf{w}^{(k_2)} & \text{otherwise}\end{cases}\]

+ The algorithm is not as good as the adaptive step methods w/ a local learning constant but easy to implement.


## Quickprop

+ [Idea](../ML/MLNN-Hinton/a12-Learning.md#841-quickprop)
  + considering the 1-dim minimization steps
  + obtaining the current and the last partial derivative of the error function in the update directions about the curvature of the error function
  + based on the independent optimization step for each weight

+ [Modeling algorithm](../ML/MLNN-Hinton/a12-Learning.md#841-quickprop)
  + Quadratic one-dimensional approximation of th error function
  + Assumption & Notations
    + $\Delta^{(k-1)} w_i$: the weight difference w/ the computed error functions at $(k-1)$-th and $k$-th steps
  + the update term for each weight at the $k$-th step obtained from a previous Quickprop or a standard gradient descent step

    \[\begin{align*}
      \Delta^{(k)} w_i &= \Delta^{(k-1)} w_i \left( \frac{\Delta_i E^{(k)}}{\Delta_i E^{(k=1)} - \Delta_i E^{(k)}} \right) \tag{10} \\
       &= - \frac{\Delta_i E^{(k-1)}}{(\Delta_i E^{(k)} - \Delta_i E^{(k)}) / \Delta^{(k-1)} w_i} \tag{11}
    \end{align*}\]

  + Eq. (11) same as the weight update in Eq. (9)
  + secant steps:
    + the denominator: a discrete approximation to the second-order derivative $\partial^2 E(\mathbf{w}) / \partial w_i^2$
    + a discrete pseudo-Newton method

+ [Problematic situations](../ML/MLNN-Hinton/a12-Learning.md#841-quickprop)
  + update issue
  + Convergence issue


## QRProp

+ Idea: adaptively switches between the Manhattan method used by Rprop and local 1-dim secant steps as used by Quickprop
  
+ [Qprop brief description](../ML/MLNN-Hinton/a12-Learning.md#842-qrprop)
  + using the individual learning rate strategy of Rprop if two consecutive error function gradient components, $\Delta_i E^{(k)}$ and $\Delta_i E{(k-1)}$ w/ the same sign or one of them equals zero
  + a fast approach to a region of minimum error
  + if sign changed, overshoot a local minimum in the specific weight direction, take a second-order step (Quickprop)
  + assume that the direction of the error function independent from all other weights, a step based on a quadratic approximation far more accurate than just stepping back half way as done by Rprop
  + constraining the size of the secant step to avoid large oscillations of the weights by
    + the error function depends on all weights
    + the quadratic approximation will be better the closer the two investigated points lie together

+ [Algorithm](../ML/MLNN-Hinton/a12-Learning.md#842-qrprop)
  1. $\Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} > 0$: perform Rprop steps (assume that a local minimum lies ahead)
  2. $\Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} < 0$:
    + indicate that a local minimum has been overshot
    + neither the individual learning rate $\gamma_i$ nor the weight $w_i$ are changed
    + a marker defined by setting $\Delta_i E^{(k)} := 0$
    + the secant step is performed in the subsequent iteration
  3. $\Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} = 0$:
    + either a marker was set in the previous step or one of the gradient components is zero (a local minimum has been directly hit))
    + near a local minimum
    + perform a second-order step
    + the secant approximation: using the gradient information provided by $\Delta_i E{(k)}$ and $\Delta_i E^{(k-2)} > 0$
    + the second-order approximation is still a better choice than just stepping halfway back when near a local minimum (and very likely overshot in the previous step)
  4. the quadratic approximation in the secant step

    \[q_i := |\Delta_i E^{(k)} / (\Delta_i E^{(k)} - \Delta_i E^{(k-2)})|\]

    is constrained to a certain interval to avoid very large or very small updates

+ [The $k$-th iteration of Qprop](../ML/MLNN-Hinton/a12-Learning.md#842-qrprop)
  + set constants: $d, u, \gamma_{min}$ and $\gamma_{max}$
  + Step 1: update the individual learning rates

    if $(\Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} = 0)$ then <br/>
    <span style="margin-left: 1em;">if $(\Delta_i E^{(k)} \neq \Delta_i E^{(k-1)})$ then</span></br>
    <span style="margin-left: 2em;">$q_i = \max \left(d, \min \left(1/u, \left|\frac{\Delta_i E^{(k)}}{\Delta_i E^{(k)} - \Delta_i E^{(k-2)}} \right|\right)\right)$</span></br>
    <span style="margin-left: 1em;">else</span></br>
    <span style="margin-left: 2em;">$q_i = 1/u$</span></br>
    <span style="margin-left: 1em;">endif</span></br>
    endif

    <br/>

    \[\gamma_i^{(k)} = \begin{cases}
      \min(u \cdot \gamma_i^{(k-1)}, \gamma_{max}) & \text{if } \Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} > 0 \\
      \gamma_i^{(k-1)} & \text{if } \Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} < 0 \\
      \max(q_i \cdot \gamma_i^{(k-1)}, \gamma_{min}) & \text{if } \Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} = 0
    \end{cases}\]

  + Step 2: update the weight

    \[w_i^{(k+1)} = \begin{cases}
      w_i^{(k)} - \gamma^{(k)} \cdot \text{sgn}(\Delta_i E^{(k)}) & \text{if } \Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} \geq 0 \\
      w_i^{(k)} & \text{otherwise}
    \end{cases}\]

    If $(\Delta_i E^{(k)} \cdot \Delta_i E^{(k-1)} < 0)$ set $\Delta_i E^{(k)} := 0$

