# Large Scale Machine Learning

## Gradient Descent with Large Datasets

### Learning With Large Datasets

#### Lecture Notes

+ Machine learning and data
  + Why do we want large data set?
    + This is evident when we take a low-bias learning algorithm and train it on a lot of data
  + Example: how “I ate two (two) eggs” shows how the algorithm performs well when we feed it a lot of data
    + Classify between confusable words
    + E.g., {to, two, too}, {then, than}
    + For breakfast I ate __________ eggs.

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.semanticscholar.org/paper/Scaling-to-Very-Very-Large-Corpora-for-Natural-Banko-Brill/639b4cac06148e9f91736ba36ad4a1b97fcdfd6a">
      <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/639b4cac06148e9f91736ba36ad4a1b97fcdfd6a/2-Figure1-1.png" style="margin: 0.1em;" alt="Accuracy of algorithms vs. size of dataset" title="Learning Curves for Confusion Set Disambiguation" width="250">
    </a></div>
  </div>

  + "It's not who has the best algorithm that wins. It's who has the most data."

+ Learning with large data sets has computational problems
  + $m = 100 \times 10^6$, to sum over $10^8$ entries to compute one step of gradient descent
  + Train with linear regression model

    $$\theta_j := \theta_j - \alpha \dfrac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

  + IVQ: Suppose you are facing a supervised learning problem and have a very large data set ($m = 100 \times 10^6$), how can you tell if the data is likely to perform much better than using a small subset ($m = 1000$) of the data?

    1. There is no need to verify this; using a larger dataset always gives much better performance.
    2. Plot $J_\text{train}(\theta)$ as a function of the number of iterations of the optimization algorithm (such as gradient descent).
    3. Plot a learning curve ($J_\text{train}(\theta)$ and $J_\text{CV}(\theta)$, plotted as a function of $m$) for some range of values of $m$ (say up to $m = 1,000$) and verify that the algorithm has bias when m is small.
    4. Plot a learning curve for a range of values of $m$ and verify that the algorithm has high variance when $m$ is small.

    Ans: 4

  + High variance: adding more training examples would increase the accuracy
  + High bias: need to plot to a large value of $m$ -> add extra features or units (in neural networks)

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.ritchieng.com/machine-learning-large-scale/#1-gradient-descent-with-large-data-sets">
      <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w10_large_scale_ml/largescaleml1.png" style="margin: 0.1em;" alt="Relationship between size of dataset error: left diagram - high variance (overfitting), right diagram - high bias (underfitting)" title="Relationship between size of dataset error: left diagram - high variance (overfitting), right diagram - high bias (underfitting)" width="350">
    </a></div>
  </div>


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/18.1-LargeScaleMachineLearning-LearningWithLargeDatasets.06644550b22b11e487451d0772c554c0/full/360p/index.mp4?Expires=1558310400&Signature=UagqJUiz0nlcgkVQUeEjAjL1ftBBMAxt3~T88w0N7hFjMP7VNlBbFuaasknggorOWQP5VyWaTZv6W0P-nhmf3FgkqOX01iYxj8KA7z7vTlsavZE2qDyHVLj0zFU9enIViuK3VJAOsbp48ogF30~xrm6MU~S2kz1tY8TUC4qM1ns_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/In93d44xSpK_d3eOMdqS3g?expiry=1558310400000&hmac=hqZvlZNj8ZXPVFLvs8trwFnPejSX2NSICfWsqvOic3I&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Stochastic Gradient Descent

#### Lecture Notes

+ Linear regression with gradient descent

  + Hypothesis and Cost functions

    $$\begin{array}{rcl} h_\theta(x) & = & \displaystyle \sum_{j=0}^n \theta_j x_j \\ J_{train}(\theta) & = & \dfrac{1}{2m} \displaystyle \sum_{i=1}^m \left( h_\theta(x^{(i)} - y^{(i)} \right)^2 \end{array}$$

  + Algorithm

    Repeat { <br/>
    <span style="padding-left: 2em;" /> $\theta_j := \theta_j - \alpha \dfrac{1}{m} \displaystyle \sum_{i=1}^m \left( h_\theta(x^{(i)} - y^{(i)}) \right) x^{(i)} \quad (\forall j = 0, \dots, n)$ <br/>
    }

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.coursera.org/learn/machine-learning/lecture/DoRHJ/stochastic-gradient-descent">
      <img src="images/m02-06.png" style="margin: 0.1em;" alt="Contour of Cost function and Weighting parameters (theta1, theta2)" title="Contour of Cost function and Weighting parameters ($\theta_1, \theta_2$)" width="350">
      <img src="images/m17-02.png" style="margin: 0.1em;" alt="Contour of Cost function with weighting parameters, theta_1 & theta_2" title="Contour of cost function with weighting parameters, \theta_1 & \theta_2" width="300">
    </a></div>
  </div>

  + With large $m$ (said $m = 300,000,000$), sum across all the examples
  + __Batch gradient descent__: observe all the training examples at a time

+ Batch vs. Stochastic gradient descent
  + Batch gradient descent

    $$J_{train}(\theta) = \dfrac{1}{2m} \displaystyle \sum_{i=1}^m \left( h_\theta(x^{(i)} - y^{(i)} \right)^2$$

    Repeat { <br/>
    <span style="padding-left: 2em;"/> $\theta_j := \theta_j - \alpha \underbrace{\frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)} - y^{(i)}) \right) x^{(i)}}_{\frac{\partial}{\partial \theta_j}J_{train} (\theta)} \quad (\forall j = 0, \dots, n)$ <br/>
    }
  + Stochastic gradient descent

    $$\begin{array}{rcl} cost \left( \theta, (x^{(i)}, y^{(i)}) \right) &=& \frac{1}{2} \left( h_\theta(x^{(i)}) - y^{(i)}) \right)^2 \\\\ J_{train}(\theta) &=& \frac{1}{m} \sum_{i=1}^m cost \left( \theta, (x^{(i)}, y^{(i)}) \right) \end{array}$$

    + General Algorithm
      1. Randomly shuffle (reorder) training examples
      2. Repeat { <br/>
        <span style="padding-left: 1em;"/> for $i=1, \dots, m$ { <br/>
        <span style="padding-left: 2em;"/> $\theta_j := \theta_j - \alpha \underbrace{\left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}}_{\frac{\partial}{\partial \theta_j} cost \left(\theta, (x^{(i)}, y^{(i)}) \right)} \quad (\forall j=0, \dots, n) \Rightarrow (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), (x^{(3)}, y^{(3)}), \dots$ <br/>
        <span style="padding-left: 1em;"/>} <br/>
        } <br/>

  + IVQ: Which of the following statements about stochastic gradient descent are true? Check all that apply.

    1. When the training set size $m$ is very large, stochastic gradient descent can be much faster than gradient descent.
    2. The cost function $J_\text{train}(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$ should go down with every iteration of batch gradient descent (assuming a well-tuned learning rate $\alpha$) but not necessarily with stochastic gradient descent.
    3. Stochastic gradient descent is applicable only to linear regression but not to other models (such as logistic regression or neural networks).
    4. Before beginning the main loop of stochastic gradient descent, it is a good idea to "shuffle" your training data into a random order.

    Ans: 124

  + Differences
    + Rather than waiting to take the parts of all the training examples (batch gradient descent), we look at a single training example and we are making progress towards moving to the global minimum
    + Batch gradient descent (red path)
    + Stochastic gradient descent (magenta path with a more random-looking path where it wonders around near the global minimum)
    + In practice, as long as the parameters close to the global minimum, it’s sufficient (within a region of global minimum)
    + repeat the loop maybe 1 to 10 times depending on the size of training set
    + It is possible even with 1 loop, where your $m$ is large, you can have good parameters
    + the $J_{train}$ (cost function) may not decrease with every iteration for stochastic gradient descent



#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/18.2-LargeScaleMachineLearning-StochasticGradientDescent.bf8834b0b22b11e4960bf70a8782e569/full/360p/index.mp4?Expires=1558310400&Signature=Ydu7JmdsPLPPZVsbOIFK3iOxTQj-XzsbMnMRkoMV-AeTYFl-cKJ2vxM8vDeO2PzznGFaTzWSfrhbrrIcbUEEHmhKZuDld774HOeRUly6yiZLE6o93u6e2iI0BqpLRsqgwb2RAtFq7a2NvSZtJsYOWIFixsvIz3-P0EFBpBO0G3g_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/Ma7B-4oJRqSuwfuKCdakNQ?expiry=1558310400000&hmac=XY-0Yf5CgkNIYvcMgmjOe0XrvCL_FwFdLMev8TS-Jmc&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Mini-Batch Gradient Descent

#### Lecture Notes

+ Comparisons of gradient descent methods
  + Batch gradient descent: use all $m$ examples in each iteration
  + Stochastic gradient descent: use $1$ example in each iteration
  + Min-batch gradient descent: use $b$ examples in each iteration
    + sometimes faster than stochastic gradient descent
    + $b = \;$ mini-batch size; e.g., $b \in [2, 100]$, typical $b = 10$,
    + Get $b=10$ examples, $(x^{(i)}, y^{(i)}), \dots, (x^{(i+9)}, y^{(i+9)})$, then perform

      $$\theta_j := \theta_j - \alpha \dfrac{1}{10} \displaystyle \sum_{k=i}^{i+9} (h_\theta(x^{(k)}) - y^{(k)}) x_j^{(k)}$$

+ Mini-batch gradient descent

  Say $b=10, m = 1000$ <br/>
  Repeat { <br/>
  <span style="padding-left: 1em;"/> for $i = 1, 11, 21, 31, \dots, 991$ { <br/>
  <span style="padding-left: 2em;"/> $\theta_j := \theta_j - \alpha \dfrac{1}{10} \displaystyle \sum_{k=i}^{i+9} \left( h_\theta(x^{(k)}) - y^{(k)} \right) x_j^{(k)} \quad (\forall j=0,\dots,n)$ <br/>
  <span style="padding-left: 1em;"/>} <br/>
  }

  + vectorization with $b$ examples
  + IVQ: Suppose you use mini-batch gradient descent on a training set of size $m$, and you use a mini-batch size of $b$. The algorithm becomes the same as batch gradient descent if:

    1. b = 1
    2. b = m / 2
    3. b = m
    4. None of the above

    Ans: 3


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/18.2-LargeScaleMachineLearning-StochasticGradientDescent.bf8834b0b22b11e4960bf70a8782e569/full/360p/index.mp4?Expires=1558310400&Signature=Ydu7JmdsPLPPZVsbOIFK3iOxTQj-XzsbMnMRkoMV-AeTYFl-cKJ2vxM8vDeO2PzznGFaTzWSfrhbrrIcbUEEHmhKZuDld774HOeRUly6yiZLE6o93u6e2iI0BqpLRsqgwb2RAtFq7a2NvSZtJsYOWIFixsvIz3-P0EFBpBO0G3g_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/Ma7B-4oJRqSuwfuKCdakNQ?expiry=1558310400000&hmac=XY-0Yf5CgkNIYvcMgmjOe0XrvCL_FwFdLMev8TS-Jmc&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Stochastic Gradient Descent Convergence


#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Advanced Topics


### Online Learning


#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Map Reduce and Data Parallelism


#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Review

### Lecture Slides

#### Learning with Large Datasets

We mainly benefit from a very large dataset when our algorithm has high variance when m is small. Recall that if our algorithm has high bias, more data will not have any benefit.

Datasets can often approach such sizes as m = 100,000,000. In this case, our gradient descent step will have to make a summation over all one hundred million examples. We will want to try to avoid this -- the approaches for doing so are described below.


#### Stochastic Gradient Descent

Stochastic gradient descent is an alternative to classic (or batch) gradient descent and is more efficient and scalable to large data sets.

Stochastic gradient descent is written out in a different but similar way:

$$cost(\theta,(x^{(i)}, y^{(i)})) = \dfrac{1}{2}(h_{\theta}(x^{(i)}) - y^{(i)})^2$$

The only difference in the above cost function is the elimination of the m constant within $\dfrac{1}{2}$.

$$J_{train}(\theta) = \dfrac{1}{m} \displaystyle \sum_{i=1}^m cost(\theta, (x^{(i)}, y^{(i)}))$$

$J_{train}$ is now just the average of the cost applied to all of our training examples.

The algorithm is as follows

1. Randomly 'shuffle' the dataset
2. For $i = 1\dots m$

  $$\Theta_j := \Theta_j − \lambda (h_{\Theta}(x^{(i)}) − y^{(i)}) \cdot x^{(i)}_j$$

This algorithm will only try to fit one training example at a time. This way we can make progress in gradient descent without having to scan all m training examples first. Stochastic gradient descent will be unlikely to converge at the global minimum and will instead wander around it randomly, but usually yields a result that is close enough. Stochastic gradient descent will usually take 1-10 passes through your data set to get near the global minimum.


#### Mini-Batch Gradient Descent

Mini-batch gradient descent can sometimes be even faster than stochastic gradient descent. Instead of using all m examples as in batch gradient descent, and instead of using only 1 example as in stochastic gradient descent, we will use some in-between number of examples b.

Typical values for b range from 2-100 or so.

For example, with b=10 and m=1000:

Repeat:

For $i = 1,11,21,31,\dots,991$

$$\theta_j := \theta_j - \alpha \dfrac{1}{10} \displaystyle \sum_{k=i}^{i+9} (h_\theta(x^{(k)}) - y^{(k)})x_j^{(k)}$$

We're simply summing over ten examples at a time. The advantage of computing more than one example at a time is that we can use vectorized implementations over the $b$ examples.


#### Stochastic Gradient Descent Convergence

How do we choose the learning rate α for stochastic gradient descent? Also, how do we debug stochastic gradient descent to make sure it is getting as close as possible to the global optimum?

One strategy is to plot the average cost of the hypothesis applied to every 1000 or so training examples. We can compute and save these costs during the gradient descent iterations.

With a smaller learning rate, it is __possible__ that you may get a slightly better solution with stochastic gradient descent. That is because stochastic gradient descent will oscillate and jump around the global minimum, and it will make smaller random jumps with a smaller learning rate.

If you increase the number of examples you average over to plot the performance of your algorithm, the plot's line will become smoother.

With a very small number of examples for the average, the line will be too noisy and it will be difficult to find the trend.

One strategy for trying to actually converge at the global minimum is to __slowly decrease $\alpha$ over time.__ For example $\alpha = \dfrac{const1}{iterationNumber + const2}$ 

However, this is not often done because people don't want to have to fiddle with even more parameters.


#### Online Learning

With a continuous stream of users to a website, we can run an endless loop that gets $(x,y)$, where we collect some user actions for the features in x to predict some behavior y.

You can update θ for each individual (x,y) pair as you collect them. This way, you can adapt to new pools of users, since you are continuously updating theta.


#### Map Reduce and Data Parallelism

We can divide up batch gradient descent and dispatch the cost function for a subset of the data to many different machines so that we can train our algorithm in parallel.

You can split your training set into z subsets corresponding to the number of machines you have. On each of those machines calculate $\displaystyle \sum_{i=p}^{q}(h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$, where we've split the data starting at $p$ and ending at $q$.

MapReduce will take all these dispatched (or 'mapped') jobs and 'reduce' them by calculating:

$$\Theta_j := \Theta_j − \alpha \frac{1}{z} (temp^{(1)}_j + temp^{(2)}_j + \dots + temp^{(z)}_j) \qquad \forall j = 0, \dots, n$$

This is simply taking the computed cost from all the machines, calculating their average, multiplying by the learning rate, and updating theta.

Your learning algorithm is MapReduceable if it can be _expressed as computing sums of functions over the training set_. Linear regression and logistic regression are easily parallelizable.

For neural networks, you can compute forward propagation and back propagation on subsets of your data on many machines. Those machines can report their derivatives back to a 'master' server that will combine them.


### Quiz: Large Scale Machine Learning






