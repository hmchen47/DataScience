# Advice for Applying Machine Learning

## Evaluating a Learning Algorithm

### Deciding What to Try Next

### Lecture Notes

+ Debugging a learning algorithm

  Suppose you have implemented regularized linear regression to predict housing prices.

  $$J(\theta) = \dfrac{1}{2m} \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^m \theta_j^2 \right]$$

  + However, when you test your hypothesis your hypothesis on new set of houses, you find that it makes unacceptably large errors. You can do the following

    + Get more training data
    + Smaller set of features
    + Get additional features
    + Try adding polynomial features ($x_1^2, x_2^2, x_1x_2,$ etc.)
    + Try decreasing lambda
    + Try increasing lambda

  + Typically people randomly choose these avenues and then figure out it may not be suitable
  + There is a simple technique to weed out avenues that are not suitable


+ Machine Learning Diagnostic
  + Test that you can run to gain insight what is or isn’t working with a learning algorithm and gain guidance as to how best to improve its performance
  + Diagnostics can take time to implement, but doing so can be a very good use of your time
  + But it’s worth the time compared to spending months on unsuitable avenues
  + IVQ: Which of the following statements about diagnostics are true? Check all that apply.

    1. It’s hard to tell what will work to improve a learning algorithm, so the best approach is to go with gut feeling and just see what works.
    2. Diagnostics can give guidance as to what might be more fruitful things to try to improve a learning algorithm.
    3. Diagnostics can be time-consuming to implement and try, but they can still be a very good use of your time.
    4. A diagnostic can sometimes rule out certain courses of action (changes to your learning algorithm) as being unlikely to improve its performance significantly.

    Ans: 234


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/10.1-AdviceForApplyingMachineLearning-DecidingWhatToTryNext.c45ffb80b22b11e49f072fa475844d6b/full/360p/index.mp4?Expires=1555027200&Signature=G6qZQfvjvanJYSPo9p1fGHEzj4gYPiyjtD9rrhTYx2HYDDXiS4ZMXzTp6Gr4zkRpgA5-DrSvgvt7smx5AnVaNrUTCEh5gfkLgHyfOZcQG2-jOJcrgKKEhR8GthkXr9fxZEDIiCWm0HQ92lhimEg6n~W-dnnZz4-IQP9tJF07Nj8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/gW252s0RSIStudrNETiEow?expiry=1555027200000&hmac=e3fCh0AMP6u9UMsU9iJSfz_8xOIligRLLGGkLXpgCVY&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Evaluating a Hypothesis

#### Lecture Notes

+ Evaluating your hypothesis
  
  Fails to generalize to new examples not in training set

  $$\begin{array}{rclcrcl} x_1 & = & \text{size of house} & \qquad & x_2 & = & \text{no. of bedrooms} \\ x_3 & = & \text{no. of floors} &\qquad & x_4 & = & \text{age of house} \\ x_5 & = & \text{average income in neighborhood} & \qquad & x_6 & = & \text{kitchen size} \\ & & & \cdots \\ x_{100} & & \cdots \end{array}$$

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://d3c33hcgiwev3.cloudfront.net/_b0cf48c6b7bc9f194310e6bc90dec220_Lecture10.pdf?Expires=1555113600&Signature=OWZGJ7XPJwSarMa5gOxrdIsbg9MrLI3PYOoU0xIUmL6-mFSvDZiEI4lNbfluPLil-D9IJ-UZsfqYXxfS~lJWQjCSp8ViScW120f3TI8xP9ap7OKvfuy5lCRNxldId0P75~PLG02kcda72mRsQllNBELdztCt3l99AUaFPopIRlM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A">
      <img src="images/m10-01.png" style="margin: 0.1em;" alt="Overfit with non-linear regression function on house prices" title="Non-linear regression function for house prices" width="250">
      <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w6_ml_design/test.png" style="margin: 0.1em;" alt="Exhibit the splitting of dataset for training and test" title="Dataset splitting" width="450">
    </a></div>
  </div>

  + $m_{test}\;$: no. of test sample
  + $(x^{(i)}_{test},\; y^{(i)})\;$: $i$-th sample of the test samples
  + Typical proportions of training and test dataset: $70\%$ vs $30\%$
  + IVQ: Suppose an implementation of linear regression (without regularization) is badly overfitting the training set. In this case, we would expect:
    
    1. The training error $J(\theta)$ to be __low__ and the test error $J_{\text{test}}(\theta)$ to be __high__
    2. The training error $J(\theta)$ to be __low__ and the test error $J_{\text{test}}(\theta)$ to be __low__
    3. The training error $J(\theta)$ to be __high__ and the test error $J_{\text{test}}(\theta)$ to be __low__
    4. The training error $J(\theta)$ to be __high__ and the test error $J_{\text{test}}(\theta)$ to be __high__

    Ans: 1


+ Training/testing procedure for linear regression
  + Learn parameter $\theta$ from training data (minimizing training error $J(\theta)$); $70\%$ of the whole dataset
  + Compute test set error (another $30\%$): $J_{test} (\theta) = \dfrac{1}{2m_{test}} \displaystyle \sum_{i=1}^{m_{test}} (h_\theta(x^{(i)}_{test}) - y^{(i)}_{test})^2$

+ Training/testing procedure for logistic regression
  + Learn parameter $\theta$ from training data
  + compute test set error:

    $$J_{test}(\theta) = 1\dfrac{1}{m_{test}} \sum_{i=1}^{m_{test}} \left[ y^{(i)}_{test} \log(h_\theta(x_{test}^{(i)})) + (1 - y_{test}^{(i)}) \log(1 - h_\theta(x_{test}^{(i)}))\right]$$

  + Misclassification error (0/1 misclassification error):

    $$\begin{array}{rcl} err(h_\theta(x),\; y) &=& \begin{cases} 1 & \text{if }\; h_\theta(x) \geq 0.5), \; y= 0 \\ & \text{or if } \;h_\theta(x) < 0.5, \;y = 1 \\ 0 & \text{otherwise} \end{cases} \\\\ \text{Test error} & = & \dfrac{1}{m_{test}} \sum_{i=1}^{m_{test}} err(h_\theta(x_{test}^{(i)}, \;y^{(i)} \end{array}$$



-----------------------------------------------------------

Once we have done some trouble shooting for errors in our predictions by:

+ Getting more training examples
+ Trying smaller sets of features
+ Trying additional features
+ Trying polynomial features
+ Increasing or decreasing $\lambda$

We can move on to evaluate our new hypothesis.

A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a training set and a test set. Typically, the training set consists of 70% of your data and the test set is the remaining 30%.

The new procedure using these two sets is then:

+ Learn $\theta$ and minimize $J_{train}(\theta)$ using the training set
+ Compute the test set error $J_{test}(\theta)$


__The test set error__

1. For linear regression: $J_{test}(\theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (h_\theta(x^{(i)}_{test}) − y^{(i)}_{test})^2$
2. For classification ~ Misclassification error (aka 0/1 misclassification error):

  $$err(h_\theta(x), \;y) = \begin{cases} 1 & \text{if } h_\theta(x) \geq 0.5 \text{ and } y=0 \\ & \text{or } h_\theta(x)< 0.5 \text{ and } y=1 \\ 0 & \text{otherwise} \end{cases}$$

This gives us a binary 0 or 1 error result based on a misclassification.

The average test error for the test set is

$$\text{Test Error } = \dfrac{1}{m_{test}} \sum_{i=1}^{m_{test}} err(h_\theta(x^{(i)}_{test}),y^{(i)}_{test})$$

This gives us the proportion of the test data that was misclassified.


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/10.2-AdviceForApplyingMachineLearning-EvaluatingAHypothesis.c45721e0b22b11e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1555113600&Signature=AOmEgzbFWI4VWblh9HkZ0dc0vbrjOGoBPMo5sexdwaYRAZQVRtLIJkuuplZs3XsG8gxMhQk0JVSr6h-5OpGSpVMUxh8hR6gFq3s2MsMO3KJMWabn6h6IdlMsbI4LVSV1QSEVTBYJnMAhYB4d8O61XMCNNonW3KtylV7guAMC3Xc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/dXB5OF6rRQWweTheq9UFfg?expiry=1555113600000&hmac=4YMcKQ7w0fWeNrFHD26ofOvS9lilneN4vNiAgYl6IQw&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Model Selection and Train/Validation/Test Sets

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Bias vs. Variance

### Diagnosing Bias vs. Variance

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Regularization and Bias/Variance

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Learning Curves

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Deciding What to Do Next Revisited

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Review

### [Lecture Slides](https://d3c33hcgiwev3.cloudfront.net/_b0cf48c6b7bc9f194310e6bc90dec220_Lecture10.pdf?Expires=1555027200&Signature=JZ5d8vEr1Me54~P93Q3pQxhCk3~BRCg26sqE3QdqzXRSb25g0wP0aeJY31mgyux6AA0AG8WBQUXjWnl2shHqen3Ska2AUuKwR-6VorXFwB6ClNWA-9r~KQeOD3V~HHIm-vSlYyqT2zGirZHCSn8l~eaRGBJmVgbp7Otrb1vI~GM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)


#### Deciding What to Try Next

Errors in your predictions can be troubleshooted by:

+ Getting more training examples
+ Trying smaller sets of features
+ Trying additional features
+ Trying polynomial features
+ Increasing or decreasing \lambda

Don't just pick one of these avenues at random. We'll explore diagnostic techniques for choosing one of the above solutions in the following sections.

#### Evaluating a Hypothesis

A hypothesis may have low error for the training examples but still be inaccurate (because of overfitting).

With a given dataset of training examples, we can split up the data into two sets: a __training set__ and a __test set__.

The new procedure using these two sets is then:

1. Learn \theta and minimize $J_{train}(\theta)$ using the training set
2. Compute the test set error $J_{test}(\theta)$


__The test set error__

1. For linear regression: $J_{test}(\theta) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (h_\theta(x^{(i)}_{test}) − y^{(i)}_{test})^2$
2. For classification ~ Misclassification error (aka 0/1 misclassification error):

  $$err(h_\theta(x),y) = \begin{cases} 1 & \text{if } h_\theta(x) \geq 0.5 \text{ and } y=0 \text{ or } h_\theta(x)<0.5 \text{ and } y=1 \\ 0 & \text{otherwise} \end{cases}$$

This gives us a binary 0 or 1 error result based on a misclassification.

The average test error for the test set is

$$\text{Test Error } = \dfrac{1}{m_{test}} \sum_{i=1}^{test} err(h_\theta(x^{(i)}_{test}),y^{(i)}_{test})$$

This gives us the proportion of the test data that was misclassified.

#### Model Selection and Train/Validation/Test Sets

+ Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis.
+ The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than any other data set.

In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

__Without the Validation Set (note: this is a bad method - do not use it)__

1. Optimize the parameters in \theta using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the test set.
3. Estimate the generalization error also using the test set with $J_{test}(\theta^{(d)})$, (d = theta from polynomial with lower error);

In this case, we have trained one variable, d, or the degree of the polynomial, using the test set. This will cause our error value to be greater for any other set of data.


__Use of the CV set__

To solve this, we can introduce a third set, the __Cross Validation Set__, to serve as an intermediate set that we can train d with. Then our test set will give us an accurate, non-optimistic error.

One example way to break down our dataset into the three sets is:

+ Training set: 60%
+ Cross validation set: 20%
+ Test set: 20%

We can now calculate three separate error values for the three different sets.

__With the Validation Set (note: this method presumes we do not also use the CV set for regularization)__

1. Optimize the parameters in \theta using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with $J_{test}(\theta^{(d)})$, (d = theta from polynomial with lower error);

This way, the degree of the polynomial d has not been trained using the test set.

(Mentor note: be aware that using the CV set to select 'd' means that we cannot also use it for the validation curve process of setting the lambda value).

#### Diagnosing Bias vs. Variance

In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.

+ We need to distinguish whether __bias__ or __variance__ is the problem contributing to bad predictions.
+ High bias is underfitting and high variance is overfitting. We need to find a golden mean between these two.

The training error will tend to __decrease__ as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to __decrease__ as we increase d up to a point, and then it will __increase__ as d is increased, forming a convex curve.

__High bias (underfitting)__: both $J_{train}(\theta)$ and $J_{CV}(\theta)$ will be high. Also, $J_{CV}(\theta) \approx J_{train}(\theta)$.

__High variance (overfitting)__: $J_{train}(\theta)$ will be low and $J_{CV}(\theta)$ will be much greater than $J_{train}(\theta)$.

The is represented in the figure below:

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="http://www.saberismywife.com/2016/12/13/Machine-Learning-6/">
    <img src="http://www.saberismywife.com/2016/12/13/Machine-Learning-6/5.png" style="margin: 0.1em;" alt="the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis. We need to distinguish whether bias or variance is the problem contributing to bad predictions. High bias is underfitting and high variance is overfitting. We need to find a golden mean between these two." title="Diagnosing Bias vs. Variance" width="450">
  </a></div>
</div>


#### Regularization and Bias/Variance

Instead of looking at the degree d contributing to bias/variance, now we will look at the regularization parameter \lambda.

+ Large $\lambda\;$: High bias (underfitting)
+ Intermediate $\lambda\;$: just right
+ Small $\lambda\;$: High variance (overfitting)

A large lambda heavily penalizes all the \theta parameters, which greatly simplifies the line of our resulting function, so causes underfitting.

The relationship of \lambda to the training set and the variance set is as follows:

+ Low $\lambda\;$: $J_{train}(\theta)$ is low and $J_{CV}(\theta)$ is high (high variance/overfitting).
+ Intermediate $\lambda\;$: $J_{train}(\theta)$ and $J_{CV}(\theta)$ are somewhat low and $J_{train}(\theta) \approx J_{CV}(\theta)$.
+ Large $\lambda\;$: both $J_{train}(\theta)$ and $J_{CV}(\theta)$ will be high (underfitting /high bias)

The figure below illustrates the relationship between lambda and the hypothesis:

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="http://www.saberismywife.com/2016/12/13/Machine-Learning-6/">
    <img src="http://www.saberismywife.com/2016/12/13/Machine-Learning-6/7.png" style="margin: 0.1em;" alt="regularization parameter \lambda. Large \lambda: High bias (underfitting); Intermediate \lambda: just right; Small \lambda: High variance (overfitting); A large lambda heavily penalizes all the \theta parameters, which greatly simplifies the line of our resulting function, so causes underfitting." title="Regularization and Bias/Variance" width="350">
  </a></div>
</div>

In order to choose the model and the regularization \lambda, we need:

1. Create a list of lambdas (i.e. $\lambda \in \{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24\}$);
2. Create a set of models with different degrees or any other variants.
3. Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\theta$.
4. Compute the cross validation error using the learned $\theta$ (computed with $\lambda$) on the $J_{CV}(\theta)$ without regularization or $\lambda = 0$.
5. Select the best combo that produces the lowest error on the cross validation set.
6. Using the best combo $\theta$ and $\lambda$, apply it on $J_{test}(\theta)$ to see if it has a good generalization of the problem.

#### Learning Curves

Training 3 examples will easily have 0 errors because we can always find a quadratic curve that exactly touches 3 points.

+ As the training set gets larger, the error for a quadratic function increases.
+ The error value will plateau out after a certain m, or training set size.


__With high bias__

+ Low training set size: causes $J_{train}(\theta)$ to be low and $J_{CV}(\theta)$ to be high.
+ Large training set size: causes both $J_{train}(\theta)$ and $J_{CV}(\theta)$ to be high with $J_{train}(\theta) \approx J_{CV}(\theta)$.

If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

For high variance, we have the following relationships in terms of the training set size:

__With high variance__

+ Low training set size: $J_{train}(\theta)$ will be low and $J_{CV}(\theta)$ will be high.
+ Large training set size: $J_{train}(\theta)$ increases with training set size and $J_{CV}(\theta)$ continues to decrease without leveling off. Also, $J_{train}(\theta) < J_{CV}(\theta)$ but the difference between them remains significant.

If a learning algorithm is suffering from __high variance__, getting more training data is __likely to help__.

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/resources/LIZza">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/58yAjHteEeaNlA6zo4Pi2Q_9f0aa85680aff49e1624155c277bb926_300px-Learning2.png?expiry=1555027200000&hmac=NKBG_J3IgnUF4fR_sf8N2DJCaKkx4hWhh0xcopoGIHw" style="margin: 0.1em;" alt="text" title="caption" width="350">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ITu3antfEeam4BLcQYZr8Q_37fe6be97e7b0740d1871ba99d4c2ed9_300px-Learning1.png?expiry=1555027200000&hmac=DoPOFXLXoGI6eBs0LQ-ZKdSCM7Sr7fs8dw2EX-8orbY" style="margin: 0.1em;" alt="text" title="caption" width="380">
  </a></div>
</div>


#### Deciding What to Do Next Revisited

Our decision process can be broken down as follows:

+ Getting more training examples: Fixes high variance
+ Trying smaller sets of features: Fixes high variance
+ Adding features: Fixes high bias
+ Adding polynomial features: Fixes high bias
+ Decreasing \lambda: Fixes high bias
+ Increasing \lambda: Fixes high variance

##### Diagnosing Neural Networks

+ A neural network with fewer parameters is __prone to underfitting__. It is also __computationally cheaper__.
+ A large neural network with more parameters is __prone to overfitting__. It is also __computationally expensive__. In this case you can use regularization (increase \lambda) to address the overfitting.

Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set.

##### Model Selection:

Choosing M the order of polynomials.

How can we tell which parameters \theta to leave in the model (known as "model selection")?

There are several ways to solve this problem:

+ Get more data (very difficult).
+ Choose the model which best fits the data without overfitting (very difficult).
+ Reduce the opportunity for overfitting through regularization.

__Bias: approximation error (Difference between expected value and optimal value)__

+ High Bias = UnderFitting (BU)
+ $J_{train}(\theta)$ and $J_{CV}(\theta)$ both will be high and $J_{train}(\theta) \approx J_{CV}(\theta)$

__Variance: estimation error due to finite data__

+ High Variance = OverFitting (VO)
+ $J_{train}(\theta)$ is low and $J_{CV}(\theta)\gg J_{train}(\theta)$

__Intuition for the bias-variance trade-off:__

+ Complex model $\implies$ sensitive to data $\implies$> much affected by changes in X $\implies$ high variance, low bias.
+ Simple model $\implies$ more rigid $\implies$ does not change as much with changes in X $\implies$ low variance, high bias.

One of the most important goals in learning: finding a model that is just right in the bias-variance trade-off.

__Regularization Effects:__

+ Small values of \lambda allow model to become finely tuned to noise leading to large variance $\implies$ overfitting.
+ Large values of \lambda pull weight parameters to zero leading to large bias $\implies$ underfitting.

__Model Complexity Effects:__

+ Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
+ Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
+ In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

__A typical rule of thumb when running diagnostics is:__

+ More training examples fixes high variance but not high bias.
+ Fewer features fixes high variance but not high bias.
+ Additional features fixes high bias but not high variance.
+ The addition of polynomial and interaction features fixes high bias but not high variance.
+ When using gradient descent, decreasing lambda can fix high bias and increasing lambda can fix high variance (lambda is the regularization parameter).
+ When using neural networks, small neural networks are more prone to under-fitting and big neural networks are prone to over-fitting. Cross-validation of network size is a way to choose alternatives.



### Errata

#### Errata in the Graded Quizzes

Quiz questions in Week 6 should refer to linear regression, not logistic regression (typo only).

#### Errata in the Video Lectures

In the "Regularization and Bias/Variance" video

The slide "Linear Regression with Regularization" has an error in the formula for $J(\theta)\;$: the regularization term should go from $j=1$ up to $n$ (and not $m$), that is $\frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$. The quiz in the video "Regularization and Bias/Variance" has regularization terms for $J_{train}$ and $J_{CV}$, while the rest of the video stresses that these should not be there. Also, the quiz says "Consider regularized logistic regression," but exhibits cost functions for regularized linear regression.

At around 5:58, Prof. Ng says, "picking theta-5, the fifth order polynomial". Instead, he should have said the fifth value of $\lambda (0.08)$, because in this example, the polynomial degree is fixed at $d = 4$ and we are varying $\lambda$.

In the "Advice for applying ML" set of videos

Often (if not always) the sums corresponding to the regularization terms in J(\theta) are (erroneously) written with j running from 1 to m. In fact, j should run from 1 to n, that is, the regularization term should be $\lambda \sum_{j=1}^n \theta_j^2\lambda \sum_{j=1}^n \theta_j^2$. The variable m is the number of $(x,y)$ pairs in the set used to calculate the cost, while n is the largest index of $j$ in the $\theta_j$ parameters or in the elements $x_j$ of the vector of features.

In the "Advice for Applying Machine Learning" section, the figure that illustrates the relationship between lambda and the hypothesis. used to detect high variance or high bias, is incorrect. $J_{train}$ is low when lambda is small (indicating a high variance problem) and high when lambda is high (indicating a high bias problem).

Video (10-2: Advice for Applying Machine Learning -- hypothesis testing)

The slide that introduces Training/Testing procedure for logistic regression, (around 04:50) the cost function is incorrect. It should be:

$$J_{test}(\theta) = −\dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} (y^{(i)}_{test} \cdot \log(h\theta(x^{(i)}_{test}))+(1−y^{(i)}_{test}) \cdot \log(1−h\theta(x^{(i)}_{test})))$$

Video Regularization and Bias/Variance (00:48)

Regularization term is wrong. Should be $\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$ and not sum over $m$.

Videos 10-4 and 10-5: current subtitles are mistimed

Looks like the videos were updated in Sept 2014, but the subtitles were not updated accordingly. (10-3 was also updated in Aug 2014, but the subtitles were updated)

#### Errata in the ex5 programming exercise

In ex5.m at line 104, the reference to "slide 8 in ML-advice.pdf" should be "Figure 3 in ex5.pdf".


### Quiz: Advice for Applying Machine Learning



