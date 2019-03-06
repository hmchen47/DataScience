(setq markdown-css-paths '("https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.css"))


# Model and Cost Function

## Model Representation

### Lecture Notes

+ Housing Prices (Portland, OR)
    <a href="https://d3c33hcgiwev3.cloudfront.net/_ec21cea314b2ac7d9e627706501b5baa_Lecture2.pdf?Expires=1552003200&Signature=TGQs5L1O0PHw2SBFtYJ4q5n3rNLp0mWpKigVJHX~vOlMdHSuPvqHnyuQGUSCPtonZ-IFHiq3F~SWhBMrwxbzerZQlLdy9-SGe5UBrMDE0rOLj-mj5VO3QchKzHbRLnmyxGu-C65y2r-CV8wmRqvN5JpKOKeqGzpWT0mV8InqUoQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A"> <br/>
        <img src="images/m02-01.png" style="display: block; margin: auto; background-color: black" alt="We're going to use a data set of housing prices from the city of Portland, Oregon. And here I'm gonna plot my data set of a number of houses that were different sizes that were sold for a range of different prices. Let's say that given this data set, you have a friend that's trying to sell a house and let's see if friend's house is size of 1250 square feet and you want to tell them how much they might be able to sell the house for. Well one thing you could do is fit a model. Maybe fit a straight line to this data. Looks something like that and based on that, maybe you could tell your friend that let's say maybe he can sell the house for around $220,000. So this is an example of a supervised learning algorithm." title="Example: Housing Prices (Portland, OR)" width="350" >
    </a>
    + Supervised Learning: Given the "right answer" for each example in the data
    + Regression Problem: Predict real-valued output
    + cf: Classification - discrete-valued output
    + Training set of housing prices (Portland, OR)

        | Size in feet$^2$ (x) | Price (\$) in 1000's (y) |
        |----------------------|--------------------------|
        | 2104 | 460 |
        | 1416 | 232 |
        | 1534 | 315 |
        | 852 | 178 |
        | ... | ... |
    + Notation: 
        + $m$: Number of training examples
        + $x$: "input" variables / features
        + $y$: "output" variable / "target" feature
    + $(x, y)$: one training example
    + $(x^{(i)}, y^{(i)})$: ith training example, e.g., $x^{(1)} = 2104$, $y^{(1)} = 460$, $x^{(2)} = 1416$, $y^{(1)} = 232$
    + IVQ: Consider the training set shown below. $(x^{(i)}, y^{(i)})$ is the $i^{th}$ training example. What is $y^{(3)}$?

        Ans: 315

+ Modeling
    <a href="https://d3c33hcgiwev3.cloudfront.net/_ec21cea314b2ac7d9e627706501b5baa_Lecture2.pdf?Expires=1552003200&Signature=TGQs5L1O0PHw2SBFtYJ4q5n3rNLp0mWpKigVJHX~vOlMdHSuPvqHnyuQGUSCPtonZ-IFHiq3F~SWhBMrwxbzerZQlLdy9-SGe5UBrMDE0rOLj-mj5VO3QchKzHbRLnmyxGu-C65y2r-CV8wmRqvN5JpKOKeqGzpWT0mV8InqUoQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A"> <br/>
        <img src="images/m02-02.png" alt="So here's how this supervised learning algorithm works. We saw that with the training set like our training set of housing prices and we feed that to our learning algorithm. Is the job of a learning algorithm to then output a function which by convention is usually denoted lowercase h and h stands for hypothesis And what the job of the hypothesis is, is, is a function that takes as input the size of a house like maybe the size of the new house your friend's trying to sell so it takes in the value of x and it tries to output the estimated value of y for the corresponding house. So h is a function that maps from x's to y's. People often ask me, you know, why is this function called hypothesis. Some of you may know the meaning of the term hypothesis, from the dictionary or from science or whatever. It turns out that in machine learning, this is a name that was used in the early days of machine learning and it kinda stuck. 'Cause maybe not a great name for this sort of function, for mapping from sizes of houses to the predictions, that you know.... I think the term hypothesis, maybe isn't the best possible name for this, but this is the standard terminology that people use in machine learning." title="ML Modeling Flow" width="250"> &nbsp;&nbsp;
        <img src="images/m02-03.png" alt="the next thing we need to decide is how do we represent this hypothesis h. For this and the next few videos, I'm going to choose our initial choice , for representing the hypothesis, will be the following. We're going to represent h as follows. And we will write this as h<u>theta(x) equals theta<u>0</u></u> plus theta<u>1 of x. And as a shorthand, sometimes instead of writing, you</u> know, h subscript theta of x, sometimes there's a shorthand, I'll just write as a h of x. But more often I'll write it as a subscript theta over there. And plotting this in the pictures, all this means is that, we are going to predict that y is a linear function of x. Right, so that's the data set and what this function is doing, is predicting that y is some straight line function of x. That's h of x equals theta 0 plus theta 1 x, okay? And why a linear function? Well, sometimes we'll want to fit more complicated, perhaps non-linear functions as well. But since this linear case is the simple building block, we will start with this example first of fitting linear functions, and we will build on this to eventually have more complex models, and more complex learning algorithms. Let me also give this particular model a name. This model is called linear regression or this, for example, is actually linear regression with one variable, with the variable being x. Predicting all the prices as functions of one variable X. And another name for this model is univariate linear regression. And univariate is just a fancy way of saying one variable. So, that's linear regression." title="How to represent h?" width="300" >
    </a>
    + Machine Learning Flowchart (right fig)
    + How do we represent $h$? (left fig)
    + $h$: hypothesis
    + Linear regression with one variable $(x)$
    + Univariate linear regression

-----------------------------

To establish notation for future use, we’ll use $x^{(i)}$ to denote the “input” variables (living area in this example), also called input features, and $y^{(i)}$ to denote the “output” or target variable that we are trying to predict (price). A pair $(x^{(i)} , y^{(i)})$ is called a __training example__, and the dataset that we’ll be using to learn—a list of m training examples $(x(i),y(i)); i=1, \ldots,m$-is called a __training set__. Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use $X$ to denote the space of input values, and $Y$ to denote the space of output values. In this example, $X = Y = ℝ$.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X \implies Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function $h$ is called a __hypothesis__. Seen pictorially, the process is therefore like this:

<a href="https://www.coursera.org/learn/machine-learning/supplement/cRa2m/model-representation"> <br/>
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1552003200000&hmac=04jS07nGMhKt8IEBPIDY0EvFbbQUXPyTxOfRcEr3-pg" style="display: block; margin: auto; background-color: black" alt="Flowchart" title="Modeling Process" width="300" >
</a>

When the target variable that we’re trying to predict is _continuous_, such as in our housing example, we call the learning problem a __regression problem__. When $y$ can take on only a small number of _discrete_ values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a __classification problem__.


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/02.1-V2-LinearRegressionWithOneVariable-ModelRepresentation.b2ac9470b22b11e4bb7e93e7536260ed/full/360p/index.mp4?Expires=1552003200&Signature=I15Zax3xiG0hbz~UeSR2dgBsxA9impcIIpjeafrZZ8A21I5xIkaX8YUAA93xoi2vgyGevBiNwbyOKHGSxtbnxxihxEvF8~Clp8aw4VGp1h5KmduLVg79lbWy5GW2~DVkBnDC0idTT1F1Tiz7MDMfIKsHfm94prc9PcM9-2a6DpU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/NYch6VFJQ1mHIelRSTNZ4w?expiry=1552003200000&hmac=blzFIgr38ulWIqkQe1RaEKvy_2xm5BnD8rBEeszkPuY&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>

## Cost Function

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Cost Function - Intuition I

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Cost Function - Intuition II

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Parameter Learning

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Gradient Descent

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Gradient Descent Intuition

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Gradient Descent For Linear Regression

### Lecture Notes


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video> <br/>


## Review

### Lecture Slides

#### ML:Linear Regression with One Variable

__Model Representation__

Recall that in _regression problems_, we are taking input variables and trying to fit the output onto a continuous expected result function.

Linear regression with one variable is also known as "univariate linear regression."

Univariate linear regression is used when you want to predict a __single output__ value y from a __single input__ value x. We're doing __supervised learning__ here, so that means we already have an idea about what the input/output cause and effect should be.

__The Hypothesis Function__

Our hypothesis function has the general form:

$$ \hat{y} = h_\theta(x) = \theta_0 + \theta_1 x$$

Note that this is like the equation of a straight line. We give to $h_\theta(x)$ values for $\theta_0$ and $\theta_1$ to get our estimated output $\hat{y}$. In other words, we are trying to create a function called $h_\theta$ that is trying to map our input data (the x's) to our output data (the y's).

Example:

    Suppose we have the following set of training data:

<table style="text-align: center;"><tbody>
  <tr><th>input x</th><th>output y</th></tr>
  <tr><td>0</td><td>4</td></tr>
  <tr><td>1</td><td>7</td></tr>
  <tr><td>2</td><td>7</td></tr>
  <tr><td>3</td><td>8</td></tr>
</tbody></table>


Now we can make a random guess about our $h_\theta$ function: $\theta_0=2$ and $\theta_1 = 2$. The hypothesis function becomes $h_\theta(x)=2+2x$.

So for input of 1 to our hypothesis, y will be 4. This is off by 3. Note that we will be trying out various values of $\theta_0$ and $\theta_1$ to try to find values which provide the best possible "fit" or the most representative "straight line" through the data points mapped on the x-y plane.


#### Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's compared to the actual output y's.

$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$
 

To break it apart, it is $\frac{1}{2} \bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta (x_{i}) - y_{i}$, or the difference between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\left(\frac{1}{2m}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.

Now we are able to concretely measure the accuracy of our predictor function against the correct results we have so that we can predict new results we don't have.

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make straight line (defined by $h_\theta(x)$) which passes through this scattered set of data. Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. In the best case, the line should pass through all the points of our training data set. In such a case the value of $J(\theta_0, \theta_1)$ will be 0.


#### ML:Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$ (actually we are graphing the cost function as a function of the parameter estimates). This can be kind of confusing; we are moving up to a higher level of abstraction. We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting particular set of parameters.

We put $\theta_0$ on the x axis and $\theta_1$  on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters.

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent, and the size of each step is determined by the parameter α, which is called the learning rate.

The gradient descent algorithm is:

repeat until convergence:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$

where

$j=0,1$ represents the feature index number.

Intuitively, this could be thought of as:

repeat until convergence:

$\theta_j := \theta_j - \alpha$ [\text{Slope of tangent aka derivative in j dimension}]


__Gradient Descent for Linear Regression__

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to (the derivation of the formulas are out of the scope of this course, but a really great one can be found here):

repeat until convergence: 

$$\begin{array}{rcl}
    \theta_0 & := & \theta_0 - \alpha \dfrac{1}{m} \displaystyle \sum^{m}_{i=0} (h_\theta (x_i) - y_i) \\ \\
    \theta_1 & := & \theta_1 - \alpha \dfrac{1}{m} \displaystyle \sum_{i=0}^m ((h_\theta (x_i) - y_i) x_i)
\end{array}$$

where $m$ is the size of the training set, $\theta_0$ a constant that will be changing simultaneously with $\theta_1$ and $x_{i}$, $y_{i}$, are values of the given training set (data).

Note that we have separated out the two cases for $\theta_j$ into separate equations for $\theta_0$ and $\theta_1$; and that for $\theta_1$ we are multiplying $x_{i}$ at the end due to the derivative.

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

Gradient Descent for Linear Regression: visual worked example
Some may find the following [video](https://www.youtube.com/watch?v=WnqQrPNYz5Q) useful as it visualizes the improvement of the hypothesis as the error function reduces.

### Errata

#### Linear Regression With One Variable

+ A general note about the graphs that Prof Ng sketches when discussing the cost function. The vertical axis can be labeled either 'y' or 'h(x)' interchangeably. 'y' is the true value of the training example, and is indicated with a marker. 'h(x)' is the hypothesis, and is typically drawn as a curve. The scale of the vertical axis is the same, so both can be plotted on the same axis.
+ In the video "Cost Function - Intuition I", at about 6:34, the value given for J(0.5) is incorrect.
+ Parameter Learning: Video "Gradient Descent for Linear Regression": At 6:15, the equation Prof Ng writes in blue "h(x) = -900 - 0.1x" is incorrect, it should use "+900".


#### Gradient Descent for Linear Regression

+ At Timestamp 3:27 of this video lecture, the equation for θ1 is wrong, please refer to first line of Page 6 of ex1.pdf (Week 2 programming Assignment) for model equation (The last x is X superscript i, subscript j (Which is 1 in this case, as it is of θ1)). θ0 is correct as it will be multiplied by 1 anyways(value of X superscript i, subscript 0 is 1), as per the model equation.

<br/>

## Quiz: Linear Regression with One Variable




