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

+ Training set -> Linear Regression <br/>
    Hypothesis: $h_\theta (x) = \theta_0 + \theta_1 \cdot x$
    + $\theta_i$: parameters
    + How to choose $\theta_i$'s?
    <a href="https://www.coursera.org/learn/machine-learning/supplement/cRa2m/model-representation"> <br/>
      <img src="images/m02-04.png" style="display: block; margin: auto; background-color: black" alt="Flowchart" title="Modeling Process" width="450" >
    </a>
    + IVQ: Consider the plot below of $h_\theta(x) = \theta_0 + \theta_1x$. What are $\theta_0$ and $\theta_1$?

        <a href="url"> <br/>
            <img src="http://spark-public.s3.amazonaws.com/ml/images/2.2-quiz-1-fig.jpg" style="display: block; margin: auto; background-color: black" alt="A line of $h_\theta(x)$ as a function of $$x$$. The line goes through points (0, 0.5), (1, 1.5), and (2, 2.5)." title="caption" width="150" >
        </a>

        a. $\theta_0 = 0, \theta_1 = 1$ <br/>
        b. $\theta_0 = 0.5, \theta_1 = 1$ <br/>
        c. $\theta_0 = 1, \theta_1 = 0.5$ <br/> 
        d. $\theta_0 = 1, \theta_1 = 1$

        Ans: b

+ The cost function
  + Idea: Choose $\theta_0$, $\theta_1$ so that $h_\theta (x)$ is close to $y$ for our training examples $(x, y)$
  + Objective:

    $$\min_{\theta_0, \theta_1} \frac{1}{2m} \displaystyle \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2$$

    where $h_\theta (x^{(i)}) = \theta_0 + \theta_1 \cdot x^{(i)}$
  + Cost function = Squared error function: $J(\theta_0, \theta_1)$

    $$J(\theta_0, \theta_1) = \frac{1}{2m} \displaystyle \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2$$

    <br/>

    $$\min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$$

----------------------------------------

We can measure the accuracy of our hypothesis function by using a __cost function__. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$
 

To break it apart, it is $\frac{1}{2} \bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta (x_{i}) - y_{i}$, or the difference between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\left(\frac{1}{2}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ ​	 term. The following image summarizes what the cost function does:

<a href="https://www.coursera.org/learn/machine-learning/supplement/nhzyF/cost-function">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/R2YF5Lj3EeajLxLfjQiSjg_110c901f58043f995a35b31431935290_Screen-Shot-2016-12-02-at-5.23.31-PM.png?expiry=1552003200000&hmac=yrnrGUcnPynJQzUJwxp0F_61mbGjDb6HPGzPJOk5ohs" style="display: block; margin: auto; background-color: black" alt="text" title="Cost Function Analysis" width="450">
</a>

### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Cost Function - Intuition I

### Lecture Notes

+ Linear Regression
    + Hypothesis: $h_\theta (x) = \theta_0 + \theta_1 \cdot x$
    + Parameters: $\theta_0$, $\theta_1$
    + Cost Function: $J(\theta_0, \theta_1) = \frac{1}{2m} \displaystyle \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$
    + Goal: $\displaystyle \min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$

+ Simplified Linear Regression - $\theta_0 = 0$
    + Hypothesis: $h_\theta (x) = \theta_1 \cdot x$
    + Parameter: $\theta_1$
    + Cost Function: $J(\theta_0, \theta_1) = \frac{1}{2m} \displaystyle \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$ &nbsp;&nbsp; with &nbsp;$h_\theta (x^{(i)}) = \theta_1 \cdot x^{(i)}$
    + Goal: $\displaystyle \min_{\theta_0, \theta_1} J(\theta_1)$

+ Example: samples - $(1, 1), (2, 2), (3, 3)$
    + $h_\theta (x)$: $\forall \;$ fixed $\theta_1 \implies$ a function of $x$
    + $J(\theta_1)$: function of the parameter $\theta_1$
    + $\displaystyle \theta_1 = 1 \Longrightarrow J(\theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2 = \displaystyle \frac{1}{2m} \sum_{i=1}^m (\theta_1 x^{(i)} - y^{(i)})^2 = \frac{1}{2m} (0 + 0 + 0)^2 = 0$.
    + $\displaystyle \theta_1 = 0.5 \Longrightarrow J(\theta_1) = \frac{1}{2m} [(0.5 - 1)^2 + (1 - 2)^2 + (1.5 - 3)^2] = 0.58$
    + IVQ: Suppose we have a training set with m=3 examples, plotted below. Our hypothesis representation is $h_\theta(x) = \theta_1 x$, with parameter $\theta_1$. The cost function $J(\theta_1)$ is $J(\theta_1) = \frac{1}{2m} \sum^m_{i=1} (h_\theta (x^{(i)}) - y^{(i)})^2$. What is $J(0)$?

        Ans: $\displaystyle \theta_1 = 0 \Longrightarrow J(0) = \frac{1}{2m} [(0 - 1)^2 + (0 - 2)^2 + (0 - 3)^2] = \frac{1}{6} \cdot 14 \approx 2.3$
    + Cost Function:
        <a href="https://www.coursera.org/learn/machine-learning/supplement/u3qF5/cost-function-intuition-i"> <br/>
            <img src="http://spark-public.s3.amazonaws.com/ml/images/2.3-quiz-1-fig.jpg" style="background-color: black; margin-right: 2em;" alt="Sample values" title="Samples" width="250" >
            <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1552003200000&hmac=Bs3Bof2Tuoxu6hHIq48mjhBGq_eZilTn0oEsYxm8EFQ" style="background-color: black" alt="Cost function" title="Cost function" width="250" >
        </a>
    + Simplified hypothesis: $\theta_1 = 1$ with minimized cost function

---------------------------

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by $h_\theta(x)$ which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ will be $0$. The following example shows the ideal situation where we have a cost function of $0$.

<a href="https://www.coursera.org/learn/machine-learning/supplement/u3qF5/cost-function-intuition-i">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1552003200000&hmac=3jx6NYib8V-a-WwcXcN1QI9yLqv--lI2AT17INmnS9Y" style="display: block; margin: auto; background-color: black" alt="text" title="caption" width="450" >
</a>

When $\theta_1 = 1$, we get a slope of $1$ which goes through every single data point in our model. Conversely, when $\theta_1 = 0.5$, we see the vertical distance from our fit to the data points increase.

<a href="https://www.coursera.org/learn/machine-learning/supplement/u3qF5/cost-function-intuition-i">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1552003200000&hmac=qVn9V1G6K8TjG9YOIMmcE2CHRAfI6ng4Kp4HMH1ID3o" style="display: block; margin: auto; background-color: black" alt="text" title="caption" width="450" >
</a>

This increases our cost function to 0.58. Plotting several other points yields to the following graph:

<a href="https://www.coursera.org/learn/machine-learning/supplement/u3qF5/cost-function-intuition-i"> <br/>
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1552003200000&hmac=Bs3Bof2Tuoxu6hHIq48mjhBGq_eZilTn0oEsYxm8EFQ" style="display: block; margin: auto; background-color: black" alt="text" title="caption" width="250" >
</a>

Thus as a goal, we should try to minimize the cost function. In this case, $\theta_1 = 1$ is our global minimum.

### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/02.3-V2-LinearRegressionWithOneVariable-CostFunctionIntuitionI.b1dc4c20b22b11e4bb7e93e7536260ed/full/360p/index.mp4?Expires=1552003200&Signature=HSeW2MqsxSgZG03tokcoijSAUcSWiPPEQJvlwjU5HNzUYd22lToIC8uHqBt1~W8Pli-iUB5MTirXtF-puAPDFIClaOuLliaZiI9~~S8C2KDbHBIkb-KedMyypMFaKWNLwHPTvvFEhe0-bbGgCap5laf2BlYFmjUHCs67gDu45Xs_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/TO-3vcj1Eea4ORI689i_OA?expiry=1552003200000&hmac=An8mckESG8bPNWVKpnB-64UEFawDuUpqoEglCczCLbg&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>

## Cost Function - Intuition II

### Lecture Notes

+ Linear Regression
  + Hypothesis: $h_\theta (x) = \theta_0 + \theta_1 \cdot x$
    + Parameters: $\theta_0$, $\theta_1$
    + Cost Function: $\displaystyle J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$
    + Goal: $\displaystyle \min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$
  + Example: $h_\theta (x) = 50 + 0.06 \cdot x$
    <a href="https://www.coursera.org/learn/machine-learning/lecture/nwpe2/cost-function-intuition-ii"> <br/>
      <img src="images/m02-05.png" alt="Samples, Cost Function" title="Linear regression: 50 + 0.06x" width="300">
      <img src="images/m02-06.png" alt="Cost functionwith $\theta_0$ and $\theta_1$" title="Linear regression: 50 + 0.06x" width="260">
    </a>

+ Examples of Cost Function with $\theta_0$ & $\theta_1$
  + $\theta_0 = 800$ and $\theta_1 = -0.15$ (left fig)
  + $\theta_0 = 360$ and $\theta_1 = 0$ (right fig)

  <a href="https://www.coursera.org/learn/machine-learning/supplement/9SEeJ/cost-function-intuition-ii">
      <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1552089600000&hmac=gYmT8EQiUJeiGPlZ_4W7fJySu6qHvu2x0K8sEdcxEhI" alt="Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for $J(\theta_0,\theta_1)$ and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when $\theta_0 = 800$ and $\theta_1 = -0.15$." title="Value of cost function w/ $\theta_0 = 800$ and $\theta_1 = -0.15$" width="300">
      <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1552089600000&hmac=BqDr2d3GZL3h8tUTwBdHeT7PBqvlPAWB1SFXChKP7KA" alt="When $\theta_0 = 360$ and $\theta_1 = 0$, the value of $J(\theta_0,\theta_1)$ in the contour plot gets closer to the center thus reducing the cost function error. " title="Value of cost function w/ $\theta_0 = 800$ and $\theta_1 = -0.15$" width="285" >
  </a>

+ Minimized cost function: $\theta_0 = 250$ and $\theta_1 = .12$

  <a href="https://www.coursera.org/learn/machine-learning/supplement/9SEeJ/cost-function-intuition-ii">
      <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1552089600000&hmac=k28KfhXiOqnzzRcJHIwjyCLSzJ7hqPzkgST2xsergns" style="display: block; margin: auto; background-color: black;" alt="The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_1$ and $\theta_0$ tend to be around 0.12 and 250 respectively." title="Minimized Cost function with $\theta_0 = 250$ and $\theta_1 = .12$" width="350" >
  </a>

----------------------------------

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

<a href="https://www.coursera.org/learn/machine-learning/supplement/9SEeJ/cost-function-intuition-ii">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1552089600000&hmac=gYmT8EQiUJeiGPlZ_4W7fJySu6qHvu2x0K8sEdcxEhI" style="display: block; margin: auto; background-color: black;" alt="the three green points found on the green line above have the same value for $J(\theta_0,\theta_1)$ and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when $\theta_0 = 800$ and $\theta_1 = -0.15$." title="Vaueof cost function w/ $\theta_0 = 800$ and $\theta_1 = -0.15$" width="350">
</a>

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for $J(\theta_0,\theta_1)$ and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when $\theta_0 = 800$ and $\theta_1 = -0.15$. Taking another $h(x)$ and plotting its contour plot, one gets the following graphs:

<a href="https://www.coursera.org/learn/machine-learning/supplement/9SEeJ/cost-function-intuition-ii">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1552089600000&hmac=BqDr2d3GZL3h8tUTwBdHeT7PBqvlPAWB1SFXChKP7KA" style="display: block; margin: auto; background-color: black;" alt="When $\theta_0 = 360$ and $\theta_1 = 0$, the value of $J(\theta_0,\theta_1)$ in the contour plot gets closer to the center thus reducing the cost function error." title="Value of cost function w/ $\theta_0 = 800$ and $\theta_1 = -0.15$" width="350" >
</a>

When $\theta_0 = 360$ and $\theta_1 = 0$, the value of $J(\theta_0,\theta_1)$ in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

<a href="https://www.coursera.org/learn/machine-learning/supplement/9SEeJ/cost-function-intuition-ii">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1552089600000&hmac=k28KfhXiOqnzzRcJHIwjyCLSzJ7hqPzkgST2xsergns" style="display: block; margin: auto; background-color: black;" alt="The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_1$ and $\theta_0$ tend to be around 0.12 and 250 respectively." title="Minimized Cost function with $\theta_0 = 250$ and $\theta_1 = .12$" width="350" >
</a>

The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_1$ and $\theta_0$ tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>

## Parameter Learning


## Gradient Descent

### Lecture Notes

+ Simpliest Gradient descent
  + Objective: Have some function $J(\theta_0, \theta_1)$ <br/>
    Want $\displaystyle \min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$
  + Outline
    + start with some $\theta_0, \theta_1$
    + keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up with at a minimum

+ Generalized gradient decent
  + Objective: Have some function $J(\Theta)$ where $\Theta = (\theta_0, \theta_1, \ldots, \theta_n)$ <br/>
    Want $\displaystyle \min_{\Theta} J(\Theta)$
  + Outline
    + start with some $\Theta$
    + keep changing $\Theta$ to reduce $J(\Theta)$ until we hopefully end up with at a minimum

+ Examples of Gradient Decent
  <a href="url"> <br/>
      <img src="images/m02-07.png" alt="start at some point (star sign), then take a step to a lowest point around, and repeat to find the lowest point of the countour as reaching a local optimum" title="Gradient descent start at a given point 1" width="310" >
      <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1552089600000&hmac=JlEz_gvm4vxzrv4qHvdM79jBfV6i4M2G-YeRgkDB9tY" alt="start at some point (star sign), then take a step to a lowest point around, and repeat to find the lowest point of the countour as reaching a local optimum" title="Gradient descent start at a given point 2" width="300" >
  </a>

+ Gradient descent algorithm
  + Def: repeat until convergence

    $$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) \;\; \text{for } j = 0, 1$$
    + $:=$: assignment, take the right-hand side value asn assign to the symbol right-hand side
    + $=$: truth association, comparison
    + $\alpha$: learning rate, step size
  + Correct: Simultaneous update

    $$\begin{array}{rcl} \text{temp0} & := & \theta_0 - \alpha \displaystyle \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) \\ \\
    \text{temp1} & := & \theta_1 - \alpha \displaystyle \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) \\ \\
    \theta_0 & := & \text{temp0} \\ \\
    \theta_1 & := & \text{temp1} \end{array}$$
  + Incorrect execution order:

    $$\begin{array}{lrcl} 1. &\text{temp0} & := & \theta_0 - \alpha \displaystyle \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) \\ \\
    2. &\theta_0 & := & \text{temp0} \\ \\
    3. &\text{temp1} & := & \theta_1 - \alpha \displaystyle \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) \\ \\
    4. & \theta_1 & := & \text{temp1} \end{array}$$
  + IVQ: Suppose $\theta_0= 1$, $\theta_1= 2$, and we simultaneously update $\theta_0$ and $\theta_1$ using the rule: $\theta_j := \theta_j + \sqrt{\theta_0 \theta_1}$ (for j = 0 and j=1) What are the resulting values of $\theta_0$ and $\theta_1$?

      a. $\theta_0 = 1, \theta_1 =2$ <br/>
      b. $\theta_0 = 1+\sqrt{2}$, $\theta_1 =2 + \sqrt{2}$ <br/>
      c. $\theta_0 = 2 + \sqrt{2}$, $\theta_1 =1 + \sqrt{2}$ <br/>
      d. $\theta_0 = 1+\sqrt{2}$, $\theta_1 =2 + \sqrt{(1 + \sqrt{2})\cdot 2}$ <br/>

      Ans: b


-------------------------------------------------

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where __gradient descent__ comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put $\theta_0$ on the x axis and $\theta_1$ on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

<a href="https://www.coursera.org/learn/machine-learning/supplement/2GnUg/gradient-descent">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1552089600000&hmac=JlEz_gvm4vxzrv4qHvdM79jBfV6i4M2G-YeRgkDB9tY" style="display: block; margin: auto; background-color: black;" alt="text" title="caption" width="350" >
</a>

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

repeat until convergence:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$

where $j=0,1$ represents the feature index number.

At each iteration j, one should simultaneously update the parameters $\theta_1, \theta_2,\cdots,\theta_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.

<a href="https://www.coursera.org/learn/machine-learning/supplement/2GnUg/gradient-descent">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56.png?expiry=1552089600000&hmac=xG3dZq13G0Z6bFReQHcR34QVWgKpGkeGjCWfaQ6S2hU" style="display: block; margin: auto; background-color: black;" alt="text" title="caption" width="450" >
</a>


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/02.5-V2-LinearRegressionWithOneVariable-GradientDescent.c89f04c0b22b11e4964ea1de90934151/full/360p/index.mp4?Expires=1552089600&Signature=OYexzUWHuwS-sEwyjF5QPgFtbtz8NuDt1NNwnkOYR~jrgO~hPVx~H1xVVdE2I58APwHu8yctYhYJ~HVXeIUtW81sXtnw5bRPN5veRU3KO-MhXyw09cWWA-Pw5ViNyzGgY7DB-TIgIwfT5QLF86X4kYjyb6b6-THcF576BoCZsmw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/hxyZ4qoIQbacmeKqCIG2-g?expiry=1552089600000&hmac=3bFuP_SKsOc5I3K_RmTXgw1dJmv2TAcD0cIwJTimJXY&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>

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




