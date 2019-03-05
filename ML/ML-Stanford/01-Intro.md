# Introduction

## Welcome

### Lecture Notes

+ Machine Learning
    + Grew out of work in AI
    + New capability for computers
+ Examples:
    + Database mining
        + Large datasets from growth of automation/web.
        + E.g., Web click data, medical records, biology, engineering
    + Applications can’t program by hand.
        + E.g., Autonomous helicopter, handwriting recognition, most of Natural Language Processing (NLP), Computer Vision.
    + Self-customizing programs
        + E.g., Amazon, Netflix product recommendations
    + Understanding human learning (brain, real AI).

### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/01.1-V3-Introduction-Welcome.49abef20b22b11e49c064db6ead92550/full/360p/index.mp4?Expires=1551916800&Signature=Qa8m7P7kXSOcM2PJ~BeRk3A0b4S-LVATIBZFxlgIbbEkollCLevcEBnsfYiJq~J~n~DXLQl4XLeytpCeQUllz2feuyRvBb220yOKtGkGAoLvDjb1P9URygWS3HHbiSnXeqCtgzBfWA3kWfS2DnPIzWqSzhuWEai-xzylPFIZzfo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180" alt="Welcome">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/OajtTUNVSJqo7U1DVcia_w?expiry=1551916800000&hmac=doYGysJbj4AzkrX31UeeSwa4OvGFWxEVtaXucb3cu30&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>


## What is Machine Learning?

### Lecture Notes

+ Machine Learning definition
    + Arthur Samuel (1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
    + Tom Mitchell (1998) Well-posed Learning Problem: A computer program is said to learn from experience __E__ with respect to some task __T__ and some performance measure __P__, if its performance on T, as measured by P, improves with experience E.

+ IVQ: Suppose your email program watches which emails you do or do not mark as spam, and based on that learns how to better filter spam. What is the task T in this setting?

    a. Classifying emails as spam or not spam. <br/>
    b. Watching you label emails as spam or not spam. <br/>
    c. The number (or fraction) of emails correctly classified as spam/not spam. <br/>
    d. None of the above—this is not a machine learning problem.

    Ans: a <br/>
    Explanation: “A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.” <br/>
    a. classifying emails -> T; b. label emails -> E; c. correctly classified -> P

+ Algorithm classification
    + Machine learning algorithms:
        + Supervised learning
        + Unsupervised learning
    + Others: Reinforcement learning, recommender systems.


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/IRPs0hQ3EeelIwrMcHgCog.processed/full/360p/index.mp4?Expires=1551916800&Signature=BFHUgjno0G0OyRStcxYdXmab5HFjIJGczhyBeowgH0JSdfiJgR9sWXyxVWbdd15hZVRlLXzCTpKacaTSJd~fjTl60y-2Ao3ZpTqXtE7v~ERI5zFqTxgSkni7KRT80Z2pNjYBQBCjnnsYwRFniy-0NaqNh5-5xMsmfOrsKZA0dA4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/Nwmp7RSHEee9zwpiIySM9A?expiry=1551916800000&hmac=w_zH9VmrykQfpuJ5e0r9vgADjKWUi43r3al-Hnbx-lk&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>


## How to Use Discussion Forums



## Supervised Learning




## Unsupervised Learning



## Unsupervised Learning



## Who are Mentors?



## Get to Know Your Classmates



## Frequently Asked Questions



## Lecture Slides

### [Week 1 Lecture Notes](https://www.coursera.org/learn/machine-learning/resources/JXWWS)


#### ML:Introduction

__What is Machine Learning?__

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

    + E = the experience of playing many games of checkers
    + T = the task of playing checkers.
    + P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

    supervised learning, OR

    unsupervised learning.


__Supervised Learning__

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. Here is a description on Math is Fun on Continuous and Discrete Data.


__Example 1:__

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.


__Example 2:__

(a) Regression - Given a picture of Male/Female, We have to predict his/her age on the basis of given picture.

(b) Classification - Given a picture of Male/Female, We have to predict Whether He/She is of High school, College, Graduate age. Another Example for Classification - Banks have to decide whether or not to give a loan to someone on the basis of his credit history.


__Unsupervised Learning__

Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results, i.e., there is no teacher to correct you.

Example:

Clustering: Take a collection of 1000 essays written on the US Economy, and find a way to automatically group these essays into a small number that are somehow similar or related by different variables, such as word frequency, sentence length, page count, and so on.

Non-clustering: The "Cocktail Party Algorithm", which can find structure in messy data (such as the identification of individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)). Here is an [answer](https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms) on Quora to enhance your understanding.


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

#### ML:Linear Algebra Review

Khan Academy has excellent [Linear Algebra Tutorials](https://www.khanacademy.org/#linear-algebra)

#### Matrices and Vectors

Matrices are 2-dimensional arrays:

$$\left [ \begin{array}{ccc}
    a & b & c \\ d & e & f \\ g & h & i \\ j & k & l
\end{array} \right ]$$

The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

$$\left [ \begin{array}{c} w \\ x\\ y\\ z \end{array} \right ]$$

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.

__Notation and terms:__

+ $A_{ij}$ refers to the element in the ith row and jth column of matrix $A$.
+ A vector with 'n' rows is referred to as an 'n'-dimensional vector
+ $v_i$ refers to the element in the ith row of the vector.
+ In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
+ Matrices are usually denoted by uppercase names while vectors are lowercase.
+ "Scalar" means that an object is a single value, not a vector or matrix.
+ $\mathbb{R}$ refers to the set of scalar real numbers
+ $\mathbb{R^n}$ refers to the set of n-dimensional vectors of real numbers


#### Addition and Scalar Multiplication

Addition and subtraction are __element-wise__, so you simply add or subtract each corresponding element:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] + \left [ \begin{array}{cc} w & x \\ y & z \end{array} \right ] = \left [ \begin{array}{cc} a+w & b+x \\ c+y & d+z \end{array} \right ]$$

To add or subtract two matrices, their dimensions must be __the same__.

In scalar multiplication, we simply multiply every element by the scalar value:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] * x = \left [ \begin{array}{cc} a*x & b*x \\ c*x & d*x \end{array} \right ]$$


#### Matrix-Vector Multiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$ \left [ \begin{array}{cc} a & b \\ c & d \\ e & f \end{array} \right ] * \left [ \begin{array}{c} x \\ y \end{array} \right ] = \left [ \begin{array}{cc} a*x + b*y \\ c*x + d*y \\ e*x+f*y \end{array} \right ]$$

The result is a __vector__. The vector must be the __second__ term of the multiplication. The number of __columns__ of the matrix must equal the number of __rows__ of the vector.

An __m x n matrix__ multiplied by an __n x 1__ vector results in an __m x 1__ vector.


#### Matrix-Matrix Multiplication

We multiply two matrices by breaking it into several vector multiplications and concatenating the result

$$\left [ \begin{array}{cc} a & b \\ c & d \\ e & f \end{array} \right ] * \left [ \begin{array}{cc} w & x \\ y & z \end{array} \right ] = \left [ \begin{array}{cc} a*w + b*y & a*x+b*z \\ c*w+d*y & dc*x+d*z \\ e*w+f*y & e*x+f*z \end{array} \right ] $$

An __m x n matrix__ multiplied by an __n x o matrix__ results in an __m x o__ matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of __columns__ of the first matrix must equal the number of __rows__ of the second matrix.

#### Matrix Multiplication Properties

+ Not commutative. $A∗B \neq B∗A$
+ Associative. $(A∗B)∗C=A∗(B∗C)$

The __identity matrix__, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$$\left [ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{array} \right ] $$

When multiplying the identity matrix after some matrix $(A∗I)$, the square identity matrix should match the other matrix's __columns__. When multiplying the identity matrix before some other matrix $(I∗A)$, the square identity matrix should match the other matrix's __rows__.


#### Inverse and Transpose

The __inverse__ of a matrix A is denoted A−1. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the pinv(A) function [1] and in matlab with the inv(A) function. Matrices that don't have an inverse are singular or degenerate.

The __transposition__ of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or A':

$$A = \left [ \begin{array}{cc} a & b \\ c & d \\ e & f \end{array} \right ]$$

$$ A^T = \left [ \begin{array}{ccc} a & c & e \\ b & d & f \end{array} \right ]$$


### Errata Week 1

#### Introduction

+ Supervised Learning: 1:25: Describing the curve as quadratic is confusing since the independent variable is price, but the plot's X-axis represents area.
+ Unsupervised Learning: 6:56 - the mouse does not point to the correct audio sample being played on the slide. Each subsequent audio sample has the mouse pointing to the previous sample.
+ Unsupervised Learning: 12:50 - the slide shows first option "Given email labelled as span/not spam, learn a spam filter" as one of the answers as well. Whereas, in the audio Professor puts it in Supervised Learning category.


#### Linear Regression With One Variable

+ A general note about the graphs that Prof Ng sketches when discussing the cost function. The vertical axis can be labeled either 'y' or 'h(x)' interchangeably. 'y' is the true value of the training example, and is indicated with a marker. 'h(x)' is the hypothesis, and is typically drawn as a curve. The scale of the vertical axis is the same, so both can be plotted on the same axis.
+ In the video "Cost Function - Intuition I", at about 6:34, the value given for J(0.5) is incorrect.
+ Parameter Learning: Video "Gradient Descent for Linear Regression": At 6:15, the equation Prof Ng writes in blue "h(x) = -900 - 0.1x" is incorrect, it should use "+900".


#### Gradient Descent for Linear Regression

+ At Timestamp 3:27 of this video lecture, the equation for θ1 is wrong, please refer to first line of Page 6 of ex1.pdf (Week 2 programming Assignment) for model equation (The last x is X superscript i, subscript j (Which is 1 in this case, as it is of θ1)). θ0 is correct as it will be multiplied by 1 anyways(value of X superscript i, subscript 0 is 1), as per the model equation.


#### Linear Algebra Review

+ Matrix-Matrix Multiplication: 7:14 to 7:33 - While exploring a matrix multiplication, Andrew solved the problem correctly below, but when he tried to rewrite the answer in the original problem, one of the numbers was written incorrectly. The correct result was (matrix 9 15) and (matrix 7 12), but when it was rewritten above it was written as (matrix 9 15) and (matrix 4 12). The 4 should have been a 7. (Thanks to John Kemp and others). This has been partially corrected in the video - third subresult matrix shows 7 but the sound is still 4 for both subresult and result matrices. Subtitle at 6:48 should be “two is seven and two”, and subtitle at 7:14 should be “seven twelve and you”.
+ 3.4: Matrix-Matrix Multiplication: 8:12 - Andrew says that the matrix on the bottom left shows the housing prices, but those are the house sizes as written above
+ 3.6: Transpose and Inverse: 9:23 - While demonstrating a transpose, an example was used to identify B(subscript 12) and A(subscript 21). The correct number 3 was circled in both cases above, but when it was written below, it was written as a 2. The 2 should have been a 3. (Thanks to John Kemp and others)
Addition and scalar multiplication video
Spanish subtitles for this video are wrong. Seems that those subtitles are from another video.


## Quiz: Introduction



