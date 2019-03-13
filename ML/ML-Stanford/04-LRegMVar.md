# Linear Regression with Multiple Variables

## Multivariate Linear Regression

### Multiple Features

#### Lecture Notes

+ Multiple features

  | Size ($\text{feet}^2)$, $x_1$ | Number of bedrooms, $x_2$ | Number of floors, $x_3$ | Age of home (years), $x_4$ | Price (\$1000), y |
  |:--:|:--:|:--:|:--:|:--:|
  | 2104 | 5 | 1 | 45 | 460 |
  | 1416 | 3 | 2 | 40 | 232 |
  | 1534 | 3 | 2 | 30 | 315 |
  | 852 | 2 | 1 | 36 |  |
  | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |
+ Notation:
  + $n$ = number of features; e.g. $n=4$
  + $x^{(i)}$ = input (features) of $i^{th}$ training example, e.g. $\quad x^{(2)} = \begin{bmatrix} 1416 \\ 3 \\ 2\\ 40 \end{bmatrix}$
  + $x^{(i)}_j$ = input features $j$ of $i^{th}$ training example; e.g. $\quad x_3^{(2)} = 2$
+IVQ: In the training set above, what is $x_1^{(4)}$?

  1. The size (in $\text{feet}^2$) of the 1st home in the training set
  2. The age (in years) of the 1st home in the training set
  3. The size (in $\text{feet}^2$) of the 4th home in the training set
  4. The age (in years) of the 4th home in the training set

  Ans: 3

+ Example: hypothesis function

  $h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_4 x_4 = h_\theta (x) = 80 + 0.2 x_1 + 0.01 x+2 + 3 x_3 - 2 x_4$

+ Hypothesis:

  $$h_\theta (x) = \theta_0 + \theta_1 x_1 + \ldots + \theta_n x_n$$

  For convenience of notation, define $x_0 = 1$ (a.k.a. $x_0^{(i)}=1 \; \text{ for } i = 1, \ldots, m$)

  $X = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^{n+1} \quad \quad \quad \Theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} \in \mathbb{R}^{n+1}$

  $\Theta^T = \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n \end{bmatrix}$ as a $1 \times (n+1)$ matrix

  $\begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix}$

  $\therefore h_\theta (x) = \theta_0 x_0 + \theta_1 x_1 + \ldots + \theta_n x_n = \Theta^T X \Longrightarrow$ Multivariate linear regression.

---------------------------------------

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

+ $x_j^{(i)}$ = value of feature $j$ in the $i^{th}$ training example
+ $x^{(i)}$ = the column vector of all the feature inputs of the $i^{th}$ training example
+ $m$ = the number of training examples
+ $n=∣x^{(i)}∣$: the number of features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$

In order to develop intuition about this function, we can think about $\theta_0$ as the basic price of a house, $\theta_1$ as the price per square meter, $\theta_2$ as the price per floor, etc. $x_1$ will be the number of square meters in the house, $x_2$ the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

$h_\theta(x)= \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} = \theta^T x$

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume $x_{0}^{(i)} = 1 \text{ for } (i\in { 1,\dots, m } )$. This allows us to do matrix operations with theta and x. Hence making the two vectors '$\theta$' and $x^{(i)}$ match each other element-wise (that is, have the same number of elements: $n+1$).]


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/04.1-LinearRegressionWithMultipleVariables-MultipleFeatures.35214c30b22b11e49f072fa475844d6b/full/360p/index.mp4?Expires=1552608000&Signature=KD5dVQ8p5nVqXIVpC2ss~0ID-PnQFSIzbHntms25lkL-xAoMAdTPxAZPibn-686wp3JcJUrrhy1YlNGXG4wdfjHfXCJIdKSCsvI3Vo7r17-RBksAEzN~A-71MmrIuAjxxeBC~uBBJrNthfyFrxjbmnW7OqRTC5tmHgCZ~i1WrRI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/HpFJLMwbEeaTLA5NOVzoSA?expiry=1552608000000&hmac=6NtQvD-CDimjRuqMwlOLdy20G1PflNLx8a_JPyqFLGg&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Gradient Descent for Multiple 

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Gradient Descent in Practice I - Feature 

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Gradient Descent in Practice II - Learning Rate

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Features and Polynomial Regression

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Computing Parameters Analytically


### Normal Equation

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Normal Equation

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Submitting Programming Assignments

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Working on and Submitting Programming Assignments

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Programming tips from Mentors

#### Lecture Notes


---------------------------------------


#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Review

### Lecture Slides




### Quiz: Linear Regression with Multiple Variables





