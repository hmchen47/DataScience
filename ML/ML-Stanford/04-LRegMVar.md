(setq markdown-css-paths '("https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.css"))

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

Linear regression with multiple variables is also known as "__multivariate linear regression__".

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

+ Linear Regression Model
  + Hypothesis function

    $$h_\theta(x) =\theta^T x = \theta_0 x_0 + \theta_1 x_1 + \ldots + \theta_n x_n \Longrightarrow \Theta^T \cdot X$$

  + Parameters: 
  
    $$\quad \theta_0, \theta_1, \ldots, \theta_n \Longrightarrow \Theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \ldots \\ \theta_n \end{bmatrix}\quad$$

    $\Theta$: a $(n+1)$-dimensional vector

  + Cost function: 

    $$\begin{array}{cc}
      J(\theta_0, \theta_1, \ldots, \theta_n) & = & \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2 \\ \\
      J(\Theta) &= & \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2
    \end{array}$$

  + IVQ: When there are n features, we define the cost function as

    $$\displaystyle J(\Theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$.

    For linear regression, which of the following are also equivalent and correct definitions of $J(\theta)$?

    $$\begin{array}{lrcl}
      1. & J(\Theta) & = & \frac{1}{2m} \sum_{i=1}^m (\Theta^T x^{(i)} - y^{(i)})^2 \\\\
      2. & J(\Theta) & = & \frac{1}{2m} \sum_{i=1}^m ((\sum_{j=0}^n \theta_j x_j^{(i)}) - y^{(i)})^2 \text{(inner sum starts at 0)} \\\\
      3. & J(\Theta) & = & \frac{1}{2m} \sum_{i=1}^m ((\sum_{j=1}^n \theta_j x_j^{(i)}) - y^{(i)})^2 \text{(inner sum starts at 1)} \\\\
      4. & J(\Theta) & = & \frac{1}{2m} \sum_{i=1}^m ((\sum_{j=0}^n \theta_j x_j^{(i)}) - (\sum_{j=0}^n y_j^{(i)}))^2 \text{(inner sum starts at 0)} \\\\
    \end{array}$$

    Ans: 12

  + Gradient descent:

    Repeat {

    $$\theta_j := \theta_j -\alpha \frac{\partial}{\partial \theta_j} J(\Theta) = \theta_j -\alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1, \ldots, \theta_n)$$
    <div style="text-align: center; padding-top: 0.5em;padding-left: calc(50vw - 5em);"> (simultaneously update for every j = 0, 1, ..., n$) </div><br/>}

+ Gradient Descent Algorithm
  + Linear regression: $n = 1$

    Repeat {

      $$\begin{array}{ccc}\theta_0 &:=& \theta_j -\alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})= \theta_0 -\alpha \frac{\partial}{\partial \theta_0} J(\Theta) \\\\
      \theta_1 & := & \theta_j -\alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})= \theta_0 -\alpha \frac{\partial}{\partial \theta_1} J(\Theta) \end{array}$$
      <span style="padding-top: 0.5em; padding-left: calc(50vw - 5em);"> (simultaneously update for </span> $\theta_0, \theta_1$)<br/>
    }
  
  + Multivariate linear repression ($n \geq 1$)

    Repeat {

      $$\theta_j := \theta_j -\alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})= \theta_j -\alpha \frac{\partial}{\partial \theta_j} J(\Theta)$$
      <span style="padding-top: 0.5em; padding-left: calc(50vw - 5em);"> (simultaneously update </span> $\theta_j \;$ for $j=0, 1, \ldots, n$)<br/>
    }
  
    Extended version: with $x_0^{(i)} = 1$

      $$\begin{array}{ccc}
        \theta_0 &:=& \theta_0 -\alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_0^{(i)} \\\\
        \theta_1 &:=& \theta_1 -\alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \\\\
        \theta_2 &:=& \theta_2 -\alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \\\\
        & \cdots &
      \end{array}$$


---------------------------------------

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

repeat until convergence:{

  $$\begin{array}{ccc}
    \theta_0 & := & \theta_0 - \alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_0^{(i)} \\
    \theta_1 & := & \theta_1 - \alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \\
    \theta_2 & := & \theta_2 - \alpha \displaystyle \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \\
    & \cdots &
  \end{array}$$
}

In other words:

repeat until convergence:{
  
  $$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \quad\quad \text{ for } j:=0,1, \ldots, n$$
}

The following image compares gradient descent with one variable to gradient descent with multiple variables:

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/supplement/aEN5G/gradient-descent-for-multiple-variables">
    <img src="images/m04-01.png" style="margin: 0.1em;" alt="Compariosons of gradient descent with one variable to gradient descent with multiple variables" title="Compariosons of gradient descent with one variable to gradient descent with multiple variables" width="500">
  </a></div>
</div>


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/04.2-LinearRegressionWithMultipleVariables-GradientDescentForMultipleVariables.0f58c050b22b11e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1552608000&Signature=P-2a7Ej5Iowtrpld~JV3wEGBqdb3qaEDTUqQ0RMXD93OENziZV1bWqjDWgA0X9myzKQw5Jy~RSabXMN5a0lTUjsbIVn-UprqKNpSti6OVU5ubBqw0FHnJdJUQnUP5jCh1ieCnnm8~IV~WgDMigTZRXQdhqOYrC2anCNHF9fFCcQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/sLOuPyjYTC-zrj8o2HwvZw?expiry=1552608000000&hmac=vjbkm5A-sImaZ0AyBXk8H_oLtvauI42SjRQc7QWteu0&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
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





