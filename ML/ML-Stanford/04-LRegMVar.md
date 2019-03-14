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

+ Feature Scaling
  + Idea: Make sure features are on a similar scale.
  + Example: 
    + $x_1 = \text{size } (0-2000 \text{ feet}^2) \Longrightarrow x_1 = \dfrac{\text{ size } (\text{feet}^2)}{2000} \rightarrow 0 \leq x_1 \leq 1$
    + $x_2 = \text{ number of bedrooms} (1-5) \Longrightarrow x_2 = \dfrac{\text{  number of bedrooms  }}{5} \rightarrow 0 \leq x_1 \leq 1$
  + General rule: Get every feature into approximately a $-1 \leq x_i \leq 1$ range.
  + Example:

    $$\begin{array}{rclcl}
      0 & \leq x_1 \leq & 3 & \rightarrow & \surd \text{ (close enough)} \\
      -2 & \leq x_2 \leq & 0.5 & \rightarrow & \surd \text{ (close enough)}\\
      -100 & \leq x_3 \leq & 100 & \rightarrow & \text{ scaling} \\
      -0.0001 & \leq x_4 \leq & 0.0001 & \rightarrow & \text{ scaling}
    \end{array}$$
  + General rule of thumb: (acceptable ranges)
    + $-3$ to $3$
    + $-1/3$ to $1/3$

+ Mean Normalization
  + Replace $x_i$ with $x_i - \mu_i$ to make features have approximately zero mean (Do not apply to $x_0 = 1$).
  + Examples:
    + average size = 1000: $\quad x_1 = \dfrac{size - 1000}{2000}$

    + 1~5 bedrooms: $\quad x_2 = \dfrac{\#bedrooms - 2}{5}$ (acceptable though $4 = 5 -1$)
  + General rule:

    $$ x_i := \dfrac{x_i - \mu_i}{s_i}$$
    + $\mu_i\;$: average value of $x_i$ in training set
    + $s_i\;$: range ($\max - \min$) or standard deviation ($\sigma_i$)of $x_i$ value set
  + IVQ: Suppose you are using a learning algorithm to estimate the price of houses in a city. You want one of your features $x_i$ to capture the age of the house. In your training set, all of your houses have an age between 30 and 50 years, with an average age of 38 years. Which of the following would you use as features, assuming you use feature scaling and mean normalization?
    1. $x_1 = \text{ age of house}$
    2. $x_i = \text{ age of house} / 50$
    3. $x_i = (\text{ age of house } - 38) / 50$
    4. $x_i = (\text{ age of house } - 38) / 20$

    Ans: 4


---------------------------------------

We can speed up gradient descent by having each of our input values in roughly the same range. This is because $\theta$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

$$−1 \leq x_{(i)} \leq 1 \quad\quad \text{or} \quad\quad −0.5 \leq x_{(i)} \leq 0.5$$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are __feature scaling__ and __mean normalization__. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

$$x_i := \dfrac{x_i - \mu_i}{s_i}$$​

Where $μ_i$ is the average of all the values for feature ($i$) and $s_i$ is the range of values $(\max - \min)$, or $s_i$ is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if $x_i$ represents housing prices with a range of 100 to 2000 and a mean value of 1000, then, $x_i := \dfrac{\text{price}-1000}{1900}$.


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/04.3-LinearRegressionWithMultipleVariables-GradientDescentInPracticeIFeatureScaling.1d388a20b22b11e4bb7e93e7536260ed/full/360p/index.mp4?Expires=1552608000&Signature=YuT6lbftkWikDFbV13T27VCb09dNRotJYj6k-uQbsQ4lwMNoCtJzhKSiQc98N2S1Vyslph3E-C0l0pU1NOTBcfaaopbqDGLak0GDMVitZH9fJW7C3d5fjOfmmgNnTsn-7ULF0MffgwpeU6FGZ8yI7bFj4ylzBiL1vwEldb5xjFE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/sTG3UqavTLuxt1Kmr2y7LA?expiry=1552608000000&hmac=rnsTdJwMKZMxRoaaOuWb2Wf_XvbNY6g6EPDlAC-CPC8&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Gradient Descent in Practice II - Learning Rate

#### Lecture Notes

+ Gradient descent

  $$\theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j} J(\Theta)$$
  + "Debugging": how to make suer gradient descent is working correctly
  + How to choose learning rate $\alpha$

+ Automatic convergence test: Convergence threshold

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.coursera.org/learn/machine-learning/supplement/TnHvV/gradient-descent-in-practice-ii-learning-rate">
      <img src="images/m04-02.png" style="margin: 0.1em;" alt="text" title="No of iterations vs $\min J(\theta)$" width="350">
    </a></div>
  </div>

  + $J(\Theta)$ should decrease each iteration
  + Declare convergence if $J(\Theta)$ decreases by less than $\varepsilon = 10^{-3}$ in one iteration.

+ Learning rate
    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div><a href="https://www.coursera.org/learn/machine-learning/supplement/TnHvV/gradient-descent-in-practice-ii-learning-rate">
        <img src="images/m04-03.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
      </a></div>
    </div>

    + $J(\Theta) \uparrow$ as number of iterations $\uparrow \quad \Longrightarrow \quad$ gradient not working, $\alpha$ too big
    + For sufficiently small $\alpha$, $J(\Theta)$ should decrease on every iteration.
    + If $\alpha$ is too small, gradient descent can be slow to converge.
    + IVQ: Suppose a friend ran gradient descent three times, with $\alpha = 0.01$, $\alpha = 0.1$, and $\alpha = 1$, and got the following three plots (labeled A, B, and C):

    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div><a href="url">
        <img src="http://spark-public.s3.amazonaws.com/ml/images/4.4-quiz-1-plots.png" style="margin: 0.1em;" alt="Plot A shows $$J(\theta)$$ sharply decreasing after the first few iterations, and then leveling off. Plot B shows $J(\theta)$ slowly decreasing at every iteration, and still decreasing at the last iteration. Plot C shows $$J(\theta)$$ increasing every iteration, with the increase growing in every iteration." title="Diagrams for IVQ" width="450">
      </a></div>
    </div>

    Which plots corresponds to which values of \alphaα?
    1. A is $\alpha=0.01$, B is $\alpha = 0.1$, C is $\alpha =1$.
    2. A is $\alpha=0.1$, B is $\alpha = 0.01$, C is $\alpha =1$.
    3. A is $\alpha=1$, B is $\alpha = 0.01$, C is $\alpha =0.1$.
    4. A is $\alpha=1$, B is $\alpha = 0.1$, C is $\alpha =0.01$.

    Ans: 2 <br/>
    Explanation: In graph C, the cost function is increasing, so the learning rate is set too high. Both graphs A and B converge to an optimum of the cost function, but graph B does so very slowly, so its learning rate is set too low. Graph A lies between the two.


+ Summary
  + If $\alpha$ too small: slow convergence.
  + If $\alpha$ too large: $J(\Theta)$ may not decrease on every iteration; may not converge; slow converge also possible
  + Best practice: to choose $\alpha$, try

    $$\ldots, 0.001, 0.003, , 0.01, 0.03, 0.1, 0.3, 1, \ldots$$



---------------------------------------

__Debugging gradient descent__. Make a plot with number of iterations on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

__Automatic convergence test__. Declare convergence if $J(\theta)$ decreases by less than $E$ in one iteration, where $E$ is some small value such as $10^{−3}$. However in practice it's difficult to choose this threshold value.

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/supplement/TnHvV/gradient-descent-in-practice-ii-learning-rate">
    <img src="images/m04-02.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
  </a></div>
</div>

It has been proven that if learning rate α is sufficiently small, then J(θ) will decrease on every iteration.

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/supplement/TnHvV/gradient-descent-in-practice-ii-learning-rate">
    <img src="images/m04-03.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
  </a></div>
</div>

To summarize:

+ If $\alpha$ is too small: slow convergence.
+ If $\alpha$ is too large: ￼may not decrease on every iteration and thus may not converge.


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/04.4-LinearRegressionWithMultipleVariables-GradientDescentInPracticeIILearningRate.0f202420b22b11e4901abd97e8288176/full/360p/index.mp4?Expires=1552694400&Signature=Ivon6YjQJpvHHPHkviOja3BtxADiZMMZQAvX4BVTygCC0QkogFWqvuLFqHS-fWh11BHIQaHRU0fISam21DJQ9hmSzYu1VWFWtVzdYb4fdrmDsxM~GFRE5OG1lqYpQ1-YYIrP5-oI6wygrbzvOay4esGxd36w5SBFZiU0wjrqknU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/R7F5CEt0R8uxeQhLdNfLQA?expiry=1552694400000&hmac=XWt-7i8N_5-hQeBGd_WftWywfOOpIKoiVgJhfNzBoSY&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
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





