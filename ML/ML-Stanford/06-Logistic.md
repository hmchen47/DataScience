# Logistic Regression

## Classification and Representation


### Classification

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Hypothesis Representation

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Decision Boundary

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Logistic Regression Model


### Cost Function

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Simplified Cost Function and Gradient Descent

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


### Advanced Optimization

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Multiclass Classification

#### Lecture Notes




-------------------------------------------------




#### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Review

### Lecture Slides

Now we are switching from regression problems to __classification problems__. Don't be confused by the name "Logistic Regression"; it is named that way for historical reasons and is actually an approach to classification problems, not regression problems.

#### Binary Classification

Instead of our output vector y being a continuous range of values, it will only be 0 or 1.

$y \in \{0,1\}$

Where $0$ is usually taken as the "negative class" and $1$ as the "positive class", but you are free to assign any representation to it.

We're only doing two classes for now, called a "Binary Classification Problem."

One method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. This method doesn't work well because classification is not actually a linear function.

##### Hypothesis Representation

Our hypothesis should satisfy:

$$0 \leq h_\theta (x) \leq 1$$

Our new form uses the "__Sigmoid Function__," also called the "__Logistic Function__":

$$\begin{array}{rcl} h_\theta(x) & = & g(\theta^Tx) \\\\ z & = & \theta^Tx \\\\ g(z) & = & \dfrac{1}{1+e^{−z}} \end{array}$$

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/resources/Zi29t"><br/>
    <img src="images/m06-01.png" style="margin: 0.1em;" alt="text" title="caption" width="450">
  </a></div>
</div>

The function $g(z)$, shown here, maps any real number to the $(0, 1)$ interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification. Try playing with [interactive plot of sigmoid function](https://www.desmos.com/calculator/bgontvxotm).

We start with our old hypothesis (linear regression), except that we want to restrict the range to 0 and 1. This is accomplished by plugging $\theta^Tx$ into the Logistic Function.

$h_\theta$ will give us the __probability__ that our output is 1. For example, $h_\theta(x)=0.7$ gives us the probability of $70\%$ that our output is 1.

$$h_\theta(x)=P(y=1|x;\theta)=1−P(y=0|x;\theta)$$
<br/>

$$P(y=0|x;\theta)+P(y=1|x;\theta)=1$$

Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is $70\%$, then the probability that it is 0 is $30\%$).

#### Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

$$\begin{array}{rcl} h_\theta(x) \geq 0.5 & \longrightarrow & y = 1 \\ h_\theta(x)< 0.5 &\longrightarrow & y=0 \end{array}$$

The way our logistic function $g$ behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

$g(z) \geq 0.5$ when $z \geq 0$

Remember.-

$$\begin{array}{rcl} z=0,e0=1 & \longrightarrow & g(z)=1/2 \\ z \rightarrow \infty, e^{−\infty} \rightarrow 0 & \longrightarrow & g(z)=1 \\ z→−\infty,e^{\infty} \rightarrow \infty & \longrightarrow & g(z)=0 \end{array}$$

So if our input to $g$ is $\theta^T X$, then that means:

$$h_\theta(x)=g(\theta^Tx) \geq 0.5$$

when $\theta^Tx \geq 0$

From these statements we can now say:

$$\begin{array}{rcl} \theta^Tx \geq 0 \Rightarrow y=1 \\ \theta^Tx < 0 \Rightarrow y=0 \end{array}$$

The __decision boundary__ is the line that separates the area where $y = 0$ and where $y = 1$. It is created by our hypothesis function.

Example:

$$\begin{array}{c} \theta = \begin{bmatrix} 5 \\ −1 \\ 0 \end{bmatrix} \\\\ y = 1 \text{  if  } 5+(−1)x_1 + 0x_2 \geq 0 \;\rightarrow\; 5−x_1 \geq 0 \;\rightarrow\; −x_1 \geq −5 \;\rightarrow\; x_1  \leq  5 \end{array}$$

In this case, our decision boundary is a straight vertical line placed on the graph where $x_1 = 5$, and everything to the left of that denotes $y = 1$, while everything to the right denotes $y = 0$.

Again, the input to the sigmoid function $g(z)$ (e.g. $\theta^T X$) doesn't need to be linear, and could be a function that describes a circle (e.g. $z = \theta_0 + \theta_1 x_1^2 +\theta_2 x_2^2$) or any shape to fit our data.


#### Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:

$$\begin{array}{lcl}J(\theta) & = & \dfrac{1}{m} \displaystyle \sum_{i=1}^{m} \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\\\ \mathrm{Cost}(h_\theta(x),y) & = & \left\{ \begin{array}{ll} −\log(h_\theta(x)), & \text{ if } y=1, \\  −\log(1−h_\theta(x)), & \text{  if  } y = 0 \end{array} \right. \end{array}$$

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/resources/Zi29t"><br/>
    <img src="images/m06-02.png" style="margin: 0.1em;" alt="text" title="caption" width="250">
    <img src="images/m06-03.png" style="margin: 0.1em;" alt="text" title="caption" width="218">
  </a></div>
</div>

The more our hypothesis is off from y, the larger the cost function output. If our hypothesis is equal to y, then our cost is 0:

$$\begin{array}{lcl} \mathrm{Cost}(h_\theta(x),y) & = & 0 \quad \text{ if } \quad h_\theta(x)=y \\\\ \mathrm{Cost}(h_\theta(x),y) & \rightarrow & \infty \left \{ \begin{array}{l} \text{ if } y=0 \text{ and } h_\theta(x) \rightarrow 1 \\ \text{ if } y=1 \text{ and } h_\theta(x) \rightarrow 0 \end{array} \right. \end{array}$$

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that $J(\theta)$ is convex for logistic regression.


#### Simplified Cost Function and Gradient Descent

We can compress our cost function's two conditional cases into one case:

$$\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

Notice that when y is equal to 1, then the second term $(1-y)\log(1-h_\theta(x))$ will be zero and will not affect the result. If y is equal to 0, then the first term $-y \log(h_\theta(x))−y$ will be zero and will not affect the result.

We can fully write out our entire cost function as follows:

$$J(\theta) = - \dfrac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$$

A vectorized implementation is:

$$\begin{array}{rcl} h & = & g(X\theta) \\\\ J(\theta) & = & \dfrac{1}{m}\cdot (−y^T \log(h) − (1−y)^T \log(1−h)) \end{array}$$


##### Gradient Descent

Remember that the general form of gradient descent is:

Repeat{

$$\theta_j := \theta_j − \alpha \dfrac{\partial}{\partial \theta_j} J(\theta)$$

}

We can work out the derivative part using calculus to get:

Repeat{
  
$$\theta_j := \theta_j − \dfrac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)})−y^{(i)})x^{(i)}_j$$

}

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in $\theta$.

A vectorized implementation is:

$$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$$

)

##### Partial derivative of $J(\theta)$

First calculate derivative of sigmoid function (it will be useful while finding partial derivative of $J(\theta))$:

$$\begin{array}{rcl} \sigma(x)^\prime & = & (\dfrac{1}{1+e^{−x}})^\prime = \dfrac{−(1+e^{−x})^\prime}{(1+e^{−x})^2} = \dfrac{−1^\prime − (e^{−x})^\prime}{(1+e^{−x})^2} = \dfrac{0 − (−x)^\prime (e^{−x})}{(1+e^{−x})^2} = \dfrac{−(−1)(e^{−x})}{(1+e^{−x})^2} = \dfrac{e^{−x}}{(1+e^{−x})^2} \\\\ & = & (\dfrac{1}{1+e^{−x}})(\dfrac{e^{−x}}{1+e^{−x}}) = \sigma(x)(\dfrac{+1−1+e^{-x}}{1+e^{-x}}) = \sigma(x)(\dfrac{1+e^{-x}}{1+e^{-x}} − \dfrac{1}{1+e^{-x}}) = \sigma(x)(1−\sigma(x)) \end{array}$$

Now we are ready to find out resulting partial derivative:

$$\begin{array}{rcl} \dfrac{\partial}{\partial \theta_j} J(\theta) & = & \dfrac{\partial}{\partial \theta_j} \dfrac{-1}{m} \displaystyle \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)}))+(1−y^{(i)}) \log(1−h_\theta(x^{(i)}))] \\\\ & = &  -\dfrac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)} \dfrac{\partial}{\partial \theta_j} \log(h_\theta(x^{(i)}))+(1−y^{(i)}) \dfrac{\partial}{\partial \theta_j} \log(1−h_\theta(x^{(i)}))] \\\\ & = & -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ \dfrac{y^{(i)}\frac{\partial}{\partial \theta_j} h_\theta(x^{(i)})}{h_\theta(x^{(i)})} + \dfrac{(1−y^{(i)}) \frac{\partial}{\partial \theta_j}(1−h_\theta(x^{(i)}))}{1−h_\theta(x^{(i)})} \right] \\\\ & = & -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ \dfrac{y^{(i)}\frac{\partial}{\partial \theta_j} \sigma(\theta^Tx^{(i)})}{h_\theta(x^{(i)})} + \dfrac{(1−y^{(i)})\frac{\partial}{\partial \theta_j} (1−\sigma(\theta^Tx^{(i)}))}{1−h_\theta(x^{(i)})} \right] \\\\ & = & -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ \dfrac{y^{(i)}\sigma(\theta^Tx^{(i)})(1−\sigma(\theta^Tx^{(i)}))\frac{\partial}{\partial \theta_j} \theta^Tx^{(i)}}{h_\theta(x^{(i)})} + \dfrac{−(1−y^{(i)})\sigma(\theta^Tx^{(i)})(1−\sigma(\theta^Tx^{(i)}))\frac{\partial}{\partial \theta_j} \theta^Tx^{(i)}}{1−h_\theta(x^{(i)})} \right] \\\\ & = & -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ \dfrac{y^{(i)}h_\theta(x^{(i)})(1−h_\theta(x^{(i)}))\frac{\partial}{\partial \theta_j} \theta^Tx^{(i)}}{h_\theta(x^{(i)})} − \dfrac{(1−y^{(i)})h_\theta(x^{(i)})(1−h_\theta(x^{(i)}))\frac{\partial}{\partial \theta_j} \theta^Tx^{(i)}}{1−h_\theta(x^{(i)})} \right] \\\\ & = & -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ y^{(i)}(1−h_\theta(x^{(i)}))x^{(i)}j−(1−y^{(i)})h_\theta(x^{(i)})x^{(i)}_j \right] = -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ y^{(i)}(1−h_\theta(x^{(i)}))−(1−y^{(i)})h_\theta(x^{(i)}) \right] x^{(i)}_j \\\\ & = & -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ y^{(i)}−y^{(i)}h_\theta(x^{(i)})−h_\theta(x^{(i)})+y^{(i)}h_\theta(x^{(i)}) \right] x^{(i)}_j = -\dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ y^{(i)}−h_\theta(x^{(i)}) \right] x^{(i)}_j \\\\ & = & \dfrac{1}{m} \displaystyle \sum_{i=1}^m \left[ h_\theta(x^{(i)})−y^{(i)} \right] x^{(i)}_j \end{array}$$

The vectorized version;

$$\nabla J(\theta) = \frac{1}{m} \cdot X^T \cdot \left(g\left(X\cdot\theta\right) - \vec{y}\right)$$

#### Advanced Optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize $\theta$ that can be used instead of gradient descent. A. Ng suggests not to write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value $\theta$:

$$J(\theta) \quad\quad \text{and} \quad\quad \dfrac{\partial}{\partial \theta_j} J(\theta)$$

We can write a single function that returns both of these:

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use octave's `fminunc()` optimization algorithm along with the `optimset()` function that creates an object containing the options we want to send to `fminunc()`. (Note: the value for MaxIter should be an integer, not a character string - errata in the video at 7:30)

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
      initialTheta = zeros(2,1);
      [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function `fminunc()` our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

#### Multiclass Classification: One-vs-all

Now we will approach the classification of data into more than two categories. Instead of y = {0,1} we will expand our definition so that $y = \{0,1 \cdots n\}$.

In this case we divide our problem into $n+1$ (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

$$\begin{array}{ccc} y & \in & \{0,1 \cdots n\} \\\\ h^{(0)}_\theta(x) & = & P(y=0|x;\theta) \\ h^{(1)}_\theta(x) & = & P(y=1|x;\theta) \\  & \vdots & \\ h^{(n)}_\theta(x) & = & P(y=n|x;\theta) \\\\ \text{prediction } & = & \displaystyle \max_i (h^{(i)}\theta(x)) \end{array}$$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.



### Errata

#### Decision Boundary

At 1:56 in the transcript, it should read 'sigmoid function' instead of 'sec y function'.


#### Cost Function

The section between 8:30 and 9:20 is then repeated from 9:20 to the quiz. The case for y=0 is explained twice.

#### Simplified Cost Function and Gradient Descent

These following mistakes also exist in the video:

+ 6.5: On page 19 in the PDF, the leftmost square bracket seems to be slightly misplaced.
+ 6.5: It seems that the factor 1/m is accidentally omitted between pages 20 and 21 when the handwritten expression is converted to a typeset one (starting at 6:53 of the video)


#### Advanced Optimization

In the video at 7:30, the notation for specifying MaxIter is incorrect. The value provided should be an integer, not a character string. So (...'MaxIter', '100') is incorrect. It should be (...'MaxIter', 100). This error only exists in the video - the exercise script files are correct.


### Quiz: Logistic Regression



