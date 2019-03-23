# Regularization: Solving the Problem of Overfitting

## The Problem of Overfitting

### Lecture Notes




-------------------------------------------




### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Cost Function

### Lecture Notes




-------------------------------------------




### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Regularized Linear Regression

### Lecture Notes




-------------------------------------------




### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Regularized Logistic Regression

### Lecture Notes




-------------------------------------------




### Lecture Video 

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Review

### Lecture Slides

#### The Problem of Overfitting

Regularization is designed to address the problem of overfitting.

High bias or underfitting is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. eg. if we take $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$ then we are making an initial assumption that a linear model will fit the training data well and will be able to generalize but that may not be the case.

At the other extreme, overfitting or high variance is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:
    1) Manually select which features to keep.
    2) Use a model selection algorithm (studied later in the course).
2) Regularization

Keep all the features, but reduce the parameters $\theta_j$.

Regularization works well when we have a lot of slightly useful features.


#### Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

$$\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$$ 

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:

$$\min_\theta\ \dfrac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\cdot\theta_3^2 + 1000\cdot\theta_4^2$$

We've added two extra terms at the end to inflate the cost of $\theta_3$ and $\theta_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near zero. This will in turn greatly reduce the values of $\theta_3x^3$ and $\theta_4x^4$ in our hypothesis function.

We could also regularize all of our theta parameters in a single summation:

$$\min_\theta\ \dfrac{1}{2m}\ \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2 \right]$$

The $\lambda$, or lambda, is the __regularization parameter__. It determines how much the costs of our theta parameters are inflated. You can visualize the effect of regularization in this [interactive plot](https://www.desmos.com/calculator/1hexc8ntqp)

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting.


#### Regularized Linear Regression

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

##### Gradient Descent

We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

Repeat {
  
$$\begin{array}{rcl} \theta_0 & := & \theta_0 − \alpha \dfrac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})−y^{(i)})x^{(i)}_0 \\\\ \theta_j & := & \theta_j − \alpha \left[ (\dfrac{1}{m} \displaystyle \sum_{i=1}^m (h_\theta(x^{(i)}) − y^{(i)}) x^{(i)}j) + \dfrac{λ}{m} \theta_j \right] \quad\quad  j \in \{1,2, \cdots, n\} \end{array}$$
}

The term $\frac{\lambda}{m} \theta_j$ performs our regularization.

With some manipulation our update rule can also be represented as:

$$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

The first term in the above equation, $1 - \alpha \frac{\lambda}{m}$ will always be less than 1. Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update.

Notice that the second term is now exactly the same as it was before.


##### Normal Equation

Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

$$\theta = (X^TX + \lambda \cdot L)^{−1} X^Ty \quad\quad \text{ where } \quad L = \begin{bmatrix} 0 & & & & \\ & 1 & & & \\ & & 1 & & \\ & & & \ddots & \\ & & & & 1 \end{bmatrix}$$

$L$ is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension $(n+1) \times (n+1)$. Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number $\lambda$.

Recall that if $m  \leq  n$, then $X^TX$ is non-invertible. However, when we add the term $λ \cdot L$, then $X^TX + λ \cdot L$ becomes invertible.


#### Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. Let's start with the cost function.


##### Cost Function

Recall that our cost function for logistic regression was:

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) \large]$$

We can regularize this equation by adding a term to the end:

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$
​
__Note Well__: The second sum, $\sum_{j=1}^n \theta_j^2$ __means to explicitly exclude__ the bias term, $\theta_0$. I.e. the \theta vector is indexed from 0 to n (holding $n+1$ values, $\theta_0$ through $\theta_n$), and this sum explicitly skips $\theta_0$, by running from 1 to n, skipping 0.


##### Gradient Descent

Just like with linear regression, we will want to separately update $\theta_0$ and the rest of the parameters because we do not want to regularize $\theta_0$.

Repeat {
  
$$\begin{array}{rcl} \theta_0 & := & \theta_0−\alpha \dfrac{1}{m} \displaystyle \sum_{i=1}^m (h_\theta(x^{(i)}) − y^{(i)}) x^{(i)}_0 \\\\ \theta_j & := & \theta_j − \alpha \left[ (\dfrac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) − y^{(i)})x^{(i)}_j) + \dfrac{\lambda}{m} \theta_j \right] \quad\quad j \in \{1,2, \ldots, n\} \end{array}$$

}

This is identical to the gradient descent function presented for linear regression.


### Initial Ones Feature Vector

#### Constant Feature

As it turns out it is crucial to add a constant feature to your pool of features before starting any training of your machine. Normally that feature is just a set of ones for all your training examples.

Concretely, if $X$ is your feature matrix then $X_0$ is a vector with ones.

Below are some insights to explain the reason for this constant feature. The first part draws some analogies from electrical engineering concept, the second looks at understanding the ones vector by using a simple machine learning example.


#### Electrical Engineering

From electrical engineering, in particular signal processing, this can be explained as DC and AC.

The initial feature vector X without the constant term captures the dynamics of your model. That means those features particularly record changes in your output y - in other words changing some feature $X_i$ where $i \neq 0$ will have a change on the output $y$. AC is normally made out of many components or harmonics; hence we also have many features (yet we have one DC term).

The constant feature represents the DC component. In control engineering this can also be the steady state.

Interestingly removing the DC term is easily done by differentiating your signal - or simply taking a difference between consecutive points of a discrete signal (it should be noted that at this point the analogy is implying time-based signals - so this will also make sense for machine learning application with a time basis - e.g. forecasting stock exchange trends).

Another interesting note: if you were to play and AC+DC signal as well as an AC only signal where both AC components are the same then they would sound exactly the same. That is because we only hear changes in signals and $\Delta(AC+DC) = \Delta(AC)$.


#### Housing price example

Suppose you design a machine which predicts the price of a house based on some features. In this case what does the ones vector help with?

Let's assume a simple model which has features that are directly proportional to the expected price i.e. if feature Xi increases so the expected price y will also increase. So as an example we could have two features: namely the size of the house in [m2], and the number of rooms.

When you train your machine you will start by prepending a ones vector $X_0$. You may then find after training that the weight for your initial feature of ones is some value \theta0. As it turns, when applying your hypothesis function $h_{\theta}(X)$ - in the case of the initial feature you will just be multiplying by a constant (most probably $\theta_0$ if you not applying any other functions such as sigmoids). This constant (let's say it's $\theta_0$ for argument's sake) is the DC term. It is a constant that doesn't change.

But what does it mean for this example? Well, let's suppose that someone knows that you have a working model for housing prices. It turns out that for this example, if they ask you how much money they can expect if they sell the house you can say that they need at least $\theta_0$ dollars (or rands) before you even use your learning machine. As with the above analogy, your constant $\theta_0$ is somewhat of a steady state where all your inputs are zeros. Concretely, this is the price of a house with no rooms which takes up no space.

However this explanation has some holes because if you have some features which decrease the price e.g. age, then the DC term may not be an absolute minimum of the price. This is because the age may make the price go even lower.

Theoretically if you were to train a machine without a ones vector $f_{AC}(X)$, it's output may not match the output of a machine which had a ones vector $f_{DC}(X)$. However, $f_{AC}(X)$ may have exactly the same trend as $f_{DC}(X)$ i.e. if you were to plot both machine's output you would find that they may look exactly the same except that it seems one output has just been shifted (by a constant). With reference to the housing price problem: suppose you make predictions on two houses $house_A$ and $house_B$ using both machines. It turns out while the outputs from the two machines would different, the difference between houseA and houseB's predictions according to both machines could be exactly the same. Realistically, that means a machine trained without the ones vector $f_AC$ could actually be very useful if you have just one benchmark point. This is because you can find out the missing constant by simply taking a difference between the machine's prediction an actual price - then when making predictions you simply add that constant to what even output you get. That is: if $house_{benchmark}$ is your benchmark then the DC component is simply $price(house_{benchmark}) - f_{AC}(features(house_{benchmark}))$

A more simple and crude way of putting it is that the DC component of your model represents the inherent bias of the model. The other features then cause tension in order to move away from that bias position.

Kholofelo Moyaba


#### A simpler approach

A "bias" feature is simply a way to move the "best fit" learned vector to better fit the data. For example, consider a learning problem with a single feature $X_1$. The formula without the $X_0$ feature is just $theta_1 * X_1 = y$. This is graphed as a line that always passes through the origin, with slope y/theta. The $x_0$ term allows the line to pass through a different point on the y axis. This will almost always give a better fit. Not all best fit lines go through the origin $(0,0)$ right?

Joe Cotton


### Errata

#### The Problem of Overfitting

At 2:07, a curve is drawn using predicting function $\theta_0 + \theta_1 x + \theta_2 x^2$, which is said as "just right". But when size of house is large enough, the prediction of this function will increase much faster than linear if $\theta_2 > 0$, or will decrease to $−\infty$ if $\theta_2 < 0$, which neither could correspond to reality. Instead, $\theta_0+\theta_1 x+\theta_2 \sqrt{x}$ may be "just right".

At 2:28, a curve is drawn using a quartic (degree 4) polynomial predicting function $\theta_0+\theta_1 x+\theta_2 x^2 +\theta_3 x^3 +\theta_4 x^4$; however, the curve drawn is at least quintic (degree 5).


#### Cost Function

In the video at 5:17, the sum of the regularization term should use 'j' instead of 'i', giving $\sum_{j=1}^{n} \theta _j ^2$ instead of $\sum_{i=1}^{n} \theta _j ^2$.


#### Regularized linear regression

In the video starting at 8:04, Prof Ng discusses the Normal Equation and invertibility. He states that $X$ is non-invertible if $m \leq$. The correct statement should be that $X$ is non-invertible if $m < n$, and may be non-invertible if $m = n$.


#### Regularized logistic regression

In the video at 3:52, the lecturer mistakenly said "gradient descent for regularized linear regression". Indeed, it should be "gradient descent for regularized logistic regression".

In the video at 5:21, the cost function is missing a pair of parentheses around the second log argument. It should be $J(\theta)=J(\theta)=[−\frac{1}{m} \sum^m{i=1} y^{(i)} \log(h_\theta(x^{(i)})+(1−y^{(i)}) \log(1−h_\theta(x^{(i)}))]+ \frac{\lambda}{2m} \sum^n_{j=1} \theta^2_j$

In the original videos for the course (ML-001 through ML-008), there were typos in the equation for regularized logistic regression in both the video lecture and the PDF lecture notes. In the slides for "Gradient descent" and "advanced optimization", there should be positive signs for the regularization term of the gradient. The formula on page 10 of 'ex2.pdf' is correct. These issues in the video were corrected for the 'on-demand' format of the course.


#### Quizzes

+ Typo "it's" in question «Because logistic regression outputs values $0≤h_\theta(x) \leq 1$, it's range [...]»
+ 1/m factor missing in the definition of the gradient in question «For logistic regression, the gradient is given by $\frac{\partial}{\partial \theta_j} J(θ) = \sum^m_{i=1}$. It should be $\frac{\partial}{\partial \theta_j} J(θ) = \frac{1}{m} \sum^m_{i=1}$.



### Quiz: Regularization



