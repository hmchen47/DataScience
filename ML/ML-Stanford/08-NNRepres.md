# Neural Networks: Representation

## Non-linear Hypotheses

#### Lecture Notes


+ Non-linear Classification
  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="url">
      <img src="images/m08-01.png" style="margin: 0.1em;" alt="non-linear classification with two classes" title="Non-linear classification" width="250">
    </a></div>
  </div>

  $$g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1x_2 + \theta_4 x_1^2x_2 + \theta_5 x_1^3x_2 + \theta_6 x_1x_2^2 + \ldots)$$

  For $n = 100, x_1, x_2, \ldots, x_{100}$
  + $x_1^2, x_1x_2, x_1, x_3, \ldots, x_1x_{100}, x_2^2, x_2x_3, \ldots \approx \frac{n^2}{2} \approx 5000$ features $\quad \Rightarrow \mathcal{O}(n^2)$
  + $x_1^2, x_2^2, \ldots, x_{100}^2 \Rightarrow 100$ features, but not good enough
  + $x_1x_2x_3, x_1^2x_2, X_{10}x_{11}x_{17}, \ldots \approx 170,000 \quad\Rightarrow \mathcal{O}(n^3)$

+ Computer Vision: Car Detection
  + What is this?
    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div><a href="url">
        <img src="images/m08-02.png" style="margin: 0.1em;" alt="A car that sees from a camera as a combination of pixels" title="Deta for a car" width="350">
      </a></div>
    </div>
  + Testing
    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div><a href="url">
        <img src="images/m08-03.png" style="margin: 0.1em;" alt="One sample puts into learning algorithm into training" title="Training for Learning Algorithm" width="250">
        <img src="images/m08-04.png" style="margin: 0.1em;" alt="The diagram depicts the results of 2-dim pixels with the label of car" title="Diagram to separate the label of car and non-car objects." width="150">
      </a></div>
    </div>

  + $50 \times 50$ pixel images $\rightarrow 2500$ pixels, $\;\;\therefore n=2500$, (if RGB, $n=7500$)

    $$x = \begin{bmatrix} \text{pixel 1 intensity} \\ \text{pixel 2 intensity} \\ \vdots \\ \text{pixel 2500 intensity}\end{bmatrix}$$

  + Quadratic features $(x_i \times x_j): \;\approx 3$ million features
  + IVQ: Suppose you are learning to recognize cars from $100 \times 100$ pixel images (grayscale, not RGB). Let the features be pixel intensity values. If you train logistic regression including all the quadratic terms ($x_ix_j$) as features, about how many features will you have?

    1) 5,000
    2) 100,000
    3) 50 million ($5\times10^7$)
    4) 5 billion ($5\times10^9$)

    Ans: 3

#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/08.1-NeuralNetworksRepresentation-NonLinearHypotheses-new.8f376d70b23611e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1553904000&Signature=jIpXIeyhNzV1xzu8Af7kn1e-lqmIUZwA9l47EsaC0-22Ku4wJaEHrz7Zacpi-z0n4x8IWoSPWpaiZ8eqFr4X0KSK1paY1aB1CyfsHmdtVOCASKcLnbjf5A~mcTsLqNry6C9RoJOFOxaUHpBhrMmRsETj0ScEHE3WJ6u21WK7Kvc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/RVjTsODRQpKY07Dg0UKSYw?expiry=1553904000000&hmac=TQ7KpcmTUiBmm6ddadPcBBgWISYhHc83gOzbMLK3e4U&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Motivations

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Neurons and the Brain

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Neural Networks

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Model Representation I

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Model Representation II

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Applications


### Examples and Intuitions I

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Examples and Intuitions II

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Multiclass Classification

#### Lecture Notes



--------------------------------------




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Review

### Lecture Slides

#### Non-linear Hypotheses

Performing linear regression with a complex set of data with many features is very unwieldy. Say you wanted to create a hypothesis from three (3) features that included all the quadratic terms:

$$g(\theta_0+\theta_1x^2_1+\theta_2x_1x_2+\theta_3x_1x_3+\theta_4x_2^2+\theta_5x_2x_3+\theta_6x^2_3)$$

That gives us 6 features. The exact way to calculate how many features for all polynomial terms is the [combination function with repetition](http://www.mathsisfun.com/combinatorics/combinations-permutations.html) $\dfrac{(n+r−1)!}{r!(n−1)!}$. In this case we are taking all two-element combinations of three features: $\dfrac{(3+2−1)!}{(2! \cdot (3−1)!)} = \dfrac{4!}{4} = 6$. (__Note__: you do not have to know these formulas, I just found it helpful for understanding).

For 100 features, if we wanted to make them quadratic we would get $\dfrac{(100+2−1)!}{(2⋅(100−1)!)} = 5050$ resulting new features.

We can approximate the growth of the number of new features we get with all quadratic terms with $\mathcal{O}(n^2/2)$. And if you wanted to include all cubic terms in your hypothesis, the features would grow asymptotically at $\mathcal{O}(n^3)$. These are very steep growths, so as the number of our features increase, the number of quadratic or cubic features increase very rapidly and becomes quickly impractical.

Example: let our training set be a collection of $50 \times 50$ pixel black-and-white photographs, and our goal will be to classify which ones are photos of cars. Our feature set size is then $n = 2500$ if we compare every pair of pixels.

Now let's say we need to make a quadratic hypothesis function. With quadratic features, our growth is $\mathcal{O}(n^2/2)$. So our total features will be about $2500^2 / 2 = 3125000$, which is very impractical.

Neural networks offers an alternate way to perform machine learning when we have complex hypotheses with many features.


#### Neurons and the Brain

Neural networks are limited imitations of how our own brains work. They've had a big recent resurgence because of advances in computer hardware.

There is evidence that the brain uses only one "learning algorithm" for all its different functions. Scientists have tried cutting (in an animal brain) the connection between the ears and the auditory cortex and rewiring the optical nerve with the auditory cortex to find that the auditory cortex literally learns to see.

This principle is called "neuroplasticity" and has many examples and experimental evidence.


#### Model Representation I

Let's examine how we will represent a hypothesis function using neural networks.

At a very simple level, neurons are basically computational units that take input (__dendrites__) as electrical input (called "spikes") that are channeled to outputs (__axons__).

In our model, our dendrites are like the input features $x_1 \ldots x_n$, and the output is the result of our hypothesis function:

In this model our $x_0$ input node is sometimes called the "bias unit." It is always equal to 1.

In neural networks, we use the same logistic function as in classification: $\frac{1}{1 + e^{-\theta^Tx}}$. In neural networks however we sometimes call it a sigmoid (logistic) __activation__ function.

Our "theta" parameters are sometimes instead called "weights" in the neural networks model.

Visually, a simplistic representation looks like:

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} \rightarrow \begin{bmatrix} & \end{bmatrix} \rightarrow h_\theta(x)$$

Our input nodes (layer 1) go into another node (layer 2), and are output as the hypothesis function.

The first layer is called the "input layer" and the final layer the "output layer," which gives the final value computed on the hypothesis.

We can have intermediate layers of nodes between the input and output layers called the "hidden layer."

We label these intermediate or "hidden" layer nodes $a^2_0 \ldots a^2_n$ and call them "activation units."

$$\begin{array}{rcl} a^{(j)}_i &=& \text{ "activation" of unit i in layer } j \\\\ \Theta^{(j)} &=& \text{ matrix of weights controlling function mapping from layer } j \text{ to layer } j+1 \end{array}$$

If we had one hidden layer, it would look visually something like:

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} \;\rightarrow\; \begin{bmatrix} a^{(2)}_1 \\ a^{(2)}_2 \\ a^{(2)}_3 \end{bmatrix} \;\rightarrow\; h_\theta(x)$$

The values for each of the "activation" nodes is obtained as follows:

$$\begin{array}{rcccl} & & a^{(2)}_1 &=& g(\Theta^{(1)}_{10} x_ 0 + \Theta^{(1)}_{11} x_1 + \Theta^{(1)}_{12} x_2 + \Theta^{(1)}_{13} x_3) \\\\ & & a^{(2)}_2 &=& g(\Theta^{(1)}_{20} x_0 + \Theta^{(1)}_{21} x_1 + \Theta^{(1)}_{22} x_2 + \Theta^{(1)}_{23} x_3) \\\\ & & a^{(2)}_3 &=& g(\Theta^{(1)}_{30} x_0 + \Theta^{(1)}_{31} x_1 + \Theta^{(1)}_{32} x_2 + \Theta^{(1)}_{33} x_3) \\\\ h_\theta(x) &=& a^{(3)}_1 & = & g(\Theta^{(2)}_{10} a^{(2)}_0 + \Theta^{(2)}_{11} a^{(2)}_1 + \Theta^{(2)}_{12} a^{(2)}_2 + \Theta^{(2)}_{13} a^{(2)}_3) \end{array}$$

This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, $\Theta^{(j)}$.

The dimensions of these matrices of weights is determined as follows:

__If network has sj units in layer j and sj+1 units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j+1)$.__

The +1 comes from the addition in $\Theta^{(j)}$ of the "bias nodes," $x_0$ and $\Theta^{(j)}_0$. In other words the output nodes will not include the bias nodes while the inputs will.

Example: layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of $\theta^{(1)}$ is going to be 4×3 where $s_j = 2$ and $s_{j+1} = 4$, so $s_{j+1} \times (s_j + 1) = 4 \times 3$.


#### Model Representation II

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $z_k^{(j)}$ that encompasses the parameters inside our g function. In our previous example if we replaced the variable z for all the parameters we would get:

$$\begin{array}{rcl} a^{(2)}_1 & = &  g(z^{(2)}_1) \\ a^{(2)}_2 & = & g(z^{(2)}_2) \\ a^{(2)}_3 & = & g(z^{(2)}_3) \end{array}$$

In other words, for layer $j=2$ and node $k$, the variable $z$ will be:

$$z^{(2)}_k = \Theta^{(1)}_{k,0} x_0 + \Theta^{(1)}_{k,1} x_1 + \ldots + \Theta^{(1)}_{k,n} x_n$$

The vector representation of $x$ and $z^{j}$ is:

$$x = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} z^{(j)} = \begin{bmatrix} z^{(j)}_1 \\ z^{(j)}_2 \\ \vdots \\ z^{(j)}_n \end{bmatrix}$$

Setting $x = a^{(1)}$, we can rewrite the equation as:

$$z^{(j)} = \Theta^{(j−1)} a^{(j−1)}$$

We are multiplying our matrix $\Theta^{(j−1)}$ with dimensions $s_j\times (n+1)$ (where $s_j$ is the number of our activation nodes) by our vector $a^{(j-1)}$ with height $(n+1)$. This gives us our vector $z^{(j)}$ with height $s_j$.

Now we can get a vector of our activation nodes for layer $j$ as follows:

$$a^{(j)} = g(z^{(j)})$$

Where our function $g$ can be applied element-wise to our vector $z^{(j)}$.

We can then add a bias unit (equal to 1) to layer $j$ after we have computed $a^{(j)}$. This will be element $a_0^{(j)}$ and will be equal to $1$.

To compute our final hypothesis, let's first compute another z vector:

$$z^{(j+1)} = \Theta^{(j)}a^{(j)}$$

We get this final $z$ vector by multiplying the next theta matrix after $\Theta^{(j−1)}$ with the values of all the activation nodes we just got.

This last theta matrix $\Theta^{(j)}$ will have only one row so that our result is a single number.

We then get our final result with:

$$h_\theta(x) = a^{(j+1)} = g(z^{(j+1)})$$

Notice that in this __last step__, between layer $j$ and layer $j+1$, we are doing __exactly the same thing__ as we did in logistic regression.

Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

#### Examples and Intuitions I

A simple example of applying neural networks is by predicting $x_1$ AND $x_2$, which is the logical 'and' operator and is only true if both $x_1$ and $x_2$ are 1.

The graph of our functions will look like:

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} \;\rightarrow\; [g(z^{(2)})] \;\rightarrow\; h_\theta(x)$$

Remember that $x_0$ is our bias variable and is always 1.

Let's set our first theta matrix as:

$\Theta^{(1)}= \begin{bmatrix} −30 & 20 & 20 \end{bmatrix}$

This will cause the output of our hypothesis to only be positive if both $x_1$ and $x_2$ are 1. In other words:

$$\begin{array}{rcl} h_\Theta(x) &=& g(−30+20x1+20x2) \\\\ x_1 = 0 & \text{ and }& x_2=0  \qquad \text{ then } \qquad g(−30) \approx 0 \\ x_1=0 & \text{ and } & x_2=1  \qquad \text{ then } \qquad  g(−10) \approx 0 \\ x_1=1 &\text{ and }& x_2=0  \qquad \text{ then }  \qquad g(−10) \approx 0 \\ x_1=1  &\text{ and }& x_ 2=1  \qquad \text{ then } \qquad  g(10) \approx 1 \end{array}$$

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates.

#### Examples and Intuitions II

The $\Theta^{(1)}$ matrices for AND, NOR, and OR are:

$$\begin{array}{rcl} \text{AND } &:& \Theta^{(1)} = \begin{bmatrix} -30 & 20 & 20 \end{bmatrix} \\ \text{NOR } &:& \Theta^{(1)}  = \begin{bmatrix} 10 & -20 & -20 \end{bmatrix} \\ \text{OR } &:& \Theta^{(1)}  = \begin{bmatrix} -10 & 20 & 20 \end{bmatrix} \end{array}$$

We can combine these to get the XNOR logical operator (which gives 1 if $x_1$ and $x_2$ are both 0 or both 1).

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} \;\rightarrow \; \begin{bmatrix} a^{(2)}_1 \\ a^{(2)}_2 \end{bmatrix} \;\rightarrow\; [a^{(3)}]\;\rightarrow\; h_\Theta(x)$$

For the transition between the first and second layer, we'll use a $\Theta^{(1)}$ matrix that combines the values for AND and NOR:

$\Theta^{(1)} = \begin{bmatrix} −30 & 10 & 20 \\ −20& 20 & −20 \end{bmatrix}$

For the transition between the second and third layer, we'll use a $\Theta^{(2)}$ matrix that uses the value for OR:

$\Theta^{(2)}= \begin{bmatrix} −10 & 20 & 20 \end{bmatrix}$

Let's write out the values for all our nodes:

$$\begin{array}{rcl} a^{(2)} & = & g(\Theta^{(1)}⋅x) \\ a^{(3)} &=& g(\Theta^{(2)} \cdot a^{(2)}) \\ h_\theta(x) &=& a^{(3)} \end{array}$$

And there we have the XNOR operator using two hidden layers!

#### Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four final resulting classes:

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ \cdots \\ x_n \end{bmatrix} \;\rightarrow\;\begin{bmatrix} a^{(2)}_0 \\ a^{(2)}_1 \\ a^{(2)}_2 \\ \cdots \end{bmatrix} \;\rightarrow\; \begin{bmatrix} a^{(3)}_0 \\ a^{(3)}_1 \\ a^{(3)}_2 \\ \cdots \end{bmatrix} \;\rightarrow\; \cdots \;\rightarrow\;\begin{bmatrix} h_\Theta(x)_1 \\ h_\Theta(x)_2 \\ h_\Theta(x)_3 \\ h_\Theta(x)_4 \end{bmatrix} \;\rightarrow\;$$

Our final layer of nodes, when multiplied by its theta matrix, will result in another vector, on which we will apply the g() logistic function to get a vector of hypothesis values.

Our resulting hypothesis for one set of inputs may look like:

$h_\Theta(x) = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}$

In which case our resulting class is the third one down, or $h_\Theta(x)_3$.

We can define our set of resulting classes as $y$:

$$y^{(i)} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix},$$

Our final value of our hypothesis for a set of inputs will be one of the elements in $y$.



### Errata

#### Errata in the video lectures

+ In the videos "Model Representation I" and "Model Representation II:, the diagram of the NN does not show the added bias units in the input and hidden layers. The bias units are represented in the equations as the variable x0.
+ In the video "Model representation I", in the in-video quiz, the figure is incorrect in that it does not show the added bias units. The bias units must be included when you calculate the size of a Theta matrix.
+ In the video "Model Representation II" at 2:42, Prof Ng mistakenly says that z(2) is a 3-dimensional vector. What he means is that the vector z(2) has three features - i.e it is size (3 x 1).


#### Errata in the programming exercise

+ In ex3.pdf at Section 1.3.2 "Vectorizing the gradient", there is a typo in the series of entries demonstrating how to compute the partial derivatives for all $\theta_j$ where $h_\theta(x) - y$ is defined. The last row in the array has $h_\theta(x^{(1)}) - y^{(m)}$ but it should be $h_\theta(x^{(m)}) - y^{(m)}$
+ Clarification: The instructions in ex3.pdf ask you to first write the unregularized portions of cost function (in Section 1.3.1 for cost and 1.3.2 for the gradients), then to add the regularized portions of the cost function (in Section 1.3.3).
+ Note: The test case for lrCostFunction() in ex3.m includes regularization, so you should first complete through Section 1.3.3.

#### Errata in the quiz

+ In question 4 of the Neural Networks: Representation quiz, one potential answer may include the variable Theta2, even though this variable is undefined (the question only defines Theta1). When answering the question, treat Theta2 as Theta with a superscript "(2)", or $\Theta_(2)$, from lecture.



### Quiz: Neural Networks: Representation






