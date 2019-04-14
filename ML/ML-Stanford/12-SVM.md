# Support Vector Machines

## Large Margin Classification

### Optimization Objective

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Large Margin Intuition

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Mathematics Behind Large Margin Classification

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Kernels


### Kernels I

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Kernels II

#### Lecture Notes



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## SVMs in Practice: Using An SVM

### Lecture Notes



### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Review

### Lecture Slides

#### Optimization Objective

The __Support Vector Machine__ (SVM) is yet another type of supervised machine learning algorithm. It is sometimes cleaner and more powerful.

Recall that in logistic regression, we use the following rules:

if $y=1$, then $h_\theta(x) \approx 1$ and $\Theta^Tx \gg 0$

if $y=0$, then $h_\theta(x) \approx 0$ and $\Theta^Tx \gg 0$

Recall the cost function for (unregularized) logistic regression:

$$\begin{array}{rcl} J(\theta) = \dfrac{1}{m} \sum_{i=1}^m −y^{(i)} \log(h_\theta(x^{(i)})) − (1−y^{(i)}) \log(1−h_\theta(x^{(i)})) \\ & = & \dfrac{1}{m} \sum_{i=1}^m −y^{(i)} \log(\dfrac{1}{1+e^{−\theta^T x^{(i)})} − (1−y^{(i)}) \log(1 − \dfrac{1}{1+e^{−\theta^Tx^{(i)}}) \end{array}$$

To make a support vector machine, we will modify the first term of the cost function $-\log(h_{\theta}(x)) = -\log \left( \dfrac{1}{1 + e^{-\theta^Tx}}\left)$ so that when $\theta^Tx$ (from now on, we shall refer to this as z) is __greater than__ 1, it outputs 0. Furthermore, for values of z less than 1, we shall use a straight decreasing line instead of the sigmoid curve.(In the literature, this is called a [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) function.)

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/resources/Es9Qo">
    <img src="images/m12-01.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
  </a></div>
</div>

Similarly, we modify the second term of the cost function $-\log(1 - h_{\theta(x)}) = -\log\left(1 - \dfrac{1}{1 + e^{-\theta^Tx}}\right)$ so that when $z$ is __less than__ -1, it outputs 0. We also modify it so that for values of z greater than -1, we use a straight increasing line instead of the sigmoid curve.

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/resources/Es9Qo">
    <img src="images/m12-02.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
  </a></div>
</div>

We shall denote these as $\text{cost}_1(z)$ and $\text{cost}_0(z)$ (respectively, note that $\text{cost}_1(z)$ is the cost for classifying when $y=1$, and $\text{cost}_\theta(z)$ is the cost for classifying when y=0), and we may define them as follows (where k is an arbitrary constant defining the magnitude of the slope of the line):

$$\begin{array}{rcl} z &=& \theta^Tx \\ cost_0(z) &=& \max(0,k(1+z)) \\ cost_1(z) &=& \max(0,k(1−z))$$

Recall the full cost function from (regularized) logistic regression:

$$J(\theta) = \dfrac{1}{m} \sum_{i=1}^m y^{(i)}(−\log(h_\theta(x^{(i)}))) + (1−y^{(i)}) (−\log(1−h_\thea(x^{(i)}))) + \dfraqc{\lambda}{2m} \sum_{j=1}^n \Theta^2_j$$

Note that the negative sign has been distributed into the sum in the above equation.

We may transform this into the cost function for support vector machines by substituting $\text{cost}_0(z)$ and $\text{cost}_1(z)$:

$$J(\theta) = \dfrac{1}{m} \sum_{i=1}^m y^{(i)} \text{cost}_1(\theta^T x^{(i)}) + (1−y^{(i)}) cost_0(\theta^T x^{(i)}) + \dfrac{\lambda}{2m} \sum_{j=1}^n \Theta^2_j$$

We can optimize this a bit by multiplying this by m (thus removing the m factor in the denominators). Note that this does not affect our optimization, since we're simply multiplying our cost function by a positive constant (for example, minimizing $(u-5)^2 + 1$ gives us 5; multiplying it by 10 to make it 10(u-5)^2 + 1010(u−5) 
2
 +10 still gives us 5 when minimized).

J(θ)=∑mi=1y(i) cost1(θTx(i))+(1−y(i)) cost0(θTx(i))+λ2∑nj=1Θ2j
Furthermore, convention dictates that we regularize using a factor C, instead of λ, like so:

J(θ)=C∑mi=1y(i) cost1(θTx(i))+(1−y(i)) cost0(θTx(i))+12∑nj=1Θ2j
This is equivalent to multiplying the equation by C = \dfrac{1}{\lambda}C= 
λ
1
​	 , and thus results in the same values when optimized. Now, when we wish to regularize more (that is, reduce overfitting), we decrease C, and when we wish to regularize less (that is, reduce underfitting), we increase C.

Finally, note that the hypothesis of the Support Vector Machine is not interpreted as the probability of y being 1 or 0 (as it is for the hypothesis of logistic regression). Instead, it outputs either 1 or 0. (In technical terms, it is a discriminant function.)

hθ(x)={10if ΘTx≥0otherwise


#### Large Margin Intuition

A useful way to think about Support Vector Machines is to think of them as Large Margin Classifiers.

If y=1, we want ΘTx≥1 (not just ≥0)

If y=0, we want ΘTx≤−1 (not just <0)

Now when we set our constant C to a very large value (e.g. 100,000), our optimizing function will constrain Θ such that the equation A (the summation of the cost of each example) equals 0. We impose the following constraints on Θ:

ΘTx≥1 if y=1 and ΘTx≤−1 if y=0.

If C is very large, we must choose Θ parameters such that:

∑mi=1y(i)cost1(ΘTx)+(1−y(i))cost0(ΘTx)=0
This reduces our cost function to:

J(θ)=C⋅0+12∑j=1nΘ2j=12∑j=1nΘ2j
Recall the decision boundary from logistic regression (the line separating the positive and negative examples). In SVMs, the decision boundary has the special property that it is as far away as possible from both the positive and the negative examples.

The distance of the decision boundary to the nearest example is called the margin. Since SVMs maximize this margin, it is often called a Large Margin Classifier.

The SVM will separate the negative and positive examples by a large margin.

This large margin is only achieved when C is very large.

Data is linearly separable when a straight line can separate the positive and negative examples.

If we have outlier examples that we don't want to affect the decision boundary, then we can reduce C.

Increasing and decreasing C is similar to respectively decreasing and increasing λ, and can simplify our decision boundary.

#### Mathematics Behind Large Margin Classification (Optional)

__Vector Inner Product__

Say we have two vectors, u and v:

u=[u1u2]v=[v1v2]
The length of vector v is denoted ||v||∣∣v∣∣, and it describes the line on a graph from origin (0,0) to (v_1,v_2)(v 
1
​	 ,v 
2
​	 ).

The length of vector v can be calculated with \sqrt{v_1^2 + v_2^2} 
v 
1
2
​	 +v 
2
2
​	 
​	 by the Pythagorean theorem.

The projection of vector v onto vector u is found by taking a right angle from u to the end of v, creating a right triangle.

p= length of projection of v onto the vector u.
u^Tv= p \cdot ||u||u 
T
 v=p⋅∣∣u∣∣
Note that u^Tv = ||u|| \cdot ||v|| \cos \thetau 
T
 v=∣∣u∣∣⋅∣∣v∣∣cosθ where θ is the angle between u and v. Also, p = ||v|| \cos \thetap=∣∣v∣∣cosθ. If you substitute p for ||v|| \cos \theta∣∣v∣∣cosθ, you get u^Tv= p \cdot ||u||u 
T
 v=p⋅∣∣u∣∣.

So the product u^Tvu 
T
 v is equal to the length of the projection times the length of vector u.

In our example, since u and v are vectors of the same length, u^Tv = v^Tuu 
T
 v=v 
T
 u.

u^Tv = v^Tu = p \cdot ||u|| = u_1v_1 + u_2v_2u 
T
 v=v 
T
 u=p⋅∣∣u∣∣=u 
1
​	 v 
1
​	 +u 
2
​	 v 
2
​	 

If the angle between the lines for v and u is greater than 90 degrees, then the projection p will be negative.

minΘ12∑j=1nΘ2j=12(Θ21+Θ22+⋯+Θ2n)=12(Θ21+Θ22+⋯+Θ2n−−−−−−−−−−−−−−−√)2=12||Θ||2
We can use the same rules to rewrite ΘTx(i):

ΘTx(i)=p(i)⋅||Θ||=Θ1x(i)1+Θ2x(i)2+⋯+Θnx(i)n
So we now have a new optimization objective by substituting p(i)⋅||Θ|| in for ΘTx(i):

If y=1, we want p(i)⋅||Θ||≥1
If y=0, we want p(i)⋅||Θ||≤−1
The reason this causes a "large margin" is because: the vector for Θ is perpendicular to the decision boundary. In order for our optimization objective (above) to hold true, we need the absolute value of our projections p^{(i)}p 
(i)
  to be as large as possible.

If Θ0=0, then all our decision boundaries will intersect (0,0). If Θ0≠0, the support vector machine will still find a large margin for the decision boundary.


__Kernels I__

Kernels allow us to make complex, non-linear classifiers using Support Vector Machines.

Given x, compute new feature depending on proximity to landmarks l^{(1)},\ l^{(2)},\ l^{(3)}l 
(1)
 , l 
(2)
 , l 
(3)
 .

To do this, we find the "similarity" of x and some landmark l^{(i)}l 
(i)
 :

fi=similarity(x,l(i))=exp(−||x−l(i)||22σ2)
This "similarity" function is called a Gaussian Kernel. It is a specific example of a kernel.

The similarity function can also be written as follows:

fi=similarity(x,l(i))=exp(−∑nj=1(xj−l(i)j)22σ2)
There are a couple properties of the similarity function:

If x \approx l^{(i)}x≈l 
(i)
 , then fi=exp(−≈022σ2)≈1
If x is far from l^{(i)}l 
(i)
 , then fi=exp(−(large number)22σ2)≈0
In other words, if x and the landmark are close, then the similarity will be close to 1, and if x and the landmark are far away from each other, the similarity will be close to 0.

Each landmark gives us the features in our hypothesis:

l(1)→f1l(2)→f2l(3)→f3…hΘ(x)=Θ1f1+Θ2f2+Θ3f3+…
\sigma^2σ 
2
  is a parameter of the Gaussian Kernel, and it can be modified to increase or decrease the drop-off of our feature f_if 
i
​	 . Combined with looking at the values inside Θ, we can choose these landmarks to get the general shape of the decision boundary.

#### Kernels II

One way to get the landmarks is to put them in the exact same locations as all the training examples. This gives us m landmarks, with one landmark per training example.

Given example x:

f_1 = similarity(x,l^{(1)})f 
1
​	 =similarity(x,l 
(1)
 ), f_2 = similarity(x,l^{(2)})f 
2
​	 =similarity(x,l 
(2)
 ), f_3 = similarity(x,l^{(3)})f 
3
​	 =similarity(x,l 
(3)
 ), and so on.

This gives us a "feature vector," f_{(i)}f 
(i)
​	  of all our features for example x_{(i)}x 
(i)
​	 . We may also set f_0 = 1f 
0
​	 =1 to correspond with Θ0. Thus given training example x_{(i)}x 
(i)
​	 :

x(i)→⎡⎣⎢⎢⎢⎢⎢⎢f(i)1=similarity(x(i),l(1))f(i)2=similarity(x(i),l(2))⋮f(i)m=similarity(x(i),l(m))⎤⎦⎥⎥⎥⎥⎥⎥
Now to get the parameters Θ we can use the SVM minimization algorithm but with f^{(i)}f 
(i)
  substituted in for x^{(i)}x 
(i)
 :

minΘC∑mi=1y(i)cost1(ΘTf(i))+(1−y(i))cost0(θTf(i))+12∑nj=1Θ2j
Using kernels to generate f(i) is not exclusive to SVMs and may also be applied to logistic regression. However, because of computational optimizations on SVMs, kernels combined with SVMs is much faster than with other algorithms, so kernels are almost always found combined only with SVMs.


__Choosing SVM Parameters__

Choosing C (recall that C = \dfrac{1}{\lambda}C= 
λ
1
​	 

If C is large, then we get higher variance/lower bias
If C is small, then we get lower variance/higher bias
The other parameter we must choose is σ2 from the Gaussian Kernel function:

With a large σ2, the features fi vary more smoothly, causing higher bias and lower variance.

With a small σ2, the features fi vary less smoothly, causing lower bias and higher variance.


__Using An SVM__

There are lots of good SVM libraries already written. A. Ng often uses 'liblinear' and 'libsvm'. In practical application, you should use one of these libraries rather than rewrite the functions.

In practical application, the choices you do need to make are:

+ Choice of parameter C
+ Choice of kernel (similarity function)
+ No kernel ("linear" kernel) -- gives standard linear classifier
+ Choose when n is large and when m is small
+ Gaussian Kernel (above) -- need to choose σ2
+ Choose when n is small and m is large

The library may ask you to provide the kernel function.

Note: do perform feature scaling before using the Gaussian Kernel.

Note: not all similarity functions are valid kernels. They must satisfy "Mercer's Theorem" which guarantees that the SVM package's optimizations run correctly and do not diverge.

You want to train C and the parameters for the kernel function using the training and cross-validation datasets.


__Multi-class Classification__

Many SVM libraries have multi-class classification built-in.

You can use the one-vs-all method just like we did for logistic regression, where y∈1,2,3,…,K with Θ(1),Θ(2),…,Θ(K). We pick class i with the largest (Θ(i))Tx.


__Logistic Regression vs. SVMs__

If n is large (relative to m), then use logistic regression, or SVM without a kernel (the "linear kernel")

If n is small and m is intermediate, then use SVM with a Gaussian Kernel

If n is small and m is large, then manually create/add more features, then use logistic regression or SVM without a kernel.

In the first case, we don't have enough examples to need a complicated polynomial hypothesis. In the second example, we have enough examples that we may need a complex non-linear hypothesis. In the last case, we want to increase our features so that logistic regression becomes applicable.

Note: a neural network is likely to work well for any of these situations, but may be slower to train.

#### Additional references

"[An Idiot's Guide to Support Vector Machines](http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)"


### Errata



### Quiz: Support Vector Machines





