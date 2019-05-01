# Recommender Systems

## Predicting Movie Ratings


### Problem Formulation

#### Lecture Note




#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Content Based Recommendations

#### Lecture Note




#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Collaborative Filtering

### Collaborative Filtering

#### Lecture Note




#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Collaborative Filtering Algorithm

#### Lecture Note




#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Low Rank Matrix Factorization


### Vectorization: Low Rank Matrix Factorization

#### Lecture Note




#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Implementational Detail: Mean Normalization

#### Lecture Note




#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Review

### Lecture Slides

#### Problem Formulation

Recommendation is currently a very popular application of machine learning.

Say we are trying to recommend movies to customers. We can use the following definitions

+ $n_u = \;$ number of users
+ $n_m = \;$  number of movies
+ $r(i,j) = 1\;$ if user j has rated movie $i$
+ $y(i,j) =\;$ rating given by user $j$ to movie $i$ (defined only if $r(i,j)=1$)


#### Content Based Recommendations

We can introduce two features, $x_1$ and $x_2$ which represents how much romance or how much action a movie may have (on a scale of 0−1).

One approach is that we could do linear regression for every single user. For each user $j$, learn a parameter $\theta^{(j)} \in \mathbb{R}^3$. Predict user $j$ as rating movie $i$ with $(\theta^{(j)})^Tx^{(i)}$ stars.

+ $\theta^{(j)} =\;$ parameter vector for user $j$
+ $x^{(i)} =\;$ = feature vector for movie $i$

For user $j$, movie $i$, predicted rating: $(\theta^{(j)})^T(x^{(i)})$
+ $m^{(j)} = \;$  number of movies rated by user $j$

To learn $\theta^{(j)}$, we do the following

$$\min_{\theta^{(j)}} = \dfrac{1}{2}\displaystyle \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2} \sum_{k=1}^n \left(\theta_k^{(j)}\right)^2$$

This is our familiar linear regression. The base of the first summation is choosing all i such that $r(i,j) = 1$.

To get the parameters for all our users, we do the following:

$$\min_{\theta^{(1)},\dots,\theta^{(n_u)}} = \dfrac{1}{2}\displaystyle \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left(\theta_k^{(j)}\right)^2$$

We can apply our linear regression gradient descent update using the above cost function.

The only real difference is that we __eliminate the constant__ $\dfrac{1}{m}$.


#### Collaborative Filtering

It can be very difficult to find features such as "amount of romance" or "amount of action" in a movie. To figure this out, we can use __feature finders__.

We can let the users tell us how much they like the different genres, providing their parameter vector immediately for us.

To infer the features from given parameters, we use the squared error function with regularization over all the users:

$$\min_{x^{(1)},\dots,x^{(n_m)}} \dfrac{1}{2} \displaystyle \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} \left( (\theta^{(j)})^T x^{(i)} - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} \left(x_k^{(i)}\right)^2$$
 

You can also randomly guess the values for theta to guess the features repeatedly. You will actually converge to a good set of features.


#### Collaborative Filtering Algorithm

To speed things up, we can simultaneously minimize our features and our parameters:

$$J(x,\theta) = \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j)=1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \dfrac{\lambda}{2}\sum_{j=1}^{n_u} \sum_{k=1}^{n} \left(\theta_k^{(j)}\right)^2$$
 
It looks very complicated, but we've only combined the cost function for theta and the cost function for $x$.

Because the algorithm can learn them itself, the bias units where $x_0=1$ have been removed, therefore $x \in \mathbb{R}^n$ and $\theta \in \mathbb{R}^n$.

These are the steps in the algorithm:

1. Initialize $x^{(i)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}$ to small random values. This serves to break symmetry and ensures that the algorithm learns features $x^{(i)},...,x^{(n_m)}$ that are different from each other.
2. Minimize $J(x^{(i)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})$ using gradient descent (or an advanced optimization algorithm).E.g. for every $j=1,\dots,n_u,i=1,\dots n_m$:

  $$x_k^{(i)} := x_k^{(i)} - \alpha\left (\displaystyle \sum_{j:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \theta_k^{(j)}} + \lambda x_k^{(i)} \right) \theta_k^{(j)} := \theta_k^{(j)} - \alpha\left (\displaystyle \sum_{i:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x_k^{(i)}} + \lambda \theta_k^{(j)} \right)$$
​	 
3. For a user with parameters θ and a movie with (learned) features $x$, predict a star rating of $\theta^Tx$.


#### Vectorization: Low Rank Matrix Factorization

Given matrices $X$ (each row containing features of a particular movie) and $\Theta$ (each row containing the weights for those features for a given user), then the full matrix $Y$ of all predicted ratings of all movies by all users is given simply by: $Y = X \Theta^T$.

Predicting how similar two movies $i$ and $j$ are can be done using the distance between their respective feature vectors $x$. Specifically, we are looking for a small value of $\parallel x^{(i)} - x^{(j)}\parallel$.


#### Implementation Detail: Mean Normalization

If the ranking system for movies is used from the previous lectures, then new users (who have watched no movies), will be assigned new movies incorrectly. Specifically, they will be assigned $\theta$ with all components equal to zero due to the minimization of the regularization term. That is, we assume that the new user will rank all movies 0, which does not seem intuitively correct.

We rectify this problem by normalizing the data relative to the mean. First, we use a matrix $Y$ to store the data from previous ratings, where the ith row of $Y$ is the ratings for the $i$th movie and the $j$th column corresponds to the ratings for the $j$th user.

We can now define a vector

$$\mu = \begin{bmatrix} \mu_1 & \mu_2 & \dots & \mu_{n_m} \end{bmatrix}$$

such that

$$\mu_i = \frac{\sum_{j:r(i,j)=1}{Y_{i,j}}}{\sum_{j}{r(i,j)}}$$

Which is effectively the mean of the previous ratings for the ith movie (where only movies that have been watched by users are counted). We now can normalize the data by subtracting $u$, the mean rating, from the actual ratings for each user (column in matrix $Y$):

As an example, consider the following matrix $Y$ and mean ratings $\mu$:

$$Y = \begin{bmatrix} 5 & 5 & 0 & 0 \\ 4 & ? & ? & 0 \\ 0 & 0 & 5 & 4 \\ 0 & 0 & 5 & 0 \end{bmatrix}, \qquad \mu = \begin{bmatrix} 2.5 \\ 2 \\ 2.25 \\ 1.25 \end{bmatrix}$$

The resulting $Y^{\prime}$ vector is:

$$Y^\prime = \begin{bmatrix} 2.5 & 2.5 & -2.5 & -2.5 \\ 2 & ? & ? & -2 \\ -2.25 & -2.25 & 3.75 & 1.25 \\ -1.25 & -1.25 & 3.75 & -1.25 \end{bmatrix}$$

Now we must slightly modify the linear regression prediction to include the mean normalization term:

$$(\theta^{(j)})^T x^{(i)} + \mu_i$$
​	 

Now, for a new user, the initial predicted values will be equal to the μ term instead of simply being initialized to zero, which is more accurate.




### Errata

In review questions, question 5 in option starting "Recall that the cost function for the content-based recommendation system is" the right side of the formula should be divided by $m$ where $m$ is number of movies. That would mean that the formula will no longer be standard cost function for the content-based recommendation system. However without this change correct answer is marked as incorrect and vice-versa. This description is not very clear but being more specific would mean breaking the honour code.

In the Problem Formulation video the review question states that the no. of movies is $n_m = 1$. The correct value for $n_m= 2$.

In "Collaborative Filtering" video, review question 2: "Which of the following is a correct gradient descent update rule for $i \neq 0$?"; Instead of $i \neq 0$ it should be $k  \neq 0$.

In lesson 5 "Vectorization: Low Rank Matrix Factorization" and in lesson 6 "Implementation detail: Mean normalization" the matrix $Y$ contains a mistake. The element $Y^{(5,4)}$ (Dave's opinion on "Sword vs Karate") should be a question mark but is incorrectly given as 0.

In lesson 6 this mistake is propagated to the calculation of $\mu$. When $\mu$ is calculated the 5th movie is given an average rating of 1.25 because (0+0+5+0)/4=1.25, but it should be (0+0+5)/3=1.67. This the affects the new values in the matrix $Y$.

In `ex8_cofi.m` at line 199, where theta is trained using `fmincg()` for the movie ratings, the use of "Y" in the function call should be "Ynorm". Y is normalized in line 181, creating Ynorm, but then it is never used. The video lecture "Implementation Detail: Mean Normalization" at 5:34 makes it pretty clear that the normalized Y matrix should be used for calculating theta.

In `ex8.pdf` section 2, "collaborative fitlering" should be "collaborative filtering"

In `ex8.pdf` section 2.2.1, “it will later by called” should be “it will later be called”

In `checkCostFunction.m` it prints "If your backpropagation implementation is correct...", but in this exercise there is no backpropagation.

In the quiz, question 4 has an invalid phrase: "Even if you each user has rated only a small fraction of all of your products (so $r(i,j)=0$ for the vast majority of $(i,j)$ pairs), you can still build a recommender system by using collaborative filtering." The word "you" seems misplaced, "your" or none.

In the quiz, question 4, one of answer options has a typo "For collaborative filtering, it is possible to use one of the advanced optimization algoirthms"

In `ex8.pdf` at the bottom of page 8, the text says that the number of features used by ex8_cofi.m is 100. Actually the number of features is 10, not 100.



### Quiz: Recommender Systems







