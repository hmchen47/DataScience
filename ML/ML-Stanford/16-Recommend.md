# Recommender Systems

## Predicting Movie Ratings


### Problem Formulation

#### Lecture Note

+ Example: Predicting movie ratings
  + User rates movies using zero to five starts

    | Movie | Alice (1) | Bob (2) | Carol (3) | Dave (4) |
    |-------|-----------|---------|-----------|----------|
    | Love at last | 5 | 5 | 0 | 0 |
    |Romance forever | 5 | ? | ? | 0 |
    | Cute puppies of love | ? | 4 | 0 | ? |
    | Nonstop car chases | 0 | 0 | 5 | 4 |
    | Swords vs. karate | 0 | 0 | 5 | ? |

  + Notations:
    + $n_u\;$: no. users
    + $n_m\;$: no. movies
    + $r(i, j)\;$: 1 if user $j$ has rated movie $i$; 0 for others
    + $y^{(i, j)}\;$: rating given by user $j$ to movie $i$ (defined only if $r(i, j) = 1$); $y^{(i, j)} \in \{1, 2, 3, 4, 5\}$

  + Parameters: $n_u = 4, n_m = 5$

  + IVQ: In our notation, $r(i,j)=1$ if user $j$ has rated movie $i$, and $y^{(i,j)}$ is his rating on that movie. Consider the following example (no. of movies $n_m=2$, no. of users $n_u=3$
  
    | Movie  | User 1 | User 2 | User 3 |
    |--------|--------|--------|--------|
    | Movie 1 | 0 | 1 | ?
    | Movie 2 | ? | 5 | 5

    What is $r(2,1)$? How about $y^{(2,1)}$?

    1. $r(2,1) = 0,\ y^{(2,1)} = 1$
    2. $r(2,1) = 1,\ y^{(2,1)} = 1$
    3. $r(2,1) = 0,\ y^{(2,1)} = \text{undefined}$
    4. $r(2,1) = 1,\ y^{(2,1)} = \text{undefined}$

    Ans: 3

#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/17.1-RecommenderSystems-ProblemFormulation.68db7c30b22b11e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1556755200&Signature=CtDjHXBd4CsmFyCc6uQUK7WPZ-3MoRxQcpzjkaKWiIGyeCRC60y0GfhxNAGO4TdM5p0HYqBfk2CvFbeCanPY2vzIcp~p~phuyvkGRvn5hLVVBHdeVcvUEcdtSxoiDhgHf2hACM99AoW~MTY71EscQY4VkYep00VL7KPLUlfTHEk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/ktecjTO-SX6XnI0zvrl-sA?expiry=1556755200000&hmac=mMdYynsAyrnhY2f3UENYH7w985vZj6K9y2rXZL706rg&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Content Based Recommendations

#### Lecture Note

+ Content-based recommender systems

  | Example  | Movie | Alice (1) <br/> $\theta^{(1)}$ | Bob (2) <br/> $\theta^{(2)}$ | Carol (3) <br/> $\theta^{(3)}$  | Dave (4) <br/> $\theta^{(4)}$  | $x_1$ <br/> (romance) | $x_2$ <br/> (action) |
  |:---------:|:-----:|:--------:|:-------:|:---------:|:-------:|:---:|:---:|
  | $x^{(1)}$ | Love at last | 5 | 5 | 0 | 0 | 0.9 | 0 |
  | $x^{(2)}$ |Romance forever | 5 | ? | ? | 0 | 1.0 | 0.01 |
  | $x^{(3)}$ | Cute puppies of love | ? | 4 | 0 | ? | 0.99 | 0 |
  | $x^{(4)}$ | Nonstop car chases | 0 | 0 | 5 | 4 | 0.1 | 1.0 |
  | $x^{(5)}$ | Swords vs. karate | 0 | 0 | 5 | ? | 0 | 0.9 |

  + Parameters: $n_u = 4, n_m = 5, x_0 = 1, n = 2$
  + For each user $j$, learn a parameter $\theta^{(j)} \in \mathbb{R}^3$.  Predict user $j$ as rating movie $i$ with $(\theta^{(j)})^Tx^{(i)}$ starts. ($\theta^{(j)}$ values discussed later)

    $$x^{(3)} = \begin{bmatrix} 1 \\ 0.99 \\ 0 \end{bmatrix} \quad\longleftrightarrow\quad \theta^{(1)} = \begin{bmatrix} 0 \\ 5 \\ 0 \end{bmatrix} \implies (\theta^{(1)})^Tx^{(3)} = 5 \times 0.99 = 4.95$$
  + IVQ: Consider the following set of movie ratings (see above table). Which of the following is a reasonable value for $\theta^{(3)}$? Recall that $x_0 = 1$.

    1. $\theta^{(3)} = \begin{bmatrix} 0 \\ 5 \\ 0 \end{bmatrix}$
    2. $\theta^{(3)} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$
    3. $\theta^{(3)} = \begin{bmatrix} 1 \\ 0 \\ 4 \end{bmatrix}$
    4. $\theta^{(3)} = \begin{bmatrix} 0 \\ 0 \\ 5 \end{bmatrix}$

    Ans: 4


+ Problem formulation
  + $r(i, j)$ = 1 if user $j$ has rated movie $i$ (0 otherwise)
  + $y^{(i, j)}$ = rating by user $j$ on movie $i$ (if defiuned)
  + $\theta^{(j)}$ = parameter vector for user $j$; $\theta^{(j)} \in \mathbb{R}^{n+1}$
  + $x^{(i)}$ = feature vector for movie $i$
  + Predicted rating: $(\theta^{(j)})^T(x^{(i)})$ for user $j$, movie $i$
  + $m^{(j)}$ = no. of movies rated by user $j$
  + Objective: to learn $\theta^{(j)}$

    $$\min_{\theta^{(i, j)}} \dfrac{1}{2m^{(j)}} \sum_{i: r(i, j) = 1} \left( (\theta^{(i)})^T(x^{(i)}) - y^{(i, j)}) \right)^2 + \dfrac{\lambda}{2m^{(j)}} \sum_{k=1}^n  \left(\theta_k^{(j)}\right)^2$$

+ Optimization objective:
  + To learn $\theta^{(j)}$ (parameter for user $j$): (with $m^{(j)}$ factor removed)

    $$\min_{\theta^{(i, j)}} \underbrace{\dfrac{1}{2} \sum_{i: r(i, j) = 1} \left( (\theta^{(i)})^T(x^{(i)}) - y^{(i, j)}) \right)^2}_{\text{cost function}} + \underbrace{\dfrac{\lambda}{2} \sum_{k=1}^n  \left(\theta_k^{(j)}\right)^2}_{\text{regularization}}$$
  + To learn $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)}$:

    $$\min_{\theta^{(1)},\dots,\theta^{(n_u)}} \dfrac{1}{2}\displaystyle \sum_{j=1}^{n_u} \underbrace{\sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)} \right)^2}_{\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)}} + \dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left(\theta_k^{(j)}\right)^2$$

+ Optimization algorithm:
  + Objective: 

    $$\min_{\theta^{(1)},\dots,\theta^{(n_u)}} \underbrace{\dfrac{1}{2}\displaystyle \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left(\theta_k^{(j)}\right)^2}_{J(\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)})}$$

  + Gradient descent update:

    $$\begin{array}{rcll} \theta_k^{(j)} &:=& \theta_k^{(j)} - \alpha \sum_{i: r(i, j) = 1} \left( (\theta^{(i, j)})^T x^{(i)} - y^{(i, j)} \right)^2 &\quad (\text{for } k = 0) \\\\ \theta_k^{(j)} &:=& \theta_k^{(j)} - \alpha \underbrace{\left( \sum_{i: r(i, j) = 1} ((\theta^{(i, j)})^T x^{(i)} - y^{(i, j)})x_k^{(i)} + \lambda \theta_k^{(j)} \right)}_{\frac{\partial}{\partial \theta_k^{(j)}} J(\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)})} & \quad (\text{for } k \neq 0) \end{array}$$


#### Lecture Video


<video src="https://d3c33hcgiwev3.cloudfront.net/17.2-RecommenderSystems-ContentBasedRecommendations.0d0f1e70b22b11e48803b9598c8534ce/full/360p/index.mp4?Expires=1556841600&Signature=FcnKwZMJhqzIR-UddYAgH-bgTt~ZHO0LdWZNPKbOSERakpLUOgUNYfpC9sHoL42J-OlnjB21Q0u~H6ez-XPVm3KNeJ0jO1GfPGki90ucpe2GRqE0OmebTeuAlI4g1XsQBlAP~02S4O7hhrGSv57xsag3oHp7XmqWp3Z2GMdKk-0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/dT8f9QRDSA2_H_UEQxgNWw?expiry=1556841600000&hmac=F3gtOz7Hwv3oI7eHC4WiNDTqiDH5pFtoGCpj1QoIJDo&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Collaborative Filtering

### Collaborative Filtering

#### Lecture Note

+ Problem Motivation

  | Example  | Movie | Alice (1) <br/> $\theta^{(1)}$ | Bob (2) <br/> $\theta^{(2)}$ | Carol (3) <br/> $\theta^{(3)}$  | Dave (4) <br/> $\theta^{(4)}$  | $x_1$ <br/> (romance) | $x_2$ <br/> (action) |
  |:---------:|:-----:|:--------:|:-------:|:---------:|:-------:|:---:|:---:|
  | $x^{(1)}$ | Love at last | 5 | 5 | 0 | 0 | ? | ? |
  | $x^{(2)}$ |Romance forever | 5 | ? | ? | 0 | ? | ? |
  | $x^{(3)}$ | Cute puppies of love | ? | 4 | 0 | ? | ? | ? |
  | $x^{(4)}$ | Nonstop car chases | 0 | 0 | 5 | 4 | ? | ? |
  | $x^{(5)}$ | Swords vs. karate | 0 | 0 | 5 | ? | ? | ? |

  + Initial guess based on the info from table

    $$\theta^{(1)} = \begin{bmatrix} 0 \\ 5 \\ 0 \end{bmatrix}, \quad \theta^{(2)} = \begin{bmatrix} 0 \\5 \\ 0 \end{bmatrix}, \quad \theta^{(3)} = \begin{bmatrix} 0 \\ 0 \\ 5 \end{bmatrix}, \quad \theta^{(4)} = \begin{bmatrix} 0 \\ 0 \\ 5 \end{bmatrix} \implies \theta^{(j)}$$
    <br/>
  
  + Guess $x^{(i))}$ based on the info in rows
    + $x^{(1)} = \begin{bmatrix} 1 \\ 1.0 \\ 0.0 \end{bmatrix} \longrightarrow$ the 1st sample rated as romance than action

  + Based on the guess, expect to have the following result

    $$(\theta^{(1)})^T x^{(1)} \approx 5 \qquad (\theta^{(2)})^T x^{(2)} \approx 5  \qquad (\theta^{(3)})^T x^{(3)} \approx 0 \qquad (\theta^{(4)})^T x^{(4)} \approx 0$$

  + IVQ: Consider the following movie ratings:

    | . | User 1 | User 2 | User 3 | (romance) |
    |--|--|--|--|--|
    | Movie 1 | 0 | 1.5 | 2.5 | ? |
    
    Note that there is only one feature $x_1$. Suppose that:

    $$\theta^{(1)} = \begin{bmatrix} 0 \\0 \end{bmatrix}, \ \theta^{(2)} = \begin{bmatrix} 0 \\ 3 \end{bmatrix}, \ \theta^{(3)} = \begin{bmatrix} 0 \\ 5 \end{bmatrix}$$

    What would be a reasonable value for $x_1^{(1)}$ (the value denoted "?" in the table above)?

    1. 0.5
    2. 1
    3. 2
    4. Any of these values would be equally reasonable.

    Ans: 1


+ Optimization algorithm
  + Given $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)}$. to learn $x^{(i)}$:

    $$\min_{x^{(i)}} \dfrac{1}{2} \sum_{j: r(i, j) = 1} \left( (\theta^{(i)})^T(x^{(i)}) - y^{(i, j)}) \right)^2 + \dfrac{\lambda}{2} \sum_{k=1}^n  \left(\theta_k^{(j)}\right)^2$$

  + Given $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n_u)}$. to learn $x^{(i)}. \dots, x^{(n_m)}$:

    $$\min_{x^{(1)}, \dots, x^{(n_m)}} \dfrac{1}{2} \sum_{i=1}^{n_m} \sum_{j: r(i, j) = 1} \left( (\theta^{(i)})^T(x^{(i)}) - y^{(i, j)}) \right)^2 + \dfrac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n  \left(\theta_k^{(j)}\right)^2$$

  + IVQ: Suppose you use gradient descent to minimize:

    $$\min_{x^{(1)}, \dots, x^{(n_m)}} \dfrac{1}{2} \sum_{i=1}^{n_m} \sum_{j: r(i, j) = 1} \left( (\theta^{(i)})^T(x^{(i)}) - y^{(i, j)}) \right)^2 + \dfrac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n  \left(\theta_k^{(j)}\right)^2$$

    Which of the following is a correct gradient descent update rule for $i\neq 0$?

    1. $x_k^{(i)} := x_k^{(i)} + \alpha\left(\sum_{j:r(i,j)=1}\left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right)\theta_k^{(j)}\right)$
    2. $x_k^{(i)} := x_k^{(i)} - \alpha\left(\sum_{j:r(i,j)=1}\left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right)\theta_k^{(j)}\right)$
    3. $x_k^{(i)} := x_k^{(i)} + \alpha\left(\sum_{j:r(i,j)=1}\left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right)\theta_k^{(j)} + \lambda x_k^{(i)}\right)$
    4. $x_k^{(i)} := x_k^{(i)} - \alpha\left(\sum_{j:r(i,j)=1}\left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)}\right)\theta_k^{(j)}+ \lambda x_k^{(i)}\right)$

    Ans: 4


+ Collaborative filtering
  + Given $x^{(1)}, \dots, x^{(n_m)}$ (and movie ratings), can estimate $\theta^{(1)}, \dots, \theta^{(n_u)}$
  + Given $\theta^{(1)}, \dots, \theta^{(n_u)}$, can estimate  $x^{(1)}, \dots, x^{(n_m)}$
  + Guess $\theta \;\rightarrow\; x \;\rightarrow\; \theta \;\rightarrow\; x \;\rightarrow\; \theta \;\rightarrow\; x \;\rightarrow\; \theta \;\rightarrow\; x \;\rightarrow\; \dots$


#### Lecture Video


<video src="https://d3c33hcgiwev3.cloudfront.net/17.3-RecommenderSystems-CollaborativeFiltering-V1.b51cf830b22b11e49f072fa475844d6b/full/360p/index.mp4?Expires=1556841600&Signature=Qaz73CPbRjOh-ndSh77gNvUzfrEtFF90O8xNvJQU460DuKGj6DJRzYuh6ySV9cGMmV1zAqsF6ZtEBD5ooJpdopIqI7EGxOg052yT~eAc9h~B2K-0pplw2N~dXaqfLXOHddEFPOEVOMhTJmOf73cNL8PIIl4f2oBC~tKlGZAb~Bg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/lv6r7GOkSjO-q-xjpCozkg?expiry=1556841600000&hmac=2yUD_IBSJYDZGqEsBqwxpOUJQBmefSBL1c2JsxQn_4o&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Collaborative Filtering Algorithm

#### Lecture Note

+ Collaboration filtering optimization objective
  + Given $x^{(1)}, \dots, x^{(n_m)}$ (and movie ratings), can estimate $\theta^{(1)}, \dots, \theta^{(n_u)}$

    $$\min_{\theta^{(1)},\dots,\theta^{(n_u)}} \dfrac{1}{2}\displaystyle \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left(\theta_k^{(j)}\right)^2$$

  + Given $\theta^{(1)}, \dots, \theta^{(n_u)}$, can estimate  $x^{(1)}, \dots, x^{(n_m)}$

    $$\min_{x^{(1)}, \dots, x^{(n_m)}} \dfrac{1}{2} \sum_{i=1}^{n_m} \sum_{j: r(i, j) = 1} \left( (\theta^{(i)})^T(x^{(i)}) - y^{(i, j)}) \right)^2 + \dfrac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n  \left(\theta_k^{(j)}\right)^2$$

  + Minimizing $x^{(1)}, \dots, x^{(n_m)}$ and $\theta^{(1)}, \dots, \theta^{(n_u)}$ simultaneously:

    $$J(x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}) = \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j)=1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \dfrac{\lambda}{2}\sum_{j=1}^{n_u} \sum_{k=1}^{n} \left(\theta_k^{(j)}\right)^2$$
    <br/>

    $$\min_{\substack{x^{(1)}, \dots, x^{(n_m)},\\ \theta^{(1)}, \dots, \theta^{(n_u)}}} J(x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)})$$
  
  + $\theta_0$ and $x_0$ are not required: $x \in \mathbb{R}^n, \theta \in \mathbb{R}^n$

+ Collaborative filtering algorithm
  1. Initialize $x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}$ to small random values
  2. Minimize $J(x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)})$ using gradient decent (or an advanced optimization algorithm)., e.g., for every $j = 1, \dots, n_u, i=1, \dots, n_m$:

    $$\begin{array}{rcl} x_k^{(i)} &:=& x_k^{(i)} - \alpha\left (\displaystyle \sum_{j:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \theta_k^{(j)}} + \lambda x_k^{(i)} \right) \\\\ \theta_k^{(j)} &:=& \theta_k^{(j)} - \alpha\left (\displaystyle \sum_{i:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x_k^{(i)}} + \lambda \theta_k^{(j)} \right)\end{array}$$
  3. For a user with parameters $\theta$ and a movie with (learned) features $x$, predict a start rating of $\theta^Tx$. [rating for user $j$ and movie $i$: $(\theta^{(j)})^T(x^{(i)})$]

  + IVQ: In the algorithm we described, we initialized $x^{(1)}, \dots, x^{(n_m)}$ and $\theta^{(1)},\dots,\theta^{(n_u)}$ to small random values. Why is this?

    1. This step is optional. Initializing to all 0’s would work just as well.
    2. Random initialization is always necessary when using gradient descent on any problem.
    3. This ensures that $x^{(i)} \neq \theta^{(j)}$ for any $i,j$.
    4. This serves as symmetry breaking (similar to the random initialization of a neural network’s parameters) and ensures the algorithm learns features $x^{(1)}, \dots, x^{(n_m)}$ that are different from each other.

    Ans: 4



#### Lecture Video


<video src="https://d18ky98rnyall9.cloudfront.net/17.4-RecommenderSystems-CollaborativeFilteringAlgorithm.c265b220b22b11e4a416e948628da1fd/full/360p/index.mp4?Expires=1556841600&Signature=MKZnUQ5YdDRv7~1~0UACFRKPucREISV~9GlF8PjSt6tmnLo5FlFwyA1KIXDaJb8aBoDU4NOGVdIHqh~3zalLArwqkAH~UwURb48DsCuHzPsc4C-b5k7FRLkthAGFyUxixOx4ZhN8msaWNbKH1GCAuVN4yCUczNqje4Nh2y67l~k_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/pnw6TUaXTLa8Ok1Glxy2sw?expiry=1556841600000&hmac=1jjKVTNL6Brr-WnLl_sCrgQrc6iYnnmgYQ0HVdV2lHQ&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Low Rank Matrix Factorization


### Vectorization: Low Rank Matrix Factorization

#### Lecture Note

+ Example for Collaborative filtering

  | Movie | Alice (1) | Bob (2) | Carol (3) | Dave (4) |
  |-------|-----------|---------|-----------|----------|
  | Love at last | 5 | 5 | 0 | 0 |
  |Romance forever | 5 | ? | ? | 0 |
  | Cute puppies of love | ? | 4 | 0 | ? |
  | Nonstop car chases | 0 | 0 | 5 | 4 |
  | Swords vs. karate | 0 | 0 | 5 | ? |
  <br/>

  $$Y = [y^{(i, j)}] = \begin{bmatrix} 5 & 5 & 0 & 0 \\ 5 & ? & ? & 0 \\ ? & 4 & 0 & ? \\ 0 & 0 & 5 & 4 \\ 0 & 0 & 5 & 0 \end{bmatrix}$$

+ Collaborative filtering
  + Predicted ratings: $(i,j) \rightarrow (\theta^{(j)})^T(x^{(i)})$

    $$X \Theta^T = \begin{bmatrix} (\theta^{(1)})^T(x^{(1)}) & (\theta^{(2)})^T(x^{(1)}) & \cdots & (\theta^{(n_u)})^T(x^{(1)}) \\ (\theta^{(1)})^T(x^{(2)}) & (\theta^{(2)})^T(x^{(2)}) & \cdots & (\theta^{(n_u)})^T(x^{(2)}) \\ \vdots & \vdots & \ddots & \vdots \\ (\theta^{(1)})^T(x^{(n_m)}) & (\theta^{(2)})^T(x^{(n_m)}) & \cdots & (\theta^{(n_u)})^T(x^{(n_m)}) \end{bmatrix}$$ 
    <br/>

    $$X = \begin{bmatrix} - & (x^{(1)})^T & - \\ - & (x^{(2)})^T & - \\ & \vdots & \\ - & (x^{(n_m)})^T & - \end{bmatrix} \qquad\qquad \Theta = \begin{bmatrix} - & (\theta^{(1)})^T & - \\ - & (\theta^{(2)})^T & - \\ & \vdots & \\ - & (\theta^{(n_u)})^T & - \end{bmatrix}$$
  + a.k.a Low rank matrix factorization

  + IVQ: Let $X = \begin{bmatrix} - & (x^{(1)})^T & - \\ - & (x^{(2)})^T & - \\ & \vdots & \\ - & (x^{(n_m)})^T & - \end{bmatrix} \qquad\qquad \Theta = \begin{bmatrix} - & (\theta^{(1)})^T & - \\ - & (\theta^{(2)})^T & - \\ & \vdots & \\ - & (\theta^{(n_u)})^T & - \end{bmatrix}$

    What is another way of writing the following:

    $$\begin{bmatrix} (x^{(1)})^T(θ^{(1)}) & \cdots & (x(1))T(θ(n_u)) \\ \vdots & \ddots & \vdots \\ (x(n_m))^T(θ(1)) & \cdots & (x(n_m))^T(θ(n_u)) \end{bmatrix}$$

    1. $X\Theta$
    2. $X^T\Theta$
    3. $X\Theta^T$
    4. $\Theta^TX^T$

    Ans: 3

+ Finding related movies
  + For each product $i$, we learn a feature vector $x^{(i)} \in \mathbb{R}^n$; e.g., $x_1$ = romance, $x_2$ = action, $x_3$ = comedy, $x_4 = \dots$
  + How to fidn movies $j$ related to movie $i$?

    $$\parallel x^{(i)} - x^{(j)} \parallel \rightarrow 0 \implies \text{movie } j \text{ and } i \text { are "similar"}$$
  + 5 most similar movies to movie $i$: find the 5 movies with the smallest $\parallel x^{(i)} - x^{(j)} \parallel$



#### Lecture Video


<video src="https://d18ky98rnyall9.cloudfront.net/17.5-RecommenderSystems-VectorizationLowRankMatrixFactorization.3f9b07a0b22b11e49f072fa475844d6b/full/360p/index.mp4?Expires=1556841600&Signature=Meauz1tCZb0teRaNL2AGdYX~m6lk1qzsnbXiXeUKlWCNqBCby5xR8KykNZuUdJj6TIgFREIiHl85ZXq1raKDs9ErPUwqY4zlTZyzA5kRkU~mARS2YZ1v-G08mWYDQ2VCeXHJE9p3ajj-yuzHysyReCbtZiX1vY9q1Wc1jXdeQP4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/IJUQPlfaSVyVED5X2vlctg?expiry=1556841600000&hmac=dq4186hWUBr93LyJYukbbB4jY4Xw55vLZLdMPxHPskk&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Implementational Detail: Mean Normalization

#### Lecture Note

+ users who have not rated any movies

  | Example  | Movie | Alice (1) <br/> $\theta^{(1)}$ | Bob (2) <br/> $\theta^{(2)}$ | Carol (3) <br/> $\theta^{(3)}$  | Dave (4) <br/> $\theta^{(4)}$  | Eve(5) <br/> $\theta^{(5)}$ |
  |:---------:|:-----:|:--------:|:-------:|:---------:|:-------:|:---:|
  | $x^{(1)}$ | Love at last | 5 | 5 | 0 | 0 | ? |
  | $x^{(2)}$ |Romance forever | 5 | ? | ? | 0 | ? |
  | $x^{(3)}$ | Cute puppies of love | ? | 4 | 0 | ? | ? |
  | $x^{(4)}$ | Nonstop car chases | 0 | 0 | 5 | 4 | ? |
  | $x^{(5)}$ | Swords vs. karate | 0 | 0 | 5 | ? | ? |
  <br/>

  $$Y =  \begin{bmatrix} 5 & 5 & 0 & 0 & ? \\ 5 & ? & ? & 0 & ? \\ ? & 4 & 0 & ? & ? \\ 0 & 0 & 5 & 4 & ? \\ 0 & 0 & 5 & 0 & ? \end{bmatrix}$$
  <br/>

  $$\min_{\substack{x^{(1)}, \dots, x^{(n_m)},\\ \theta^{(1)}, \dots, \theta^{(n_u)}}} \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j)=1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \dfrac{\lambda}{2}\sum_{j=1}^{n_u} \sum_{k=1}^{n} \left(\theta_k^{(j)}\right)^2$$

  + Example: $n=2, \theta^{(5)} \in \mathbb{R}^2 \longrightarrow \theta^{(5)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$
  
    $$\min_{\substack{x^{(1)}, \dots, x^{(n_m)},\\ \theta^{(1)}, \dots, \theta^{(n_u)}}} \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j)=1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i,j)} \right)^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \underbrace{\dfrac{\lambda}{2}\sum_{j=1}^{n_u} \sum_{k=1}^{n} \left(\theta_k^{(j)}\right)^2}_{\frac{\lambda}{2} \left[ (\theta_1^{(5)})^2 + (\theta_2^{(5)})^2 \right]}$$
    <br/>

    $$\theta^{(5)})^Tx^{(i)} = 0 \rightarrow \text{ all rated } 0$$
  
+ Mean normalization

  $$Y = \begin{bmatrix} 5 & 5 & 0 & 0 & ? \\ 5 & ? & ? & 0 & ? \\ ? & 4 & 0 & ? & ? \\ 0 & 0 & 5 & 4 & ? \\ 0 & 0 & 5 & 0 & ? \end{bmatrix} \qquad\qquad \mu = \begin{bmatrix} 2.5 \\ 2.5 \\2 \\ 2.25 \\ 1.25  \end{bmatrix} \rightarrow Y = \begin{bmatrix} 2.5 & 2.5 & -2.5 & -2.5 & ? \\2.5 & ? & ? & -2.5 & ? \\ ? & 2 & -2 & ? & ? \\ -2.25 & -2.25 & 2.75 & 1.75 & ? \\ -1.25 & -1.25 & 3.75 & -1.25 & ? \end{bmatrix} \rightarrow \text{ learn } \theta^{(j)}, x^{(i)}$$

  + For user $j$, on movie $i$ predict: 
  
    $$(\theta^{(j)})^T (x^{(i)}) + \mu_i$$

  + User 5 (Eve): 

    $$\theta^{(5)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \;\rightarrow\; \underbrace{(\theta^{(j)})^T (x^{(i)})}_{ = 0} + \mu_i = \mu_i$$

  + IVQ: We talked about mean normalization. However, unlike some other applications of feature scaling, we did not scale the movie ratings by dividing by the range (max – min value). This is because:

    1. This sort of scaling is not useful when the value being predicted is real-valued.
    2. All the movie ratings are already comparable (e.g., 0 to 5 stars), so they are already on similar scales.
    3. Subtracting the mean is mathematically equivalent to dividing by the range.
    4. This makes the overall algorithm significantly more computationally efficient.

  Ans: 2


#### Lecture Video


<video src="https://d18ky98rnyall9.cloudfront.net/17.6-RecommenderSystems-ImplementationalDetailMeanNormalization.536281a0b22b11e48803b9598c8534ce/full/360p/index.mp4?Expires=1556841600&Signature=eiXk0JTMr83ujViKtzRyxjKgNsQKeki1F0j4S3~PC4RwYeGcpwjph28ihx7h10yotfQwaDvv1mmWrJLefkzN4gv4grICCb8qSgHc-l6-nhBaJV9q2y6uynd97-0LuFZEqK3t6B6N1BsT0wTvWt5qB5UHkfiCLHICwyhiYuysDDU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/HzBcIU_4QDGwXCFP-IAxRw?expiry=1556841600000&hmac=hXWjhVy7tE9qct10o9EFUiNSKfVKKd8gxKYpDUTYKuE&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
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

  $$\begin{array}{rcl} x_k^{(i)} &:=& x_k^{(i)} - \alpha\left (\displaystyle \sum_{j:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \theta_k^{(j)}} + \lambda x_k^{(i)} \right) \\\\ \theta_k^{(j)} &:=& \theta_k^{(j)} - \alpha\left (\displaystyle \sum_{i:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x_k^{(i)}} + \lambda \theta_k^{(j)} \right)\end{array}$$
​ |  
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
​ |  

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


1. Suppose you run a bookstore, and have ratings (1 to 5 stars) of books. Your collaborative filtering algorithm has learned a parameter vector $\theta^{(j)}$ for user $j$, and a feature vector $x^{(i)}$ for each book. You would like to compute the 
"training error", meaning the average squared error of your system's predictions on all the ratings that you have gotten from your users. Which of these are correct ways of doing so (check all that apply)?

    For this problem, let mm be the total number of ratings you have gotten from your users. (Another way of saying this is that $m = \sum_{i=1}^{n_m} \sum_{j=1}^{n_u} r(i,j)$. [Hint: Two of the four options below are correct.]

    1. $\frac{1}{m} \sum_{(i,j):r(i,j)=1} \sum_{k=1}^n (( \theta^{(j)})_k x^{(i)}_k - y^{(i,j)} )^2$
    2. $\frac{1}{m} \sum_{(i,j):r(i,j)=1} (\sum_{k=1}^n (\theta^{(j)})_k x^{(i)}_k - y^{(i,j)} )^2$
    3. $\frac{1}{m} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} (( \theta^{(j)})_i x^{(i)}_j - y^{(i,j)} )^2$
    4. $\frac{1}{m} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} ( \sum_{k=1}^n (\theta^{(j)})_k x^{(i)}_k - y^{(i,j)} )^2$
  
    Ans: x14, X23 (4312), x13 (1234)



2. In which of the following situations will a collaborative filtering system be the most appropriate learning algorithm (compared to linear or logistic regression)?

    1. You run an online news aggregator, and for every user, you know some subset of articles that the user likes and some different subset that the user dislikes. You'd want to use this to find other articles that the user likes.
    2. You manage an online bookstore and you have the book ratings from many users. For each user, you want to recommend other books she will enjoy, based on her own ratings and the ratings of other users.
    3. You manage an online bookstore and you have the book ratings from many users. You want to learn to predict the expected sales volume (number of books sold) as a function of the average rating of a book.
    4. You've written a piece of software that has downloaded news articles from many news websites. In your system, you also keep track of which articles you personally like vs. dislike, and the system also stores away features of these articles (e.g., word counts, name of author). Using this information, you want to build a system to try to find additional new articles that you personally will like.
    5. You manage an online bookstore and you have the book ratings from many users. You want to learn to predict the expected sales volume (number of books sold) as a function of the average rating of a book.
    6. You own a clothing store that sells many styles and brands of jeans. You have collected reviews of the different styles and brands from frequent shoppers, and you want to use these reviews to offer those shoppers discounts on the jeans you think they are most likely to purchase
    7. You run an online bookstore and collect the ratings of many users. You want to use this to identify what books are "similar" to each other (i.e., if one user likes a certain book, what are other books that she might also like?)
    8. You're an artist and hand-paint portraits for your clients. Each client gets a different portrait (of themselves) and gives you 1-5 star rating feedback, and each client purchases at most 1 portrait. You'd like to predict what rating your next customer will give you.

    Ans: 67 (7658) X78 (5678), 12 (1234)




3. You run a movie empire, and want to build a movie recommendation system based on collaborative filtering. There were three popular review websites (which we'll call $A$, $B$ and $C$) which users to go to rate movies, and you have just acquired all three companies that run these websites. You'd like to merge the three companies' datasets together to build a single/unified system. On website $A$, users rank a movie as having 1 through 5 stars. On website $B$, users rank on a scale of 1 - 10, and decimal values (e.g., 7.5) are allowed. On website $C$, the ratings are from 1 to 100. You also have enough information to identify users/movies on one website with users/movies on a different website. Which of the following statements is true?

    1. You can merge the three datasets into one, but you should first normalize each dataset's ratings (say rescale each dataset's ratings to a 1-100 range).
    2. Assuming that there is at least one movie/user in one database that doesn't also appear in a second database, there is no sound way to merge the datasets, because of the missing data.
    3. It is not possible to combine these websites' data. You must build three separate recommendation systems.
    4. You can combine all three training sets into one without any modification and expect high performance from a recommendation system.
    5. You can merge the three datasets into one, but you should first normalize each dataset separately by subtracting the mean and then dividing by (max - min) where the max and min (5-1) or (10-1) or (100-1) for the three websites respectively.
    6. You can combine all three training sets into one as long as your perform mean normalization and feature scaling after you merge the data.

    Ans: 5 (5374), 1 (1234)


4. Which of the following are true of collaborative filtering systems? Check all that apply.

    1. When using gradient descent to train a collaborative filtering system, it is okay to initialize all the parameters $(x^{(i)}$ and $\theta^{(j)}$ to zero.
    2. Recall that the cost function for the content-based recommendation system is $J(\theta) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j) =1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (\theta_k^{(j)})^2$. Suppose there is only one user and he has rated every movie in the training set. This implies that $n_u = 1$ and $r(i,j) = 1$ for every $i,j$. In this case, the cost function $J(\theta)$ is equivalent to the one used for regularized linear regression.
    3. If you have a dataset of users ratings' on some products, you can use these to predict one user's preferences on products he has not rated.
    4. To use collaborative filtering, you need to manually design a feature vector for every item (e.g., movie) in your dataset, that describes that item's most important properties.
    5. Suppose you are writing a recommender system to predict a user's book preferences. In order to build such a system, you need that user to rate all the other books in your training set.
    6. For collaborative filtering, the optimization algorithm you should use is gradient descent. In particular, you cannot use more advanced optimization algorithms (L-BFGS/conjugate gradient/etc.) for collaborative filtering, since you have to solve for both the $x^{(i)}$'s and $\theta^{(j)}$'s simultaneously.
    7. Even if each user has rated only a small fraction of all of your products (so $r(i,j)=0$ for the vast majority of $(i,j)$ pairs), you can still build a recommender system by using collaborative filtering.
    8. For collaborative filtering, it is possible to use one of the advanced optimization algoirthms (L-BFGS/conjugate gradient/etc.) to solve for both the $x^{(i)}$'s and $\theta^{(j)}$'s simultaneously.

    Ans: 23 (2134), 78 (5678), 23 (1234)



5. Suppose you have two matrices AA and BB, where AA is 5x3 and BB is 3x5. Their product is $C = AB$, a 5x5 matrix. Furthermore, you have a 5x5 matrix $\mathbb{R}$ where every entry is 0 or 1. You want to find the sum of all elements $C(i,j)$ for which the corresponding $R(i,j)$ is 1, and ignore all elements $C(i,j)$ where $R(i,j) = 0$. One way to do so is the following code:

    Which of the following pieces of Octave code will also correctly compute this total? Check all that apply. Assume all options are in code.

    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div><a href="https://www.coursera.org/learn/machine-learning/exam/3HGvu/recommender-systems">
        <img src="iamges/m16-01.png" style="margin: 0.1em;" alt="Diagram for Q5 in Mod16" title="Diagram for Q5 in Mod16" width="350">
      </a></div>
    </div>

    1. `total = sum(sum((A * B) .* R))`
    2. `C = (A * B) .* R; total = sum(C(:));`
    3. `total = sum(sum((A * B) * R));`
    4. `C = (A * B) * R; total = sum(C(:));`

    Ans: 12



