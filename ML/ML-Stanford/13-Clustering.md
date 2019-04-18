# Unsupervised Learning: Clustering

## Unsupervised Learning: Introduction

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## K-Means Algorithm

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Optimization Objective

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Random Initialization

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Choosing the Number of Clusters

### Lecture Notes




### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Review

### Lecture Slides

#### Unsupervised Learning: Introduction

Unsupervised learning is contrasted from supervised learning because it uses an unlabeled training set rather than a labeled one.

In other words, we don't have the vector y of expected results, we only have a dataset of features where we can find structure.

Clustering is good for:

+ Market segmentation
+ Social network analysis
+ Organizing computer clusters
+ Astronomical data analysis


#### K-Means Algorithm

The K-Means Algorithm is the most popular and widely used algorithm for automatically grouping data into coherent subsets.

1. Randomly initialize two points in the dataset called the cluster centroids.
2. Cluster assignment: assign all examples into one of two groups based on which cluster centroid the example is closest to.
3. Move centroid: compute the averages for all the points inside each of the two cluster centroid groups, then move the cluster centroid points to those averages.
4. Re-run (2) and (3) until we have found our clusters.

Our main variables are:

+ K (number of clusters)
+ Training set $\{x^{(1)}, x^{(2)}, \dots,x^{(m)}\}$
+ Where $x^{(i)} \in \mathbb{R}^n$

Note that we __will not use__ the x0=1 convention.

The algorithm:

```matlab
Randomly initialize K cluster centroids mu(1), mu(2), ..., mu(K)
Repeat:
   for i = 1 to m:
      c(i):= index (from 1 to K) of cluster centroid closest to x(i)
   for k = 1 to K:
      mu(k):= average (mean) of points assigned to cluster k
```

The first for-loop is the 'Cluster Assignment' step. We make a vector $c$ where $c(i)$ represents the centroid assigned to example x(i).

We can write the operation of the Cluster Assignment step more mathematically as follows:

$$c^{(i)} = \arg\min_k\ \parallel x^{(i)} - \mu_k \parallel^2$
 
That is, each $c^{(i)}$ contains the index of the centroid that has minimal distance to $x^{(i)}$.

By convention, we square the right-hand-side, which makes the function we are trying to minimize more sharply increasing. It is mostly just a convention. But a convention that helps reduce the computation load because the Euclidean distance requires a square root but it is canceled.

Without the square:

$$\parallel x^{(i)} - \mu_k \parallel = \parallel\quad\sqrt{(x_1^i - \mu_{1(k)})^2 + (x_2^i - \mu_{2(k)})^2 + (x_3^i - \mu_{3(k)})^2 + ...}\quad\parallel$$

With the square:

$$\parallel x^{(i)} - \mu_k \parallel^2 = \parallel\quad(x_1^i - \mu_{1(k)})^2 + (x_2^i - \mu_{2(k)})^2 + (x_3^i - \mu_{3(k)})^2 + ...\quad\parallel$$

...so the square convention serves two purposes, minimize more sharply and less computation.

The __second for-loop__ is the 'Move Centroid' step where we move each centroid to the average of its group.

More formally, the equation for this loop is as follows:

$$\mu_k = \1n [x(k1)+x(k2)+⋯+x(kn)] \in \mathbb{R}^n$$

Where each of $x^{(k_1)}, x^{(k_2)}, \dots, x^{(k_n)}$ are the training examples assigned to group $m_{\mu_k}$.

If you have a cluster centroid with 0 points assigned to it, you can randomly re-initialize that centroid to a new point. You can also simply eliminate that cluster group.

After a number of iterations the algorithm will converge, where new iterations do not affect the clusters.

Note on non-separated clusters: some datasets have no real inner separation or natural structure. K-means can still evenly segment your data into $K$ subsets, so can still be useful in this case.


#### Optimization Objective

Recall some of the parameters we used in our algorithm:

+ $c^{(i)}\;$ = index of cluster $(1,2,...,K)$ to which example x(i) is currently assigned
+ $\mu_k\;$ = cluster centroid k ($\mu_k \in \mathbb{R}^n$)
+ $\mu_{c^{(i)}}\;$ = cluster centroid of cluster to which example x(i) has been assigned

Using these variables we can define our __cost function__:

$$J(c^{(i)},\dots,c^{(m)},\mu_1,\dots,\mu_K) = \dfrac{1}{m}\sum_{i=1}^m \parallelx^{(i)} - \mu_{c^{(i)}}\parallel^2$$
 

Our __optimization objective__ is to minimize all our parameters using the above cost function:

$$\min_{c,\mu}\ J(c,\mu)$$

That is, we are finding all the values in sets $c$, representing all our clusters, and $\mu$, representing all our centroids, that will minimize __the average of the distances__ of every training example to its corresponding cluster centroid.

The above cost function is often called the __distortion__ of the training examples.

In the cluster assignment step, our goal is to:

Minimize $J(\ldots)$ with $c^{(1)},\dots,c^{(m)}$ (holding $\mu_1,\dots,\mu_K$ fixed)

In the move centroid step, our goal is to:

Minimize $J(\ldots)$ with $\mu_1,\dots,\mu_K$

With k-means, it is not possible for the cost function to sometimes increase. It should always descend.


#### Random Initialization

There's one particular recommended method for randomly initializing your cluster centroids.

+ Randomly pick $K$ training examples. (Not mentioned in the lecture, but also be sure the selected examples are unique).
+ Have $K < m$. That is, make sure the number of your clusters is less than the number of your training examples.
+ Set $\mu_1,\dots,\mu_K$ equal to these K examples.

K-means __can get stuck in local optima__. To decrease the chance of this happening, you can run the algorithm on many different random initializations. In cases where K<10 it is strongly recommended to run a loop of random initializations.

```matlab
for i = 1 to 100:
   randomly initialize k-means
   run k-means to get 'c' and 'm'
   compute the cost function (distortion) J(c,m)
pick the clustering that gave us the lowest cost
```


#### Choosing the Number of Clusters

Choosing K can be quite arbitrary and ambiguous.

__The elbow method__: plot the cost J and the number of clusters K. The cost function should reduce as we increase the number of clusters, and then flatten out. Choose K at the point where the cost function starts to flatten out.

However, fairly often, the curve is __very gradual__, so there's no clear elbow.

__Note__: J will __always__ decrease as K is increased. The one exception is if k-means gets stuck at a bad local optimum.

Another way to choose K is to observe how well k-means performs on a __downstream purpose__. In other words, you choose K that proves to be most useful for some goal you're trying to achieve from using these clusters.

Bonus: Discussion of the drawbacks of K-Means
This [links](http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means) to a discussion that shows various situations in which K-means gives totally correct but unexpected results


### Errata



### Quiz: Unsupervised Learning



