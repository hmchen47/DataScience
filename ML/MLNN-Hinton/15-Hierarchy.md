# 15. Hierarchical Structure with Neural Networks

## 15.1 From principal components analysis to autoencoders

### Lecture Notes

+ Principal Components Analysis (PCA) -Intro
  + widely used technique in signal processing
  + higher dimensional data represented by a much lower dimensional code
  + situation: a data lying a linear manifold in the high dimensional space
  + task: finding a data manifold and projecting the data onto the manifold = representation on the manifold $\ni$ orthogonal directions not variation much in the data $\implies$ not losing much information
  + operation:
    + standard principal components methods: efficient
    + neural network w/ one linear hidden layer and linear output layer: inefficient
  + advantage of using neural networks:
    + generalizing the technique by using deep neural networks where the code is a nonlinear function of the input
    + reconstructing the data from the code as a nonlinear function of the input vector
    + able to deal w/ curved manifold in the input space

+ Principal Components Analysis
  + finding the $M$ orthogonal directions
    + $\exists\; N$-dimensional data, representing the data w/ less than $N$ numbers, said $M$
    + the direction where data w/ the most variance
    + ignoring the directions where the data not varying much
    + $M$ principal directions forming a lower-dimensional subspace
    + representing an $N$-dimensional datapoint by its projections onto the $M$ principal directions
    + losing all information about where the datapoint located in the remaining orthogonal directions but not much
  + reconstructing by using the mean value (over all the data)
    + the mean value w/ $N-M$ directions not represented w/ $M$ orthogonal directions
    + reconstruction error = sum over all these unrepresented directions of the squared differences of the datapoint from the mean
  + example: PCA w/ $N=2$ and $M=1$ (see diagram)
    + 2-dimensional data distributed according to an elongated Gaussian
    + the ellipse: a kind of one standard deviation contour of the Gaussian
    + the green point on PC directions representing the data on the red point
      + using PCA w/ a single component
      + the component as the direction in the data w/ greatest variance
      + representing the red point = representing how far along that direction
      + $\therefore$ representing the projection of the red point onto that line; i.e., the green point
    + reconstruction of the red point: an error equal to the squared difference btw red and green points
      + using all the mean values of the data points ignored
      + representing a point on the black line
      + the loss on the construction = the squared difference btw the red point and the green point
      + the loss = the difference btw the data point and the mean values of all the data in the direction ignored
      + minimizing the loss = choosing to ignore the directions w/ less variance

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/39K9qaJ" ismap target="_blank">
      <img src="img/m15-01.png" style="margin: 0.1em;" alt="PCA example w/ N=2 and M=1" title="PCA example w/ N=2 and M=1" width=350>
    </a>
  </div>

+ Implementing PCA w/ backpropagation
  + inefficient implementation
  + task: making output = the input in a network w/ a central bottleneck
    + making a network in which the output of the network as the reconstruction of input
    + trying to minimize the squared error in the reconstruction
    + the network w/ a central bottleneck
    + $\exists\; M$ hidden units corresponding to the principal components
    + input vector projected to the code vector
    + from code vector to construct the output vector
    + goal: making the output vector as similar as the input vector

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://bit.ly/39K9qaJ" ismap target="_blank">
        <img src="img/m15-02.png" style="margin: 0.1em;" alt="PCA example w/ N=2 and M=1" title="PCA example w/ N=2 and M=1" width=200>
      </a>
    </div>

  + efficient code = the activities of the hidden units $\ni$ the bottleneck
    + the activities of the hidden unit forming a bottleneck
    + the code vector = a compressed representation of the input vector
  + linear hidden and output layers $\implies$ autoencoder
    + autoencoder
      + learning hidden units w/ a linear function of the data
      + minimizing the squared reconstruction error
    + exactly what PCA does
      + exact the same reconstruction error as PCA does
      + not necessary w/ hiddent units corresponding exactly to the principal components
  + $M$ hidden units
    + spanning the same space at the first $M$ components found by PCA
    + weight vectors probably not orthogonal
    + tending to have equal variances
    + probably rotating and skewing of those axes
    + the incoming vectors of code units $\neq$ the directions of the components $\implies$ orthogonal
    + the space spanned by the incoming weight vectors of those code units = the space spanned by the $M$ principal components
    + $\therefore\;$ the networks $\equiv$ principal components
    + performance: the stochastic gradient descent learning for the network < PCA algorithm

+ Generalizing PCA w/ backpropagation
  + purpose: generalizing PCA
    + able to represent data w/ a curved manifold rather than a linear manifold in a high dimensional space
  + adding nonlinear layers before and after the code: encoding and decoding weights
    + encoder: converting coordinates in the input space to coordinates on the manifold
    + decoder: inverting the mapping of encoder
    + nonlinear layers: possibly efficiently representing data that lies on or near a nonlinear manifold
  + learned $\ni$ mapping on both directions  
  + network architecture (see diagram)
    + adding one ore more layers of nonlinear hidden units, typically using logistic units
    + the code layer: linear units
    + following a one or more layers of nonlinear units
    + output layer trained as similar as possible to the input vector
    + using supervisor learning algorithm to do unsupervised learning

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/39K9qaJ" ismap target="_blank">
      <img src="img/m15-03.png" style="margin: 0.1em;" alt="PCA example w/ N=2 and M=1" title="PCA example w/ N=2 and M=1" width=200>
    </a>
  </div>


### Lecture Video

<a href="https://bit.ly/2UOSCv3" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 15.2 Deep Autoencoders

### Lecture Notes

+ Deep autoencoders
  + always looking as a nice way to do nonlinear dimensional reduction
    + providing flexible mapping both ways
    + linear (or better) learning time in the number of training cases
    + final encoding model: fairly compact and fast
  + difficulties
    + very difficult to optimize deep autoencoders using backpropagation
    + small initial weights $\ni$ backpropagation gradient vanished
  + Solutions
    + unspervised layer-by-layer pre-training
    + initializing the weights carefully as in Echo-state nets

+ Encoder network
  + G. E. Hinton*, R. R. Salakhutdinov, [Reducing the Dimensionality of Data with Neural Networks](https://bit.ly/2xbMHXZ), Science, 28 Jul 2006
  + network architecture:
    + training a stack of 4 RBM's and then unrolling them
    + fine-tuning w/ gen gentle backpropagation
  + comparisons of methods for compressing digit images to 30 real numbers
    + real data
    + 30-D deep autoencoder
    + 30D PCA

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/39K9qaJ" ismap target="_blank">
      <img src="img/m15-04.png" style="margin: 0.1em;" alt="Autoencoder example w/ 4 RBMs" title="Autoencoder example w/ 4 RBMs" width=350>
      <img src="img/m15-05.png" style="margin: 0.1em;" alt="Comparison of autoencoder results" title="Comparison of autoencoder results" width=350>
    </a>
  </div>



### Lecture Video

<a href="https://bit.ly/3b04kbG" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 15.3 Deep autoencoders for document retrieval and visualization

### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 15.4 Semantic hashing

### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 15.5 Learning binary codes for image retrieval

### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 15.6 Shallow autoencoders for pre-training

### Lecture Notes





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>

