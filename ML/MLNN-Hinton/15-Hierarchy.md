# 15. Hierarchical Structure with Neural Networks

## 15.1 From principal components analysis to autoencoders

### Lecture Notes

+ Principal Components Analysis
	+ finding the $M$ orthogonal directions
		+ taking $N$-dimensional data
		+ the direction where data w/ the most variance
		+ $M$ principle directions forming a lower-dimensional subspace
		+ able to represent an $N$-dimensional datapoint by its projections onto the $M$ principle directions
		+ lossing all information about where the datapoint located in the remaining orthogonal directions
	+ reconstrcting by using the mean value (over all the data)
		+ the mean value w/ $N-M$ directions not represented w/ $M$ orthogonal directions
		+ reconstruction error = sum over all these unrepresented directions of the squared differences of the datapoint from the mean
	+ example: PCA w/ $N=2$ and $M=1$ (see diagram)
		+ green point on PCA directions representing red point
    + reconstruction of the red point: an error equal to the squared btw red and green points

	<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
		<a href="https://bit.ly/39K9qaJ" ismap target="_blank">
			<img src="img/m15-01.png" style="margin: 0.1em;" alt="PCA example w/ N=2 and M=1" title="PCA example w/ N=2 and M=1" width=350>
		</a>
	</div>

+ Implementing PCA w/ backpropagation
  + inefficient implementation
  + task: making output = the input in a network w/ a central bottleneck

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://bit.ly/39K9qaJ" ismap target="_blank">
        <img src="img/m15-02.png" style="margin: 0.1em;" alt="PCA example w/ N=2 and M=1" title="PCA example w/ N=2 and M=1" width=200>
      </a>
    </div>

  + efficient code: the activities of the hidden units in the bottleneck
  + linear hidden and output layers $\implies$ learning hidden units w/ a linear function of the data
    + minimizing the squared reconstruction error
    + exactly what PCA does
  + $M$ hidden units
    + spanning the same space at the first $M$ components found by PCA
    + weight vectors probably not orthogonal
    + tending to have equal variances

+ Generalizing PCA w/ backpropagation
  + adding non-linear layers before and after the code: encoding and decoding weights
  + non-linear layers: possibly efficiently representing data that lies on or near a non-linear manifold
  + encoder: converting coordinates in the input space to coordinates on the manifold
  + decoder: inverting the mapping of encoder

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





### Lecture Video

<a href="url" target="_BLANK">
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

