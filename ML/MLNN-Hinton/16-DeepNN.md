# 16. Deep Neural Networks
  
## 16.1 Learning a joint model of images and captions

### Lecture Notes

+ Modeling the joint density of images and captions
  + N. Srivastava and R. Salakhutdinov, [Multimodal Learning with Deep Boltzmann Machines](https://tinyurl.com/wzsknt8), NIPS 2012
  + goal: to build a joint density model of captions and standard computer vision feature vectors extracted from real photographs
  + issue: requiring more computation than building a joint density model of labels and digit images
  + procedure:
    + training a multilayer model of images
    + training a separated multilayer model of word-count vectors
    + adding a new top layer connected to the top layers of both individual models
      + using further joint training of the whole system to allow each modality to improve the earlier layers of other modality
  + using a deep Boltzmann machine 
    + instead of using a deep belief net
    + symmetric connection btw all pairs of layers
    + further joint training of the whole DBM allows each modality to improve the earily layers of the otehr modality $\to$ used a DBM
    + probably used a DBM and done generative fine-tuning w/ contrastive wake-sleep
  + mechanism of pre-training on the hidden layers of the DBM
    + standard pre-training $\implies$ a composite model w/ DBM $\to$ not DBM

+ Combining 3 RBMs to make a DBM
  + network architecture (see diagram)
  + the top and bottom RBMs pre-trained w/ the weights in one direction twice as big as in the other direction
  + the middle layers: geometric model averaging

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/u3whuvf" ismap target="_blank">
      <img src="img/m16-01.png" style="margin: 0.1em;" alt="Combining RBMs to a DBM" title="Combining RBMs to a DBM" width=300>
    </a>
  </div>


### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 16.2 Hierarchical coordinate frames

### Lecture Notes






### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 16.3 Bayesian optimization of neural network hyperparameters

### Lecture Notes






### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 16.4 The fog of progress

### Lecture Notes






### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>

