# Neural Style Transfer and Visualization of Convolutional Networks

Author: Matthew Stewart

[Original Article](https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b)


## Introduction

+ neural style transfer (NST)
  + Def: Artistic generation of high perceptual quality images that combines the style or texture of some input image, and the elements or content from a different one.
  + producing an image using NST, two images required
    1. the one wishing to transfer the style of
    2. the image to transform using the style of the first image to morph the two images
  
+ Example
  + Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “A neural algorithm of artistic style,” Aug. 2015.
  + Image A: original image of a riverside town
  + Image B: after image translation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1728/1*ATqRZmWHZWcma2lg64aPAA.png" style="margin: 0.1em;" alt="Example of NST w/ a riverside town" title="Example of NST" width=550>
    </a>
  </div>


## Visualizing Convolutional Networks

+ Purpose of visualizing CNN
  + little insight about the learning and internal operation
  + through visualization might be able to
    + observe how input stimuli excite the individual feature maps
    + observe the evolution of features
    + make more substantiated designs

+ Neural network for NST
  + Architecture used
    + Similar to AlexNet: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105.
    + Architecture (see diagram)
      + input: images size 256 x 256 x 3
      + convolutional layers
      + max-pooling layers
      + full-connected layer at the end
      + detail outlined in "Matthew D. Zeiler and Rob Fergus, “Visualizing and understanding convolutional networks” in Computer Vision. 2014, pp. 818–833, Springer"
  + Dataset Imagenet 2012 training database for 1,000 classes

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/2095/1*q6LNA5oP62l6Hr5yR4TnIg.png" style="margin: 0.1em;" alt="The architecture used for NST" title="The architecture used for NST" width=650>
    </a>
  </div>

+ Deconvolution network
  + objective: project hidden feature maps into the original input space
  + benefit: able to visualize the activations of a specific filter
  + Note: NOT performing any deconvolutions
  + Matthew D Zeiler, Graham W Taylor, and Rob Fergus, “Adaptive deconvolutional networks for mid and high-level feature learning,” in IEEE International Conference on Computer Vision (ICCV), 2011, pp. 2018–2025.

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1665/1*47N0hNMbZtbpRXnOzZH4rw.png" style="margin: 0.1em;" alt="Deconvolutional network" title="Deconvolutional network" width=450>
    </a>
  </div>


### Deconvolutional Network Description

+ Aspects of the deconvolution networks
  + unpooling
  + rectification
  + filtering

+ Unpooling layer
  + max-pooling operation non-invertible
  + switch variables - record the locations of maxima
  + placing the reconstructed features into the recorded locations

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1763/1*o2xfitlqjjru1moC-5eBOw.png" style="margin: 0.1em;" alt="Unpooling layer" title="Unpooling layer" width=450>
    </a>
  </div>

+ Rectification layer: signals go through a ReLu operation

+ Filtering layer: us of transposed convolution
  + flipped horizontally and vertically
  + transposed convolution projects feature maps back to input space
  + transposed convolution corresponds to the backpropagation of the gradient

+ How do we perform feature visualization?
  1. evaluate the validation database on the trained network
  2. record the nine highest activation values of each filter's output
  3. project the recorded 9 outputs into input space for every neuron
    + projecting: all other activation units in the given layer set to zero
    + only observing the gradient of a single channel
    + switch variables used in the unpooling layers
  + earlier layers learn more fundamental features such as lines and shapes
  + latter layers learn more complex features

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/2095/1*ANPwlSZ36smDSgiYXAcAbQ.png" style="margin: 0.1em;" alt="AlexNet first layer" title="AlexNet first layer" height=200>
      <img src="https://miro.medium.com/max/2095/1*nrrIv5uriFmn8sRBlLAswQ.png" style="margin: 0.1em;" alt="AlexNet second layer" title="AlexNet second layer" height=200>
    </a>
  </div>
  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/2095/1*DQxut5xSVZ4qrexLbijPZA.png" style="margin: 0.1em;" alt="AlexNet fourth layer" title="AlexNet fourth layer" height=200>
      <img src="https://miro.medium.com/max/2095/1*8eqV8qklluFcD_q1J5uZnQ.png" style="margin: 0.1em;" alt="AlexNet fifth layer" title="AlexNet fifth layer" height=200>
    </a>
  </div>




### How do we test feature evolution during training?




### How do we know this is the best architecture?





## Image reconstruction




### Texture Synthesis





### Generating new textures





### Process description





## Neural Style Transfer




### Code Implementation




### DeepDream





### Inceptionism: Going Deeper into Neural Networks




## Final Comments



