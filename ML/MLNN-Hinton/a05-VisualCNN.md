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

+ How do we test feature evolution during training?
  + the feature evolution after 1, 2, 5, 10, 20, 39, 40 and 64 (see diagram)
  + notes about the network
    + lower layers converge soon after a few single passes
    + fifth layer not converged until a very large numver of epochs
    + lower layers may change their features correspondence after converging

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/2323/1*exTfLFv6y0LYIseZRUsJ1g.png" style="margin: 0.1em;" alt="Example output of five layers at a specified number of epochs" title="Example output of five layers at a specified number of epochs" width=800>
    </a>
  </div>

+ How do we know this is the best architecture?
  + comparison of two architectures (see diagram)
  + less dead unit on the modified (left) network
  + more defined features on modified network

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1818/1*i0DWeHo8uqAVr_586BIWUw.png" style="margin: 0.1em;" alt="Left picture used filters 7 × 7 instead of 11 × 11, and reduced the stride from 4 to 2" title="Left picture used filters 7 × 7 instead of 11 × 11, and reduced the stride from 4 to 2" width=500>
    </a>
  </div>


## Image reconstruction

+ Deep image representation
  + Aravindh Mahendran and Andrea Vedaldi, “Understanding deep image representations by inverting them,” Nov. 2014.
  + able to reconstruct an image from latent features
  + training network to retain an accuracy photographic representation about the image, retaining geometric and photometric invariance

+ Mathematical representation
  + Assumptions & Notations
    + $a^{[l]}$: the latent representation of layer $l$
  + Optimization problem

    \[\hat{x} = \underset{\mathbf{y}}{\operatorname{arg min}} J_C^{[l]}(\mathbf{x}, \mathbf{y}) + \lambda R(\mathbf{y})\]

    \[J_C^{[l]}(\mathbf{x}, \mathbf{y}) = \left\| a^{[l](G)} - a^{[l](C)} \right\|_{\mathcal{F}}^2\]

  + Regularization w/ $\alpha$-norm regularizer

    \[R_{\alpha} (\mathbf{y}) = \lambda_\alpha \left\| \mathbf{y} \right\|_{\alpha}^{\alpha}\]

  + Regularization w/ total variation regularizer

    \[R_{V_\beta} (\mathbf{y}) = \lambda_{V_\beta} \sum_{i, j, k} \left( \left(y_{i, j+1, k} - y_{i, j, k}\right)^2 + \left(y_{i+1, j, k} - y_{i, j, k}\right)^2 \right)^{\beta/2}\]

+ Procedure for image reconstruction
  1. initialize $\mathbf{y}$ with random noise
  2. feedforward pass the image
  3. computing the loss function
  4. computing the gradients of the cost and backpropagate to input space
  5. updating general image $G$ w/ a gradient step

  + Example

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
        <img src="https://miro.medium.com/max/1520/1*905yAP40kq9lhZ8_9PcDAg.png" style="margin: 0.1em;" alt="Example of image reconstruction" title="Example of image reconstruction" height=250>
      </a>
    </div>
    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
        <img src="https://miro.medium.com/max/2103/1*JYRkcMkObMUi4w2Md4RALw.png" style="margin: 0.1em;" alt="Example of image reconstruction" title="Example of image reconstruction" height=300>
      </a>
    </div>


### Texture Synthesis

+ textture synthesis
  + purpose: to generate high perceptual quality images that imitate a given texture
  + using a trained convolutional neural network for object classification
  + employing correlation of features among layers as a generative process
  + Example

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
        <img src="https://miro.medium.com/max/1613/1*-kBlSklZBtqVGkECajwF3w.png" style="margin: 0.1em;" alt="Example of texture synthesis" title="Example of texture synthesis" width=450>
      </a>
    </div>
  
+ Output of a given layer

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1035/1*PBwlcoL2R7zvQ1SBIOT_zA.png" style="margin: 0.1em;" alt="Output layer of texture synthesis" title="Output layer of texture synthesis" width=300>
    </a>
  </div>

+ Computing the cross-correlation of the feature maps
  + Notations
    + $a_{ijk}^{[l]}$: the output of a given filter $k$ at layer $l$
  + the cross-correlation btw output and a different channel $k$

    \[G_{kk^\prime}^{[l]} = \sum_{i=1}^{n_{H}^{[l]}} \sum_{j=1}^{n_W^{[l]}} a_{ijk}^{[l]} a_{ijk^\prime}^{[l]}\]

  + The Gram matrix: vectorized cross-correlation

    \[G^{[l]} = A^{[l]}(A^{[l]})^T\]

    \[\left(A^{[l]}\right)^T = (a_{::1}^{[l]}, \dots, a_{::n_C^{[l]}}^{[l]})\]


### Generating new textures

+ Notations
  + $G^{[l](S)}$: the Gram matrix of the style image
  + $G^{[l](G)}$: the Gram matrix of the newly generated matrix
  + $\|G\|_\mathcal{F}$: the Frobenius norm

+ creating new texture
  + synthesize an image w/ similar correlation to the one to reproduce
  
  \[J^{[l]}_S (G^{[l]}(S), G^{[l](G)}) = \frac{1}{4\left(n_W^{[l]} n_H^{[l]}\right)^2} \left\| G^{[l](S)} - G^{[l](G)} \right\|_\mathcal{F}^2\]

+ global cost function: with given weights $\lambda_1, \dots, \lambda_L$

  \[J_S(\mathbf{x}, \mathbf{y}) = \sum_{l=0}^L \lambda_l J_S^{[l]} \left(G^{[l](S)}, G^{[l](G)}\right)\]


### Process description





## Neural Style Transfer




### Code Implementation




### DeepDream





### Inceptionism: Going Deeper into Neural Networks




## Final Comments



