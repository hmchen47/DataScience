# Neural Style Transfer and Visualization of Convolutional Networks

Author: Matthew Stewart

[Original Article](https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b)


## Introduction

+ Neural Style Transfer (NST)
  + Def: Artistic generation of high perceptual quality images that combines the style or texture of some input image, and the elements or content from a different one.
  + producing an image using NST, two images required
    1. the one wishing to transfer the style of
    2. the image to transform using the style of the first image to morph the two images
  
+ Example
  + Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “A neural algorithm of artistic style,” Aug. 2015. ([Paper](https://arxiv.org/abs/1508.06576))
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
    + Similar to AlexNet: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105.([Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ))
    + Architecture (see diagram)
      + input: images size 256 x 256 x 3
      + convolutional layers
      + max-pooling layers
      + full-connected layer at the end
      + detail outlined in "Matthew D. Zeiler and Rob Fergus, “Visualizing and understanding convolutional networks” in Computer Vision. 2014, pp. 818–833, Springer. ([Paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf))
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
  + Matthew D Zeiler, Graham W Taylor, and Rob Fergus, “Adaptive deconvolutional networks for mid and high-level feature learning,” in IEEE International Conference on Computer Vision (ICCV), 2011, pp. 2018–2025. ([Paper](https://www.matthewzeiler.com/mattzeiler/adaptivedeconvolutional.pdf))

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

+ Filtering layer: use of transposed convolution
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
      <img src="https://miro.medium.com/max/2095/1*ANPwlSZ36smDSgiYXAcAbQ.png" style="margin: 0.1em;" alt="AlexNet first layer" title="AlexNet first layer" height=180>
      <img src="https://miro.medium.com/max/2095/1*nrrIv5uriFmn8sRBlLAswQ.png" style="margin: 0.1em;" alt="AlexNet second layer" title="AlexNet second layer" height=180>
    </a>
  </div>
  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/2095/1*DQxut5xSVZ4qrexLbijPZA.png" style="margin: 0.1em;" alt="AlexNet fourth layer" title="AlexNet fourth layer" height=180>
      <img src="https://miro.medium.com/max/2095/1*8eqV8qklluFcD_q1J5uZnQ.png" style="margin: 0.1em;" alt="AlexNet fifth layer" title="AlexNet fifth layer" height=180>
    </a>
  </div>

+ How do we test feature evolution during training?
  + the feature evolution after 1, 2, 5, 10, 20, 39, 40 and 64 (see diagram)
  + notes about the network
    + lower layers converge soon after a few single passes
    + fifth layer not converged until a very large number of epochs
    + lower layers may change their features correspondence after converging

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/2323/1*exTfLFv6y0LYIseZRUsJ1g.png" style="margin: 0.1em;" alt="Example output of five layers at a specified number of epochs" title="Example output of five layers at a specified number of epochs" width=600>
    </a>
  </div>

+ How do we know this is the best architecture?
  + comparison of two architectures (see diagram)
  + less dead unit on the modified (left) network
  + more defined features on modified network

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1818/1*i0DWeHo8uqAVr_586BIWUw.png" style="margin: 0.1em;" alt="Left picture used filters 7 × 7 instead of 11 × 11, and reduced the stride from 4 to 2" title="Left picture used filters 7 × 7 instead of 11 × 11, and reduced the stride from 4 to 2" width=400>
    </a>
  </div>


## Image reconstruction

+ Deep image representation
  + Aravindh Mahendran and Andrea Vedaldi, “Understanding deep image representations by inverting them,” Nov. 2014. ([Paper](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/mahendran15understanding.pdf))
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
        <img src="https://miro.medium.com/max/1520/1*905yAP40kq9lhZ8_9PcDAg.png" style="margin: 0.1em;" alt="Example of image reconstruction" title="Example of image reconstruction" height=200>
      </a>
    </div>
    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
        <img src="https://miro.medium.com/max/2103/1*JYRkcMkObMUi4w2Md4RALw.png" style="margin: 0.1em;" alt="Example of image reconstruction" title="Example of image reconstruction" height=200>
      </a>
    </div>


### Texture Synthesis

+ Texture synthesis
  + purpose: to generate high perceptual quality images that imitate a given texture
  + using a trained convolutional neural network for object classification
  + employing correlation of features among layers as a generative process
  + Example

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
        <img src="https://miro.medium.com/max/1613/1*-kBlSklZBtqVGkECajwF3w.png" style="margin: 0.1em;" alt="Example of texture synthesis" title="Example of texture synthesis" width=350>
      </a>
    </div>
  
+ Output of a given layer

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1035/1*PBwlcoL2R7zvQ1SBIOT_zA.png" style="margin: 0.1em;" alt="Output layer of texture synthesis" title="Output layer of texture synthesis" width=250>
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

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
    <img src="https://miro.medium.com/max/1485/1*cXUh1aw7Q3i5ZRq1CIlyng.png" style="margin: 0.1em;" alt="Full texture synthesis process" title="Full texture synthesis process" width=600>
  </a>
</div>

+ Ref: Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “Texture synthesis using convolutional neural networks” ([Paper](http://papers.nips.cc/paper/5633-texture-synthesis-using-convolutional-neural-networks))


## Neural Style Transfer

+ Neural Style Transfer
  + first publication: Gatys, Leon A.; Ecker, Alexander S.; Bethge, Matthias (26 August 2015). “A Neural Algorithm of Artistic Style”
  + combining content and style reconstruction
  + procedure of NST
    + choose a layer (or set of layers) to represent content - the middle layers recommended (not too shall, not too deep) for best results
    + minimizing the total cost by using backpropagation

      \[J_{total}(\mathbf{x}, \mathbf{y}) = \alpha J_C^{[l]}(\mathbf{x}, \mathbf{y}) + \beta J_S(\mathbf{x}, \mathbf{y})\]

    + initializing the input with random noise (necessary of generating gradients)
    + replacing max-pool layers with average pooling to improve the gradient flow and to produce more appealing pictures
  + Example

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1343/1*CAMHku2Bars3sxUUktH9aA.png" style="margin: 0.1em;" alt="Examples of neural style transfer" title="Examples of neural style transfer" height=240> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1883/1*90t9BwVJ--zzGzJnwXJkgQ.png" style="margin: 0.1em;" alt="Procedure of neural style transfer" title="procedure of neural style transfer" height=240>
    </a>
  </div>


### Code Implementation

+ the [Jupyter notebook](https://github.com/mrdragonbear/Neural-Networks/blob/master/Neural-Style-Transfer/Neural-Style-Transfer.ipynb) located in the GitHub repository

+ [local Neural Style Transfer Jupyter notebook](src\Neural-Style-Transfer/Neural-Style-Transfer.ipynb)

+ Code Implementation
  + Part 1: import necessary functions
    ```python
    import time
    import numpy as np

    from keras import backend as K
    from keras.applications import vgg16, vgg19
    from keras.preprocessing.image import load_img

    from scipy.misc import imsave
    from scipy.optimize import fmin_l_bfgs_b

    # preprocessing
    from utils import preprocess_image, deprocess_image
    ```

  + Part 2: content loss
    + combining the content and style of a pair w/ a loss function that incorporates this information
      + mimic the specific activations of a certain layer for the content image
      + mimic the style
    + the variable to optimize on the loss function will be generated image that aims to minimize the proposed cost
    + performing gradient descent on the pixel values, rather than on the neural network weights
    + trained neural network: VGG-16
    + using the activation values obtained for an image of interest to represent the content and style
    + measuring how much the feature map pf the generated image differs from the feature map of the source image

  + Part 3: style loss
    + the style measures the similarity among filters in a set of layers
    + similarity: compute the Gram matrix of the activation values for the style layers
    + Gram matrix: related to the empirical covariance matrix and reflecting the statistics of the activation values

  + Part 4: style loss - layer's loss
    + compute the style loss at a set of layers rather than just a single layer
    + total style loss: the sum of style losses at each layer

  + Part 5: total-variation regularizer
    + encouraging smoothness in the image using a total-variation regularizer
    + penalty: reducing variation among the neighboring pixel values

  + Part 6: style transfer
    + put all together to generate some images
    + style_transfer function: combining the losses and optimizing for the image to minimize the total loss

  + Part 6: generate picture
    + run you own compositions and test out variations of hyper-parameters
    + list of hyper-parameters to vary
      + base_img_path: filename of content image
      + style_img_path: filename of style image
      + compute_img_path: filename of the generated image
      + convnet: specifying the neural network weights, VGG-16 or VGG-19
      + content_layer: specifying layer to use for content loss
      + content_weight:
        + weighting the content loss in the overall composite loss function
        + increasing the value of this parameter will make the final image look more realistic (closer to the original content)
      + style_layers: specifying a list of which layers to use for the style loss
      + style_weight
        + specifying a list of weights to use for each layer in style_layers (each of which will contribute a term to the overall style loss)
        + using higher weights for the earlier style layers
          + describing more local/smaller-scale features
          + local features: more important to texture than features over larger receptive fields
        + increasing these weights to make the resulting image look less like the original content and more distorted towards the appearance of the style image
      + tv_weight:
        + specifying the weighting of total variation regularization in the overall loss function
        + increasing the value makes the resulting image look smoother and less jagged, at the cost of lower fidelity to style and content

+ Examples

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1224/1*f3gT72gXAbOgtQ9a2sXDGA.png" style="margin: 0.1em;" alt="Style of ‘Escher Sphere’ used to transform an image of the Goldengate Bridge" title="Style of ‘Escher Sphere’ used to transform an image of the Goldengate Bridge" height=200>
      <img src="https://miro.medium.com/max/1280/1*8s1L0x7HOS6saNomlSXagA.png" style="margin: 0.1em;" alt="Style of ‘Seated Nude’ used to transform an image of the riverbank town image" title="Style of ‘Seated Nude’ used to transform an image of the riverbank town image" height=200>
    </a>
  </div>


### DeepDream

+ DeepDreamer
  + a computer vision program created by Alexander Mordvintsev at Google
  + using a convolutional neural network to find and enhance patterns in images via algorithm pareidolia
  + creating a dream-like hallucinogenic appearance in the deliberately over-processed images
  + dreaming:
    + the generation of images that produce desired activation in a trained deep network
    + a collection of related approaches

+ Example

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1045/1*iADSYV8WCtqGBhkqNqNlnQ.png" style="margin: 0.1em;" alt="Example of DeeDream" title="Example of DeepDream" height=300>
    </a>
  </div>


### Inceptionism: Going Deeper into Neural Networks

+ Inceptionism
  + having a reasonable intuition about what types of features encapsulated by each of the layers in a neural network
  + network
    + first layer: edges or corners
    + intermediate layers: interpreting the basic features to look for overall shapes or components, like a door or a leaf
    + final layer: assembling shapes or components into complete interpretations, like trees, building, etc.

+ Example
  + what kind of image would result in a banana
  + one way: turning the neural network upside down, starting with an image full of random noise, and then gradually tweak the image toward what the neural neural network considers a banana
  + if imposing a prior constraint, the image should have similar characteristics to natural images
  + a correlation between neighboring pixels (see diagram)
  + neural networks trained to discriminate between different image classes
  + having a substantial amount if information needed to generate image too

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1238/1*SilWmiEZXN6CnQmrasARyg.png" style="margin: 0.1em;" alt="Example of Inceptionism" title="Example of Inceptionism" width=400>
    </a>
  </div>

+ class generation
  + purpose: flipping the discriminative model into a generative model
  + useful to ensure that the network is learning the right features and not cheating

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1668/1*StKK6yiXhSrlSjGwSGgxQg.png" style="margin: 0.1em;" alt="Example of class generation" title="Example of class generation" width=300>
    </a>
  </div>

+ visualizing mistakes
  + cheating with dumbbells
  + training a network w/ a set of pictures of dumbbells
  + using random noise w/ prior constraints to imagine some dumbbells (see diagram)
  + failed tto completely distill the essence of a dumbbell
  + none of the training pictures having any weightlifters
  + visualization help to correct these kinds of training mishaps

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1280/1*A6jsI8cxorXp-7fhBo6WaA.png" style="margin: 0.1em;" alt="Example of visualizing mistakes" title="Example of visualizing mistakes" width=300>
    </a>
  </div>

+ Enhancing feature maps
  + feeding an image and then picking a layer and asking the network to enhance whatever it detect
  + lower layer: producing strokes and simple ornament-like patters (top left diagram)
  + higher layer: emerging into complex features or even whole objects (top right diagram)
  + training w/ pictures of animals, look more like an animal (middle diagram)
  + features entered bias as the network toward certain interpretations (bottom left diagram)
  + applying the algorithm iteratively on its own outputs and zooming after each iteration (bottom right diagram)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1498/1*b9hjaHnItz168bZ3YE-8XQ.png" style="margin: 0.1em;" alt="Enhancing feature map: lower layer" title="Enhancing feature map: lower layer" height=130>
      <img src="https://miro.medium.com/max/1193/1*feBHqMc81crBnZFhSKo1Ag.png" style="margin: 0.1em;" alt="Enhancing feature map: higher layer" title="Enhancing feature map: higher layer" height=130>
    </a>
  </div>
  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1390/1*jqvzS7yIE1NUbRyIENyLlw.png" style="margin: 0.1em;" alt="Examples of clouds w/ animals" title="Examples of clouds w/ animals" height=150>
    </a>
  </div>
  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b" ismap target="_blank">
      <img src="https://miro.medium.com/max/1173/1*KQ3uF6qStNlQ421qj5f9Fw.png" style="margin: 0.1em;" alt="Horizon lines tend to get filled with towers and pagodas" title="Horizon lines tend to get filled with towers and pagodas" height=200>
      <img src="https://miro.medium.com/max/1490/1*e8w7E1mAMsJhL9C2NBOJqw.png" style="margin: 0.1em;" alt="Examples of applied output and some zooming after each iteration" title="Examples of applied output and some zooming after each iteration" height=200>
    </a>
  </div>


## References

1. Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “A neural algorithm of artistic style,” Aug. 2015. ([Paper](https://arxiv.org/abs/1508.06576))
2. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105. ([Paper])(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ)
3. Matthew D. Zeiler and Rob Fergus, “Visualizing and understanding convolutional networks” in Computer Vision. 2014, pp. 818–833, Springer. ([Paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf))
4. Matthew D Zeiler, Graham W Taylor, and Rob Fergus, “Adaptive deconvolutional networks for mid and high-level feature learning,” in IEEE International Conference on Computer Vision (ICCV), 2011, pp. 2018–2025. ([Paper](https://www.matthewzeiler.com/mattzeiler/adaptivedeconvolutional.pdf))
5. Aravindh Mahendran and Andrea Vedaldi, “Understanding deep image representations by inverting them,” Nov. 2014. ([Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.html))
6. Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “Texture synthesis using convolutional neural networks”. ([Paper](http://papers.nips.cc/paper/5633-texture-synthesis-using-convolutional-neural-networks))
7. Gatys, Leon A.; Ecker, Alexander S.; Bethge, Matthias (26 August 2015). “A Neural Algorithm of Artistic Style”. ([Paper](https://arxiv.org/abs/1508.06576))



