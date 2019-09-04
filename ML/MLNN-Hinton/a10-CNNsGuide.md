# A Beginner's Guide To Understanding Convolutional Neural Networks

Author: Adit Deshpande

[Part 1](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

[Part 2](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

## Introduction

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/Cover.png" style="margin: 0.1em;" alt="text" title="caption" width=550>
  </a>
</div>

+ Convolutional neural networks
  + some of the most influential innovations in the field of computer vision
  + Alex Krizhevsky (2012) used them to win that year’s ImageNet competition
    + classification error: 26% to 15%
  + Company applications
    + Facebook: automatic tagging algorithms
    + Google: photo search
    + Amazon: product recommendations
    + Pinterest: home feed personalization
    + Instagram: search infrastructure


## The problem space

+ Image classification:
  + the task of taking an input image and outputting a class (a cat, dog, etc) or a probability of classes that best describes the image
  + humans: one of the first skills learned from the moment we are born and one that comes naturally and effortlessly as adults
    + able to immediately characterize the scene and give each object a label, all without even consciously noticing
    + able to quickly recognize patterns, generalize from prior knowledge, and adapt to different image environments


### Inputs and Outputs

+ Input
  + Image: an array of pixel values
  + resolution & size: e.g., 32 x 32 x 3 (RGB) array of numbers
  + intensity at a point: values from 0 to 255

+ Output: the probability of the image being a certain class


### What We Want the Computer to Do

+ able to differentiate between all the images it’s given and figure out the unique features that make a dog a dog or that make a cat a ca

+ able perform image classification by looking for low level features such as edges and curves, and then building up to more abstract concepts through a series of convolutional layers


### Biological Connection

+ Visual cortex
  + CNNs do take a biological inspiration from the visual cortex.
  + small regions of cells sensitive to specific regions of the visual field

+ Experiment by Hubel and Wiesel in 1962
  + some individual neuronal cells in the brain responded (or fired) only in the presence of edges of a certain orientation
  + all of these neurons were organized in a columnar architecture and that together, they were able to produce visual perception

+ the basis behind CNNs: specialized components inside of a system having specific tasks (the neuronal cells in the visual cortex looking for specific characteristics) is one that machines use as well



## Structure


### First Layer - Math Part

+ Convolutional Layer
  + always the first layer in CNNs
  + remember what the input to this convolutional layer is
  + Definition
    + filter: the flashlight (or sometimes referred to as a neuron or a kernel)
    + receptive field: the region shining over
    + weights or parameters: numbers containing in the filter
    + convolving: sliding
    + activation map or feature map: all the numbers of multiplications
  + Analogy for convolutional layer
    + a flashlight that is shining over the top left of the image
    + flashlight sliding across all the areas of the input image
    + the depth of this filter has to be the same as the depth of the input
    + multiplying the values in the filter with the original pixel values of the image (aka computing element wise multiplications)
    + summed up all multiplications
  + Example
    + Input: 32 x 32 x 3 array of pixel values
    + the light this flashlight shines covers a 5 x 5 area
    + the dimensions of this filter: 5 x 5 x 3
    + total multiplications for one snapshot: 5 x 5 x 3 = 75
    + activation map: (32 - 5 + 1) x (32 - 5 + 1) x 1 = 784

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/ActivationMap.png" style="margin: 0.1em;" alt="Visualization of 5 x 5 filter convolving around an input and producing an activation map" title="Visualization of 5 x 5 filter convolving around an input and producing an activation map" width=450>
    </a>
  </div>

+ Spatial dimension concern
  + using more filters to preserve the spatial dimensions better
  + The more filters, the greater the depth of the activation map, and the more information about the input volume.
  + eg. total output volume: 28 x 28 x 2 with 2 filters


### First Layer - High Level Perspective

+ Interpretation fo Convolutional Layer
  + feature identifier = filter
  + features: straight edges, simple colors, and curves
  + curve detector: a pixel structure w/ higher numerical values along the area that is a shape of a curve
  + Example
    + first filter (curve detector): 7 x 7 x 3
    + only considering the top depth slice of the filter

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/Filter.png" style="margin: 0.1em;" alt="Visualization of a curve detector filter and its pixel representation" title="Visualization of a curve detector filter and its pixel representation" width=350>
    </a>
  </div>

+ Example for first layer
  + feature identifier: 7 x 7
  + put filter at the top left corner
  + multiply the values in the filter with the original pixel values of the image
  + remove receptive field with the filter (summed multiplication = 0)
  + activation map: the output of this convolutional layer (26 x 26 x 1)
  + The filters on the first layer convolve around the input image and “activate” (or compute high values) when the specific feature it is looking for is in the input volume.

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/OriginalAndFilter.png" style="margin: 0.1em;" alt="Original image and visualization of the filter" title="Original image and visualization of the filter" height=150>
      <img src="https://adeshpande3.github.io/assets/FirstPixelMulitiplication.png" style="margin: 0.1em;" alt="Pixel representation of the receptive field and filter" title="Pixel representation of the receptive field and filter" height=150>
      <img src="https://adeshpande3.github.io/assets/SecondMultiplication.png" style="margin: 0.1em;" alt="Another pixel representation of the receptive field and filter" title="Another pixel representation of the receptive field and filter" height=150>
      <img src="https://adeshpande3.github.io/assets/FirstLayers.png" style="margin: 0.1em;" alt="Visualization of filters" title="Visualization of filters" height=150>
    </a>
  </div>



### Going Deeper Through the Network

+ Traditional convolutional neural network architecture
  + A classic CNN architecture

    input $\rightarrow$ Conv $\rightarrow$ ReLU $\rightarrow$ Conv $\rightarrow$ ReLU $\rightarrow$ Pool $\rightarrow$ ReLU $\rightarrow$ Conv $\rightarrow$ ReLU $\rightarrow$ Fully Connected

  + The first convolutional layer
    + filters in this layer designed to detect
    + detect low level features such as edges and curves
  + network needs to recognize higher level features such as hands or paws or ears
  + 2nd convolutional layer
    + the output of the first convolutional layer as the input
    + input of 1st convolutional layer: original image
    + input of 2nd convolutional layer: activation map of 1st convolutional layer
  + each layer of the input is basically describing the locations in the original image for where certain low level features appear.
  + output of higher level features: semicircles (combination of a curve and straight edge) or squares (combination of several straight edges)
  + activation maps that represent more and more complex features as more convolutional layer went through
  + as deeper into the network, the filters w/ a larger and larger receptive
  + able to consider information from a larger area of the original input volume (more responsive to a larger region of pixel space)

+ References
  + [Matt Zeiler and Rob Fergus](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
  + [Jason Yosinski](https://www.youtube.com/watch?v=AgkfIQ4IGaM)



### Fully Connected Layer




### ReLU (Rectified Linear Units) Layers





### Pooling Layers




### Dropout Layers




### Network in Network layers






## Training (aka: What Makes this Stuff Work)




## Testing





## How Companies Use CNNs




## Stride and Padding





## Choosing Hyperparameters






## Classification, Localization, Detection, Segmentation




## Transfer Learning




## Data Argumentation Techniques











