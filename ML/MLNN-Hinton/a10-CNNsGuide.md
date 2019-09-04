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
  + using more filters to preserrve the spatial dimensions better
  + eg. total output volume: 28 x 28 x 2 with 2 filters


### First Layer - High Level Perspective





### Going Deeper Through the Network





### Fully Connected Layer





## Training (aka: What Makes this Stuff Work)




## Testing





## How Companies Use CNNs




## Stride and Padding





## Choosing Hyperparameters





## ReLU (Rectified Linear Units) Layers





## Pooling Layers




## Dropout Layers




## Network in Network layers






## Classification, Localization, Detection, Segmentation




## Transfer Learning




## Data Argumentation Techniques











