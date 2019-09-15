# The 9 Deep Learning Papers You Need to Know About (Understanding CNNs Part 3)

Author: Adit Deshpande

[URL](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)


## Introduction

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/Cover3rd.png" style="margin: 0.1em;" alt="Example Convolutional Neural Network" title="Example Convolutional Neural Network" width=650>
  </a>
</div>

+ some of the most important papers that have been published over the last 5 years

## AlexNet (2012)

+ A. Krizhevsky, I. Sutskever, and G. Hinton, [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

+ a CNN was used to achieve a top 5 test error rate of 15.4% (Top 5 error is the rate at which, given an image, the model does not output the correct label with its top 5 predictions)

+ Architecture
  + 5 convolutional layers
  + max-pooling layers
  + dropout layers
  + 3 fully connected layers

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/AlexNet.png" style="margin: 0.1em;" alt="AlexNet architecture" title="AlexNet architecture" width=550>
    </a>
  </div>

+ Main Points
  + training on ImageNet w/ 15 million annotated images from a total of over 22,000 categories
  + used ReLU for nonlinearity functions
  + used data argumentation techniques w/ image translations, horizontal reflections, and patch extractions
  + implemented dropout layers to combat the problem of overfitting to the training data
  + trained the model using batch stochastic gradient descent, w/ specific values for momentum and weight decay
  + trained on 2 GTX 580 GPUs for 5 or 6 days

+ Why it's important
  + the neural network was the coming out party for CNN's in the computer vision community
  + the first time a model performed so well on a historically difficult ImageNet dataset
  + techniques used still using today, including data argumentation and dropout
  + illustrated the benefits of CNNs and backed up with record breaking performance iin the competition


## ZF Net (2013)

+ M. Zeiler and R. Fergus, [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf)

+ a large increase in the number of CNN models

+ model achieved an 11.2% error rate

+ developed some very keys ideas about improving performance

+ a good amount of time explaining a lot of the intuition behind ConvNets and showing how to visualize the filters and weights correctly

+ the renewed interest in CNNs is due to the accessibility of large training sets and increased computational power with the usage of GPUs

+ the limited knowledge that researchers had on inner mechanisms of these models, saying that without this insight, the “development of better models is reduced to trial and error”

+ main contributions: details of a slightly modified AlexNet model and a very interesting way of visualizing feature maps

+ Architecture

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/zfnet.png" style="margin: 0.1em;" alt="ZF Net Architecture" title="ZF Net Architecture" width=750>
    </a>
  </div>

+ Main Points
  + Very similar architecture to AlexNet, except for a few minor modifications.
  + AlexNet trained on 15 million images, while ZF Net trained on only 1.3 million images.
  + using filters of size 7x7 and a decreased stride value in the first layer instead of using 11x11 sized filters
    + a smaller filter size in the first convolutional layer helps retain a lot of original pixel information in the input volume
    + skipping a lot of relevant information
  + a rise in the number of filters used as the network growth
  + using ReLUs for their activation functions, cross-entropy loss for the error function, and trained using batch stochastic gradient descent
  + training on a GTX 580 GPU for twelve days
  + Developed a visualization technique named Deconvolutional Network
    + help to examine different feature activations and their relation to the input space
    + deconvnet: mapping features to pixels (the opposite of what a convolutional layer does)

+ DeConvNet
  + at every layer of the trained CNN, attach a “deconvnet” which has a path back to the image pixels
  + forward pass: an input image is fed into the CNN and activations are computed at each level
  + examine the activations of a certain feature in the 4th convolutional layer
  + store the activations of this one feature map, but set all of the other activations in the layer to 0
  + pass this feature map as the input into the deconvnet
  + this deconvnet has the same filters as the original CNN
  + the input then goes through a series of unpool (reverse maxpooling), rectify, and filter operations for each preceding layer until the input space is reached

+ Visualization of layers

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/deconvnet.png" style="margin: 0.1em;" alt="Visualization of Layer 1 & 2" title="Visualization of Layer 1 & 2" width=500>
      <img src="https://adeshpande3.github.io/assets/deconvnet2.png" style="margin: 0.1em;" alt="Visualization of Layer 3, 4 & 5" title="Visualization of Layer 3, 4 & 5" width=400>
    </a>
  </div>

  + a pooling layer that downsamples the image
  + the 2nd layer has a broader scope of what it can see in the original image
  + [Demo of DeConvNet](https://www.youtube.com/watch?v=ghEmQSxT6tw)

+ Why It’s Important
  + provide great intuition as to the workings on CNNs and illustrated more ways to improve performance
  + visualization approach: not only to explain the inner workings of CNNs, but also provides insight for improvements to network architectures


## VGG Net (2014)

+ K. Simonyan and A. Zisserman, [Very Deep Convolutional Networks for Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf)

+ the winners of ILSVRC 2014, best utilized with its 7.3% error rate

+ Architecture: a 19 layer CNN that strictly used 3x3 filters with stride and pad of 1, along with 2x2 maxpooling layers with stride 2

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/VGGNet.png" style="margin: 0.1em;" alt="The 6 different architectures of VGG Net" title="The 6 different architectures of VGG Net" width=450>
  </a>
</div>

+ Main points
  + use of only 3x3 sized filters
    + the combination of two 3x3 convolutional layers has an effective receptive field of 5x5
    + simulating a larger filter while keeping the benefits of smaller filter sizes
    + benefit: a decrease in the number of parameters
    + with two convolutional layers, able to use two ReLU layers instead of one
  + 3 convolutional layers back to back have an effective receptive field of 7x7
  + As the spatial size of the input volumes at each layer decrease (result of the convolutional and pool layers), the depth of the volumes increase due to the increased number of filters as you go down the network.
  + the number of filters doubles after each maxpool layer, reinforcing the idea of shrinking spatial dimensions, but growing depth
  + worked well on both image classification and localization tasks, used a form of localization as regression
  + built model with the Caffe toolbox
  + used scale jittering as one data augmentation technique during training
  + used ReLU layers after each convolutional layer and trained with batch gradient descent
  + trained on 4 Nvidia Titan Black GPUs for two to three weeks

+ Why It’s Important
  + one of the most influential papers
  + reinforcing the notion that convolutional neural networks have to have a deep network of layers in order for this hierarchical representation of visual data to work



