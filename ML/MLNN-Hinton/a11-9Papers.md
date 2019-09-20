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


## GoogLeNet (2015)

+ C. Szegedy, et. al., [Going Deeper with Convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)

+ GoogLeNet: a 22 layer CNN

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/GoogleNet.gif" style="margin: 0.1em;" alt="GoogLeNet architecture" title="GoogLeNet architecture" height=250>
      <img src="https://adeshpande3.github.io/assets/GoogLeNet.png" style="margin: 0.1em;" alt="Another view of GoogLeNet architecture" title="Another view of GoogLeNet architecture" height=200>
    </a>
  </div>

+ a top 5 error rate of 6.7%

+ one of the first CNN architectures that really strayed from the general approach of simply stacking conv and pooling layers on top of each other in a sequential structure

+ this new model places notable consideration on memory and power usage

### Inception Module

+ Inception module

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/GoogLeNet2.png" style="margin: 0.1em;" alt="Parallel region of GoogLeNet as inception module" title="Parallel region of GoogLeNet as inception module" height=250>
      <img src="https://adeshpande3.github.io/assets/GoogLeNet3.png" style="margin: 0.1em;" alt="Full inception module" title="Full inception module" height=250>
      <img src="https://adeshpande3.github.io/assets/GoogLeNet4.png" style="margin: 0.1em;" alt="Naive idea of an Inception module" title="Naive idea of an Inception module" height=250>
    </a>
  </div>

  + Previous layer (bottom green box): input
  + Filter concatenation (top green box): output
  + traditional ConvNet:make a choice of whether to have a pooling operation or a convolutional operation (the choice of filter size)
  + Inception module: perform all of these operations in parallel

+ Naive inception module vs. Inception Module
  + Naive inception module: too many output end up with extremely large depth channel for the output volume
  + 1x1 convolutions: providing a method of dimensionality reduction
  + eg., an input volume of 100x100x60 and applying 20 filters of 1x1 convolution would reduce the volume to 100x100x20
  + Adding 1x1 convolutional operations before the 3x3 adn 5x5 layers $\longrightarrow$ the 3x3 and 5x5 convolutions won't have as large of a volume to deal with.
    + Analogy: a "pooling of features" - reducing the depth of the volume
    + Similar to how er reduce the dimensions of height and width with normal maxpooling layers.
  + 1x1 convolutional layers followed by ReLU units
    + [the effectiveness of 1x1 convolutions](http://iamaaditya.github.io/2016/03/one-by-one-convolution/)
    + [visualization of the filter concatenation](https://www.youtube.com/watch?v=VxhSouuSZDY)

+ How dows this architecture help?
  + a module consisting of a network in network layer, a medium sized filter convolution and a pool operation
  + network in network convolution: able to extract information about the very find grain details in the volume
  + the 5x5 filter: able to cover a large receptive field of the input and thus able to extract its information as well
  + pooling operation: reducing spatial sizes and combating overfitting
  + ReLU: improving the nonlinearity of the network
  + able to perform the functions of these different operations and still remain computationally considerate

### Main Points

+ used 9 inception modules in the whole architectures, with over 100 layers in total.
+ no use of fully connected layers
+ using an average pool instead, to go from a 7x7x1024 volume to a 1x1x1024 volume
+ using 12x fewer parameters than AlexNet
+ during testing, multiple corps of the same image were created, fed into the network and the maxsoft probabilities were averages to give us the final solution
+ utilized concepts from R-CNN for their detection module
+ there are updated versions to the Inception module 9version 6 & 7)
+ trained on a few high-end GPUs within a week


### Why it's important

+ one of the first models that introduced the idea that CNN layers didn't always have to stacked up sequentially

+ the Inception module: a creative structuring of layers can lead to improved performance and computationally efficiency


## Microsoft ResNet (2015)

+ K. He, X. Zhang, S. Ren, and J. Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)

+ ResNet: a new 152 layer network architecture that set new records in classification, detection, and localization through one architecture

+ ILSVRC 2015 with an incredible error rate of 3.6%

+ humans generally hover around a 5-10% error rate

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/ResNet.gif" style="margin: 0.1em;" alt="ResNet" title="ResNet" width=250>
  </a>
</div>

### Residual Block

+ idea behind a residual block: input x goes through conv-relu-conv series

+ given $F(x)$ then adding to the original input $x$. the hypothesis function is $H(x) = F(x) + x$

+ traditional CNNs: $H(x) = F(x)$, computing $F(x)$
  + complete new representation w.r.t. input $x$

+ Residual module: computing $F(x) + x$
  + computing a "delta" or a slight change to the original input to get a slightly altered representation
  + easier to optimize the residual mapping than to optimize the original

+ might be effective during the backward pass of backpropagation

+ the gradient will flow easily through the graph because addition operations distribute the gradient

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/ResNet.png" style="margin: 0.1em;" alt="A residual block" title="A residual block" width=350>
  </a>
</div>


### Main Points

+ "Ultra-deep" - Yann LeCun
+ 152 layers
+ after fist 2 layers, the spatial size gets compressed from an input volume of 224x224 to a 56x56 volume
+ a naive increase of layers in plain nets result in higher training and test error
+ tried 1024-layer network but a lower test accuracy, presumably due to overfitting
+ trained on an 8 CPU machine for 2~3 weeks


### Why it's important

+ 3.6% error rate
+ the best CNN architecture so far
+ stacking more layers on top of each other isn't going to result in a substantial performance boost

+ Ref: K. Zhang, et. al., [Residual Networks of Residual Networks: Multilevel Residual Networks](https://arxiv.org/pdf/1608.02908.pdf)


## Region Based CNNs (R-CNN - 2013, Fast R-CNN - 2015, Faster R-CNN - 2015)

+ R. Girshick, J. Donahue, T. Darrell, and J. Malik, [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524v5.pdf), 2013

+ R. Girshick, [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf), 2015

+ S. Ren, K. He, R. Girshick, and J. Sun, [Faster R-CNN: Toward Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1504.08083.pdf), 2015

+ R-CNN
  + one of the most impactful advancement in computer vision
  + Fast R-CNN and Faster R-CNN: making the model faster and better for modern object detection tasks
  + Purpose of R-CNNs
    + solving the problem of object detection
    + able to draw bounding boxes over all the objects
  + split into two general components:
    + the region proposal step
    + the classification step

+ Region proposal method
  + any class agnostic region proposal method should fit
  + Selective Search:
    + J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders, [Selective Search for Object Recognition](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)
    + used in particular for R-CNN
  + perform the function of generating 2000 different regions that have the highest probability of containing an object
  + obtained a set of region proposals
  + proposals wrapped into an an image size that can be fed into a trained CNN (AlexNet in this case) that extracts a feature vector of each region
  + vector used as the input to a set of linear SVMs trained for each class and output a classification
  + the vector gets into a bounding box regressor to obtain the most accurate coordinates

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/rcnn.png" style="margin: 0.1em;" alt="R-CNN workflow" title="R-CNN workflow" width=550>
    </a>
  </div>

  + non-maxima suppression: used to suppress bounding boxes that have a significant overlap with each other

### Fast R-CNN

+ R-CNN problems
  + multiple stages (ConvNets to SVMs to bounding box regressors)
  + computationally expensive
  + extremely slow: R-CNN - 53 seconds per image

+ Fast R-CNN
  + able to solve the problem of speed
  + basically sharing computation of convolutional layers between different proposals and swapping the order of generating region proposals and running th CNN
  + first fed through a ConvNet to obtain features of the region proposals from the last feature map of the ConvNet
  + lastly connected layers as well as regression and classification heads

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/FastRCNN.png" style="margin: 0.1em;" alt="Fast R-CNN workflow" title="Fast R-CNN workflow" width=550>
  </a>
</div>


### Faster R-CNN

+ Combat the complex training pipeline that both R-CNN and Fast R-CNN exhibited

+ inserting a region proposal network (RPN) after the last convolutional layer

+ able to just look at the last convolutional feature map and produce region proposals from that

+ the same pipeline as R-CNN used (ROI pooling, FC, and then classification and regression heads)

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/FasterRCNN.png" style="margin: 0.1em;" alt="Faster R-CNN workflow" title="Faster R-CNN workflow" width=350>
  </a>
</div>


### What it's Important

+ able to determine that a specific object in an image is one thing, but able to determine that object's exact location is a huge jump in knowledge for the computer

+ Fast R-CNN: the standard for object detection programs today



## Generative Adversarial Networks (2014)

+ I. Goodfellow, et. al., [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf), 2014

+ The next big development - Yann LeCun

+ Example
  + taking an example image and applying a perturbation, or a slight modification
  + the prediction error maximized
  + the object category of the prediction changes, while the image itself looks the same when compared to the image without the perturbation

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/Adversarial.png" style="margin: 0.1em;" alt="ConvNet examples with distortion" title="ConvNet examples with distortion" width=550>
    </a>
  </div>

  + ConvNet makes drastic errors in classification even the difference btw the images imperceptible to human

+ C. Szegedy, et. al., [Intriguing properties of neural networks](https://arxiv.org/pdf/1312.6199v4.pdf)

+ Generative adversarial networks
  + consisting of two models
    + a generative model
    + a discriminative model
  + generative model
    + determining whether a given image looks natural (an image from the dataset) or artificially created
    + similar to "a team of counterfeiters, trying to produce and use fake currency"
    + the generator trying to fool the discriminator
  + discriminative model
    + creating images so that the discriminator gets trained to produce the correct outputs
    + similar to "the police, trying to detect the counterfeit currency"
    + the discriminator trying to not get indistinguishable the genuine articles

+ Why it's important
  + the discriminator is aware of the "internal representation of the data"
  + the discriminator trained to understand the differences btw real images from the dataset and artificially created ones
  + used as a feature extractor using in CNN


## Generating Image Descriptions (2014)

+ A. Karpathy and F. Li, [Deep visual semantic alignment for Generating Image Descriptions](https://arxiv.org/pdf/1412.2306v2.pdf), 2015

+ A. Karpathy, A. Joulin and F. Li, [Deep fragment embeddings for bidirectional image sentence mapping](https://arxiv.org/pdf/1406.5679v1.pdf), 2014

+ Combining CNNs and bidirectional RNNs (Recurrent Neural Networks) to generate natural language descriptions of different image regions

+ Example of Generating image descriptions

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/Caption.png" style="margin: 0.1em;" alt="Example output of the model" title="Example output of the model" width=450>
    </a>
  </div>

+ Comparison w/ normal CNNs
  + traditional CNNs: a single clear label associated with each image in the training data
  + image descriptions:
    + having a sentence (or caption) associated with each image
    + weak label: segments of the sentence refer to (unknown) parts of the image

+ Generating image descriptions
  + Using this training data, a deep neural network "infers the latent alignment between segments of the sentences and the region that they described".
  + Another neural network takes in the image as input and generates a description in text.
  + Two components: alignment and generation

### Alignment model

+ goal: align the visual and textual data

+ Alignment model
  + input: an image and a sentence
  + output: a score for how well they match
  + trained on compatible and incompatible image-sentence pairs

+ Representation:
  + information about image
    + feeding the image into an R-CNN to detect the individual objects
    + trained on ImageNet data
    + the top 19 (plus the original image) object regions are embedded to a 500 dimensional space
    + having 20 different 500 dimensional vectors for each image
  + information about the sentence
    + embedding words into the same multimodal space
    + using a bidirectional recurrent neural network
    + illustrating information about the context of words in a given sentence
  + information about the picture and the sentence in the same space
  + compute inner products to show a measure of similarity


### Generative Model

+ the main purpose of creating a dataset where you have a set of image regions (found by the R-CNN) and corresponding text (BRNN)

+ learning from that dataset to generate descriptions given an image

+ taking in an image abd feeding it through a CNN

+ softmax layer: disregarded as the outputs of the fully connected layer become the inputs to another RNN

+ RNN: the function basically form probability distributions on the different words in a sentence

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
    <img src="https://adeshpande3.github.io/assets/GeneratingImageDescriptions.png" style="margin: 0.1em;" alt="Workflow of alignment and generative model" title="orkflow of alignment and generative model" width=850>
  </a>
</div>

### What it's Important

+ Using seemingly different RNN and CNN models to create a very useful application that in a way combines the fields of Computer Vision and Natural Language Processing

+ New idea in terms of how to make computers and models smarter when dealing with tasks that cross different fields


## Spatial Transformer Networks (2015)

+ M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu, [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf), 2016

+ The basic idea: transform the input image in a way so that the subsequent layers have an easier time making a classification

+ issue: making changes on image before fed into the specific convolutional layer

+ this module intends to correct: pose normalization (scenarios where the object is tilted or scaled) and spatial attention (brining attention to the correct object in a crowded image)

+ traditional CNNs: make model invariant to images with different scales and rotations requiring a lot of training examples for the model to learn properly

+ Spatial transformer
  + traditional CNN models
    + the maxpooling layer dealing with spatial invariant
    + knowing a specific feature in the original input volume
    + relative location not as important as exact location
  + dynamically to produce different behavior (different distortions/transformations) for each input image
  + module components
    + localization network
      + take input volume and output parameters of the spatial transformation applied
      + parameters $\theta$: 6 dimensions for an affine transformation
    + grid generator
      + creation of a sampling grid
      + the result of wrapping the regular grid with affine transformation ($\theta$) created in the localization network
    + sampler: perform a wrapping of the input feature map
  + the module dropped into a CNN at any point and basically helps the network learn how to transform feature maps in a way that minimizes the cost fiunction during training
  + example of spatial transformer

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html" ismap target="_blank">
      <img src="https://adeshpande3.github.io/assets/SpatialTransformer.png" style="margin: 0.1em;" alt="A spatial transformer module" title="A spatial transformer module" height=200>
      <img src="https://adeshpande3.github.io/assets/SpatialTransformer2.png" style="margin: 0.1em;" alt="Overview of the function of a spatial transformer module" title="Overview of the function of a spatial transformer module" height=200>
    </a>
  </div>



