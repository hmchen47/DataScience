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




