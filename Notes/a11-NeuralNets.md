# Understanding and coding Neural Networks From Scratch in Python and R

Author: Sunil Ray

Date: 202-07-24

[Original](https://tinyurl.com/ydcgnt8j)

## 0. Introduction

+ Learn and practice a concept in two ways
  + option 1
    + learn the entire theory on a particular subject
    + find ways to apply those concepts
    + goals: how an entire algorithms, maths, assumptions, limitations, and applying
  + options 2
    + start w/ simple basics and picking an intuition on the subject
    + pick a problem and solve it
    + goals: how to apply an algorithm w/ different parameters, values, limits to understand the algorithm


## 1. Simple intuition behind Neural networks

+ Simple intuition behind Neural networks
  + forward propagation: an estimation process
    + taking several inputs
    + processing inputs through multiple neurons from multiple hidden layers
    + returning result using an output layer
  + backward propagation
    + trying to minimize the error of actual (desired) output and forward propagation output
    + by adjusting the value/weight of neurons to minimize their contributions
  + gradient descent
    + a common algorithm to reduce the number of iterations to minimize the error
    + optimizing the task quickly and efficiently


## 2. Multi-Layer Perceptron and its basics

+ Multi-Layer Perceptron Architecture
  + taking multiple inputs and produces one output
  + example: structure w/ three inputs and produces one output

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/yayrxdbq"
        alt    ="Perceptron"
        title  ="Perceptron"
      />
    </figure>

  + three ways of creating input-output relationships
    + directly combining the input and computing the output based on a threshold value
      + given $x_1 = 0$, $x_2 = 1$, $x_3 = 1$ and threshold $\rho = 0$
      + $x_1 + x_2 + x_3  > 0 \implies O = 1$, otherwise $O = 0$
    + add weights to the input
      + weight providing importance of an input
      + assign $w_1 = 2, w_2 = 3$, and $w_3 = 4$ to $x_1, x_2$, and $x_3$, respectively
      + multiply input with respective weights
      + compare w/ threshold value as $w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 > \rho$
    + add bias
      + bias as how much flexible the preceptron is
      + similar to the constant $b$ of a linear function $y = ax + b$
      + move the line up and down to fit the prediction w/ the data better
      + $b = 0$: the predict line through the origin $(0, 0)$
      + the predict line: $w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + 1 \cdot b$
  + artificial neuron = perceptron
  + neuron: applying non-linear transformation (activation function) to the inputs and biases

+ Activation function
  + the sum of weighted input: $w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + 1 \cdot b$
  + let $1 \to x_0$ and $b \to w_0$
  + output of the neuron

    \[ a = f\left( \sum__{i=0}^N w_i x_i \right) \]

  + used to make a non-linear transformation to fit nonlinear hypothesis or to estimate the complex functions
  + possible activation function: Sigmoid, Tanh, ReLu and others

+ Forward propagation, back propagation, and epoch
  + forward propagation: the process from input to output
  + error: the difference btw estimated output and the actual output
  + back propagation (BP):
    + the weight and bias updating process
    + update weight and bias based on the error
    + working by determining the loss (or error) at the output
    + propagating back into the network
    + weight updated to minimize the error resulting from each neuron
  + epoch: one round of forwarding and back propagation iteration

+ Multi-layer perceptron (MLP)
  + consisting of multiple layers: hidden layers stacked in btw the input layer and the output layer

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/yasmeebl"
        alt    ="neural network, multilayer perceptron, mlp"
        title  ="neural network, multilayer perceptron, mlp"
      />
    </figure>

  + fully connected: every node in a layer (except the input and output layer) connected ti every node in the previous layer and the following layer

+ Gradient descent
  + Full Batch Gradient Descent Algorithm
    + using all the training data points to update each of the weights
    + example:
      + using 10 data points (entire training data)
      + calculate the change in $w_1$ ($\Delta w_1$) and change in $w_2$ ($\Delta w_2$)
      + update $w_1$ and $w_2$
  + Stochastic Gradient Descent
    + using 1 or more (sample) but never the entire training data to update the weight one
    + example
      + using 1st data point
      + calculate the change in $w_1$ ($\Delta w_1$) and change in $w_2$ ($\Delta w_2$)
      + update $w_1$ and $w_2$
      + using 2nd point to work on the updated weights

## 3. Steps involved in Neural Network methodology





## 4. Visualizing steps for Neural Network working methodology





## 5. Implementing NN using Numpy (Python)





## 6. Implementing NN using R





## 7. Understanding the implementation of Neural Networks from scratch in detail





## 8. [Optional] Mathematical Perspective of Back Propagation Algorithm







