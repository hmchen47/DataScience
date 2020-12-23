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
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
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

    \[ a = f\left( \sum_{i=0}^N w_i x_i \right) \]

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

+ Procedure of Multi-layer perceptron

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
      onclick="window.open('https://tinyurl.com/ydcgnt8j')"
      src    ="https://tinyurl.com/yasmeebl"
      alt    ="neural network, mlp"
      title  ="neural network, mlp"
    />
  </figure>

  + <font style="color: magenta; font-weight: bold;">Forward Propagation</font>: step 2~4
  + <font style="color: magenta; font-weight: bold;">Backward Propagation</font>: step 5~11

  0. input and output
    + $X$: input matrix
    + $Y$: output matrix
  1. initialize weights and biases w/ random values (one-time initialization)
    + $w_h$: a weight matrix to the hidden layer
    + $b_h$: bias matrix to the hidden layer
    + $w_{out}$: a weight matrix to the output layer
    + $b_{out}$: bias matrix to the output layer
  2. linear transformation: take matrix dot product of input and weights assigned to edges between the input and hidden layer then add biases of the hidden layer neurons to respective inputs

    ```python
    hidden_layer_input= matrix_dot_product(X,wh) + bh
    ```

  3. perform non-linear transformation using an activation function (Sigmoid). (Sigmoid $f(x) = \frac{1}{(1 + \exp(-x))}$)

    ```python
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    ```

  4. perform a linear transformation on hidden layer activation (take matrix dot product with weights and add a bias of the output layer neuron) then apply an activation function (again used sigmoid, but you can use any other activation function depending upon your task) to predict the output

    ```python
    output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout
    output = sigmoid(output_layer_input)
    ```

  5. compare prediction with actual output and calculate the gradient of error (Actual – Predicted). Error is the mean square loss = $((Y-t)^2)/2$

    ```python
    E = y – output
    ```

  6. compute the slope/ gradient of hidden and output layer neurons ( To compute the slope, we calculate the derivatives of non-linear activations x at each layer for each neuron). The gradient of sigmoid can be returned as $x \cdot (1 – x)$.

    ```python
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    ```

  7. compute change factor(delta) at the output layer, dependent on the gradient of error multiplied by the slope of output layer activation

    ```python
    d_output = E * slope_output_layer
    ```

  8. the error will propagate back into the network which means error at the hidden layer. take the dot product of the output layer delta with the weight parameters of edges between the hidden and output layer ($w_{out}^T$).

    ```python
    Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)
    ```

  9. compute change factor(delta) at hidden layer, multiply the error at hidden layer with slope of hidden layer activation

    ```python
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    ```

  10. update weights at the output and hidden layer: The weights in the network can be updated from the errors calculated for training example(s).

    ```python
    wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate
    wh =  wh + matrix_dot_product(X.Transpose,d_hiddenlayer)*learning_rate
    ```

  11. update biases at the output and hidden layer: The biases in the network can be updated from the aggregated errors at that neuron
    + bias at output_layer = bias at output_layer + sum of delta of output_layer at row-wise * learning_rate
    + bias at hidden_layer = bias at hidden_layer + sum of delta of output_layer at row-wise * learning_rate

    ```python
    bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
    bout = bout + sum(d_output, axis=0)*learning_rate
    ```

## 4. Visualizing steps for Neural Network working methodology

+ Visualization for MLP
  + Step 0: Read input and output

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/ya3p5m8f"
        alt    ="neural network methodology"
        title  ="neural network methodology"
      />
    </figure>

  + Step 1: Initialize weights and biases with random values (There are methods to initialize weights and biases but for now initialize with random values)

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/y8dwrydq"
        alt    ="neural network, weights"
        title  ="neural network, weights"
      />
    </figure>

  + Step 2: Calculate hidden layer input: <br/>
    hidden_layer_input= matrix_dot_product(X,wh) + bh

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/ya8kkjrg"
        alt    ="neural networks, hidden layer"
        title  ="neural networks, hidden layer"
      />
    </figure>
  
  + Step 3: Perform non-linear transformation on hidden linear input <br/>
    hiddenlayer_activations = sigmoid(hidden_layer_input)

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/y936n3by"
        alt    ="transformation. activation function"
        title  ="transformation. activation function"
      />
    </figure>
  
  + Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer<br/>
    output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout<br/>
    output = sigmoid(output_layer_input)

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/y9ex86jx"
        alt    ="neural network, activaton function"
        title  ="neural network, activaton function"
      />
    </figure> 


  + Step 5: Calculate gradient of Error(E) at output layer<br/>
    E = y-output

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/yck24qtr"
        alt    ="gradient"
        title  ="gradient"
      />
    </figure>
  
  + Step 6: Compute slope at output and hidden layer<br/>
    Slope_output_layer= derivatives_sigmoid(output)<br/>
    Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/ya95wyjc"
        alt    ="neural network, gradient slope"
        title  ="neural network, gradient slope"
      />
    </figure>
  
  + Step 7: Compute delta at output layer <br/>
    d_output = E * slope_output_layer*lr

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/yck9lrcj"
        alt    ="delta, neural network"
        title  ="delta, neural network"
      />
    </figure>

  + Step 8: Calculate Error at the hidden layer<br/>
    Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/ydbaplfp"
        alt    ="hidden layer, error"
        title  ="hidden layer, error"
      />
    </figure>
  
  + Step 9: Compute delta at hidden layer <br/>
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/ydcp6atp"
        alt    ="delta, hidden layer"
        title  ="delta, hidden layer"
      />
    </figure>

  + Step 10: Update weight at both output and hidden layer <br/>
    wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate<br/>
    wh =  wh+ matrix_dot_product(X.Transpose,d_hiddenlayer)*learning_rate

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/y8kjz5zx"
        alt    ="neural network, weights"
        title  ="neural network, weights"
      />
    </figure>
  

  + Step 11: Update biases at both output and hidden layer<br/>
    bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate<br/>
    bout = bout + sum(d_output, axis=0)*learning_rate

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick="window.open('https://tinyurl.com/ydcgnt8j')"
        src    ="https://tinyurl.com/yc752j83"
        alt    ="neural network, bias"
        title  ="neural network, bias"
      />
    </figure>

  + above only cover one iteration

## 5. Implementing NN using Numpy (Python)

[Implementation of NN using Numpy](../src/a11-NeuralNets.py)

## 6. Implementing NN using R

[Implementation of NN in R](../src/a11-NeuralNets.r)

## 7. Understanding the implementation of Neural Networks from scratch in detail

[Implementation of NN in Jupyter](../src/a11-NeuralNets.ipynb)


## 8. [Optional] Mathematical Perspective of Back Propagation Algorithm







