# Neural Networks


## Modeling

### Simple Neuron Model

+ A biological neuron with a basic mathematical mode

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.embedded-vision.com/platinum-members/cadence/embedded-vision-training/documents/pages/neuralnetworksimagerecognition" ismap target="_blank">
      <img src="https://www.embedded-vision.com/sites/default/files/technical-articles/CadenceCNN/Figure3a.jpg" style="margin: 0.1em;" alt="Illustration of a biological neuron" title="Illustration of a biological neuron" width=350>
      <img src="https://www.embedded-vision.com/sites/default/files/technical-articles/CadenceCNN/Figure3b.jpg" style="margin: 0.1em;" alt="Illustration of a biological neuron's mathematical model" title="Illustration of a biological neuron's mathematical model" width=350>
    </a>
  </div>

+ [Linear neuron](../ML/MLNN-Hinton/01-IntroML.md#some-simple-models-of-neurons)

  \[z = b + \sum_i w_i x_i\]

  + $y$: the output
  + $b$: the bias
  + $w_i$: the weight on the $i$-th input
  + $x_i$: the $i$-th input

+ [Typical Activation functions $f(\cdot)$](../ML/MLNN-Hinton/01-IntroML.md#some-simple-models-of-neurons)
  + Binary threshold

    \[z = b + \sum_i w_i x_i \implies y = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}\]

  + Rectified Linear Neurons

      \[z = b + \sum_i x_i w_i \implies y = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}\]

  + Sigmoid neurons

    \[z = b + \sum_i x_i w_i \implies y = \frac{1}{1 + e^{-z}}\]

  + Stochastic binary neurons

    \[z = b + \displaystyle \sum_i x_i w_i \implies p(s = 1) = \frac{1}{1 + e^{-z}}\]


  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://blog.zaletskyy.com/some-simple-models-of-neurons" ismap target="_blank">
      <img src="https://blog.zaletskyy.com/Media/Default/NeuralNetworks/binaryNeuron.png" style="margin: 0.1em;" alt="Binary threshold neuron" title="Binary threshold neuron" height=150>
    </a>
    <a href="https://www.bo-song.com/coursera-neural-networks-for-machine-learning/" ismap target="_blank">
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-2.png" style="margin: 0.1em;" alt="Rectified Linear Neurons" title="Rectified Linear Neurons  (ReLU)" height=150>
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-5.png" style="margin: 0.1em;" alt="Sigmoid neurons" title="Sigmoid neurons" height=150>
      <img src="https://www.bo-song.com/wp-content/uploads/2015/12/Untitled-6.png" style="margin: 0.1em;" alt="Stochastic binary neurons" title="Stochastic binary neurons" height=150>
    </a>
  </div>




## Architectures





## Activation function: Logistic and Softmax Functions





## 

