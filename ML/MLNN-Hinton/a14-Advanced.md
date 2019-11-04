# Advanced Topics in Neural Networks

Author: Matthew Stewart

[Original article](https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae)


## 1. Transfer Learning

+ Commonly applied to companies trying to develop commercial solutions that utilize computer vision

+ Issues for developing NN
  1. limited amount of data
  2. limited computing power

+ Companies releasing their state-of-the-art neural network architecture that are optimized for image recognition

+ Transfer learning
  + transfer the learning from one model to a second model which examining data is similar to the data of the original model
  + retrain the last few layers of the origin model with your own data to fine-tune the model for our specific application
  + the weight of the pre-trained model loaded into the architecture of the new model
  + weight from output layers trained while the other network layers frozen
  + Procedure
    1. build the architecture of the original model using Kersa and then load the model weights of the trained network usually `.h5` format for weight)
    2. freeze the weights of the initial layers by setting the layers to have the `trainable=False` parameter
  + the initial layers of a convolutional neural network containing  primitive information about the image
  + with deep neural network, the object becomes more complex and high-level as the network begins to differentiate more clearly between image qualities
  + not trying to teach the network to examine images but just fine=-tue it for our use-case
  + set the fully connected network layers at the output to be trainable, and perhaps the final convolutional layer (if enough data) to be trainable, then train the network with our data
  + benefit: fewer data to train the model because the number of network trainable parameters only a fraction of the total number of parameters in the network

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/1750/1*lG2SWdf1fEmkIvi6MPy9Ig.png" style="margin: 0.1em;" alt="The weights from a pre-trained model are loaded into the architecture of the new model. Weights from the output layers are trained whilst the other network layers are frozen. The resulting model will be fine-tuned for the required application." title="Transfer learning" width=450>
    </a>
  </div>


+ Example: J. Yosinski, et.al.
  + input layers (layer 1-4): relatively little impact on the output for transfer learning
  + output layers (layer 5-7): significant impact on the output

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/2208/1*ic5TQ3rTLhmRzQOKKp4qfw.png" style="margin: 0.1em;" alt="The input layers (e.g. layers 1–4) have relatively little impact on the output for transfer learning as compared with the later layers of the network (e.g. layers 6 and 7)." title="Illustration of accuracy with the network w/ different layers trained" width=450>
    </a>
  </div>

+ Limitation: only worked if two datasets very similar


## 2. Pruning

+ Model pruning
  + induce sparsity in a deep neural network's various connection matrices
  + reducing the number of non-zero-valued parameters in the model
  + originally used in decision trees where branches of the tree are pruned as a form of model regularization
  + pruning weights unimportant or rarely fired w/ little to no consequence
  + Fact: the majority of neurons relatively small impact on the model performance, i.e., achieving high accuracy even eliminating a large numbers of parameters

+ Reducing the number of parameters in a network
  + neural architectures and datasets get larger to obtain reasonable execution times of models
  + increasing important

+ Efficacy of pruning
  + research paper: [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/pdf/1710.01878.pdf)
  + examining the performance of neural networks as a function of sparsity (effectively the percentage of neurons removed)
  + even removing 75% of the neurons w/o significantly affected the model performance

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/1965/1*GdcBOqBhQw4GsA7wxfLMAg.png" style="margin: 0.1em;" alt="(a) the gradual sparsity function and expontentially decaying learning rate used for training sparse-Inception V3 model; (b) evolution of the model's accuracy during the training process" title="Illustration of sparsity and accuracy" width=400>
      <img src="https://miro.medium.com/max/1328/1*_MvV8ttzId69GzMeJCvBLQ.png" style="margin: 0.1em;" alt="Table 1: model size and accuracy tradeoff for sparse-Inception V3" title="Table for model size an daccuracy tradeoff" width=300>
    </a>
  </div>

+ Performing pruning
  + pruning typing done in convolutional neural networks
  + the majority of parameters in convolutional models occur in the fully connected (vanilla) neural layers
  + most of the parameters eliminated from this portion of the network

+ Approaches of performing pruning
  + weight pruning:
    + weight pruning rank-orders the weights by their magnitude
    + parameters with larger weights more likely to fire and thus more likely to be important
  + unit pruning
    + set entire columns in the weight matrix to zero, in effect deleting the corresponding output neuron
    + to achieve sparsity of $k\%$, ranking the columns of a weight matrix according to their L2-norm and delete teh smallest $k\%$
  + Fisher pruning
    + relying on the Fisher information
    + generating a norm known as the Fisher-Rao norm which can be used to rank-order parameters
    + conjecture: a link between the Fisher information and the redundancy of parameters

+ Two recent papers
  + Christos Louizos, Max Welling, Diederik P. Kingma (2018) [Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312)
  + Lucas Theis, Iryna Korshunova, Alykhan Tejani, Ferenc Huszár (2018) [Faster gaze prediction with dense networks and Fisher pruning](https://arxiv.org/abs/1801.05787)

+ code implementation of pruning on the standard VGG16 network: [code implementation of pruning on the standard VGG16 network](https://jacobgil.github.io/deeplearning/pruning-deep-learning?source=post_page-----f27fbcc638ae----------------------)


## Tuning the learning rate




### Cyclical Learning Rates for Neural Networks




### Estimating the Learning Rate




### SGD with Warm Restarts




### Snapshot ensembles: Train 1, get M for free




### Polyak-Ruppert averaging





## How to address overfitting




### Estimators




### Diagnosing bias-variance





## Dropout




### Diagnosing bias-variance




## Initialization Strategies




## Final Comment





