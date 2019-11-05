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



### 3.1 Cyclical Learning Rates for Neural Networks

+ Cyclical learning rates
  + main use: escape local extreme points, especially sharp local minima (overfitting)
  + saddle points:
    + abundant in high dimensions
    + convergence very slow if not impossible
  + increasing the learning rate periodically
    + a short term negative effects but help to achieve a long-term beneficial effect
  + decreasing the learning rate
    + reduce error towards the end

+ Examples of cyclical learning rates

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/780/1*kk4fq-pLk95ZWN6KyXM3RA.png" style="margin: 0.1em;" alt="Examples of cyclical learning rates" title="Examples of cyclical learning rates" width=300>
    </a>
  </div>

+ Limitations: what learning rate scheme set and the magnitude of these learning rates


### 3.2 Estimating the Learning Rate

+ starting w/ a small learning rate and increasing it on every batch exponentially

+ computing the loss function on a validation set

+ working for finding bounds for cyclic learning rates

+ Learning rates and loss function
  + the cliff region in between the two extremes
  + steadily decreasing and stable learning occurring

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/933/1*J-PZ-RanI1Ve-_kbuYzghQ.png" style="margin: 0.1em;" alt="Exponentially increasing learning rate across epochs" title="Exponentially increasing learning rate across epochs" height=200>
      <img src="https://miro.medium.com/max/1220/1*Pep5Xicj_1C1WhFQAgL-nA.png" style="margin: 0.1em;" alt="Loss function as a function of learning rate" title="Loss function as a function of learning rate" height=200>
    </a>
  </div>



### 3.3 SGD with Warm Restarts

+ Warm restarts
  + restart the learning after a specified number of epochs
  + example
    + the learning rate starts at 0.1 initially and decreases exponentially over time
    + after 30 iterations, the learning rate scheduler resets the learning rate to the same value as epoch 1
    + the learning rate scheduler repeats the same exponentially decay
  + record the best estimates each time before resetting the learning rate
  + restarts not from scratch but from the last estimate
  + providing most of the benefits as cyclical learning rates
  + able to escape extreme local minima

+ Warm restarts with cosine annealing

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/805/1*Iymc7F6RF_PKije9dhZA2g.png" style="margin: 0.1em;" alt="Warm restarts with cosine annealing done every 50 iterations of Cifar10 dataset" title="Warm restarts with cosine annealing done every 50 iterations of Cifar10 dataset" width=350>
    </a>
  </div>



### 3.4 Snapshot ensembles

+ G. Huang, et. al proposal: [Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/pdf/1704.00109)

+ Ensemble networks
  + training a single neural network with $M$ different models
  + much more robust and accurate than individual networks
  + another type of regularization technique
  + converge to $M$ different local optima and save network parameters
  + training w/ many different neural networks and then optimizing w/ major vote, or averaging of the prediction output

+ Example of snapshot ensembles
  + Left diagram: Illustration of SGD optimization with a typical learning rate schedule. The model converges to a minimum at the end of training.
  + Right diagram: Illustration of Snapshot Ensembling. The model undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima.

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/2118/1*Lp8rhR6C_TWuSIcF_QXXfA.png" style="margin: 0.1em;" alt="Left: Illustration of SGD optimization with a typical learning rate schedule. The model converges to a minimum at the end of training. Right: Illustration of Snapshot Ensembling. The model undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima." title="Snapshot ensembles" width=650>
    </a>
  </div>

+ Classification
  + often developing ensembled or blended models
  + providing superior results to any single model
  + constraint: high correlation btw models

+ Procedure
  + Model training
    + training w/ each model to reach local minimum w.r.t the training loss
    + take a snapshot of the model weights before raising the training rate
    + after $M$ cycles, $M$ model snapshots $f_1, f_2, \dots, f_M$ obtained
  + mode ensemble
    + taking average of snapshots
    + used to obtain result

+ Advantages
  + achieving a neural network w/ smoothened parameters
  + reducing the total noise and the total error
  + w/o any additional training cost: total training time of $M$ snapshots same as training a model w/ a standard schedule

+ Not perfect: different initialization points or hyperparameter choices converging to different local minimum

+ Results of the snapshot ensemble on several common datasets
  + Error rates (%) on CIFAR-10 and CIFAR-100 datasets.
  + All methods in the same group are trained for the same number of iterations.
  + Results of the ensemble method are colored in blue, and the best result for each network/dataset pair are bolded.

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://towardsdatascience.com/advanced-topics-in-neural-networks-f27fbcc638ae" ismap target="_blank">
      <img src="https://miro.medium.com/max/1695/1*aGAFkgUcikcm81E6EOX04g.png" style="margin: 0.1em;" alt="Error rates (%) on CIFAR-10 and CIFAR-100 datasets. All methods in the same group are trained for the same number of iterations. Results of the ensemble method are colored in blue, and the best result for each network/dataset pair are bolded." title="Snapshot ensembles" width=550>
    </a>
  </div>


### 3.5 Polyak-Ruppert averaging

+ Polyak averaging
  + motivation: gradient descent w/ a large learning rate unable to converge effectively to the global minimum
  + another approach to address the unstable learning issue that simulated the use of snapshot ensembles
  + using an average of the weights from multiple models seen towards the end of the training run
  + taking the time average of these parameters to obtain a smoother estimator for the true parameter, $t$ iterations

    \[ \hat{\theta}(t) = \frac{1}{t} \sum_i \hat{\theta}^{(i)} \]

  + leveraged in several ways
    + time averaging: using hte weights from the same model at several different epochs towards the end of the training run
    + ensemble averaging: using the weights from multiple models towards their individual training runs
    + hybrid approach: using the weights from snapshots and then averaging these weights for an ensemble prediction
  
+ Convergence
  + guarantee strong converge in a convex setting
  + non-convex surfaces: the parameter space differed greatly in different regions; averaging less useful
  + Considering the exponentially decaying average

    \[ \hat{\theta}^{(t)} = \alpha \hat{\theta}^{(t-1)} + (1 - \alpha) \hat{\theta}^{(t)} \quad \text{with} \quad \alpha \in [0, 1] \]

  + depending on the chosen value of $\alpha$ additional weight either placed on the newest parameter values or the older parameter values
  + the importance of the older parameters exponentially decays over time

+ Polyak averaging & snapshot ensembles: different ways of smoothing the random error manifestly present in the unstable learning process of neural networks

+ Good walkthrough of Polyak averaging applied to neural networks: [How to Calculate an Ensemble of Neural Network Model Weights in Keras (Polyak Averaging)](https://machinelearningmastery.com/polyak-neural-network-model-weight-ensemble/?source=post_page-----f27fbcc638ae----------------------)



## 4. How to address overfitting




### 4.1 Estimators




### 4.2 Diagnosing bias-variance





## 5. Dropout




### 5.1 Diagnosing bias-variance




## 6. Initialization Strategies





## 7. Further Reading





