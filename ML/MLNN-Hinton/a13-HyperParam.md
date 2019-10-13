# A Disciplined Approach to Neural Network Hyper-Parameters: Part-I - Learning Rate, Batch Size, Momentum, and Weight Decay

Author: Leslie N. Smith

[Original Article](https://arxiv.org/pdf/1803.09820)

[Original Article HTML Version](https://www.arxiv-vanity.com/papers/1803.09820/)


## Abstract

+ how to examine the training validation/test loss function for subtle clues of underfitting and overfitting

+ suggesting guidelines for moving toward the optimal balancing point

+ experiments show that it is crucial to balance every manner of regularization for each dataset and architecture

+ weight decay used as a sample regularizer to show how its optimal value is tightly coupled with the learning rate and momentum

+ Files to replicate the results](https://github.com/lnsmith54/hyperParam1)


## 1. Introduction

+ the process of setting the hyper-parameters
  + including the designing the network architecture
  + requiring expertise and extensive trial and error and time consuming

+ no simple and easy ways to set hyper-parameters
  + learning rate, batch size, momentum and weight decay
  + grid search or random search: computationally expensive and time consuming
  + James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281–305, 2012.
  + training time and final performance: highly dependent on good choice
  + choosing the standard architecture and the hyper-parameter files freely available in model zoo or from gitgub.com

+ proposed methodologies for finding optimal settings for several hyper-parameters
  
+ goal: providing practical advice that saves time and effort, yet improves performance

+ basis of the approach
  + well-known concept of the balance between underfitting and overfitting
  + examining the training's test-/validation loss for clues of underfitting and overfitting to strive for optimal set of hyper-parameters
  + paying close attention while using cyclical learning rates and cyclical momentum
  + Leslie N Smith. Cyclical learning rates for training neural networks. In Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on, pp. 464–472. IEEE, 2017.


## 2. Related Work

+ Deep learning 
  + Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT press Cambridge, 2016.
  + Yoshua Bengio. Practical recommendations for gradient-based training of deep architectures. In Neural networks: Tricks of the trade, pp. 437–478. Springer, 2012.
  + Genevieve B Orr and Klaus-Robert Müller. Neural networks: tricks of the trade. Springer, 2003.

+ early works of Leslie N. Smith
  + Leslie N Smith. No more pesky learning rate guessing games. arXiv preprint arXiv:1506.01186, 2015.
  + Leslie N Smith. [Cyclical learning rates for training neural networks](https://www.arxiv-vanity.com/papers/1506.01186/). In Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on, pp. 464–472. IEEE, 2017.
  + Samuel L Smith, Pieter-Jan Kindermans, and Quoc V Le. [Don’t decay the learning rate, increase the batch size](https://www.arxiv-vanity.com/papers/1711.00489/). arXiv preprint arXiv:1711.00489, 2017.

+ use of large learning rate and small batch size
  + Stanisław Jastrzebski, Devansh Arpit, Nicolas Ballas, Vikas Verma, Tong Che, and Yoshua Bengio. [Residual connections encourage iterative inference](https://www.arxiv-vanity.com/papers/1710.04773/). arXiv preprint arXiv:1710.04773, 2017a.
  + Stanisław Jastrzebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. [Three factors influencing minima in sgd](https://www.arxiv-vanity.com/papers/1711.04623/). arXiv preprint arXiv:1711.04623, 2017b.
  + Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. [A walk with sgd](https://www.arxiv-vanity.com/papers/1802.08770/). arXiv preprint arXiv:1802.08770, 2018.
  + different optimal setting of learning rates and batch sizes in this report

+ exploring batch sizes and correlating the optimal batch size to the learning rate, size of the dataset, and momentum
  + Samuel L Smith and Quoc V Le. [Understanding generalization and stochastic gradient descent](https://www.arxiv-vanity.com/papers/1710.06451/). arXiv preprint arXiv:1710.06451, 2017.
  + more comprehensive and more practical in Sec. 4.2

+ use of regularization by weight decay and dropout
  + Alex Hernández-García and Peter König. [Do deep nets really need weight decay and dropout?](https://www.arxiv-vanity.com/papers/1802.07042/) arXiv preprint arXiv:1802.07042, 2018.
  + this report: the total regularization needs to be in balance for a given dataset and architecture
  + experiments suggestion: only add regularization by data augmentation to replace the regularization by weight decay and dropout w/o a full study of regularization

+ approaches to learn optimal hyper-parameters by differentiating the gradient w.r.t the hyper-parameters
  + Jonathan Lorraine and David Duvenaud. [Stochastic hyperparameter optimization through hypernetworks](https://www.arxiv-vanity.com/papers/1802.09419/). arXiv preprint arXiv:1802.09419, 2018.
  + this report: simpler to perform


## 3. The Unreasonable Effectiveness of Validation/Test Loss

+ a good detective observes subtle clues that the less observant miss.

+ purpose
  + draw attention to the clues in the training process
  + provide guidance as to their meaning

+ architecture & hyper-parameters
  + observing and understanding the clues available early during training
  + tuning w/ short runs of a few epochs
  + epoch defined as once through the entire training data
  + eliminating the necessary of running complete grid or random searches

+ comparison of training loss, validation accuracy, validation loss, and generalization error (Fig. 1)
  + These runs are a learning rate range test with the resnet-56 architecture and Cifar-10 dataset.
  + Characteristic plot of training loss, validation accuracy, and validation loss (left diagram)
    + plots of the training loss, validation accuracy, and validation loss for a learning rate range test of a residual network on the Cifar dataset to find reasonable learning rates for training
    + the test loss within the black box: signs of overfitting at learning rates of $0.01 - 0.04$
    + example where the test loss provide valuable information

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.arxiv-vanity.com/papers/1803.09820/" ismap target="_blank">
      <img src="https://media.arxiv-vanity.com/render-output/1492523/testLoss1.png" style="margin: 0.1em;" alt="Figure 1(a): Characteristic plot of training loss, validation accuracy, and validation loss." title="(a) Characteristic plot of training loss, validation accuracy, and validation loss." height=250>
      <img src="https://media.arxiv-vanity.com/render-output/1492523/generalizationError.png" style="margin: 0.1em;" alt="Figure 1(b): Characteristic plot of the generalization error, which is the validation/test loss minus the training loss." title="(b) Characteristic plot of the generalization error, which is the validation/test loss minus the training loss." height=250>
    </a>
  </div>

+ __REMARK 1.__ the test/validation loss is a good indicator of the network's convergence.
  + the test/validation loss used to provide insights on the training process
  + the final test accuracy used for comparing performance


### 3.1 A Review of the Underfitting and Overfitting Trade-off

+ Underfitting
  + unable to reduce the error for either the test or training set
  + cause: an under capacity of the machine learning model
  + not powerful enough to fit the underlying complexities of the data distribution

+ Overfitting: model so powerful as to fit the training set too well and the generalization error increases

+ Pictorial explanation of the tradeoff between underfitting and overfitting (Fig. 2)
  + model complexity (the x axis) refers to the capacity or powerfulness of the machine learning model
  + the optimal capacity falls between underfitting and overfitting
  + achieving a horizontal test loss during the training of a network can also point to the optimal balance of the hyper-parameter

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.arxiv-vanity.com/papers/1803.09820/" ismap target="_blank">
      <img src="https://media.arxiv-vanity.com/render-output/1492523/under-overfitting.png" style="margin: 0.1em;" alt="Pictorial explanation of the tradeoff between underfitting and overfitting." title="Figure 2. Pictorial explanation of the tradeoff between underfitting and overfitting." height=250>
    </a>
  </div>

+ __REMARK 2.__ achieving the horizontal part of the test loss is the goal of hyper-parameter tuning
  + difficult with deep neural network
  + with networks becoming more powerful with greater depth (i.e., more layers), width (i.e., more neurons or filters per layer), and the addition of skip connections to its architecture
  + various forms of regularization, such as weight decay or dropout
  + Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014.
  + important hyper-parameters
  + using a variety of optimization methods
  + Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

+ Insight of underfitting and overfitting
  + signs of the underfitting and overfitting of the test or validation loss early in the training process useful for tuning the hyper-parameters
  + Fig. 1a: some overfitting within the black square indicates a suboptimal choice of hyper-parameters
  + well set initial values for hyper-parameters results in performing well through the entire training process
  + the test loss during the training process used to find the optimal network architecture and hyper-parameters w/o performing a full training to compare the final performance results


### 3.2 Underfitting

+ Underfitting visible during the training (Fig. 3)
  + Underfitting is characterized by a continuously decreasing test loss, rather than a horizontal plateau.
  + Fig. 3(a) (a) Test loss for the Cifar-10 dataset with a shallow 3 layer network
    + red curve
      + decreasing test loss w/ a learning rate ($LR = 0.001$)
      + Underfitting: continue to decrease
    + blue curve
      + decreasing more rapidly during the initial iterations and then is horizontal
      + __a positive clue__: the configuration producing a better final accuracy than other configuration
  + Fig. 3(b) (b) Test loss for Imagenet with two networks; resnet-50 and inception-resnet-v2
    + underfitting: underlying complexities of the data distributions
    + the test loss continues decreasing over the 100,000 iterations (about 3 epochs)
    + the inception-resnet-v2 decreasing more and becoming more horizontal
    + the inception-resnet-v2 less underfitting

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.arxiv-vanity.com/papers/1803.09820/" ismap target="_blank">
      <img src="https://media.arxiv-vanity.com/render-output/1492523/3layerLoss.png" style="margin: 0.1em;" alt="(a) Test loss for the Cifar-10 dataset with a shallow 3 layer network." title="Figure 3. (a) Test loss for the Cifar-10 dataset with a shallow 3 layer network." height=250>
      <img src="https://media.arxiv-vanity.com/render-output/1492523/imagenetTestLoss3.png" style="margin: 0.1em;" alt="(b) Test loss for Imagenet with two networks; resnet-50 and inception-resnet-v2." title="Figure 3. (b) Test loss for Imagenet with two networks; resnet-50 and inception-resnet-v2." height=250>
    </a>
  </div>


### 3.3 Overfitting

+ Previous examples
  + Overfitting more complicated than underfitting but clues are visible in the test loss
  + Fig. 2: the test loss from underfitting (deceasing) to overfitting (increasing), but overfitting in neural network is often not so simple
  + Fig. 1 the test loss (blue curve)
    + signs of overfitting at small learning rate $(0.01 \sim 0.04)$
    + decreasing at higher rates as though it is underfitting
    + too small learning rate exhibits overfitting behavior

+ Examples of overfitting (Fig. 4)
  + increasing validation/test loss indicates overfitting
  + Fig. 4(a): Cifar-10 dataset with a shallow 3 layer network
    + $WD = 10^{-4}$ (blue curve): minimizing near at $LR = 0.002$, then increasing (overfitting)
    + $WD = 4 \times 10^{-3}$: (red curve)
      + stable at a larger LR range
      + attain a lower loss value
      + better than the previous one
      + diverging at $LR = 0.008$
    + $WD = 10^{-2}$ (yellow curve): sharp increasing at $LR = 0.005$
      + not a sign of overfitting
      + caused by instabilities in the training due to the large learning rate
  + Fig. 4(b): Imagenet dataset with resnet-50 architecture
    + blue curve: underfitting w/ $LR = 0.1$ and $WD = 10^{-4}$
    + red curve: overfittign w/ a ver small $WD = 10^{-7}$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.arxiv-vanity.com/papers/1803.09820/" ismap target="_blank">
      <img src="https://media.arxiv-vanity.com/render-output/1492523/overfitting3.png" style="margin: 0.1em;" alt="(a) Cifar-10 dataset with a shallow 3 layer network." title="Figure 4(a): Cifar-10 dataset with a shallow 3 layer network." height=250>
      <img src="https://media.arxiv-vanity.com/render-output/1492523/imagenetResnetOverfitting.png" style="margin: 0.1em;" alt="(b) Imagenet dataset with resnet-50 architecture." title="Figure 4(b) Imagenet dataset with resnet-50 architecture." height=250>
    </a>
  </div>

+ Additional examples of overfitting
  + Fig. 7(a): Cyclical momentum tests for the Cifar-10 dataset with a shallow 3 layer network
    + the value of momentum matters
    + yellow curve: 


## 4. Cyclical Learning Rates, Batch Sizes, Cyclical Momentum, and Weight Decay




### 4.1 Cyclical Learning Rates and Super-convergence Revisited





### 4.2 Batch Size




### 4.3 Cyclical Momentum




### 4.4 Weight Decay




## 5. Experiments with Other Architectures and Datasets





### 5.1 Wide Resents on Cifar-10




### 5.2 Densenets on Cifar-10





### 5.3 MNIST





### 5.4 Cifar-100




### 5.5 Imagnet





## 6. Discussion





## 7. References




## A. Appendix




### A.1 Experimental Methods: Detailed Information about the Experiments to Enable Replication





### A.2 Implementation of Cyclical Momentum in Caffe





