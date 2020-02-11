# 13. Deep Belief Nets

## 13.1 The ups and downs of backpropagation

### Lecture Notes

+ A brief history of backpropagation
  + the backprogapation algorithm for learning multiple layers of features
    + Bryson A E, Jr. & Ho Y C. "Applied optimal control: optimization, estimation, and control",  Waltham, MA: Blaisdell, 1969. 481 p.
    + A. Bryson, Y.-C. Ho, and G. Siouris, [Applied Optimal Control: Optimization, Estimation, and Control](https://www.researchgate.net/profile/Y-C_Ho/publication/3116618_Applied_Optimal_Control_Optimization_Estimation_and_Control/links/5b7abdeaa6fdcc5f8b56a7df/Applied-Optimal-Control-Optimization-Estimation-and-Control.pdf), IEEE Transactions on Systems Man and Cybernetics 9(6):366 - 367 · July 1979  $\to$ linear version of backpropagation
    + P. Webos, [Beyond regression : new tools for prediction and analysis in the behavioral sciences](https://www.researchgate.net/profile/Paul_Werbos/publication/35657389_Beyond_regression_new_tools_for_prediction_and_analysis_in_the_behavioral_sciences/links/576ac78508aef2a864d20964/Beyond-regression-new-tools-for-prediction-and-analysis-in-the-behavioral-sciences.pdf), Thesis, 1974 $\to$ non-linear version and 1st version of backpropagation
    + McClelland, J. L., & Rumelhart, D. E. (1981). [An interactive activation model of context effects in letter perception: I. An account of basic findings.](https://psycnet.apa.org/doi/10.1037/0033-295X.88.5.375) Psychological Review, 88(5), 375–407. $\to$ not knowing P. Webos work and abandon due to bad performance by Hinton
    + D. Parker, “Learning-logic," Invention Report 581-64, File 1, Office of Technology Licensing, Stanford University, Stanford, CA, Oct. 1982.
    + Y. LeCun, [A Theoretical Framework for Back-Propagation](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf), proceedings of the 1988 Connectionist Models Summer School
    + D. Rumelhart, G. Hinton, R. Williams, [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf), Technical rept. Mar-Sep 1985
    + A. Kurenkov, [A 'Brief' History of Neural Nets and Deep Learning](http://www.andreykurenkov.com/writing/ai/a-brief-history-of-neural-nets-and-deep-learning/), 2015
  + backpropagation algorithm: clearly having great promise for learning multiple layers for non-linear feature detector
  + Give up at the late 1990's by most serious researchers
  + still widely used in psychological models and in practical applications suc as credit card fraud detection

+ Why failed
  + popular reasons for giving up in the late 1990's
    + not good use of multiple hidden layers of non-linear features
      + except in convolutional nets
      + except for toy examples
    + not work well in recurrent networks or deep auto-encoders
    + Support Vector Machine (SVM)
      + working better
      + required less expertise
      + produced repeatable results
      + much better and fancier theory
  + actual reasons
    + computers: thousands of times too slow
    + labeled datasets: hundreds of times too small for the regime in which backpropagation would really shine
    + deep networks:
      + too small and not initialized sensibly
      + gradient die too fast because of small initialized weights
  + these issues preenting from being successful for tasks, including vision and speech, where it would eventual be a big win

+ A spectrum of machine learning tasks
  + Typical Statistics $\longleftrightarrow$ Artificial Intelligence
  + typical statistics
    + low-dimensional data; e.g., less than 100 dimensions
    + lots of noise in the data
    + not much structure in the data; structure able to captured by a fairly simple model
    + main problem: separating true structure from noise, not thinking noise is a structure
    + solution: good for Bayesian nets but not ideal for non-Bayesian neural nets $\implies$ trying SVM or Gaussian Process (GP - regression)
  + artificial intelligence
    + high-dimensional data; e.g., more than 100 dimensions, such images and coefficients representing speech
    + noise: not the main problem
    + huge amount structure in the data, but too complicated to be represented by a simple model
    + main problem: figuring out a way to represent the complicated structure so that it can be learned
      + try to hand design appropriate representations
      + earier to resolve by letting backpropagation figure out what representations to use by given a multiple layers
      + using computation power to decide what the representation should be
    + solution: using backpropagation to figure it out

+ Support Vector Machines (SVM)
  + never a good bet for Artificial Intelligence tasks that need good representations
  + SVM: just a clever reincarnation of Perceptrons with kernel function
  + viewpoint 1:
    + expanding the input to a (very large) layer of non-linear <span style="color: re;">non-adaptive</span> features; like perceptrons w/ big layers of features
    + only one layer of adaptive weights, the weights from the features to the decision unit
    + a very efficient way of fitting the weights that controls overfitting by maximum margin hyperplane in a high dimensional space
  + viewpoint 2:
    + using each input vector in the training set to define a <span style="color: re;">non-adaptive</span> "pheature"
    + global match btw a test input and that training input, i.e., how similar the test input is to a particular training case 
    + a clever way of simultaneously doing feature selection and finding weights on the remaining features
  + Limitation: 
    + only for non-adaptive features and one layer of adaptive weights
    + unable to learn multiple layers of representation

+ Historical document from AT&T Adaptive Systems Research Dept, 1995
  + Larry Jackel, Head of AT&T Adaptive Systems Research Dept.
    + people would understand why big neural networks trained w/ backpropagation worked well on datasets
    + people would understand theoretically in terms of conditions and bounds
  + Vladimir Vapnik: by 2005 nobody will be using big neural nets to train w/ backpropagation
  + both wrong: limitations of using backpropagation
    + not because of no good theory and essentially hopeless
    + no big enough computers or big enough datasets



### Lecture Video

<a href="https://youtu.be/KSevMPjHgx4?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" target="_BLANK">
  <img style="margin-left: 2em;" src="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=100/>
</a><br/>


## 13.2 Belief nets

### Lecture Notes




### Lecture Video

<a href="https://youtu.be/6yxJeySclIc?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" target="_BLANK">
  <img style="margin-left: 2em;" src="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=100/>
</a><br/>


## 13.3 Learning Sigmoid Belief Nets

### Lecture Notes




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=100/>
</a><br/>


## 13.4 The wake-sleep algorithm

### Lecture Notes




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=100/>
</a><br/>

