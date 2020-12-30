# Support Vector Machine (SVM)


## Overview 

+ [Support Vector Machines](../ML/MLNN-Hinton/13-BeliefNets.md#131-the-ups-and-downs-of-backpropagation) (SVM)
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


## Model: SVM
 
+ [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) function

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.coursera.org/learn/machine-learning/resources/Es9Qo">
      <img src="../ML/ML-Stanford/images/m12-01.png" style="margin: 0.1em;" alt="text" title="caption" width="200">
      <img src="../ML/ML-Stanford/images/m12-02.png" style="margin: 0.1em;" alt="text" title="caption" width="200">
    </a></div>
  </div>

+ [Support vector machine: Objective](../ML/ML-Stanford/12-SVM.md#large-margin-classification)

  \[\min_\theta C \cdot \sum_{j=1}^m \left[y^{(i)} \text{cost}_1 (\theta^T x^{(i)}) + (1 - y^{(i)}) \text{cost}_0 (\theta^T x^{(i)}) \right] + \dfrac{1}{2} \sum_{j=1}^n \theta_j^2\]

+ [cost functions](../ML/ML-Stanford/12-SVM.md#large-margin-intuition)

    <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
      <div><a href="https://d3c33hcgiwev3.cloudfront.net/_246c2a4e4c249f94c895f607ea1e6407_Lecture12.pdf?Expires=1555459200&Signature=Aibx4MyH1R-oUJMXRrW3chKna-a~XoCJd-c~g3UwUpgnzRFULWlxriuLCniD~Q92GzKqNrslw0CwCyCyMBKemvQnjt-iVThjFe9Q23SDi3qmcAPq1eprZTr84Vq2IccOXYuPf7XaHwBj~r16BTEDnkiLWOZ79H1d1zTG6DBQpT0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A">
        <img src="../ML/ML-Stanford/images/m12-04.png" style="margin: 0.1em;" alt="cost functions to represent the y = 1 (left diagram) and y = 0 (right diagram)" title="Cost functions: left(y=1) right(y=0)" width="350">
      </a></div>
    </div><br/>

  + If $y=1$, we want $\theta^T x \geq 1$ (not just $\geq 0$) <br/>
    If $y=0$, we want $\theta^T x \leq -1$ (not just $< 0$)

+ [Logistic regression vs SVMs](../ML/ML-Stanford/12-SVM.md#svms-in-practice-using-an-svm)
  + logistic regression or SVM
    $n =\;$ number of features ($x \in \mathbb{R}^{n+1}$), $m = \;$ number of training examples <br/>
    if $n$ is large (relative to $m$): <br/>
    <span style="padding-left: 1em;"/>Use logistic regression, or SVM without a kernel ("linear kernel") <br/>
    <span style="padding-left: 2em;"/>if $n$ is mall, $m$ is intermediate: (e.g, n = 1~1,000, m = 10~10,000) <br/>
    <span style="padding-left: 3em;"/>Use SVM with Gaussian kernel<br/><br/>
    <span style="padding-left: 2em;"/>if $n$ is small, $m$ is large: (e.g., n = 1~1,000, m = 50,000+) <br/>
    <span style="padding-left: 3em;"/>Creat/add more features, then use logistic regression or SVM without a kernel
  
  + Neural network likely to work well for most of these settings, but may be slower to train

+ [Multi-class classification](../ML/ML-Stanford/12-SVM.md#svms-in-practice-using-an-svm)
  + classes: $y \;\in\; \{1, 2, 3, \ldots, K\}$
  + Many SVM packages already have built-in multi-class classification functionality
  + Otherwise, use one-vs-all method. (Train $K$ SVMs, one to distinguish $y=i$ from the rest, for $i=1, 2, \ldots, K$), get $\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)}$. Pick class $i$ with largest $(\theta^{(i)})^Tx$

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.ritchieng.com/machine-learning-svms-support-vector-machines/#2b-kernels-ii">
      <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w7_support_vector_machines/svm22.png" style="margin: 0.1em;" alt="Multi-class classification" title="Multi-class classification" width="350">
    </a></div>
  </div>


## Decision Boundary

+ [Simplification for Decision Boundary](../ML/ML-Stanford/12-SVM.md#large-margin-intuition)
  + Objective:

    \[\min_\theta C \underbrace{\sum_{i=1}^m \left[ y^{(i)} \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \text{cost}_0(\theta^Tx^{(i)}) \right]}_{(A)} + \dfrac{1}{2} \sum_{j=1}^n \theta_j^2\]

  + $C \gg 0$, $(A) = 0\;$ to minimize the cost function
  + Wherever $y^{(i)} = 1\;: \theta^T x^{(i)} \geq 1$ <br/>
    Wherever $y^{(i)} = 0\;: \theta^T x^{(i)} \leq -1$

    \[\begin{array}{rl} \min_\theta & C \cdot 0 + \dfrac{1}{2} \sum_{j=1}^n \theta^2_j \\\\ \text{s.t.} & \theta^T x^{(i)} \geq 1 \quad \text{if } y^{(i)} = 1 \\ & \theta^T x^{(i)} \leq -1 \quad \text{if } y^{(i)} = 0 \end{array}\]


+ [SVM decision boundary](../ML/ML-Stanford/12-SVM.md#mathematics-behind-large-margin-classification)
  + Objective

    \[\begin{array}{ll} \displaystyle \min_\theta & \dfrac{1}{2} \sum_{j=1}^n \theta^2_j \\\\ \text{s.t. } & \theta^T x^{(i)} \geq 1 \quad \text{if } y^{(i)} = 1 \\ & \theta^T x^{(i)} \leq -1 \quad \text{if } y^{(i)} = 0 \end{array}\]

  + Projections and hypothesis

    \[\begin{array}{ll} \displaystyle \min_\theta & \dfrac{1}{2} \displaystyle \sum_{j=1}^n \theta^2_j = \dfrac{1}{2} \parallel \theta \parallel^2 \\\\ \text{s.t. } & p^{(i)} \cdot \parallel \theta \parallel \geq 1 \quad \text{if } y^{(i)} = 1 \\ & p^{(i)} \cdot \parallel \theta \parallel \leq -1 \quad \text{if } y^{(i)} = 0 \end{array}\]

    where $p^{(i)}$ is the projection of $x^{(i)}$ onto the vector $\theta$.

    + Simplification: $\theta_0 = 0$ - When $\theta_0 = 0$, the vector passes through the origin.
    + $\theta$ projection: always $90^o$ to the decision boundary


## Kernels

+ [Gaussian kernel](../ML/ML-Stanford/12-SVM.md#kernels-i)
  + Given $x$, compute new feature depending on proximity to landmarks $l^{(1)}, l^{(2)}, l^{(3)}, \ldots$
  
    \[\begin{array}{rcl} f_1 & = & similarity(x, l^{(1)}) = \exp \left( -\dfrac{\parallel x - l^{(1)} \parallel^2}{2 \sigma^2} \right) \\ f_2 & = & similarity(x, l^{(2)}) = \exp \left( -\dfrac{\parallel x - l^{(2)} \parallel^2}{2 \sigma^2} \right) \\  f_3 & = & similarity(x, l^{(3)}) = \exp \left( -\dfrac{\parallel x - l^{(3)} \parallel^2}{2 \sigma^2} \right) \\ & \cdots \end{array}\]

  + manually pick 3 landmarks
  + given an example $x$, define the features as a measure of similarity between $x$ ans the landmarks
  
    \[\begin{array}{rcl} f_1 &=& similarity(x, l^{(1)}) \\ f_2 &=& similarity(x, l^{(2)}) \\ f_3 &=& similarity(x, l^{(3)}) \end{array}\]

  + kernel: $k(x, l^{(i)}) = similarity(x, l^{(i)})$
  + The similarity functions are __Gaussian kernels__, $\exp\left( - \dfrac{\parallel x - l^{(i)} \parallel^2}{2\sigma^2} \right)$.

+ [Kernels and Similarity](../ML/ML-Stanford/12-SVM.md#kernels-i)

  \[f_1 = similarity(x, l^{(1)}) = exp \left(-\dfrac{\parallel x - l^{(1)} \parallel^2}{2\sigma^2} \right) = \exp \left( -\dfrac{\sum_{j=1}^n (x_j - l_j^{(1)})^2}{2 \sigma^2} \right)\]

  + If $x \approx l^{(1)}: f_1 \approx \exp \left( -\dfrac{0^2}{2\sigma^2} \right) \approx 1$
  + If $x$ is far from $l^{(1)}: f_1 = \exp \left( - \dfrac{(\text{large number})^2}{2\sigma^2} \right) \approx 0$

+ [SVM with kernels](../ML/ML-Stanford/12-SVM.md#kernels-ii)
  + Given $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)})$, choose $l^{(1)} = x^{(1)}, l^{(2)} = x^{(2)}, \ldots, l^{(m)} = x^{(m)}$
  + Given example $x$:

    \[\begin{array}{lcl} f_0  =  1 \\f_1 = similarity(x, l^{(1)}) \\ f_1 = similarity(x, l^{(2)})  \\ \cdots \end{array} \implies f = \begin{bmatrix} f_0 \\ f_1 \\ \vdots \\ f_m \end{bmatrix}\]

  + For training example $(x^{(i)}, y^{(i)})\;$:

    \[x^{(i)} \quad\implies\quad \begin{array}{rcl} f_0^{(i)} &=& 1 \\ f_1^{(i)} &=& sim(x^{(i)}, l^{(1)}) \\ f_2^{(i)} &=& sim(x^{(i)}, l^{(2)}) \\ &\cdots& \\ f_i^{(i)} &=& sim(x^{(i)}, l^{(i)}) = \exp \left( -\dfrac{0}{2\sigma^2} \right) \\ &\cdots& \\ f_m^{(i)} &=& sim(x^{(i)}, l^{(m)}) \end{array} \implies f^{(i)} = \begin{bmatrix} f_0^{(i)} \\ f_1^{(1)} \\ \vdots \\ f_m^{(i)} \end{bmatrix}\]

  + Hypothesis: Given $x$, compute features $f \in \mathbb{R}^{m+1}$

    Predict "y=1" if $\theta^Tf = \theta_0 f_0  + \theta_1 f_1 + \ldots + \theta_m f_m \geq 0, \quad \theta \in \mathbb{R}^{m+1}$

  + Training

    \[min_\theta C \cdot \sum_{i=1}^m \left[ y^{(i)} \text{cost}_1(\theta^T f^{(i)}) + (1 - y^{(i)}) \text{cost}_0(\theta^T f^{(i)}) \right] + \dfrac{1}{2} \sum_{j=1}^{n (=m)} \theta_j^2\]

    \[\begin{array}{crl} \sum_{j} \theta_j^2 &=& \theta^T \theta = \begin{bmatrix} \theta_1 & \theta_2 & \cdots & \theta_m \end{bmatrix} \begin{bmatrix} \theta_1 \\ \theta_2 \ \vdots \\ \theta_m \end{bmatrix} = \parallel \theta \parallel^2 \\\\ &=& \theta^TM\theta = \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_m \end{bmatrix} \begin{bmatrix} 0 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \end{bmatrix} \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_m \end{bmatrix} \end{array}\]

  + applying kernel's idea to other algorithms
    + able to do so by applying the kernel's idea and define the source of features using landmarks
    + unable to generalize SVM's computational tricks to other algorithms

+ [SVM parameters](../ML/ML-Stanford/12-SVM.md#kernels-ii)
  + $C (= 1/\lambda)$
    + Large C (small $\lambda$): lower bias, high variance
    + Small C (large $\lambda$): higher bias, lower variance
  + $\sigma^2$
    + Large $\sigma^2\;$: feature $f_i$ vary more smoothly $\implies$ higher bias, lower variance
    + Small $\sigma^2\;$: feature $f_i$ vary less smoothly $\implies$ lower bias, higher variance

+ [Other choice of kernel](../ML/ML-Stanford/12-SVM.md#svms-in-practice-using-an-svm)
  + Note: not all similarity functions $similarity(x, l)$ make valid kernels. (Need to satisfy technical condition called "Mercer's Theorem" to make sure SVM packages' optimizations run correctly, and do not diverge).
  + Many off-the-shelf kernels available
    + Polynomial kernel: $k(x, l) = (x^Tl + \text{constant})^{\text{degree}}$ such as $(x^T l)^2, (x^T l)^3, (x^T l) + 1^3, (x^T l)^4, \ldots$
    + More esoteric: String kernel, chi-square kernel, histogram intersection kernel, ...



