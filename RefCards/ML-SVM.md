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

+ [kernel function](../Notes/a15-SVMa.md#so-what-is-a-kernel-anyway)
  + kernel function:
  
    \[ K(\vec{v}, \vec{w}) \text{ with } K: \Bbb{R}^N \times \Bbb{R}^K \to \Bbb{R} \]
  
  + function computed a dot product btw $\vec{v}$ and $\vec{w}$, ie, a measure of 'similarity' btw $\vec{v}$ and $\vec{w}$

+ [Binary vs. multiclass classification](../Notes/a15-SVMa.md#linear-svm-binary-classification)
  + binary classifier: simple
  + multiclass classification problem
    + more than 2 possible outcomes
    + example: train face verification system to detect the identity of a photograph from a pool of N people (where N > 2)


## Modeling SVM
 
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
    <span style="padding-left: 2em;"/>if $n$ is mall, $m$ is intermediate: (e.g, n = 1\~1,000, m = 10\~10,000) <br/>
    <span style="padding-left: 3em;"/>Use SVM with Gaussian kernel<br/><br/>
    <span style="padding-left: 2em;"/>if $n$ is small, $m$ is large: (e.g., n = 1\~1,000, m = 50,000+) <br/>
    <span style="padding-left: 3em;"/>Create/add more features, then use logistic regression or SVM without a kernel
  
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

    \[ \min_\theta C \;\underbrace{\sum_{i=1}^m \left[ y^{(i)} \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \text{cost}_0(\theta^Tx^{(i)}) \right]}_{(A)} + \dfrac{1}{2} \sum_{j=1}^n \theta_j^2 \]

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


## Binary Classification

+ [Linear SVM, Binary classification](../Notes/a15-SVMa.md#linear-svm-binary-classification)
  + $\exists$ a two-class dataset $D$
  + training a classifier $C$ to predict the class labels of future data points
  + binary classification problem: yes/no question
    + medicine: given a patient's vital data, patient $\stackrel{?}{\to}$ cold
    + computer vision: image containing a person?

## Multiclass Classification

+ [Main approaches to the multiclass problem](../Notes/a15-SVMa.md#linear-svm-binary-classification)
  + directly add a multiclass extension to a binary classifier
    + pros: a principled way of solving the multiclass problem
    + cons: much more complicated $\to$ significantly longer training and test procedures
  + combine multiple binary classifiers to create a 'mega' multiclass classifier
    + pros: simple idea, easy to implement faster than multiclass extensions
    + cons: ad-hoc method for solving the multiclass problem $\to$ probably exist datasets 
      + OVO/OVA performing poorly on
      + general multiclass classifier performing well on
  + recommend to use OVO (one-vs-One) / OVA (One-vs-All) rather than more complicated generalized multiclass classifiers


## Linear SVM 

+ [Linear SVM](../Notes/a15-SVMa.md#linear-svm-binary-classification)
  + find a hyperplane $\vec{w}$ s.t. best separating the data points in the training set by class labels, $\vec{w}$ $\implies$ decision boundary
  + classify a point $x_i \in X$ (dataset) by simply seeing which 'side' of $\vec{w}$ that $x$ lies (Fig. 1)
  + the hyperplane $\vec{w}$ (a line in $\Bbb{R}^2$) separating the space into two halves (Fig. 2)
    + the Decision Boundary (solid line) of a Linear SVM on a linearly-separable dataset
    + SVM trained on 75% of the dataset, and evaluated on the remaining 25% (circled data points from the test set)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y7g97y2e" ismap target="_blank">
      <img style="margin: 0.1em;" height=200
        src  ="https://tinyurl.com/yajsamxv"
        alt  ="Figure 1: A two-class, linearly separable dataset."
        title="Figure 1: A two-class, linearly separable dataset."
      >
      <img style="margin: 0.1em;" height=200
        src  ="https://tinyurl.com/ycrvwg9p"
        alt  ="Figure 2: The Decision Boundary of a Linear SVM on a linearly-separable dataset. The solid line is the boundary. The SVM is trained on 75% of the dataset, and evaluated on the remaining 25%. Circled data points are from the test set."
        title="Figure 2: The Decision Boundary of a Linear SVM on a linearly-separable dataset. The solid line is the boundary. The SVM is trained on 75% of the dataset, and evaluated on the remaining 25%. Circled data points are from the test set."
      >
    </a>
  </div>

## Nonlinear SVM

+ [Nonlinear dataset](../Notes/a15-SVMa.md#linear-svm-binary-classification)
  + no line in $\Bbb{R}^2$ (Fig. 3)
  + both random classifier and linear SVM perform poorly (Fig. 4)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y7g97y2e" ismap target="_blank">
      <img style="margin: 0.1em;" height=200
        src  ="https://tinyurl.com/ya6ocsx8"
        alt  ="Figure 3: A two-class dataset that is not linearly separable. The outer ring (cyan) is class '0', while the inner ring (red) is class '1'."
        title="Figure 3: A two-class dataset that is not linearly separable. The outer ring (cyan) is class '0', while the inner ring (red) is class '1'."
      >
      <img style="margin: 0.1em;" height=200
        src  ="https://tinyurl.com/yaaw3xxy"
        alt  ="Figure 4: The decision boundary of a linear SVM classifier. Because the dataset is not linearly separable, the resulting decision boundary performs and generalizes extremely poorly. Like in Figure 2, we train the SVM on 75% of the dataset, and test on the remaining 25%."
        title="Figure 4: The decision boundary of a linear SVM classifier. Because the dataset is not linearly separable, the resulting decision boundary performs and generalizes extremely poorly. Like in Figure 2, we train the SVM on 75% of the dataset, and test on the remaining 25%."
      >
    </a>
  </div>

+ [Dealing w/ non-separable data](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + assumption: separating hyperplane $\vec{w}$ as the decision boundary
  + generalizing the constraint of decision boundary, the line in the original feature space (here $\Bbb{R}^2$)
  + explicitly discover decision boundaries w/ arbitrary shape

+ [Separable in higher-dimension](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + representing a 2D version of the true dataset lives in $\Bbb{R}^3$
  + the $\Bbb{R}^3$ dataset is easily linear separable by a hyperplane
  + train a linear SVM classifier that successfully finds a good decision boundary
  + given the dataset in $\Bbb{R}^2$, find a transformation $T: \Bbb{R}^2 \to \Bbb{R}^3$ s.t. transformed dataset linearly separable in $\Bbb{R}^3$
  + assume a transformation $\phi$, the new classification pipeline
    1. transform the training set $X$ to $X'$ w/ $\phi$
    2. train a linear SVM on $X'$ to get classifier $f_{svm}$
    3. test: a new sample $\vec{x}$ to $\vec{x'} = \phi(\vec{x}) \to$ output class label determined by $f_{svm}(\vec{x'})$
  + the hyperplane learned in $\Bbb{R}^3$ is nonlinear when projected back to $\Bbb{R}^2$
  + improving the expressiveness of the linear SVM classifier by working a high-dimensional space

+ [Procedures for non-linear SVM](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + a dataset $D$, not linearly separable in a high-dimensional space $\Bbb{R}^M (M > N)$
  + $\exists$ a transformation $\phi$ that lifts the dataset $D$ to a higher-dimensional $D^\prime$ to find a decision boundary $\vec{w}$ that separates the classes in $D^\prime$
  + train a linear SVM on $D^\prime$ to find a decision boundary $\vec{w}$ that separates the classes in $D^\prime$
  + projecting the decision boundary $\vec{w}$ found in $\Bbb{R}^M$ back to the original space $\Bbb{R}^N$

+ [Caveat: impractical for large dimensions](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + consider the computational consequences of increasing the dimensional consequences of increasing the dimensionality from $\Bbb{R}^N$ to $\Bbb{R}^M$ (M > N)
  + $M$ grows very quickly w.r.t. $N$ (e.g., $M \in \mathcal{O}(2^N)$) $\implies$ learning SVMs via dataset transformations will incurr serious computational and memory problem
  + in general, a $d$-dimensional polynomial kernel maps from $\Bbb{R}^N$ to an $\binom{N+d}{d}$-dimensional space

## Kernel trick

+ [Dot products](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + the SVM has no need to explicitly work in the higher-dimensional space at training or testing time
  + during training, the optimization problem only uses the training samples to compute pair-wise dot products $(\vec{x_i}, \vec{x_j})$, where $\vec{x_i}, \vec{x_j} \in \Bbb{R}^N$

+ [Kernel trick](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + $\exists$ kernel functions, $K(\vec{v}, \vec{w}), \;\vec{v}, \vec{w} \in \Bbb{R}^N$ compute the dot product btw $\vec{v}$ and $\vec{w}$ in a higher-dimensional $\Bbb{R}^M$ w/o explicitly transform $\vec{v}$ and $\vec{w}$ to $\Bbb{R}^M$
  + implication: by using a kernel $K(\vec{x_i}, \vec{x_j})$, implicitly transform dataset to a higher-dimensional $\Bbb{R}^M$ w/o using extra memory and w/ a minimal effect on computation time

+ [Kernel functions](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + kernel function $K(\vec{v}, \vec{w})$: $K(\Bbb{R}^N \times \Bbb{R}^M) \to \Bbb{R}$ 
  + a kernel $K$ effectively computes dot products in a high-dimensional space $\Bbb{R}^M$ while remaining in $\Bbb{R}^N$
  + $\forall \,\vec{x_i}, \vec{x_j} \in \Bbb{R}^N, K(\vec{x_i}, \vec{x_j}) = \left(\phi(\vec{x_i}), \phi(\vec{x_j})\right)_M$ where $(\cdot, \cdot)_M$ = inner product of $\Bbb{R}^M, M>N$, $\phi(\vec{x})$ transforms $\vec{x}$ to $\Bbb{R}^M$; i.e., $\phi: \Bbb{R}^N \to \Bbb{R}^M$


## Popular Kernel Functions

+ [Popular kernels & `sklearn` library](../Notes/a15-SVMa.md#dealing-with-non-separable-data)
  + popular kernels: polynomial, radial basis fucntion, and sigmoid kernel
  + sklearn's SVM implementation `svm.svc`: kernel parameter - `linear`, `poly`, `rbf`, or `sigmoid`
  + let $\vec{x_i}, \vec{x_j} \in \Bbb{R}^N$ be rows from dataset $X$
    1. __polynomial kernel:__ $(\gamma \cdot \langle \vec{x_i} , \vec{x_j} \rangle + r)^d$
    2. __Radial Basis Function (RBF) Kernel:__ $\exp\left(-\gamma \cdot \lvert \vec{x_i} - \vec{x_j} \rvert ^2\right)$, where $\gamma > 0$
    3. __Sigmoid Kernel:__ $\tanh(\langle \vec{x_i}, \vec{x_j} \rangle + r)$
  + sklearn's `svm.svc` uses both `gamma` and `coef0` parameters for the `kernel = 'sigmoid'` despite the above definition only having $\gamma$
  + choosing the 'correct' kernel is a nontrivial task, and may depend on the specific task at hand
  + true the kernel parameters to get good performance from classifier
  + popular parameter-tuning techniques including K-fold cross validation


