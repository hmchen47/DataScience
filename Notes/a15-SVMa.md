# Support Vector Machine (SVM) Articles


## Support Vector Machines: A Concise Technical Overview

Author: Matthew Mayo @ KDnuggets

[Original](https://tinyurl.com/ybazrz4w)

+ Classification of ML
  + classification: building a model that separates data into distinct classes
  + classification schemes: decision tree, support vector machine, etc.

+ Support Vector Machine (SVM)
  + powerful, well-used classification algorithms
  + generating a substantial amount of attention prior to DL
  + useful, in particular, domains and remain some of the most popular classification algorithms
  + transforming the training data set into a high dimension
  + inspecting for the optimal separation boundary, or boundaries, btw classes
  + terminologies
    + __hyperplan:__ SVM boundary, identified by locating support vectors, or the instances that most essentially define classes
    + __margin:__ the line parallel to the hyperplane defined by the shortest distance btw a hyperplane and its support vectors
  + idea: w/ a high enough dimensions, a hyperplane separating a particular class from all others can always be found, thereby delineating dataset member classes
  + goal: maximum-margin hyperplane which resides equidistance from respective class support vector
  + Maximum-Margin Hyperplane and the Support Vectors

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
      onclick="window.open('https://tinyurl.com/ybazrz4w')"
      src    ="https://tinyurl.com/y8td5whg"
      alt    ="Maximum-Margin Hyperplane and the Support Vectors"
      title  ="Maximum-Margin Hyperplane and the Support Vectors"
    />
  </figure>

+ Finding the maximum-margin hyperplane
  + a hyperplane expressed as

    \[ \bf{W  \cdot X} + \bf{b} = \bf{0} \]

    + $\bf{W}$: vector of weights
    + $\bf{X}$: the training data
    + $\bf{b}$: scalar bias
  + combining the linear inequalities into a single equation and transforming them into a  constrainted quadratic optimization problem
  + using a Lagrangian formulation and solving w/ [Karush-Kuhn-Tucker conditions](https://tinyurl.com/lv88ujf)
  + maximum-margin hyperplane

    \[ \bf{x} = \bf{b + \sum_{i=1}^{n} \alpha_i y_i a(i) \cdot a} \]

    + $\bf{b}, \bf{\alpha_i}$: learned parameters
    + $n$: number of support vector instance
    + $i$: support vector instance
    + $t$: vector of training instance
    + $\bf{y_i}$: class value of a particular training instance of vector $t$
    + $\bf{a{i}}$: the vector of support vectors
  + data not linearly separable
    + transformed into a higher dimensional space
    + searched for hyperplane, another quadratic optimization problem
  + all calculation performed on the original input data $\to$ reduce complexity
  + higher dimensionality $\to$ higher computational complexity
  + the dot product of an instance and each of the support vectors need to to be calculated for each instance classification
  + __kernel function:__ map an instance to feature space created by a particular function it applies instance to lower dimensionality data
  + common kernel function: polynomial kernel $(x \cdot y)^n$ computes the dot product of 2 vectors, and raises that result to the power of n


## Support Vector Machines: A Simple Explanation

Author: Noel Bambrick @ AYLIEN

[Original](https://tinyurl.com/yaw2okww)

+ Support vector machine
  + supervised machine learning algorithm
  + classification and regression purpose
  + finding a hyperplane that best divides a dataset into two classes
  + support vectors: data points nearest the hyperplane, the points of data set that, if removed, would alter the position of dividing hyperplane

+ Hyperplane
  + a line that linearly separates and classifies a set of data
  + margin (left diagram): the distance btw the hyperrplan and the nearest data point from either set
  + goal: choose a hyperplane w/ a greatest possible margin btw the hyperplan and any point within the training set , giving a greater chance of new data being classified correctly
  + no clear hyperplane (right diagram): lifting balls represents the mapping of data into a higher dimension $\to$ kerneling

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yaw2okww" ismap target="_blank">
      <img style="margin: 0.1em;" height=150
        src  ="https://tinyurl.com/ydau7omt"
        alt  ="SVM"
        title="SVM"
      >
      <img style="margin: 0.1em;" height=150
        src  ="https://tinyurl.com/y9x59eke"
        alt  ="SVM"
        title="SVM"
      >
    </a>
  </div>

+ Pros and Cons of Support Vector Machines
  + advantages
    + accuracy
    + working well on smaller cleaner datasets
    + more efficient because it uses a subset of training points
  + limitations
    + not suitable to larger dataset
    + less efficient on noiser dataset w/ overlapping classes

+ SUM uses
  + text classification tasks, including category assignments, detecting spam and sentiment analysis
  + image recognition challenges: aspect-based recognition, color-based classification
  + handwritten digit recognition


## Why Use SVM?

Author: GregL

Date: 2017-01-24

[Original](https://tinyurl.com/y7ft7wn7)

+ What is SVM?
  + supervised machine learning algorithm
  + classification or regression problem
  + kernel trick: transform data and then based on these transformations if finds an optimal boundary btw the possible outputs

+ What makes SVM so great?
  + non-linear SVM or SVM using a non-linear kernel
  + the boundary that the algorithm calculated does not have to be a straight line
  + cons: much longer training time

+ Demo: Cows and Wolves
  + build a classified based on the position of the cows and wolves in your pasture
  + the logistic and decision tree models both make use of straight lines
  + code: [cows_and_wolves.py](src/a15-cows_and_wolves.py)

+ Let the SVM do the hard work
  + non-linear relationship
  + taking transformations btw variables ($\log(x)$, $(x^2)$) becomes much less important since it's going to be accounted for int he algorithm
  + build a model using 80% of data as training set
  + demo code: [comparisons of SVM, Logistic and Decision tree](src/a15-comparisons.py)



## Everything You Wanted to Know about the Kernel Trick

Author: Eric Kim

[Original](https://tinyurl.com/y7g97y2e), [PDF](https://tinyurl.com/y8puwov2)


### So, What is a Kernel Anyway?

+ What is a kernel anyway?
  + kernel function:
  
    \[ K(\vec{v}, \vec{w}) \text{ as a function } K: \Bbb{R}^N \times \Bbb{R}^K \to \Bbb{R} \]
  
  + function computed a dot product btw $\vec{v}$ and $\vec{w}$, ie, a measure of 'similarity' btw $\vec{v}$ and $\vec{w}$

### Linear SVM, Binary Classification

+ Linear SVM, Binary classification
  + $\exists$ a two-class dataset $D$
  + training a classifier $C$ to predict the class labels of future data points
  + binary classification problem: yes/no question
    + medicine: given a patient's vital data, patient $\stackrel{?}{\to}$ cold
    + computer vision: image containing a person?
  
+ Binary vs. multiclass classification
  + binary classifier: simple
  + multiclass classification problem
    + more than 2 possible outcomes
    + train face verification system that can detect the identity of a photograph from a pool of N people (where N > 2)

+ Main approaches to the mukticlass problem
  + directly add a multiclass extension to a binary classifier
    + pros: a principled way of solving the multiclass problem
    + cons: much more complicated $\to$ significantly longer training and test procedures
  + combine multiple binary classifiers to create a 'mega' multiclass classifier
    + pros: simple idea, easy to implement faster than multiclass extensions
    + cons: ad-hoc method for solving the multiclass problem $\to$ exist datasets for which OVO/OVA will perform poorly on
  + recommend to us OVO (one-vs-One) / OVA (One-vs-All) rather than more complicated generalized multiclass classifiers

+ Linear SVM
  + find a hyperplane $\vec{w} \to$ best separating the data points in the training set by class labels, $\vec{w}$ $\implies$ decision boundary
  + classify a point $x_i \in X$ (dataset) $\to$ simply see which 'side' of $\vec{w}$ that $x$ lies (Fig. 1)
  + the hyperplane $\vec{w}$ (a line in $\Bbb{R}$) separates the space into two halves (Fig. 2)
    + The Decision Boundary of a Linear SVM on a linearly-separable dataset. The solid line is the boundary.
    + The SVM is trained on 75% of the dataset, and evaluated on the remaining 25%. Circled data points are from the test set.

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

+ Nonlinear dataset
  + no line in $\Bbb{R}^2$ (Fig. 3)
    + A two-class dataset that is not linearly separable.
    + The outer ring (cyan) is class '0', while the inner ring (red) is class '1'.
  + both random classifier and linear SVM perform poorly (Fig. 4)
    + The decision boundary of a linear SVM classifier.
    + Because the dataset is not linearly separable, the resulting decision boundary performs and generalizes extremely poorly.
    + Like in Figure 2, we train the SVM on 75% of the dataset, and test on the remaining 25%.

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


### Dealing with Nonseparable Data

+ Dealing w/ non-separable data
  + assume that the decision boundary is a a separating hyperplane $\vec{w}$
  + generalizing the constraint of decision boundary that is line in the original feature space (here $\Bbb{R}^2$)
  + explicitly discover decision boundaries w/ arbitrary shape

+ Separable in higher-dimension
  + representing a 2D version of the true dataset lives in $\Bbb{R}^3$
  + the $\Bbb{R}^3$ dataset is easily linear separable by a hyperplane
  + train a linear SVM classifier that successfully finds a good decision boundary
  + given the dataset in $\Bbb{R}^2$, find a transformation $T: \Bbb{R}^2 \to \Bbb{R}^3$ s.t. transformed dataset is linearly separable in $\Bbb{R}^3$
  + Demo: seperation w/ higher dimensions $T([x_1, x_2]) = [x_1, x_2, x_1^2+x_2^2]$

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick="window.open('https://tinyurl.com/y7g97y2e')"
        src    ="https://tinyurl.com/y92svw8a"
        alt    ="Figure 5: (Left) A dataset in $\mathbb{R}^2$, not linearly separable. (Right) The same dataset transformed by the transformation: $[x_1, x_2] = [x_1, x_2, {x_1}^2 + {x_2}^2]$."
        title  ="Figure 5: (Left) A dataset in $\mathbb{R}^2$, not linearly separable. (Right) The same dataset transformed by the transformation: $[x_1, x_2] = [x_1, x_2, {x_1}^2 + {x_2}^2]$."
      />
      <figcaption> Figure 5: (Left) A dataset in $\mathbb{R}^2$, not linearly separable. (Right) The same dataset transformed by the transformation: $[x_1, x_2] = [x_1, x_2, {x_1}^2 + {x_2}^2]$. </figcaption>
    </figure>

  + assume a transformation $\phi$, the new classification pipeline
    1. transform the training set $X$ to $X^\prime$ w/ $\phi$
    2. train a linear SVM on $X^\prime$ to get classifier $f_{SVM}$
    3. test: a new sample $\vec{x}$ to $\vec{x^\prime} = \phi(\vec{x}) \to$ output class label determined by: $f_{SVM}(\vec{x^\prime})$
  + the hyperplane learned in $\Bbb{R}^3$ is nonlinear when projected back to $\Bbb{R}^2
  + improving the expressiveness of the linear SVM classifier by working a high-dimensional space

+ Recap
  + a dataset $D$, not linearly separable in a high-dimensional space $\Bbb{R}^M (M > N)$
  + $\exists$ a transformation $\phi$ that lifts the dataset $D$ to a higher-dimensional $D^\prime$ to find a decision boundary $\vec{w}$ that separates the classes in $D^\prime$
  + train a linear SVM on $D^\prime$ to find a decision n=boundary $\vec{w}$ that separates the classes in $D^\prime$
  + projecting the decision boundary $\vec{w}$ found in $\Bbb{R}^M$ back to the original space $\Bbb{R}^N$

+ Caveat: impractical for large dimensions
  + consider the computational consequences of increasing the dimensional consequences of increasing the dimensionality from $\Bbb{R}^N$ to $\Bbb{R}^M$ (M > N)
  + if $M$ grows very quickly w.r.t. $N$ (e.g., $M \in \mathcal{O}(2^N)$), then learning SVMs via dataset transformations will incurr serious computational and memory problem
  + polynomial kernel: $\Bbb{R}^2 \to \Bbb{R}^5$

    \[ [x_1, x_2] = [x_1^2, x_2^2, \sqrt{2} x_1x_2, \sqrt{2c}x_1, \sqrt{2c}x_2, c] \]

  + in general, a $d$-dimensional polynomial kernel maps fro $\Bbb{R}^N$ to an $\binom{n+d}{d}$-dimensional space

+ Dot products
  + the SVM has no need to explicitly work in the higher-dimensional space at training or testing time
  + during training, the optimization problem only uses the training samples to compute pair-wise dot products $(\vec{x_i}, \vec{x_j})$, where $\vec{x_i}, \vec{x_j} \in \Bbb{R}^N$

+ Kernel trick
  + $\exists$ kernel functions, $K(\vec{v}, \vec{w}), \;\vec{v}, \vec{w} \in \Bbb{R}^N$ compute the dot product btw $\vec{v}$ and $\vec{w}$ in a higher-dimensional $\Bbb{R}^M$ w/o explicitly transform $\vec{v}$ and $\vec{w}$ to $\Bbb{R}^M$
  + implication: by using a kernel $K(\vec{x_i}, \vec{x_j})$, implicitly transform dataset to a higher-dimensional $\Bbb{R}^M$ w/o using extra memory and w/ a minimal effect on computation time

+ Kernel functions
  + kernel function $K(\vec{v}, \vec{w})$: $K(\Bbb{R}^N \times \Bbb{R}^M) \to \Bbb{R}$ 
  + a kernel $K$ effectively computes dot products in a high-dimensional space $\Bbb{R}^M$ while remaining in $\Bbb{R}^N$
  + for $\vec{x_i}, \vec{x_j} \in \Bbb{R}^N, K(\vec{x_i}, \vec{x_j}) = (\phi(\vec{x_i}, \phi(\vec{x_j}))_M$ where $(\cdot, \cdot)_M$ = inner product of $\Bbb{R}^M, M>N$, $\phi(\vec{x})$ transforms $\vec{x}$ to $\Bbb{R}^M$; $\phi: \Bbb{R}^N \to \Bbb{R}^M$

+ Popular kernels
  + popular kernels: polynomial, radial basis fucntion, and sigmoid kernel
  + sklearn's SVM implementation `svm.svc`: kernel parameter - linear, poly, rbf, or sigmoid
  + let $\vec{x+t}, \vec{x_j} \in \Bbb{R}^N$ be rows from dataset $X$
    1. __polynomial kernel:__ $(\gamma \cdot \langle \vec{x_i} , \vec{x_j} \rangle + r)^d$
    2. __Radial Basis Function (RBF) Kernel:__ $\exp(-\gamma \cdot \lvert \vec{x_i} - \vec{x_j} \rvert ^2)$, where $\gamma > 0$
    3. __Sigmoid Kernel:__ $tanh(\langle \vec{x_i}, \vec{x_j} \rangle + r)$
  + sklearn's `svm.svc` uses both `gamma` and `coef0` parameters for the `kernel = 'sigmoid'` despite the above definition only having $\gamma$
  + choosing the 'correct' kernel is a nontrivial task, and may depend on the specific task at hand
  + true the kernel parameters to get good performance from classifier
  + popular parameter-tuning techniques including K-fold cross validation

+ Examples w/ kernel functions
  + polynomial kernel
  + radial basis function (RBF) kernel
  + sigmoid kernel

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y7g97y2e" ismap target="_blank">
      <img style="margin: 0.1em;" height=180
        src  ="https://tinyurl.com/y8brg9wh"
        alt  ="Figure 6: The decision boundary with a Polynomial kernel."
        title="Figure 6: The decision boundary with a Polynomial kernel."
      >
      <img style="margin: 0.1em;" height=180
        src  ="https://tinyurl.com/ycs4fg58"
        alt  ="Figure 7: The decision boundary with a Radial Basis Function (RBF) kernel."
        title="Figure 7: The decision boundary with a Radial Basis Function (RBF) kernel."
      >
      <img style="margin: 0.1em;" height=180
        src  ="https://tinyurl.com/ya33xyu9"
        alt  ="Figure 8: The decision boundary with a Sigmoid kernel."
        title="Figure 8: The decision boundary with a Sigmoid kernel."
      >
    </a>
  </div>


### Summary

1. Brief review of Linear SVMs for Binary Classification.
2. Encountered a linearly non-separable dataset that a Linear SVM is not able to handle.
3. Data which is linearly nonseparable in $R^N$ may be linearly separable in a higher-dimensional space $\mathbb{R}^M$ ( $M > N$, where $N$ is the original feature space dimensionality).
4. Unfortunately, a naive implementation of (3) would result in a massive time and space burden. For instance, for the $d$-degree polynomial kernel, the new feature space dimensionality $M$ grows at a rate of ${N + d \choose d}$.
5. However, the SVM formulation only requires dot products $\langle \vec{x_i}, \vec{x_j} \rangle$ between training examples $\vec{x_i}, \vec{x_j} \in X$ (where $X$ is the training data).
6. By replacing all dot products $\langle \vec{x_i}, \vec{x_j} \rangle$ with a kernel function $K(\vec{x_i}, \vec{x_j})$, we can implicitly work in a higher-dimensional space $\mathbb{R}^M$ ( $M > N$), without explicitly building the higher-dimensional representation. Thus, the SVM can learn a nonlinear decision boundary in the original $\mathbb{R}^N$, which corresponds to a linear decision boundary in $\mathbb{R}^M$.
    + This is the "trick" in "Kernel trick"
    + Machine Learning folks say that we have increased the "expressiveness" of Linear SVMs. Recall that the vanilla Linear SVM can only learn linear decision boundaries in $\mathbb{R}^N$. By introducing kernel methods, Linear SVMs can now learn nonlinear decision boundaries in $\mathbb{R}^N$.
      + Remember: the decision boundary will still be linear in $\mathbb{R}^M$, the feature space induced by the kernel $K$!
7. Finally, we empirically evaluated SVMs with various kernels, and observed a significant improvement when the dataset is not linearly separable.

### Source code

+ [README](src/a15-README.txt)
+ [Dataset Generation](generate_dataset.py)
+ [Data Transform](src/a15-demo_data_transform.py)
+ [Kernel trick](src/a15-kernel_trick.py)


### References

1. Jordan, Michael I., and Romain Thibaux. "[__The Kernel Trick.__](http://www.cs.berkeley.edu/~jordan/courses/281B-spring04/lectures/lec3.pdf)" Lecture Notes. 2004. Web. 5 Jan. 2013.
2. Berwick, Robert. "[__An Idiot's Guide to Support Vector Machines (SVMs)__](http://www.cs.ucf.edu/courses/cap6412/fall2009/papers/Berwick2003.pdf)". Lecture Slides. 2003. Web. 5 Jan. 2013. 
3. "[Scikit-learn: Machine Learning in Python](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)", Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. 
4. "[Scikit-learn: sklearn.svm.SVC Documentation](http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html)", Pedregosa et al.
5. Rifkin, Ryan. "[Multiclass Classification](http://www.mit.edu/~9.520/spring09/Classes/multiclass.pdf)". Lecture Slides. February 2008. Web. 6 Jan. 2013. 
6. Balcan, Maria Florina. "[8803 Machine Learning Theory: Kernels](http://www.cc.gatech.edu/~ninamf/ML10/lect0309.pdf)". Lecture Notes. 9 March. 2010. Web. 6 Jan. 2013.
7. Wikipedia contributors. "[Cross-validation (statistics).](http://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#K-fold_cross-validation)" Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 6 Jan. 2013. Web. 6 Jan. 2013.
8. Hofmann, Martin. "[Support Vector Machines -- Kernels and the Kernel Trick](http://www.cogsys.wiai.uni-bamberg.de/teaching/ss06/hs_svm/slides/SVM_Seminarbericht_Hofmann.pdf)". Notes. 26 June 2006. Web. 7 Jan. 2013.

