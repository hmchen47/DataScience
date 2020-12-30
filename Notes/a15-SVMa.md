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


