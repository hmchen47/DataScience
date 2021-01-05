# A Few Useful Things to Know about Machine Learning

Author: Pedro Domingos

[Original](https://tinyurl.com/y4fz3jja)

[Article in Communications of the ACM](https://tinyurl.com/kcd5ztd)


## 1. Introduction

+ Machine learning
  + systems automatically learning programs from data
  + a.k.a. data mining or predictive analytics
  + used in 
    + Web search
    + spam filters
    + recommender systems
    + ad placement
    + credit scoring
    + fraud detection
    + stock trading
    + drug design
    + and many other applications

+ Classifier
  + a system
  + input (typically): a vector of discrete and/or continuous feature values
  + output: a single discrete value, the class
  + example: spam filter to classify email messages
    + input: a Boolean vector $\vec{x} = (x_1, \dots, x_j, \dots, x_n)$ where $x_j = 1$ if $i$th word in the dictionary appears in in the mail and $x_j = 0$ otherwise
    + _learner_ w/ a _training set_ of _examples_ $(\bf{x}_i, y_i)$ as input
    + $\bf{x_i} = (x_{i, 1}, \dots, x_{i, d})$: an observation input
    + $y_i$: the corresponding output and the output of a classifier
    + $\bf{x_t}$:  the test input of the learner


## 2. Learning = Representation + Evaluation + Optimization

+ Components of machine learning
  + representation
    + formal language to represent the classifier
    + hypothesis space: the set of classifiers that it can possible learner
    + how to represent the inputs
  + evaluation
    + evaluation function / objective function / scoring function
    + distinguishing good classifiers
    + used internally or externally by the algorithm
    + external one used to to optimize
  + optimization
    + methods to search the classifiers for the highest-scoring one
    + key to choose:
      + the efficiency for the learner
      + determining the classifier produced if more then one optimum

+ Common learning algorithms for components
  + Representation
    + Instances
      + $K$-nearest neighbor: finding the $k$ most similar training examples and predicting the majority class among them
      + Support vector machines
    + Hyperplanes: a linear combination of the features per class and predict the class with the highest-valued combination
      + Naive Bayes
      + Logistic regression
    + Decision trees: testing one feature at each internal node, w/ one branch for each feature value, and having class predictions at the leaves
    + Set of rules
      + Propositional rules
      + Logic programs
    + Neural networks
    + Graphical models
      + Bayesian networks
      + Conditional random fields
  + Evaluation
    + Accuracy/Error rate
    + Precision and recall
    + Squared error
    + Likelihood
    + Posterior probability
    + Information probability
    + Information gain: the mutual information btw feature $x_j$ and the class $y$
    + K-L divergence
    + Cost / Utility
    + Margin
  + Optimization
    + Combination optimization
      + Greedy search: testing feature $x$ w/ $c_0$ as the child for $x = 0$ and $c_1$ as the child for $x = 1$
      + Beam search
      + Branch-and-bound
    + Continuous optimization
      + Uncostrained
        + Gradient descent
        + Conjugate gradient
        + Quasi-Newton methods
      + Constrained
        + Linear programming
        + Quadratic programming

+ Example of learning algorithm: Decision Tree induction

  <span style="font-size: 1.3em; padding-bottom: 2.0em;"><b>Algorithm 1 LearnDT</b>(<i>TrainSet</i>)</span><br/>
  <b>if</b> all examples in <i>TrainSet</i> have the same class $y_*$ <b>then</b><br/>
  <b style="padding: 1em;">return</b> MakeLeaf($y_*$)<br/>
  <b>if</b> no feature $x_j$ has InfoGain($x_j, y) > 0$ <b> then</b><br/>
  <span style="padding: 1em;">$y_* \gets$ Most frequent class in $i$TrainSet</i><br/>
  <b style="padding: 1em;">return</b> MakeLeaf(y_*)<br/>
  $x_* \gets {\operatorname{argmax}}_{x_j}$ InfoGain($x, y$)<br/>
  $TS_0 \gets$ Examples in $TrainSet$ with $x_* = 0$<br/>
  $TS_1 \gets$ Examples in $TrainSet$ with $x_* = 1$<br/>
  <b>return</b> MakeNode($x_*$, LearnDT($TS_0$), LearnDT($TS_1$))


## 3. It's Generalization that Counts

+ Generalization for machine learning
  + fundamental goal: generalization beyond the example in the training set
  + common mistake: test on the training data to get illusion of success $\to$ keep some of the data and test the classifier
  + contamination of classifier by the test data in insidious ways $\gets$ using test data to true parameter
  + cross-validation
    + randomly dividing training data into (say) 10 subsets
    + holding out each one while training on the reset
    + testing each learned classifier on the examples not seen
    + averaging out the results to see how well the particular parameters setting does
  + generalization as goal
    + an interesting consequence for machine learning
    + unable to access to the function for optimization
    + using training error as a surrogate for test error
    + fraught w/ danger
  + objective function
    + only a proxy for the true goal
    + no need to fully optimize it
    + a local optimum returned by simple greedy search probably better than the global optimum

## 4. Data Alone is not Enough




## 5. Overfitting Has Many Faces




## 6. Intuition Falls in High Dimensions





## 7. Theoretical Guarantees are not What They Seem




## 8. Feature Engineering is the Key




## 9. More Date Beats a Clever Algorithms




## 10. Learn Many Models Not Just One




## 11. Simplicity Does Not Imply Accuracy




## 12. Presentable Does Not Imply Learnable




## 13. Correlation Does Not Imply Causation



## 14. Conclusion 



## 15 References





