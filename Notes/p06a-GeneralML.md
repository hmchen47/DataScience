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

  <span style="font-size: 1.2em; padding-bottom: 2.4em;"><b>Algorithm 1 LearnDT</b>(<i>TrainSet</i>)</span><br/>
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

+ Data along not enough
  + another major consequence of generalization as goal
  + every learner must embody some knowledge or assumptions beyond the data
  + no free lunch theorems: no learner able to beat random guessing over all possible functions to be learned
  + general assumptions
    + including smoothness, similar examples having similar classes, limited dependences, or limited complexity
    + often enough to do very well
    + large part of why machine learning so successful
  
+ Induction
  + a knowledge lever
  + turning a small amount of input knowledge into a large amount of output knowledge
  + a vastly more powerful lever than deduction
  + requiring much less input knowledge to produce useful results
  + still requiring more than zero input knowledge to work
  + the more put in, the more to get out

+ Key criteria for choosing a representation
  + which kinds of knowledge easily to express in it
  + examples
    + a lot of knowledge about what makes examples similar in our domain $\implies$ instance-based methods
    + knowledge about probabilistic dependencies $\implies$ graphic models w/ a good fit
  + knowledge about what kinds of preconditions required by each class $\implies$ "ID ... THEN ..." rules as the best option
  + most useful learners
    + no assumptions hard-wired into them
    + allowing to state assumptions explicitly and varying them widely
    + incorporating assumptions automatically into the learning
  
+ Domain knowledge
  + machine learning:
    + unable to get something from nothing
    + get more from less
  + programming: build everything from scratch
  + learning as farming
    + let nature doing more of the work
    + combining knowledge w/ data to grow programs

## 5. Overfitting Has Many Faces

+ Overfitting
  + running the risk of just hallucinating a classifier (or parts of it)
  + simply encoding random quirks in the data
  + the bugbear of machine learning
  + decomposing generalization error into bias and variance

+ Bias and variance
  + bias: a learner's tendency to consistently learn the same wrong thing
  + variance: the tendency to learn random things irrespective of real signal
  + linear learner w/ high bias:
    + the frontier btw two classes not a hyperplane
    + unable to induce it
  + decision tree
    + no bias problem
    + able to represent any Boolean function
    + suffering from high variance
    + learned on different training sets generated by the same phenomenon often very different
    + the choice of optimization methods
      + beam search w/ lower bias but higher variance than greedy search
      + more powerful learner not necessary better than a less powerful one

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
      onclick="window.open('https://sites.astro.caltech.edu/~george/ay122/cacm12.pdf')"
      src    ="https://tinyurl.com/y6feccbu"
      alt    ="Bias and variance in dart-throwing"
      title  ="Bias and variance in dart-throwing"
    />
  </figure>

+ Example: Naive Bayes vs. C4.5 rules
  + powerful learner not necessary better than a less powerful one

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
      onclick="window.open('https://sites.astro.caltech.edu/~george/ay122/cacm12.pdf')"
      src    ="https://tinyurl.com/y42xcchl"
      alt    ="Naive Bayes can outperform a state-of-the-art rule learner (C4.5 rules) even when the true classifier is a set of rules."
      title  ="Naive Bayes can outperform a state-of-the-art rule learner (C4.5 rules) even when the true classifier is a set of rules."
    />
  </figure>

+ Solving overfitting
  + Cross-validation: helping to combat overfitting
  + adding regularization term to the evaluation function
    + penalizing classifiers w/ more structure
    + favoring smaller ones w/ less room to overfit
    + performing a statistical significant test before adding new structure
    + deciding whether the distribution of the class really different w/ and w/o the structure
  + skeptical claim: solving the overfit problem
    + easy to avoid overfitting (variance) by falling into the opposite error of underfitting (bias)
    + no single technique always doing best

+ Common misconception
  + overfitting caused by noise, like training example labeled w/ the wrong class
  + sever overfititng possibly occurring even in the absence of noise
  + classifier
    + a Boolean formula in disjunctive normal form
    + each term as the conjunction of the feature values of one specific training example
    + getting all the training examples right and every positive test example wrong

+ Multiple testing
  + closely related overfitting
  + standard statistical tests
    + assumption: only one hypothesis being test
    + modern learner easily test millions before they are done
  + combated by correcting the significance tests to take the number of hypotheses into account but leading to underfitting
  + false discovery rate: controlling the fraction of falsely accepted non-null hypotheses


## 6. Intuition Falls in High Dimensions

+ Curse of dimensionality
  + many algorithms working find in low dimensions but intractable w/ high dimesions
  + generalizing correctly becomes exponentially harder as the dimensionality (number of features) of the examples grows
  + a fixed-size training set covers a dwindling fraction of the input space
  + machine learning algorithms depending on (explicitly or implicitly) breaks down in high dimensions
    + $x_1 \wedge x_2$: nearest neighbor classifier w/ Hamming distance as the similarity measure
    + example
      + $\exists \;x_3, \dots, x_{100}$
      + the noise from them completely swamps the signal the signal in $x_1$ and $x_2$
      + nearest neighbor effectively making random predictions
  + all 100 feature relevant
    + $\bf{x_t}$: a test example
    + $d$-dimensional grid, $\bf{x_t}$'s $2d$ nearest example at the same distance from it
    + dimensionality increasing, more and more example become nearest neighbors of $\bf{x_t}$, until the choice of nearest neighbor is effectively

+ Only one instance of a more general problem w/ high dimensions
  + institution on 3-dimension world often not applying in high-dimensional ones
  + high dimension
    + most of the mass of a multivariate Gaussian distribution not near the mean, but in an increasingly distant "shell" around it
    + most of the volume of a high-dimensional orange in the skin, not the pulp
  + constant number of examples distributed uniformly in high-dimensional hypercube
  + beyond some dimensionality most example closer to a face of hypercube than to their nearest neighbor
  + approximating a hyper-sphere by inscribing it in a hypercube
  + almost all the volume of the hypercube outside the hyper-sphere
  + disadvantage for ML: shapes of one type often approximated by shapes of another

+ High-dimensional classifier
  + find a reasonable frontier btw examples of different classes just by visual inspection
  + hard to understand what is happening $\to$ difficult to design a good classifier
  + gathering more features never hurt but probably outweighted by the curse of dimensionality

+ Blessing of non-uniformity
  + effect partly counteracting the curse
  + not spreading uniformly throughout the instance space
  + contracted on or near a lower-dimensional manifold
  + example
    + $k$-nearest neighbor working quite well for handwritten digit recognition
    + images of digits w/ one-dimension per pixel
    + the space of digit images much smaller than the space of all possible images
  + learners
    + implicitly taking advantage if this lower effective dimension
    + explicitly reducing the dimensionality used


## 7. Theoretical Guarantees are not What They Seem




## 8. Feature Engineering is the Key




## 9. More Date Beats a Clever Algorithms




## 10. Learn Many Models Not Just One




## 11. Simplicity Does Not Imply Accuracy




## 12. Presentable Does Not Imply Learnable




## 13. Correlation Does Not Imply Causation



## 14. Conclusion 



## 15 References





