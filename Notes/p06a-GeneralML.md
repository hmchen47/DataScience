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

+ Theoretical guarantees
  + a bound on the number of examples needed to ensure good generalization
  + induction traditionally contracted w/ deduction
    + deduction: guarantee that the conclusions are correct
    + induction: guarantee to settle for probabilistic 

+ Bad classifier
  + true error rate greater than $\varepsilon$
  + the probability consistent w/ $n$ random, independent training examples less than $(1 - \varepsilon)^n$
  + $b$: the number of the bad classifiers in the learner's hypothesis space $H$
  + the probability that at least one of them is consistent $ < b(1 - \varepsilon)^n$ by the union bound
  + assumption: the learner returning a consistent classifier
  + the probability less than $|H|(1 \varepsilon)^n$, where $b \le |H|$
  + the probability less than $\delta$, $n > \ln(\delta / |H|)/\ln(1 - \varepsilon) \ge \frac{1}{\varepsilon} \left(\ln(|H| + \ln \frac{1}{\delta})\right)$
    + the required number of examples only grows logarithmically with $|H|$ and $1/\delta$
    + most interesting hypothesis spaces: doubly exponential in the number of features $d$
    + the bounds extremely loose
  + Boolean functions
    + the space of Boolean functions of $d$ Boolean variables
    + $e$ possible different examples $\implies 2^e$ possible different functions
    + $2^d$ possible different examples $\implies 2^{2^d}$ possible different functions
    + hypothesis spaces "merely" exponential
    + the bound still very loose
  + example
    + 100 Boolean features
    + the hypothesis space: decision trees w/ up to 10 levels
    + to guarantee $\delta = \varepsilon = 1\% \to$ half a million examples
    + in practice, a small fraction of the suffices for accurate 

+ Size of training set
  + given large enough training set, high probability  w/ learner
    + returning a hypothesis generalized well, or
    + unable to find a consistent hypothesis
  + bound not providing way to select a good hypothesis
  + the hypothesis space containing the true classifier $\implies$ lower probability w/ a bad classifier as training size increased
  + shrinking hypothesis space $\implies$ bound improving but shrinking the chance to contain the true classifier

+ Asymptotic guarantee
  + theoretically w/ infinite data, the learner guaranteed to output the correct classifier
  + rash to choose one learner over another because of its asympotic gaurantee
  + in practice, seldom in the asymptotic regime due to bias-variance tradeoff
  + learn A better than learner B given infinite data while learner B better than learner A given finite data

+ Role of theoretical guarantee
  + not as criterion for practical decision but as a source of understanding and driving for algorithm design
  + the close interplay of theory and practice makes rapid progress of ML
  + caveat emptor:
    + def: the principle that the buyer alone is responsible for checking the quality and suitability of goods before a purchase is made
    + learning is a complex phenomenon
    + learner w/ a theoretical justification and working in practice
    + not meant theory explaining the practice

## 8. Feature Engineering is the Key

+ Project time consuming
  + big portion:
    + gather, integrate, clean, and pre-process data
    + trial and error in feature design
  + little portion: machine learning
  + iterative process rather than one-shot process
    + running the learner
    + analyzing the results
    + modifying the data and/or the learner
    + repeating

+ Feature engineering
  + raw data not in the form amenable to learning but able to construct features from it
  + intuition, creativity and "black art" to select features
  + difficult due to domain-specific
  + learner largely general-specific
  + most useful learners facilitating incorporating knowledge.

+ Automatic feature engineering process
  + automatically generating large number of candidate features and selecting the best set by their info gain w.r.t the class
  + features irrelevant in isolation probably relevant in combination
  + learner w/ a very large number of feature
    + time-consuming
    + overfitting

## 9. More Date Beats a Clever Algorithm

+ Algorithms vs. Data
  + assumption: constructed the best set of featrues
  + issue: still not accurate
  + main choices of solutions
    + design a better learning algorithm
    + gather more data
  + researchers mainly concerned w/ the algorithms
  + pragmatically the quickest path to success often just get more data
  + rule of thumb: a dumb algorithm w/ lots of lots of data beats a clever one w/ modest amounts of it

+ Scalability
  + main resources of computer science: time & memory
  + main resources of ML: time, memory & data
  + paradox:
    + 1980's: data
    + 2010's: time
    + principle: more data $\to$ more complex classifiers
    + practice: simpler classifiers used because complex ones take too long to learn

+ Clever algorithms
  + first approximation doing all the same
  + proportional rules
    + readily encoded as neural networks
    + similar relationship hold btw other representations
  + learners
    + working by grouping nearby examples into the same class
    + key difference: the meaning of "nearby"
    + non-uniformly distributed data
      + producing widely different frontiers
      + still making the same predictions in the regions that matter
  + explaining why powerful learners unstable but still accurate

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick="window.open('https://sites.astro.caltech.edu/~george/ay122/cacm12.pdf')"
        src    ="https://tinyurl.com/yxu6khcf"
        alt    ="Very different frontiers can yield similar class predictions. (+ and - are training examples of two classes.)"
        title  ="Very different frontiers can yield similar class predictions. (+ and - are training examples of two classes.)"
      />
    </figure>

+ Rule to select the algorithms
  + trying the simplest learners first
    + naive Bayes before logistic regression
    + $k$-nearest neighbor before support vector machine
  + more sophisticated learners
    + seductive but usually harder to use
    + the curse of dimensionality: no existing amount of data enough
  + clever algorithms
    + making the most of the data and computing resource available
    + no sharp frontier btw designing learners and learning classifiers
    + any given piece of knowledge encoded in the learner and learned from data

+ Types of learners
  + fixed-size
    + parametric learners
    + e.g., linear classifier
  + variable-size
    + non-parametric learners
    + e.g., decision trees
    + principle: learning any function given sufficient data
    + practice: probably unable to learn because of limitations of the algorithm
      + e.g., greedy search falls into local optima

+ Bottlenecks of ML
  + biggest bottleneck: human cycles
  + typical evaluation: measures of accuracy & computational cost
  + more important:
    + human effort saved and insight gained
    + human-understandable output
  + organizations
    + in place and infrastructure to make experimenting w/ many learners, data sources and learning problems 
    + a close collaboration btw ML experts adnd applications domain ones

## 10. Learn Many Models Not Just One

+ Model ensembles
  + systematic empirical comparisons
    + the best learner varying from application to application
    + systems containing many different learners started to appear
  + instead of selecting the best variation found, combine many variations w/ little extra effort and much better results
  + techniques
    + _bagging_
      + simply generating random variations of the training set by resampling
      + learning a classifier on each
      + combining the results by voting
    + _boosting_
      + training examples w/ weights
      + weights varied: each new classifier focuses on the examples $\to$ the previous one tends to get wrong
    + _stacking_: the outputs of individual classifiers as the inputs of a "higher-level" learner

+ Bayesian model averaging (BMA)
  + the theoretically optimal approach to learning
  + predictions on new examples made by averaging the individual predictions of all classifiers in the hypothesis space
  + weighted by
    + how well the classifier explaining the training data
    + how much we believe in them a _priori_

+ Model ensembles vs. BMA
  + very different
  + ensembles
    + changing the hypothesis space
    + e.g., from single decision trees to linear combinations of them
    + taking a wide variety of forms
    + weights fairly even w/ the single highest-weight classifier
    + a key part of the ML toolkit
  + BMA
    + averaging weights to the hypotheses in the original space according to a fixed formula
    + weights extremely different from those produced by bagging or boosting
    + weights extremely skewed w/ the single highest-weight classifier
    + making effectively to just selecting the high-weighted classifier
    + seldom worth the trouble

## 11. Simplicity Does Not Imply Accuracy

+ Occam's razor
  + the problem-solving principle that "entities should not be multiplied without necessity"
  + the simplest explanation is usually the right one

+ Simplicity of learner
  + given two classifiers w/ the same training error
  + the simpler of the two likely w/ the lowest error
  + many counter examples
    + model ensembles: generalization error of a boosted ensemble continues to improve by adding classifiers even after the training error reaching zero
    + support vector machine: able to effectively have an infinite number parameters w/o overfitting
  + function $sign(\sim(ax))$:
    + discriminating an arbitrary large, arbitrary labeled set of points on the $x$ axis
    + even through only one parameter
  + survey:
    + simpler hypotheses preferred
    + simplicity is a virtue in its own right
    + not because of a hypothetical connection w/ accuracy

+ Complexity w/ the size of the hypothesis space
  + smaller spaces allowing hypotheses to be represented by shorter codes
  + bounds viewed as implying that shorter hypotheses generalize better
  + assigning shorter codes to the hypothesis in the space w/ some _a priori_ reference for
  + circular reasoning:
    + "proof" of a tradeoff btw accuracy & simplicity
    + preferred simpler design accurate
      + preferred one accurate
      + not the hypotheses "simple"

+ Complication of simplicity
  + few learners search exhaustively the hypothesis space
  + trying fewer hypotheses from hypothesis space less likely to overfit than trying a more hypotheses from a smaller space
  + the size of hypothesis space only a rough guide
  + relating training and test error really matters

## 12. Presentable Does Not Imply Learnable

+ Representable and Learnable
  + representations in variable-size learners
    + every function able to be represented, or approximated arbitrarily closely, using this representation
    + often ignore all others
  + representable not meant learnable
    + eg, standard decision tree learners unable to learn tree s w/ more leaves than there are training examples
  + in continuous space representing even simple functions using a fixed set of primitives often requires an infinite number of components
  + learner unable to find true function even representable if hypothesis space w/ many local optima of the evaluation function
  + standard learner only able to learn a tiny subset of all possible  w/ given finite data, time & memory
  + key question:
    + can it be learned?
    + not "can it be represented?" $\gets$ trivial
  
+ Compact representation
  + some representations exponentially more compact than others for some functions
  + $therefore$ exponentially less data required to learn those functions
  + many learners formed by combining simple basis functions
    + eg, SVM combining kernels centered at some of the training examples
  + representing parity of $n$ bits requiring $2^n$ basis functions
  + using a representation w/ more layers (i.e., more steps btw input and output), parity able to encoded in a linear classifier
  + finding methods to learn these representations as one of the major research frontiers

## 13. Correlation Does Not Imply Causation

+ Correlation and Causation
  + correlation not implying causation
  + learned correlation often treated as representing causal relations
  + the goal of learning predictive models used correlation as guides to action
  + observational data: the predictive variable not under the control of the learner
  + experimental data: the predictive variable under the control of the learner
  + some learning algorithms able to extract the causal info from observational data
  + the applicability limited
  + correlation as sign of potential causal connection
  + using correlation as a guide for further investigation

+ Causality only a convenient fiction
  + existence of causality: a deep philosophical question w/ no definite answer in sight
  + practical points of ML
    + whether called "cause" or not, predicting the effect og yje action not just correlation btw observational variables
    + obtain experimental data as possible

## 14 Conclusion

+ Further studies
  + [Pedro Domingos web page](https://homes.cs.washington.edu/~pedrod/)
  + [Predro Domingos online course](http://www.cs.washington.edu/homes/pedrod/class)
  + I. Witten, E. Frank, and M. Hall. Data Mining: Practical Machine Learning Tools and Techniques


## 15 References

1. E. Bauer and R. Kohavi. An empirical comparison of voting classification algorithms: Bagging, boosting and variants. Machine Learning, 36:105–142, 1999.
1. Y. Bengio. Learning deep architectures for AI. Foundations and Trends in Machine Learning, 2:1–127, 2009. 
1. Y. Benjamini and Y. Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society, Series B, 57:289–300, 1995.
1. J. M. Bernardo and A. F. M. Smith. Bayesian Theory. Wiley, New York, NY, 1994.
1. A. Blumer, A. Ehrenfeucht, D. Haussler, and M. K. Warmuth. Occam’s razor. Information Processing Letters, 24:377–380, 1987.
1. W. W. Cohen. Grammatically biased learning: Learning logic programs using an explicit antecedent description language. Artificial Intelligence, 68:303–366, 1994.
1. P. Domingos. The role of Occam’s razor in knowledge discovery. Data Mining and Knowledge Discovery, 3:409–425, 1999.
1. P. Domingos. Bayesian averaging of classifiers and the overfitting problem. In Proceedings of the Seventeenth International Conference on Machine Learning, pages 223–230, Stanford, CA, 2000. Morgan Kaufmann. 
1. P. Domingos. A unified bias-variance decomposition and its applications. In Proceedings of the Seventeenth International Conference on Machine Learning, pages 231–238, Stanford, CA, 2000. Morgan Kaufmann.
1. P. Domingos and M. Pazzani. On the optimality of the simple Bayesian classifier under zero-one loss. Machine Learning, 29:103–130, 1997.
1. G. Hulten and P. Domingos. Mining complex models from arbitrarily large databases in constant time. In Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 525–531, Edmonton, Canada, 2002. ACM Press.
1. D. Kibler and P. Langley. Machine learning as an experimental science. In Proceedings of the Third European Working Session on Learning, London, UK, 1988. Pitman.
1. A. J. Klockars and G. Sax. Multiple Comparisons. Sage, Beverly Hills, CA, 1986.
1. R. Kohavi, R. Longbotham, D. Sommerfield, and R. Henne. Controlled experiments on the Web: Survey and practical guide. Data Mining and Knowledge Discovery, 18:140–181, 2009.
1. J. Manyika, M. Chui, B. Brown, J. Bughin, R. Dobbs, C. Roxburgh, and A. Byers. Big data: The next frontier for innovation, competition, and productivity. Technical report, McKinsey Global Institute, 2011.
1. T. M. Mitchell. Machine Learning. McGraw-Hill, New York, NY, 1997.
1. A. Y. Ng. Preventing “overfitting” of cross-validation data. In Proceedings of the Fourteenth International Conference on Machine Learning, pages 245–253, Nashville, TN, 1997. Morgan Kaufmann.
1. J. Pearl. On the connection between the complexity and credibility of inferred models. International Journal of General Systems, 4:255–264, 1978.
1. J. Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, Cambridge, UK, 2000.
1. J. R. Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufmann, San Mateo, CA, 1993.
1. M. Richardson and P. Domingos. Markov logic networks. Machine Learning, 62:107–136, 2006.
1. J. Tenenbaum, V. Silva, and J. Langford. A global geometric framework for nonlinear dimensionality reduction. Science, 290:2319–2323, 2000.
1. V. N. Vapnik. The Nature of Statistical Learning Theory. Springer, New York, NY, 1995.
1. I. Witten, E. Frank, and M. Hall. Data Mining: Practical Machine Learning Tools and Techniques. Morgan Kaufmann, San Mateo, CA, 3rd edition, 2011.
1. D. Wolpert. The lack of a priori distinctions between learning algorithms. Neural Computation, 8:1341–1390, 1996



