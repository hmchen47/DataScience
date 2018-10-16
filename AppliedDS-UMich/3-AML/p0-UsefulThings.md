# A Few Useful Things to Know about Machine Learning

Author: Pedro Domingos
Organization: Department of Computer Science and Engineering, University of Washington

+ A __classifier__ is a system that inputs (typically) a vector of discrete and/or continuous _feature values_ and outputs a single discrete value, the _class_.

## Learning = Representation + Evaluation + Optimization

+ Combinations of just three components choosing learning algorithms
    + Representation
        + Formal language that the computer can handle
        + __hypothesis space__ of the learner: the set of classifiers that it can possibly learn
    + Evaluation
        + evaluation function (objective function or scoring function): to distinguish good classifiers from bad ones
        + used internally by the algorithm may differ from the external one that we want the classifier to optimize
    + Optimization
        + a method to search among the classifiers in the language for the highest-scoring one
        + choice of optimization technique: 
            + key to the efficiency of the learner
            + determining the classifier produced if the evaluation function has more than one optimum
        + Commonly start out using off-the-shelf optimizers, which are later replaced by custom-designed ones

+ Three components of learning algorithms
    + Representation
        + Instances: K-nearest neighbor, Support vector machines
        + Hyperplanes: Naive Bayes, Logistic regression
        + Decision trees
        + Sets of rules: Propositional rules, Logic programs
        + Neural networks
        + Graphical models: Bayesian networks, Conditional random fields
    + Evaluation
        + Accuracy/Error rate
        + Precision and recall
        + Squared error
        + Likelihood
        + Posterior probability
        + Information gain
        + K-L divergence
        + Cost/Utility
        + Margin
    + Optimization
        + Combinatorial optimization: Greedy search, Beam search, Branch-and-bound
        + Continuous optimization:
            + Unconstrained: Gradient descent, Conjugate gradient, Quasi-Newton methods
            + Constrained: Linear programming, Quadratic programming


## It's generalization that counts

+ __Fundamental Goal__ of machine learning: to _generalize_ beyond the examples in the training set
+ Most common mistake: to test on the training data and have the illusion of success
+ Contamination of classifier by test data, e.g. using test data to tune parameters and do a lot of tuning
+ __Cross-validation__: randomly dividing your training data into (say) ten subsets, holding out each one while training on the rest, testing each learned classifier on the examples it did not see, and averaging the results to see how well the particular parameter setting does
+ Notice that generalization being the goal has an interesting consequence for machine learning.
+ Objective function = proxy for the true goal -> no need to fully optimize it; in fact, a local optimum returned by simple greedy search may be better than the global optimum.


## Data alone is not enough

+ Generalization being the goal has another major consequence: data alone is not enough, no matter how much of it you have.
+ Every learner must embody some knowledge or assumptions beyond the data it’s given in order to generalize beyond it.
+ General assumptions—like smoothness, similar examples having similar classes, limited dependencies, or limited complexity—are often enough to do very well, and this is a large part of why machine learning has been so successful
+ Induction is a vastly more powerful lever than deduction, requiring much less input knowledge to produce useful results, but it still needs more than zero input knowledge to work.
+ One of the key criteria for choosing a representation is which kinds of knowledge are easily expressed in it.
    + Knowledge about what makes examples similar in our domain, instance-based methods may be a good choice.
    + Knowledge about probabilistic dependencies, graphical models are a good fit.
    + Knowledge about what kinds of preconditions are required by each class, “IF . . . THEN . . .” rules may be the the best option.
+ Programming, like all engineering, is a lot of work: we have to build everything from scratch.
+ Learning is more like farming, which lets nature do most of the work. Farmers combine seeds with nutrients to grow crops. Learners combine knowledge with data to grow programs.


## Overfitting has many faces

+ Decomposing generalization error into bias and variance - understand overfitting
    + Bias: a learner’s tendency to consistently learn the same wrong thing.
    + Variance: the tendency to learn random things irrespective of the real signal.
    <a href="http://deeplearning.lipingyang.org/category/machine-learning_tricks4better-performance/"> <br/>
        <img src="http://deeplearning.lipingyang.org/wp-content/uploads/2018/03/img_5a99cf35a18b5.png" alt="If lambda is small then we’re not using much regularization and we run a larger risk of over fitting whereas if lambda is large that is if we were on the right part of this horizontal axis then, with a large value of lambda, we run the higher risk of having a biased problem, so if you plot J train and J cv, what you find is that, for small values of lambda, you can fit the trading set relatively way cuz you’re not regularizing. So, for small values of lambda, the regularization term basically goes away, and you’re just minimizing pretty much just gray arrows. So when lambda is small, you end up with a small value for Jtrain, whereas if lambda is large, then you have a high bias problem, and you might not feel your training that well, so you end up the value up there. So Jtrain of theta will tend to increase when lambda increases, because a large value of lambda corresponds to high bias where you might not even fit your trainings that well, whereas a small value of lambda corresponds to, if you can really fit a very high degree polynomial to your data, let’s say. After the cost validation error we end up with a figure like this." title= "Bias and variance in dart-throwing" height="200">
    </a>
+ High bias: A linear learner because when the frontier between two classes is not a hyperplane the learner is unable to induce it.
+ High variance: decision trees learned on different training sets generated by the same phenomenon are often very different, when in fact they should be the same.
+ Beam search: lower bias than greedy search, but higher variance, because it tries more hypotheses.
+ Strong false assumptions can be better than weak true ones, because a learner with the latter needs more data to avoid overfitting.
+ To combat overfitting
    + Cross-validation, for example by using it to choose the best size of decision tree to learn
    + Adding a _regularization term_ to the evaluation function, for example, penalize classifiers with more structure, thereby favoring smaller ones with less room to overfit
    + Perform a statistical significance test like chi-square before adding new structure, to decide whether the distribution of the class really is different with and without this structure
+ Easy to avoid overfitting (variance) by falling into the opposite error of underfitting (bias)
+ Common misconception
    + overfitting caused by noise
    + severe overfitting can occur even in the absence of noise
    + For instance, Boolean classifier that is just the disjunction of the examples labeled “true” in the training set. The classifier is a Boolean formula in disjunctive normal form, where each term is the conjunction of the feature values of one specific training example. -> all the training examples right and every positive test example wrong, regardless of whether the training data is noisy or not.
+ __Multiple testing__ closely related to overfitting
    + modern learners can easily test millions before they are done
    + false discovery rate: control the fraction of falsely accepted non-null hypotheses


## Intuition fails in high dimensions

+ Curse of dimensionality: Bellman in 1961
    + Many algorithms that work fine in low dimensions become intractable when the input is high-dimensional
    + Generalizing correctly becomes exponentially harder as the dimensionality (number of features) of the examples grows, because a fixed-size training set covers a dwindling fraction of the input space.
+ Machine learning algorithms depend on (explicitly or implicitly) breaks down in high dimensions
    + Consider a nearest neighbor classifier with Hamming distance as the similarity measure, and suppose the class is just $x_1 \wedge x_2$.
    + 98 irrelevant features $x_3, \cdots , x_{100}$, the noise from them completely swamps the signal in $x_1$ and $x_2$, and nearest neighbor effectively makes random predictions.
    + all 100 features are relevant, Suppose that examples are laid out on a regular grid, and consider a test example $x_t$. If the grid is d-dimensional, $x_t$’s 2d nearest examples are all at the same distance from it. So as the dimensionality increases, more and more examples become nearest neighbors of $x_t$, until the choice of nearest neighbor (and therefore of class) is effectively random.
+ Intuitions from a three dimensional world, often do not apply in high-dimensional ones
    + most of the mass of a multivariate Gaussian distribution is not near the mean, but in an increasingly distant “shell” around it
    + constant number of examples is distributed uniformly in a high-dimensional hypercube, beyond some dimensionality most examples are closer to a face of the hypercube than to their nearest neighbor
+ High dimensions: hard to understand what is happening --> difficult to design a good classifier
+ Blessing of non-uniformity
    + most applications examples are not spread uniformly throughout the instance space, but are concentrated on or near a lower-dimensional manifold
    + For example, k-nearest neighbor works quite well for handwritten digit recognition even though images of digits have one dimension per pixel, because the space of digit images is much smaller than the space of all possible images.
    + Learners can implicitly take advantage of this lower effective dimension, or algorithms for explicitly reducing the dimensionality can be used


## Theoretical guarantees are not what they seem

+ Most theoretical guarantees: a bound on the number of examples needed to ensure good generalization
+ Induction is traditionally contrasted with deduction: in deduction you can guarantee that the conclusions are correct; in induction all bets are off.
+ Having guarantees on the results of induction, particularly if we’re willing to settle for probabilistic guarantees
    + Let’s say a classifier is bad if its true error rate is greater than $\epsilon$.
    + Then the probability that a bad classifier is consistent with n random, independent training examples is less than $(1 − \epsilon)^n$.
    + Let $b$ be the number of bad classifiers in the learner’s hypothesis space $H$. 
    + The probability that at least one of them is consistent is less than $b(1 − \epsilon)^n$, by the union bound. 
    + Assuming the learner always returns a consistent classifier, the probability that this classifier is bad is then less than $|H|(1 − \epsilon)^n$, where we have used the fact that $b \leq |H|$. 
    + So if this probability to be less than $\epsilon$, it suffices to make $n > \ln(\epsilon / |H|)/ \ln(1 − \epsilon) \geq \frac{1}{\epsilon} (\ln |H| + \ln \frac{1}{\epsilon})$.
+ The bounds obtained in this way are usually extremely loose
    + The required number of examples only grows logarithmically with $|H|$ and $1 / \epsilon$.
    + Most interesting hypothesis spaces are doubly exponential in the number of features $d$, which still leaves us needing a number of examples exponential in $d$
    + Consider the space of Boolean functions of d Boolean variables. If there are e possible different examples, there are $2^e$ possible different functions, so since there are $2^d$ possible examples, the total number of functions is $2^{2^d}$.
    + Even for hypothesis spaces that are “merely” exponential, the bound is still very loose, because the union bound is very pessimistic.
    + For example, if there are 100 Boolean features and the hypothesis space is decision trees with up to 10 levels, to guarantee $\delta = \esilon = 1\%$ in the bound above we need half a million examples.
+ Given a large enough training set, with high probability your learner will either return a hypothesis that generalizes well or be unable to find a consistent hypothesis.
+ The bound also says nothing about how to select a good hypothesis space. It only tells us that, if the hypothesis space contains the true classifier, then the probability that the learner outputs a bad classifier decreases with training set size.
+ Common type of theoretical guarantee is _asymptotic_: given infinite data, the learner is guaranteed to output the correct classifier.
+ In practice, seldom in the asymptotic regime (also known as “asymptopia”). 
+ Bias-variance tradeoff: if learner $A$ is better than learner $B$ given infinite data, $B$ is often better than $A$ given finite data.
+ Main role of theoretical guarantees: a source of understanding and driving force for algorithm design
+ _Caveat Emptor_: learning is a complex phenomenon, and just because a learner has a theoretical justification and works in practice doesn’t mean the former is the reason for the latter.


## Feature engineering is the key

+ Most important factor is the features used
+ Most of the effort: the raw data is not in a form that is amenable to learning, but you can construct features from it that are.
+ Time-consuming: gather data, integrate it, clean it and pre-process it, and how much trial and error can go into feature design.
+ Typical procedure: iterative process of running the learner, analyzing the results, modifying the data and/or the learner, and repeating.
+ Feature engineering is more difficult because it’s domain-specific, while learners can be largely general-purpose.
+ Current solution: automatically generating large numbers of candidate features and selecting the best by (say) their information gain with respect to the class. 
+ Features that look irrelevant in isolation may be relevant in combination.
+ For example, if the class is an XOR of k input features, each of them by itself carries no information about the class.


## More data beats a cleaverer algorithm

+ Two main choices for the best set of features but still not accurate enough:
    + design a better learning algorithm
    + gather more data (more examples, and possibly more raw features, subject to the curse of dimensionality)
+ Rule of thumb: a dumb algorithm with lots and lots of data beats a clever one with modest amounts of it
+ Scalability on resources: time, memory, and training data for ML
+ Even though in principle more data means that more complex classifiers can be learned, in practice simpler classifiers wind up being used, because complex ones take too long to learn. -> fast ways to learn complex classifiers
+ The meaning of “nearby" as key difference: all learners essentially work by grouping nearby examples into the same class
+ Powerful learners can be unstable but still accurate
+ Basic rule: try the simplest learners first, naive Bayes > logistic regression > k-nearest neighbor > support vector machines
    <a href="http://people.cs.vt.edu/liangzhe/slides/01-29-2015-elaheh.pdf"> <br/>
        <img src="images/fig2-47.png" alt="As a rule, it pays to try the simplest learners first (e.g., naive Bayes before logistic regression, k-nearest neighbor before support vector machines). More sophisticated learners are seductive, but they are usually harder to use, because they have more knobs you need to turn to get good results, and because their internals are more opaque" title= "Very different frontiers cna yield similar class predictions" height="200">
    </a>
+ More sophisticated learners are seductive, but they are usually harder to use, because they have more knobs you need to turn to get good results, and because their internals are more opaque.
+ Major types of learners:
    + fixed-size learners:
        + only take advantage of so much data
    + variable-size learners:
        + representation grows with the data
        + non-parametric learners
        + in principle learn any function given sufficient data --> limitations of the algorithm
+ clever algorithms: making the most of the data and computing resources available
+ Human cycles:
    + human effort saved and insight gained, although harder to measure, are often more important.
    + human-understandable output, e.g. rule sets
    + data sources and learning problems: easy and efficient
    + close collaboration between machine learning experts and application domain ones.


## Learn many models, not just one




## Simplicity does not imply accuracy




## Represntable does not imply learnable




## Correlation does not imply causation



