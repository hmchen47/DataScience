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




## Theoretical guarantees are not what they seem




## Feature engineering is the key




## More data beats a cleaverer algorithm




## Learn many models, not just one




## Simplicity does not imply accuracy




## Represntable does not imply learnable




## Correlation does not imply causation



