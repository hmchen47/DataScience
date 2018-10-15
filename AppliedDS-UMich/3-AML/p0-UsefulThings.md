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
+ The objective function is only a proxy for the true goal -> no need to fully optimize it; in fact, a local optimum returned by simple greedy search may be better than the global optimum.
+ 


## Data alone is not enough




## Overfitting has many faces




## Intuition fails in high dimensions




## Theoretical guarantees are not what they seem




## Feature engineering is the key




## More data beats a cleaverer algorithm




## Learn many models, not just one




## Simplicity does not imply accuracy




## Represntable does not imply learnable




## Correlation does not imply causation



