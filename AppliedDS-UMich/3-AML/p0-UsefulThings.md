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



