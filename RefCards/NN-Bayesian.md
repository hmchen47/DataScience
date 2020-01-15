# Bayesian Approach got Neural Networks

## Overview

+ [The Bayesian framework](../ML/MLNN-Hinton/09-Overfitting.md#94-introduction-to-the-bayesian-approach)
  + instead of looking for the most likely setting of the parameters of a model, consider all possible settings of the parameters
  + trying to figure out for each of these possible settings have probabilities given the data we observed
  + assumption: always have a prior distribution for everything
    + the prior may be very vague
    + with given data, combine the prior distribution w/ a likelihood term to get a posterior distribution
  + likelihood term: how probable the observed data is given the parameters of the model
    + flavor parameter settings that make the data likely
    + fight the prior
    + always win w/ enough data: even w/ the wrong prior but end up w/ the right hypothesis if awful of data

+ [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) (MLE)
  + a method of estimating the parameters of a probability distribution by maximizing a likelihood function
  + the assumed statistical model: the observed data is most probable
  + maximum likelihood estimate: the point in the parameter space that maximizes the likelihood function
  + Bayesian inference: a special case of maximum a posteriori estimation (MAP) that assumes a uniform prior distribution of the parameters
  + frequentist inference: a special case of an extremum estimator, with the objective function being the likelihood

+ [Bayes Theorem](../ML/MLNN-Hinton/09-Overfitting.md#94-introduction-to-the-bayesian-approach)
  + equivalent expression for the join probability
  
    \[ p(D)p(W|D) = p(D, W) = p(W)p(D|W) \]

    + $p(D, W)$: join probability with a set of parameters $W$ and some data $D$
    + for supervised learning, the data is going to consist of the target values
    + $p(W|D), p(D|W)$: conditional probability

  + Bayes theorem

    \[ p(W|D) = \frac{p(W) p(D|W)}{p(D)} \]

    + $p(W|D)$: posterior probability of weight vector $W$ given training data $D$
    + $p(W)$: prior probability of weight vector $W$
    + $p(D|W)$: probability of observed data given $W$
    + $p(D) = \int_W p(W)p(D|W)$




