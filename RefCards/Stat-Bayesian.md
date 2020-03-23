# Bayesian Approaches

## Overview

+ [Two competing philosophies of statistical analysis](../Notes/a05-Bayesian.md#notes)
  + the frequentist
  + the Bayesian

+ [Bayesian methods](../Notes/a05-Bayesian.md#notes)
  + based on the idea of unknown quantities w/ probability distributions
  + unknown quantities including population means and proportions
  + prior knowledge / belief
    + the probability distribution proportion
    + the knowledge from data before knowing it

+ [Frequentist vs Bayesian methods](../Notes/a05-Bayesian.md#notes)
  + frequentist methods
    + the population value: fixed, unvarying (but unknown) quantity
    + w/o a probability distribution
    + calculating confidence intervals for the quantity or significance tests of hypotheses concerning it
  + Bayesian methods
    + not allowing to widen knowledge of the problem
    + not providing what researchers seem to want
    + able to provide
      + the probability of the 95% of the population value lies within the 95% CI
      + the probability of truth of the null hypothesis less than 5% $\to p(H_0 = \text{true}) < 5\%$

+ [Issues of the Bayesian methods](../Notes/a05-Bayesian.md#notes)
  + how to decide on the prior distribution
  + intractable computational problems
  + choice of Bayesian or frequentist: unknown which existed

+ [Conclusions for computational issue](../Notes/a05-Bayesian.md#notes)
  + developing computer intensive methods of analysis
  + new approaches to very difficult statistical problems, such as the location of geographical clusters of cases of a disease
  + a change in the statistical paradigm


## Bayes Theorem



## Prior Distribution

+ [The prior distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the prior probability of a random event or an uncertain proposition: the unconditional probability assigned before any relevant evidence is taken into account
  + methods to create prior
    + determined from past information, such as previous experiments
    + elicited from the purely subjective assessment of an experienced expert
    + (uninformative) created to reflect a balance among outcomes when no information is available
    + chosen according to some principle, such as symmetry or maximizing entropy given constraints
    + (conjugate) chosen a prior from a family simplifies calculation of the posterior distribution



## The Likelihood

+ [Likelihood function / likelihood](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + measuring the goodness of fit of a statistical model to a sample of data for given values of the unknown parameters
  + describing a hypersurface whose peak represents the combination of model parameter values that maximize the probability of drawing the sample obtained
  + maximum likelihood estimation: a procedure for obtaining the arguments of the maximum of the likelihood function
  + Definition of a parameterized model: the likelihood function

      \[ \theta \to f(x | \theta) \implies \mathcal{L}(\theta | x) = f(x | \theta) \tag{Likelihood} \]

  + Definition for continuous distribution

    \[\text{argmax}_\theta \mathcal{L}(\theta, x_j) = \text{argmax}_\theta f(x_j | \theta) \]

+ [Likelihoods](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + statistical inference:
    + learning about the assumed underlying distribution of quantities observed
    + generally carried out by assuming that the probability distributions follow a particular _parametric_ form $p(y | \theta)$
    + the distribution of $Y$ depends on some currently unknown parameter $\theta$
  + Bayesian inference: considered as random variables but the usual convention of capital and lower-case letters is ignore, to no apparent detriment
  + likelihood $p(Y | \theta)$:
    + once data $y$ observed, a function of $\theta \to$ extend to which different values $\theta$ are supported by the data
    + summarizing all the information that the data $y$ able to provide about the parameter $\theta$
  + any function of $\theta$ proportional to $p(y|\theta)$ can be considered as the likelihood
  + likelihood function: the relative plausibility of different values of $\theta$
  + maximum likelihood estimate: with the value of $\theta$ for which the likelihood is a maximum
  + using a range of values which are _best_ supported by the data as an interval estimate for $\theta$
  + a reasonable range defined by values of the likelihood above $\exp(-1.96^2/2) = 14.7\%$ of the maximum value
  + in practice, constructing intervals in such a manner is laborious, and in general approximate likelihood functions by the normal distribution
  + example: Bernoulli
    + $n$ individuals and measured the $i$-th individual responds to treatment, $Y_i = 1$, or not, $Y_i = 0$
    + assume a set of independent Bernoulli trials such that the probability of response is $\theta$
    + joint distribution for all $n$ individuals

      \[\begin{align*}
        p(y_1, \dots, y_n|\theta) &= \prod_{i=1}^n p(y_i|\theta) =  \prod_{i=1}^n \theta^{y_i}(1 - \theta)^{1 - y_i} \\
          &= \underbrace{\theta^{y_1+\dots+y_n} (1 - \theta)^{(1-y_1)+\dots+(1-y_n)} = \theta^r (1-\theta)^{n - r}}_{y_1, \dots, y_n \text{ independent}}
      \end{align*}\]

      + $r = \sum_i y_i$: the number of responders
    + likelihood maximized at $\hat{\theta} = r/n$
    + the maximum likelihood estimate proportional to responders



## Posterior Distribution

+ [The posterior distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + a way to summarize what we know about uncertain quantities in Bayesian analysis
  + summarizing what you know after the data has been observed

    \[ \text{Posterior Distribution} = \text{Prior Distribution} + \text{Likelihood Function (“new evidence”)} \]

  + Posterior probability: the probability that an event will happen after all evidence or background information has been taken into account





## The Likelihood Principles





## Hierarchical Models





