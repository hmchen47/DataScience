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

+ [The Bayesian approach](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + resting on an essentially _subjective_ interpretation of probability
  + allowed to express generic _uncertainty_ or _degree of belief_ about any unknown but potentially observable quantity
  + rules of probability
    + not assumed as self-evident
    + able to derived from 'deeper' axioms of reasonable behavior of an individual
  + probabilities _for_ events rather than probabilities _of_ events
  + a reflection of personal uncertainty rather than necessarily being based on future unknown events

+ Bayesian statistics: treating subjectivity with respect by placing it in the open and under the control of the consumer of data



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
  + using likelihoods to generate estimators $\to$ the maximum likelihood estimator

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



## Posterior Distribution

+ [The posterior distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + a way to summarize what we know about uncertain quantities in Bayesian analysis
  + summarizing what you know after the data has been observed

    \[ \text{Posterior Distribution} = \text{Prior Distribution} + \text{Likelihood Function (“new evidence”)} \]

  + Posterior probability: the probability that an event will happen after all evidence or background information has been taken into account



## The Likelihood Principles

+ [The likelihood principle](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + proposition: given a statistical model, all the evidence in a sample relevant to model parameters is contained in the likelihood function
  + all the information that the data provides about the parameter is contained in the likelihood
  + data only influence the relative plausibility of an alternative hypothesis through the relative likelihood
  + Bayesian inference automatically obeys this principle
  + example -- Stopping: The likelihood principle in action



## Odds Ratios

+ [Odds ratios](../Notes/a06-OddsRatios.md#what-is-an-odds-ratio) (OR)
  + Def: A measure of association btw an exposure and an outcome
  + the odds that an outcome will occur given a particular exposure, compared to the odds of the outcome occurring in the absence of that exposure
  + most commonly used in case control studies
  + able to be used in cross-sectional and cohort study designs
  + example: logistic regression

+ [Usage of OR](/Notes/a06-OddsRatios.md#when-is-it-used)
  + used to compare
    + the relative odds of the occurrence of the outcome of interest, eg, disease or disorder
    + given exposure to the variable of interest, eg, health characteristics, aspect of medical history
  + used to determine whether a particular exposure is a risk factor for a particular outcome, and to compare the magnitude of various risk factors for that outcome
    + $OR = 1$: exposure not affecting odds of outcomes
    + $OR > 1$: exposure associated w/ higher odds of outcome
    + $OR < 1$: exposure associated w/ lower odds of outcome

+ [confidence interval of OR](/Notes/a06-OddsRatios.md#what-about-confidence-intervals)
  + 95% confidence interval (CI): used to estimate the precision of the OR
    + large CI  $\implies$ low level of precision of the OR
    + small CI $\implies$ higher precision of the OR
  + 95% CI not measuring statistical significance
  + used as a proxy for the presence of statistical significance if not overlap the null value (eg, $OR=1$)
  + inappropriate to interpret OR w/ 95% CI that spans the null value as indicating  evidence for lack of association btw the exposure and outcome

+ [confounding](/Notes/a06-OddsRatios.md#confounding)
  + Def: non-casual association observed btw a given exposure, outcome as a result of the influence of a third variable
  + confounding variable: the third variable
    + causally associated w/ the outcome of interest
    + non-causally or causally associated w/ the exposure
    + not an intermediate variable in the causal pathway btw exposure and outcome
  + methods to address confounding
    + stratification: fixing the level of confounders and producing groups within which the confounder does not vary
    + multiple regression: adjusting for (accounting for) potentially confounding variables in the model

+ [Example: case control study](/Notes/a06-OddsRatios.md#example)
  + goal: calculating (a) ORs and (b) 95% CIs
  + Calculating ORs

    <table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
      <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Two-by-two frequency table</a></caption>
      <thead>
      <tr>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;"></th>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Outcome Status</th>
      </tr>
      <tr>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;"></th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">+</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">-</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td rowspan="2">Exposure status</td> <td style="text-align: center;">+</td> <td style="text-align: center;">a</td> <td style="text-align: center;">b</td> </tr>
      <tr> <td style="text-align: center;">-</td> <td style="text-align: center;">c</td> <td style="text-align: center;">d</td> </tr>
      </tbody>
    </table>

    + a = number of exposed cases
    + b = number of exposed non-cases
    + c = number of unexposed cases
    + d = number of unexposed non-cases

    \[\begin{align*}
      OR &= \frac{a/c}{b/d} = \frac{ad}{bc} \\\\
      OR &= \frac{(n) \text{ exposed cases } / (n) \text{ unexposed cases }}{(n) \text{ exposed non-cases }/(n) \text{ unexpected non-cases }} \\\\
         &= \frac{(n) \text{ exposed non-cases } \times (n) \text{ unexpected non-cases}}{(n) \text{ exposed cases } \times (n) \text{ unexposed cases }}
    \end{align*}\]

  + conclusion: important points from example
    + presence of a positive OR for an outcome given a particular exposure does not necessarily indicate that this association is statistically significant $\implies$ determined by the confidence intervals and $p$-value
    + overall, depression is strongly linked to suicidal and suicidal attempt w/ a particular size and composition, and in the presence of other variables, the association may not be significant




## Hierarchical Models





