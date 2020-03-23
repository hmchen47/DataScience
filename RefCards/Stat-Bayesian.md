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



## Reporting



+ [Example 4](../Notes/p01-Bayesian.md#37-bayesian-analysis-with-normal-distributions) -- SBP: Bayesian analysis for normal data

+ [Example 6](../Notes/p01-Bayesian.md#38-point-estimation-interval-estimation-and-interval-hypotheses) -- GREAT (continue): Bayesian analysis of a trial of early thrombolytic therapy



## Prior Distribution

+ [The prior distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the prior probability of a random event or an uncertain proposition: the unconditional probability assigned before any relevant evidence is taken into account
  + methods to create prior
    + determined from past information, such as previous experiments
    + elicited from the purely subjective assessment of an experienced expert
    + (uninformative) created to reflect a balance among outcomes when no information is available
    + chosen according to some principle, such as symmetry or maximizing entropy given constraints
    + (conjugate) chosen a prior from a family simplifies calculation of the posterior distribution

+ [Characteristics of prior](../Notes/p01-Bayesian.md#39-the-prior-distribution)
  + not necessarily specified beforehand
  + not necessarily unique
  + not necessarily completely specified
  + not necessarily important



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

+ [The likelihood principle](../Notes/p01-Bayesian.md#33-comparing-simple-hypotheses-likelihood-ratios-and-bayes-factors)
  + the likelihoods contains all the relevant that can be extracted from the data
  + all the information that the data provide about the parameter is contained in the likelihood

+ [Bayes factor](../Notes/p01-Bayesian.md#33-comparing-simple-hypotheses-likelihood-ratios-and-bayes-factors) (BF)
  + measure of the relative likelihood of two hypotheses
  + small values being considered as both evidence _against_ $H_0$ and evidence _for_ $H_1$
  + transforming prior to posterior odds

+ [Calibration of Bayes factor](../Notes/p01-Bayesian.md#33-comparing-simple-hypotheses-likelihood-ratios-and-bayes-factors) (likelihood ratio)

  <table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="0" align="center" width=50%>
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf">Calibration of Bayes factor (likelihood ratio)</a></caption>
    <thead>
    <tr>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Bayes factor range</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:40%;">Strength of evidence in favour of $H_0$ and against $H_1$</th>
    </tr>
    </thead>
    <tbody>
    <tr style="text-align: center;"> <td> > 100 / < 1/100</td>       <td>Decisive</td> </tr>
    <tr style="text-align: center;"> <td>32 to 100 / 1/32 to 1/100</td>    <td>Very strong</td> </tr>
    <tr style="text-align: center;"> <td>10 to 32 / 1/10 to 1/3.2</td>     <td>Strong</td> </tr>
    <tr style="text-align: center;"> <td>3.2 to 10 1/3.2 to 1/10</td>     <td>Substantial</td> </tr>
    <tr style="text-align: center;"> <td>1 to 3.2 / 1 to 1/3.2</td>      <td>'Not worth more than a bare mention'</td> </tr>
    </tbody>
  </table>

+ Use of Bayes theorem: general statistical analysis
  + a parameter $\theta$ is an unknown quantity such as the mean benefit of a treatment on a specified patient
  + the prior distribution $p(\theta)$ needs to be specified
  + concern: a natural extension of the subjective interpretation of probability



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


## Exchangeability

+ [Exchangeability](../Notes/p01-Bayesian.md#34-exchangeability-and-parametric-modeling)
  + a formal expression of the idea that no systematic reason to distinguish the individual variables $Y_1, \dots, Y_n$ (similar but not identical)
  + exchangeable: the probability of $Y_1, \dots, Y_n$ assigned to any set of potential outcomes, $p(y_1, \dots, y_n)$, unaffected by permutations of the labels attached to the variables

+ [Judgment of exchangeability](../Notes/p01-Bayesian.md#34-exchangeability-and-parametric-modeling)
  + a set of binary variables $Y_1, \dots, Y_n$ exchangeable $\implies$ the marginal distribution

    \[ p(y_1, \dots, y_n) = \int \, \prod_{i=1}^{n} p(y_i | \theta) p(\theta) d\theta \tag{3} \]

  + exchangeable random quantities can be though of being i.i.d. variables drawn from some common distribution depending on an unknown parameter $\theta$ w/ a prior distribution $p(\theta)$
  + from a subjective judgment about observable quantities, one derives that whole apparatus of i.i.d. variables, conditional independence, parameters and prior distributions



## Bayesian Analysis

+ [Bayesian approach to make inference](../Notes/p01-Bayesian.md#36-bayesian-analysis-with-binary-data): combining the likelihood w/ initial evidence or opinion regarding $\theta$, as expressed in a prior distribution $p(\theta)$

+ [Prior to posterior analysis](../Notes/p01-Bayesian.md#32-bayes-theorem-for-two-hypotheses)
  + hypotheses $H_0$ and $H_1$: mutually exhaustive and exclusive
  + the prior probability for each of two hypotheses: $p(H_0)$ and $p(H_1)$
  + $y$: the result of a test
  + posterior probabilities:

    \[ p(H_0 | y) = \frac{p(y | H_0)}{p(y)} \times p(H_0)  \tag{1} \]

  + the overall probability of $y$ occurring:
  
    \[p(y) = p(y | H_0) p(H_0) + p(y | H_1) p(H_1) \]

  + the odds form of Bayes theorem

    \[\begin{align*}

      \frac{p(H_0 | y)}{p(H_1 | y)} &= \frac{p(y | H_0)}{p(y | H_1)} \times \frac{p(H_0)}{p(H_1)} \tag{2} \\
    \end{align*}\]

    + the prior odds: $p(H_0)/p(H_1)$
    + the posterior odds: $p(H_0 | y) / p(H_1 | y)$
    + the ratio of the likelihood: $p(y | H_0) / p(y | H_1)$

    \[\begin{align*}
      \text{posterior odds} &= \text{likelihood ratio} \times \text{prior odds} \\ \\
      \log(\text{posterior odds}) &= \log(\text{likelihood ratio}) + \log(\text{prior odds})
    \end{align*}\]

    + the weight of evidence: $\log(\text{likelihood ratio})$

+ [Bayesian approach](../Notes/p01-Bayesian.md#35-bayes-theorem-for-general-quantities)
  + Notations & Assumptions
    + $\theta$: unknown quantity
    + $p(\theta | H)$: the prior distribution of $\theta$; judgment about $\theta$ conditional on a context $H$
    + $y$: some observed evidence
  + $p(y | \theta)$: the (conditional) probability of $y$ for each possible value of $\theta$
  + $p(\theta | y)$: likelihood; to obtain the new, posterior, probability for different $\theta$, taking account of the evidence $y$
  + applying Bayesian theorem to a general quantity $\theta$

    \[ p(\theta | y) = \frac{p(y | \theta)}{p(y)} \times p(\theta)  \tag{4}\]

    + $p(y)$: a normalizing factor to ensure that $\int p(\theta|y)d\theta = 1$ and value not interested
  + the essence of Bayes theorem only concerns the terms involving $\theta$

    \[ p(\theta | y) \propto p(y | \theta) \times p(\theta) \tag{5} \]

  + the posterior distribution proportional to (i.e. has the same shape as) the product of the likelihood and the prior



## Bayesian Analysis w/ Binary Data

+ [Bernoulli distribution](../Notes/p01-Bayesian.md#361-binary-data-with-a-discrete-prior-distribution)
  + only a limited set of hypotheses concerning the true proportional $\theta$, corresponding to a finite list denoted $\theta_1, \dots, \theta_j$
  + the posterior probabilities for the $\theta_j$

    \[ p(\theta_j | y) \propto \theta_j^y (1 - \theta_j)^{1-y} \times p(\theta_j)  \tag{7} \]

    where the normalizing factor that ensures the posterior probabilities add to 1

    \[ p(y) = \sum_j \theta_j^y (1 - \theta_j)^{1-y} \times p(\theta_j) \]

  + the result w/ $r$ 'successes' out of $n$ trials, the relevant posterior

    \[ p(\theta_j | r) \propto \theta_j^r (1 - \theta_j)^{1-y} \times p(\theta_j) \tag{8} \]

+ [Uniform distribution/Notes/p01-Bayesian.md#362-conjugate-analysis-for-binary-data
  + assumption for prior distribution:
    + all possible values of $\theta$ equally likely
    + uniform distribution, $p(\theta) = 1 \, \text{ for } \,0 \leq \theta \leq 1$
  + applying Bayes theorem

    \[ p(\theta | y) \propto \theta^r (1-\theta)^{n-r} \times 1  \tag{9} \]

    + $r$: the number of events
    + $n$: the total number of individuals
  + the functional form of the posterior distribution proportional to a beta distribution: $Beta(r+1, n-r+1)$

+ [Beta distribution/Notes/p01-Bayesian.md#362-conjugate-analysis-for-binary-data): $Beta(a, b)$ for prior distribution
  
  \[\begin{align*}
    \text{Prior} &\propto \theta^{a-1} (1 - \theta)^{b-1} \\
    \text{Likelihood} &\propto \theta^r (1 - \theta)^{n-r} \\
    \text{Posterior} &\propto \theta^{a-1}(1 - \theta)^{b-1} \theta^r (1-\theta)^{n-r} \tag{10} \\
      &\propto \theta^{a+r-1}(1-\theta)^{b+n-r-1} = Beta(a+r, b+n-r)
  \end{align*}\]


+ [Beta-Binomial distribution w/ Bayesian considerations](../Notes/p01-Bayesian.md#3132-predictions-for-binary-data )
  + prior distribution: let $\mu = a/(a+b), M = a+b$

    \[\begin{align*}
      p(\theta | \mu, M) = Beta(M\mu, M(1-\mu)) &= \frac{\Gamma(M)}{\Gamma(M\mu) \Gamma(M(1-\mu))} \theta^{M\mu-1}(1-\theta)^{M(1-\mu)-1} \\\\
      E(\theta | \mu, M) = \mu \qquad & \qquad Var(\theta | \mu, M) = \frac{\mu(1-\mu)}{M-1}
    \end{align*}\]

  + posterior distribution

    \[\begin{align*}
      p(\theta | k) &\propto \underbrace{p(k | \theta)}_{\text{binomial}} \underbrace{p(\theta | \mu, M)}_{\text{beta}} = Beta(k+M\mu, n-k+M(1-\mu)) \\
        &= \frac{\Gamma(M)}{\Gamma(M\mu) \Gamma(M(1-\mu))} \begin{pmatrix} n \\ k \end{pmatrix} \theta^{k+M\mu-1} (1-\theta)^{n-k+M(1-\mu)-1} \\\\
      E(\theta | k) &= \frac{k+M\mu}{n+M}
    \end{align*}\]

  + marginal distribution

    \[\begin{align*}
      p(k | \mu, M) &= \int_0^1 p(k | \theta) p(\theta | \mu, M) d\theta \\\\
        &= \frac{\Gamma(M)}{\Gamma(M\mu)\Gamma(M(1-\mu))} \begin{pmatrix} n \\ k \end{pmatrix} \int_0^1 \theta^{k+M\mu-1}(1-\theta)^{n-k+M(1-\mu)-1} d\theta \\\\
        &= \frac{\Gamma(M)}{\Gamma(M\mu)\Gamma(M(1-\mu))} \begin{pmatrix} n \\ k \end{pmatrix} \frac{\Gamma(k+M\mu)\Gamma(n-k+M(1-\mu))}{\Gamma(n+M)} \\\\
      p(k | a, b) &= \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)}\frac{\Gamma(k+a)\Gamma(n-k+b)}{\Gamma(n+a+b)}\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
    \end{align*}\]



## Bayesian analysis with normal distributions

+ [Normal distribution](../Notes/p01-Bayesian.md#37-bayesian-analysis-with-normal-distributions)
  + the prior distribution $p(\theta)$

    \[ p(\theta) = N \left(\theta \left\vert \mu, \frac{\sigma^2}{n_0}\right.\right) \tag{13} \]

    + $\mu$: the prior mean
    + $\sigma$: the standard deviation for the prior and the likelihood
    + $n_0$: 'implicit' sample size that the prior based on
  + advantages of Eq.13 for prior-to-posterior analysis
    + $n_0 \to 0 \implies \sigma^2 \uparrow$ and the distribution becoming 'flatter'
    + the distribution $\to$ uniform over $(-\infty, \infty)$
    + normal prior w/ a very large variance used to represent a 'non-informative' distribution
  + posterior distribution
    + normal prior distribution: $\theta \sim N(\mu, \sigma^2/n_0)$
    + likelihood: $y_m \sim N(\theta, \sigma^2/m)$
    + posterior distribution obeys

      \[\begin{align*}
        p(\theta|y_m) &\propto p(y_m | \theta) p(\theta) \\
         &\propto \exp \left(-\frac{(y_m - \theta)^2 m}{2\sigma^2} \right) \times \exp \left(-\frac{(\theta - \mu)^2 n_o}{2\sigma^2} \right)
      \end{align*}\]

    + the term involving $\theta$ exactly that arising from a posterior distribution

      \[ p(\theta|y_m) = N \left(\theta \left\vert \frac{n_0\mu + my_m}{n_o + m}, \frac{\sigma^2}{n_0 + m}\right.\right) \tag{14}\]

    + posterior mean $(n_o \mu + m y_m)/(n_o + m)$
      + a weighted average of the prior mean $\mu$ and parameter estimate $y_m$
      + $y_m$ weighted by their precision
    + posterior variance (1/precision)
      + based on an implicit sample size equivalent to the sum of the prior 'sample size' $n_0$ and the sample size of the data $m$
      + when combining sources of evidence from the prior and the likelihood, _adding precisions_ to decrease the uncertainty

+ [General form of normal distribution](../Notes/p01-Bayesian.md#37-bayesian-analysis-with-normal-distributions)
  + general notations:
    + prior distribution: $\theta \sim N(\mu, \tau^2)$
    + likelihood: $y_m \sim N(\theta, \sigma_m^2)$
  + posterior distribution

    \[ p(\theta | y_m) = N \left( \theta \left\vert \frac{\frac{\mu}{\tau^2} + \frac{y_m}{\sigma_m^2}}{\frac{1}{\tau^2}+\frac{1}{\sigma_m^2}}, \frac{1}{\frac{1}{\tau^2}+\frac{1}{\sigma_m^2}} \right.\right) \tag{15} \]



## Parameter Estimation

+ [Point estimates](../Notes/p01-Bayesian.md#38-point-estimation-interval-estimation-and-interval-hypotheses)
  + traditional measures of location of distribution: mean, median, and mode
  + given a theoretical justification as a point estimate derived from a posterior distribution, by imposing a particular penalty on error in estimation
  + posterior distribution: symmetric and unimodal $\implies$ mean, median, and mode all coincide in a single value and no difficulty in making a choice
  + posterior distribution considerably skewed in some circumstances $\implies$ marked difference btw mean and median

+ [Interval estimates](../Notes/p01-Bayesian.md#38-point-estimation-interval-estimation-and-interval-hypotheses)
  + credible interval: any interval containing probability different from a 'Neyman-Pearson' confidence interval
  + types of intervals: assume a continuous parameter $\theta$ w/ range on $(-\infty, \infty)$ and posterior conditional on generic data $y$
    + _one-side intervals_: typical  $x = .90, .95, .99$
      + one-side upper $x \cdot 100\%$ w/ $(\theta_L, \infty)$ where $p(\theta < \theta_L| y) = x$
      + one side lower $x \cdot 100\%$ w/ $(-\infty, \theta_U)$ where $p(\theta > \theta_U | y) = x$
    + _two-sided 'equi-tail-area' intervals_: a two-sided $x \cdot 100\%$ (typical 90%, 95%, 99%) interval w/ equal probability in each tail area w/ $(\theta_L, \theta_U)$ where $p(\theta < \theta_L | y) = x/2$ and $p(\theta > \theta_U | y) = 1.0 - x/2$
    + _Highest Posterior Density (HPD) intervals_
      + typical property: skewed posterior distribution $\implies$ a two-sided interval w/ equal tail areas generally containing some parameter values having lower posterior probability than values outside the interval
      + HPD w/o such property
      + adjusting: the probability ordinates at each end of the interval are identical $\implies$ the narrowest possible interval containing the required possibility
      + posterior distribution w/ more than one mode $\implies$ HPD may be a set of disjoint intervals
  + HPD interval (Fig. 4)
    + preferable but generally difficult to compute
    + normal distributions: using tables or programs giving tail areas
    + more complicated situation: generally simulating value of $\theta$ and one and two-sided intervals constructed using the empirical distribution of simulated values

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf" ismap target="_blank">
        <img src="../Notes/img/p01-04a.png" style="margin: 0.1em;" alt="(a) a symmetric unimodal distribution in which equi-tail-area and HPD intervals coincide at -1.64 to 1.64" title="Fig. 4(a) a symmetric unimodal distribution in which equi-tail-area and HPD intervals coincide at -1.64 to 1.64" height=120>
        <img src="../Notes/img/p01-04b.png" style="margin: 0.1em;" alt="(b) a skewed unimodal distribution in which the equi-tail-area interval is 0.8 to 6.3, whereas the HPD of 0.4 to 5.5 is considerably shorter" title="Fig. 4(b) a skewed unimodal distribution in which the equi-tail-area interval is 0.8 to 6.3, whereas the HPD of 0.4 to 5.5 is considerably shorter" height=120>
      </a>
      <a href="http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf" ismap target="_blank">
        <img src="../Notes/img/p01-04c.png" style="margin: 0.1em;" alt="(c) a bimodal distribution in which the equi-tail-area interval is -3.9 to 8.6, whereas the HPD appropriately consists of two segments" title="Fig. (c) a bimodal distribution in which the equi-tail-area interval is -3.9 to 8.6, whereas the HPD appropriately consists of two segments" height=150>
      </a>
    </div>

  + traditional confidence intervals vs. Bayesian credible intervals
    1. _interpretation_ - most important
      + a 95% probability that the true $\theta$ lies in a 95% credible interval $\implies$ certainly _not_ the interpretation of a 95% confidence interval
      + a long series of 95% confidence intervals: 95% of events containing the true parameter value
      + the Bayesian interpretation: giving a probability of whether a _particular_ confidence interval contains the true value
    2. _credible interval_
      + generally narrower due to the additional information provided by the prior
      + width of posterior distribution w/ normal distribution: $U_D - L_D = 2 \times 1.96 \times \sigma/\sqrt{n_o + m}$
      + confidence interval of normal distribution: $2 \times 1.96 \times \sigma/\sqrt{m}$
    3. _care required in terminology_:
      + the width of classical confidence intervals: the standard error of the estimator
      + the width of Bayesian credible intervals: dedicated by the posterior standard deviation

+ [Interval hypotheses](../Notes/p01-Bayesian.md#38-point-estimation-interval-estimation-and-interval-hypotheses)
  + a hypothesis of interest comprises an interval $H_0: \theta_L < \theta < \theta_U$
  + posterior distribution: $p(H_0 | y) = p(\theta_L < \theta < \theta_U | y)$
  + computed w/ standard formulae or simulation methods

+ [Confidence interval of prior distribution and posterior distribution](../Notes/p01-Bayesian.md#38-point-estimation-interval-estimation-and-interval-hypotheses)
  + the (rather odd) prior belief that all values of $\theta$ were equally likely $\implies p(\theta)$ constant
  + $p(\theta | y) \propto p(y | \theta) \times p(\theta)$: the resulting posterior distribution simply proportional to the likelihood
  + $p(\theta|y_m) = N \left(\theta \left\vert \frac{n_0\mu + my_m}{n_o + m}, \frac{\sigma^2}{n_0 + m}\right.\right)$: equivalent to assuming $n_0 = 0$ in an analysis w/ normal distribution
  + traditional confidence interval: essentially equivalent to a credible interval based on the likelihood alone
  + Bayesian and classical equivalent results w/ a uniform or 'flat' prior
  + 'it is already common practice in medical statistics to interpret a frequentist confidence interval as if it did represent a Bayesian posterior probability arising from a calculation invoking a prior density that is uniform on the fundamental scale of analysis' -- P. Burton, 'Helping doctors to draw appropriate inferences from the analysis of medical studies'


## Result Interpretation

+ [Connections btw Bayes theorem and clinical trials](../Notes/p01-Bayesian.md#310-how-to-use-bayes-theorem-to-interpret-trial-results)
  + known: the prior distribution on $\theta$ should supplement the usual information ($p$-value and CI) which summarizes the likelihood
  + consideration: huge number of clinical trials carried out and finding the few clearly beneficial interventions
  
+ [Types of error](../Notes/p01-Bayesian.md#310-how-to-use-bayes-theorem-to-interpret-trial-results)
  + Type I error ($\alpha$): false positive - the chance of claiming an ineffective treatment is effective
  + Type II error ($\beta$): false negative - the chance of claiming an effective treatment is ineffective
  + the odds of formulation of Bayes theorem, when a 'significant result' observed

    \[\begin{align*}
      \frac{p(H_0 | \text{significant result})}{p(H_1 | \text{significant result})} &= \frac{p(\text{significant result} | H_0)}{p(\text{significant result} | H_1)} \times \frac{p(H_0)}{p(H_1)} \\\\
        &= \frac{p(\text{Type I error})}{1 - p(\text{Type II error})} \times \frac{p(H_0)}{p(H_1)}
    \end{align*}\]

    + $H_0$: ineffective treatment
  + truly effective treatment relative rare $\implies$ a 'statistical significant' result stands a good chance of being a false positive
  + the precise $p$-value / 'significant' and $\alpha$
    + Lee & Zelen (2000): suggested selecting $\alpha$ that the posterior probability of an effective treatment, having observed a significant result, is sufficient high, say above 0.9
    + Simon (2000) and Bryant & Day (2000): criticized solely based on the trail is 'significant', rather than the actual observed data



## Credibility Test

+ [Credibility in clinical trials](../Notes/p01-Bayesian.md#311-the-credibility-of-significant-trial-results)
  + credibility
    + the beliefability of new findings in the light of current knowledge
    + a key issue in the assessment of clinical trial outcomes
  + Bayesian methods: probability not as idealized long-run frequencies, but as degrees of belief based on all the available evidence
  + extending to ask how skeptical not to find an apparently positive treatment effective convincing
  + prior mean $y_m = 0$, reflecting initial skepticism about treatment difference, w/ the variance of the prior expressing the degree of skepticism with which we view extreme treatment effects, either positive or negative

+ [Bayesian credibility test](../Notes/p01-Bayesian.md#311-the-credibility-of-significant-trial-results)
  + critical prior distribution $\implies$ the corresponding posterior 95% interval including 0
  + observing $y_m >0$, a normal likelihood and prior w/ $\mu = 0$

    \[ \theta \sim N \left( \frac{m y_m}{n_0 + m}, \frac{\sigma^2}{n_0 + m} \right) \]

  + the upper point $u_m$ of the 95% posterior interval

    \[ u_m = \frac{m y_m}{n_0 + m} + 1.96 \frac{\sigma}{\sqrt{n_0 + m}} \]

    $\implies$ the 95% interval will overlap 0 if $u_m > 0$
  + the effective number of events in the skeptical prior leading to a 95% posterior interval including 0 (to simplify w/ equality)

    \[ n_0 > \left( \frac{m y_m}{1.96 \sigma} \right) -m = \frac{m^2}{1.96^2 \sigma^2} \left(y_m^2 - \frac{1.96^2 \sigma^2}{m} \right) \tag{16} \]

  + $l_D$, $u_D$: the lower and upper points of a 95% interval  based on the data alone, respectively

    \[\begin{align*}
      (l_D, u_D) = y_m &\pm 1.96 \sigma / \sqrt{m} \\\\
      (u_d - l_d)^2 = 4 \times 1.96^2 \sigma^2 /m \quad & \quad u_d l_D = y_m^2 - 1.96^2 \sigma^2/m
    \end{align*}\]
  
  + the critical value of $l_0$ occurs when the lower point of the 95% prior interval

    \[ l_0 = \frac{-19.6 \sigma}{\sqrt{n_0}} = - \frac{(u_D - l_D)^2}{4 \sqrt{u_D l_D}} \]

  + $l_D, u_D$ on a $\log(OR)$ scale $\to l_0 = \log(L_0), l_D = \log(L_D), u_D = \log(U_D)$

    \[ L_0 = \exp\left( \frac{-\log^2(U_D/L_D)}{4 \sqrt{\log(U_D) \log(L_D)}} \right) \tag{17} \]

  + $L_0$ and CI
    + the critical value ($L_0$) for the lower end of 95% skeptical interval $\to$ the resulting posterior distribution w/ a 95% interval including 1
    + prior belief in $(L_0, 1/L_0) \implies$ not convinced by the evidence
    + a significant trial _credible_ $\implies$ prior experience indicates that OR lying outside the critical prior interval are plausible
  
+ [Assessment of ‘credibility’ of ORs](../Notes/p01-Bayesian.md#311-the-credibility-of-significant-trial-results)
  + observing a classical 95% interval $(L_D, U_D)$ for an OR
  + $L_0$:
    + the lower end of a 95% prior interval centered on 1 expressing skepticism about large differences
    + the critical value such that the resulting posterior distribution has a 95% interval that just includes 1
    + not producing 'convincing' evidence
  + $OR >> L_0 \implies$ judged plausible based on evidence external to the study
  + the significant conclusions $\nRightarrow$ convincing

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf" ismap target="_blank">
      <img src="../Notes/img/p01-08.png" style="margin: 0.1em;" alt="Assessment of 'credibility' of odds ratios" title="Assessment of 'credibility' of odds ratios" width=350>
    </a>
  </div>

+ [Applying assessment to GREAT study](..](../Notes/p01-Bayesian.md#311-the-credibility-of-significant-trial-results))
  + 95% classical CI for $\log(OR) = (-1.45, -0.03) \to OR = (0.24, 0.97) \implies L_D = 0.24, U_D = 0.97, L_0 = 0.10$
  + $OR >> 0.1$ as plausible $\to$ the results of the GREAT study w/ caution
  + $L_D, U_D$ and $L_0$ not plausible $\implies$ not finding GREAT result 'credible'
  + characteristic of any 'just significant' results such as those observed in the GREAT trial: just a minimal amount of prior skepticism is necessary to make the Bayesian analysis 'non-significant'

+ [Credibility: Sumartriptan trial results](../Notes/p01-Bayesian.md#311-the-credibility-of-significant-trial-results)
  + interest: the results of an early study of subcutaneous sumatriptan for migraine - Matthews, 2001
  + improvement: 79% w/ sumatriptan vs. 25% w/ placebo
  + 95% of the prior belief within critical interval $\implies$ posterior 95% interval not exclude OR = 1 $\implies$ the data not 'convincing'
  + unreasonable to rule out on prior grounds advantages of greater than 19%, and hence reject the critical prior interval as being unreasonably skeptical, and accept the results as 'credible'


## Prediction

+ [Prediction w/ Bayes theorem](../Notes/p01-Bayesian.md#3131-predictions-in-the-bayesian-framework)
  + task: predict some future observations $x$ on the basis of currently observed data $y$
  + the distribution $p(x|y)$ ex tended w/ unknown parameters $\theta$ by

    \[ p(x | y) = \int p(x | y, \theta) p(\theta | y) d\theta \]

  + the posterior distribution $p(y | \theta)$
  + $x$ and $y$ conditionally independent given $\theta \implies p(x | y, \theta) = p(x | \theta)$
  + the predictive distribution: the sampling distribution of $x$ averaged over the current beliefs regarding the unknown $\theta$

    \[ p(x | y) = \int p(x|\theta) p(\theta | y) d\theta \]

+ Predictive distribution w/ binary data
  + $\theta$ as the true response rate for a set of Bernoulli trials
  + current posterior distribution of $\theta$ with mean $\mu$
  + observing the next $n$ trials to predict $Y_n$, the number of successes

    \[ E(Y_n) = E_\theta[E(Y_n | \theta)] = E_\theta[n\theta] = n\mu \tag{20} \]

  + the probability that the next observation (n=1) is success equal to $\mu$, the posterior mean of $\theta$

+ Beta-Binomial distribution
  + $\theta$ as a conjugate $Beta(a, b)$
  + the exact predictive distribution for $Y_n$, known as the beta-binomial distribution

    \[ p(y_n) = \frac{\Gamma (a+b)}{\Gamma(a)\Gamma(b)} \begin{pmatrix} n \\ y_n \end{pmatrix} \frac{\Gamma(a+y_n) \Gamma(b+n-y_n)}{\Gamma(a+b+n)} \tag{21} \]

  + w/ $E(\theta) = a/(a+b)$, the mean and variance of the distribution

    \[ E(Y_n) = n \frac{a}{a+b} \]
    \[ Var(Y_n) = \frac{nab}{(a+b)^2} \frac{a+b+n}{(a+b+1)} \tag{22} \]

  + Special cases
    + $a = b = 1$:
      + the current posterior distribution $\sim$ uniform
      + the predictive distribution for the number of successes in the next $n$ trials $\sim$ unifrom $\forall \; n = 0, 1, 2, \dots$
    + predicting the next single observation ($n = 1$), Eq. 21 simplified to a Bernoulli distribution w/ $\mu = a/(a+b)$

+ Uniform distribution
  + a prior for $\theta$ as uniform
  + observing $m$ trials w/ positive, the posterior distribution $\sim$ Beta(m+1, 1)
  + Lapace's law of success: the probability that the event will occur at the next trial is $m/(m+1)$
  + even if an event has happened in every case so far, never completely certain that it will happen at the next opportunity

+ [Normal predictive distribution](../Notes/p01-Bayesian.md#3133-predictions-for-normal-data)
  + likelihood: $Y_n \sim N(\theta, \sigma^2/n)$
  + prior distribution: $\theta \sim N(\mu, \sigma^2/n_0)$
  + predictions on future values of $Y_n$
  + consider $Y_n$ as being the sum of two independent quantities: $(Y - \theta) \sim N(0, \sigma^2/n)$ and $\theta \sim N(\mu, \sigma^2, n_0)$
  + the predictive distribution

    \[ Y_n \sim N \left( \mu, \sigma^2 \left(\frac{1}{n} + \frac{1}{n_0}\right) \right) \tag{23} \]

  + predictions: adding variances $\implies$ increasing uncertainty
  + combining sources of evidence using Bayes theorem $\implies$ increasing precision and decreasing uncertainty
  + observed data $y_m$ and the current posterior distribution $\theta \sim N((n_0\mu+my_m)/(n_o+m), \sigma^2/(n_o+m))$, the predictive distribution

    \[ Y_n|y_m \sim N \left( \frac{n_0\mu+my_m}{n_0+m}, \sigma^2 \left( \frac{1}{n_0+m} + \frac{1}{n} \right) \right) \tag{24} \]




## Hierarchical Models

+ Modeling for sequential data sets
  + $\exists$ two or more segmented observed data, $y_m$ followed by $y_n$
  + the posterior distribution of $y_m$ w/ Bayes theorem (Eq. 5)

    \[ p(\theta | y_m) \propto p(y_m | \theta) \times p(\theta) \tag{18} \]

  + using the posterior distribution as the prior distribution after observing the following data segment, $y_n$
  + the posterior conditioning on all the data

    \[\begin{align*}
      p(\theta | y_n, y_m) & \propto p(y_n | \theta, y_m) p(\theta | y_m)  \tag{19} \\
        & \propto p(y_n | \theta, y_m) p(y_m | \theta) p(\theta)
    \end{align*}\]

  + factorizing the joint likelihood

    \[ p(y_m, y_n | \theta) = p(y_n | \theta, y_m) p(y_m | \theta) \]

  + most situations, $p(y_n | \theta, y_m)$ not depending on $y_m$; i.e. $Y_n$ simply conditionally independent of $Y_m$ given $\theta$
  + $\therefore p(\theta | y_m)$ simply as the prior for a standard Bayesian update using the likelihood $p(y_n | \theta)$




