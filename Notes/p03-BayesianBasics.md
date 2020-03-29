# Basics of Bayesian Methods

Author: Sujit Ghosh

Year: 2010

[Origin: Chapter 3, pp 155-178](https://bit.ly/2WTB3va)



Book: Bang H., Zhou X., van Epps H., Mazumdar M. (eds) [Statistical Methods in Molecular Biology](https://bit.ly/39ukdpi). Methods in Molecular Biology (Methods and Protocols), vol 620. Humana Press, Totowa, NJ


## 1. Introduction

+ Bayesian inference method
  + providing a logical framework to utilize all available sources of information in making a decision when analyzing data
  + prior knowledge: experience, expert pinion, or collected data in previous studies w/ similar protocol

+ Subjectivity
  + Bayesian statistical methods criticized as being subjective
  + almost any scientific knowledge and theories at best: subjective in nature
  + S. Press and M. Tanur, [Subjectivity of Scientist and the Bayesian Approach](https://bit.ly/39ygXJS), 2001
    + Subjectivity occurs, and should occurs, in the works of scientists
    + Total objectivity in science is a myth.  Good science inevitably involves a mixture of subjective and objective parts.
  + Bayesian inferential framework: a logical foundation in data analysis to accommodate
    + objective (by modeling observed data)
    + subjective (by using prior distribution for parameter)
  + classical (frequentist) statistical methods
    + not objective as often as claimed in practice
    + e.g., $p$-value < 0.05 $\implies$ reject $H_0$, why 0.05?
  + non-parametric methods
    + not completely objective
    + assumption: the existence of variance within a linear model framework but data w/ long-tail distribution
    + assumption: independence among observations
  + assumptions built around any scientific methods $\implies$ subjective
  + validating assumptions w/ sensitivity analysis

+ Task, Notations and Assumptions
  + experimental goal: to infer about plausible value(s) of the parameter vector $\theta$
  + $Y = (y_1, y_2, \dots, y_n)$: observed response data
  + making inference about such hidden cause(s), $\theta$, based on observing the effect(s), $y$
  + $p(y|\theta)$: a conditional density of the observation $y$ defined on the _sample space_ $\mathbb{Y}$; the _sampling density_ of $y$ given $\theta$
  + $p(\theta)$: the marginal density of $\theta$ to be the _prior density_ of $\theta$ defined on the _parameter space_ $\Theta$
  + actual determination of the prior density = the determination of the sampling density $p(y|\theta)$
  + $\{f(y|\theta): \theta \in \Theta\}$: the parametric class of sampling densities
  + assumption: the specific form of $p(\cdot | \theta)$ completely known once $\theta$ determined
  + assumption: $\Theta \subseteq \mathbb{R}^m$; i.e. finite-dimensional _parametric space_


### 1.1 The Bayes' Rule

+ Bayes model
  + scalar form
    + $p(y|\theta)$: the sampling density of $y \in \mathbb{Y}$ for a given value $\theta \in \Theta$
    + $p(\theta)$: the prior density of $\theta$
    + the conditional density of $\theta$ given $y$

      \[ p(\theta|y) = \frac{p(y|\theta) p(\theta)}{\int p(y|\theta)p(\theta)d\theta} = \frac{p(y|\theta)p(\theta)}{m(y)} \tag{1} \]

      + $p(\theta|y)$: the posterior density of $\theta$ given y
      + $m(y) = \int p(y|\theta) p(\theta)$: marginal density of $y$

  + vector form
    + $Y = (y_1, y_2, \cdots, y_n)$:  a vector of response
    + $p(Y|\theta)$: the joint density of $Y$ given $\theta$
    + the conditional density of $\theta$ given $Y$

      \[ p(\theta|Y) = \frac{p(Y|\theta) p(\theta)}{\int p(Y|\theta)p(\theta)d\theta} = \frac{p(Y|\theta)p(\theta)}{m(Y)} \]

  + independently and identically distributed (iid) observations
    + $y_i|\theta \stackrel{\text{iid}}{\sim} p(y|\theta),\; \forall i=1, \dots, n \implies$ $p(Y|\theta) = \prod_{i=1}^n p(y_i | \theta)$
    + generality: $\exists\; p_i(y_1, \dots, y_{i-1}, \theta)$ of $y_i$ given $y_1, \dots, y_{i-1}$ and $\theta$ for $i=2, 3, \dots, n \implies f(Y|\theta) = p_1(y_1|\theta) \prod_{i=1}^n p(y_i|\theta)$
    + vector form: replacing $p(y|\theta)$ by $p(Y|\theta)$ for Eq.(1)
  + an observed data set $Y$ consisting two quantities
    + $p(Y|\theta)$: sampling density
    + $p(\theta)$: prior density
  + summary

    \[\begin{array}{ll}
      \text{prior density for } \theta: & p(\theta) \\
      \text{sampling density of } y \text{ given } \theta: & p(y|\theta) \\
      \text{marginal density of } y: & m(y) = \int p(y|\theta)p(\theta) d\theta \\
      \text{posterior density of } \theta \text{ given } y: & p(\theta | y) = p(y|\theta)/m(y)
    \end{array}\]

+ Marginal density $m(y)$
  + discrete version: replacing integration by summation
  + the marginal density and the posterior density w.r.t a $\sigma$-finite measure; e.g., Lebesgue measure or counting measure
  + solution: suppressing the domain of integration of $\Theta$ while defining $m(y)$ by assuming  that $p(\theta) = 0\;\forall\; \theta \notin \Theta$

+ Kernel functions
  + prior kernel function
    + $p(\theta) = C \cdot k(\theta)$
      + $C = (\int k(\theta)d\theta)^{-1} > 0$: normalizing constant
      + $k(\cdot)$: a non-negative function defined on the parameter space $\Theta$
    + examples
      + $\theta^{5.2} (1-\theta)^{2.5} I_{(0, 1)}(\theta)$:
        + a kernel function of beta distribution w/ shape parameters 6.2 and 3.5
        + $k(\theta) = \theta^{5.2} (1-\theta)^{2.5}$ and $C = I_{(0, 1)}(\theta)$
      + $\theta^5e^{-\theta} I_{(0, \infty)}(\theta)$: a kernel function of gamma distribution w/ parameters 6 and 1
      + indicator function of the set $A$

        \[I_A(\theta) = \begin{cases} 1 & \theta \in A \\ 0 & \theta \notin A \end{cases}\]

  + likelihood function
    + satisfied

      \[ p(Y|\theta) =  C(Y)\mathcal{L}(\theta; Y) \]

      + $C(Y) > 0$: a function of data $Y$ only
      + $\mathcal{L}(\theta; Y)$: the likelihood function of $\theta$
    + examples
      + $y_i|\theta \stackrel{\text{iid}}{\sim} N(\theta_1, \theta_2) ,\; \theta = (\theta_1, \theta_2) \in \Theta = \mathbb{R} \times (0, \infty) \implies y_i \sim N(\theta_1, \theta_2)$
      + posterior density: $p(y|\theta) = (2\pi \theta_2)^{-1/2} \exp\left(-(x - \theta_1)^2/2\theta_2^2 \right)$
      + likelihood function: $\mathcal{L}(\theta; Y) = \theta_2^{-n/2} \exp\left(-\sum_{i=1}^n (y_i - \theta_1)^2/2\theta_2^2 \right)$
  + posterior kernel function
    + the posterior density determined by

      \[ K(\theta; Y) = \mathcal{L}(\theta; Y) k(\theta) \tag{2} \]

    + $\mathcal{L}(\theta; Y)$: the likelihood function of $\theta$
    + $k(\theta)$: a prior kernel function of $\theta$

+ The posterior density
  + representation w/ kernel functions

    \[ p(\theta|Y) = \frac{K(\theta; Y)}{\int K(\theta; Y) d\theta} = \frac{\mathcal{L}(\theta; Y) k(\theta)}{\int \mathcal{L}(\theta; Y) k(\theta) d\theta} \tag{3} \]

  + Bayesian Mantra: posterior is proportional to likelihood times prior kernel

    \[p(\theta|Y) \propto \mathcal{L}(\theta; Y) k(\theta) \]

+ Improper prior
  + def: prior kernel function $k(\theta) \geq 0$ for which $\int C \cdot k(\theta) d\theta  = \infty$
  + an improper prior is determined only up to a constant because $\int k(\theta) d\theta \implies \int C \cdot k(\theta) = \infty \;\forall\; C > 0$
  + not an improper prior (probability) distribution any more as $p(\theta)$ no longer a (probability) density function
  + checking the posterior still proper for almost all data if the improper prior used
  + using improper prior $p(\theta) = C \cdot k(\theta) \implies$ verifying the posterior kernel finitely integrable $\int K(\theta; Y) d\theta < \infty$ for almost all $Y$
  + Applying Fubini's theorem (on interchanging order of integration):
    + improper prior $\iff$ improper marginal
    + $\int k(\theta) d\theta = \infty \iff  \int m(Y) dY = \infty$
  + __Lemma__: If the likelihood function is bounded below, i.e., $\inf_\theta \mathcal{L}(\theta; Y) \geq \mathcal{L}_0(Y)$ for some $\mathcal{L}_0(Y) >0 \implies$ any improper prior leads to an improper posterior
  + the posterior distribution not necessary proper if the prior improper [[Wiki](https://bit.ly/2vYuixh)]
  + using improper priors as uninformative priors; e.g., $p(m, v) \sim 1/v \implies$ any value for the mean "equally likely" and a value for the positive variance "less likely" [[Wiki](https://bit.ly/3bDybXF)]
  + [examples](https://bit.ly/2xzRqSW)
    + uniform distribution on an infinite interval; i.e., a half line or entire real line
    + $Beta(0,0)$ (uniform distribution on log-odds scale)
    + logarithmic prior on the positive reals (uniform distribution on log scale)
  + improper priors not true probability distributions [[Degroot & Schervish](https://bit.ly/2QVlXSb)]

+ Example: Does vitamin C cure common cold?
  + intervention: a randomly chosen group of patients suffering from common cold took (same amount of) vitamin C for 1 week and response whether or not vitamin C cured common cold immediately following the week
  + Notations & Assumptions
    + $y$: response w/ 1 as common cold cured and 0 otherwise
    + $\theta$ = \Pr[common cold cured within a week]: the parameter of interest
    + outcome of a given patient: Bernoulli distribution $p(y|\theta) = \theta^y(1-\theta)^{1-y}$ w/ $x = 0, 1$ and $\theta \in [0, 1]$
    + $n$ iid observations: $y_i \in [0, 1], i=1, \dots, n$
    + $Y = (y_1, y_2, \cdots, y_n)$: the response vector
    + $s$: the total number of patents (out of n) cured within a week on taking vitamin C
  + the joint density of the response vector $Y$: $p(Y|\theta) = \prod_{i=1}^n p(y_i|\theta) = \theta^s(1-\theta)^{n-s} \implies$ $s \sim Bin(n, \theta)$, binomial distribution w/ density function

    \[p(s|\theta) = {n \choose k} \theta^s (1-\theta)^{n-s}, \quad\text{ for } s=0, 1, \cdots, n\]

  + the posterior density w/ any prior density $p(\theta)$

    \[p(\theta|Y) = \frac{\theta^s(1-\theta)^{n-s} p(\theta)}{\int_0^1 \theta^s (1-\theta)^{n-s} p(\theta) d\theta} \tag{4}\]
  + remark
    + $s = \sum_i y_i$: the _sufficient statistic_; the posterior density $p(\theta|Y)$ depends on the data $Y$ only through $s$
    + $p(\theta|Y) = p(s|\theta) \implies$ the conditional density of $\theta$ given only the sum $s$ (and sample size $n$)
    + applying Fisher-Neyman factorization theorem: $p(Y|\theta)$ = the joint density of $Y$ given $\theta$ & $S = S(Y)$ = a sufficient statistic (vector) $\implies p(\theta | Y) = p(\theta|S) \;\forall\; p(\theta)$

+ Prior conjugate family
  + conjugate family: prior densities and their leading posterior densities belonging to the same family
  + the choice of conjugate family not unique
  + example of beta distribution
    + prior distribution: $p(\theta) = C \theta^{a-1}(1-\theta)^{b-1}$ w/ $C = C(a, b)$ and $a > 0, b > 0$ unknown quantities
    + posterior distribution: $p(y|\theta) = C^\ast \theta^{a^\ast -1} (1-\theta)^{b^\ast - 1}$ w/ $C^\ast = C(a^\ast, b^\ast), \; a^\ast = (a+s), b^\ast = (b + s)$
    + $C(a, b) = \Gamma(a+b)/\Gamma(a)\Gamma(b)$,  where gamma function $\Gamma(a) = \int_0^\infty t^{a-1}e^{-t} dt$
    + the prior $\theta \sim Beta(a, b) \implies$ the posterior $\theta|Y \sim Beta(a^\ast, b^\ast)$
  + example: binomial sampling density
    + $s|\theta \sim Bin(n, \theta) \in \{Beta(a, b): a > 0, b>0\}$ forming a conjugate family of beta densities
    + $\{C(a, b)\theta^{a-1} (1-\theta)^{b-1} p_0(\theta): a>0, b>0\}$ forms a conjugate family
      + $p_0(\cdot) \in [0, 1]$: any nonnegative continuous function
      + $C(a, b) = \left(\int_0^1 \theta^{a-1}(1-\theta)^{b-1} p_0(\theta) d\theta\right)^{-1}$: normalizing constant
  + exponential family of densities: sampling density w/ a sufficient statistic of constant dimension always finds a conjugate family of prior densities
  + _natural conjugate family_:
    + constructing a class of such family for iid observations obtaining from a sampling density $p(y|\theta)$ for $y \in \mathbb{Y}$ and $\theta \in \Theta$
    + by simply defining the family of prior densities as $\{\prod_{j=1}^m p(y_j^0|\theta): y_j^0 \in \mathbb{Y}, m = m_0, m_0 + 1, \dots\}$
    + provided $\exists \; m_0 \in \mathbb{N} \to \int \prod_{j=1}^{m_0} p(y_j^0 | \theta) d\theta < \infty$
  + _subjective prior of informative prior_: the parameters of the prior density elicited using a previously collected data or expert knowledge
  + _noninformative prior_:
    + no such prior information available or very little knowledge available about the parameter $\theta$
    + example:
      + invariant under monotone transformation
      + maximum likelihood prior
      + reference prior among others

+ Numerical methods
  + no need of conjugate family of prior densities
  + computing posterior summary estimates
  + binomial sampling density:
    + Zellner's prior: $p(\theta) = C\theta^\theta (1-\theta)^{1-\theta}$ for $\theta \in (0, 1)$ and C = 1.61857
    + Jeffery's prior: $Beta(a = 0.5, b = 0.5)$
  + Haldane's prior: improper prior $p(\theta) = [\theta(1-\theta)]^{-1}$
  + uniform prior: same as $Beta(a=1, b=1)$ w/ $\theta$ using binomial sampling density
  + posterior inference about $\theta$ or odds ratio $\rho = \theta/(1-\theta)$ or log-odds ratio $\eta = \log \rho$ relative insensitive to all previous priors


### 1.2 Hierarchical and Empirical Approaches

+ Un-pure Bayesian method
  + issue: difficult to completely specify the prior density based on available expert opinion or background scientific information
  + utilizing hyperparametes to resolve
    + $\{p_1(\theta|\lambda): \lambda \in \Lambda\}$: a class of prior densities
    + $\lambda$: hyper-parameter
    + $p_1(\cdot | \cdot)$: a conditional density of $\theta$ given $\lambda$
  + example: beta distribution for vitamin C 
    + $\{Beta(a, b): (a, b) \in (0, \infty) \times (0, \infty)\}$: a class of beta densities
    + $(a, b)$: hyper-parameter for the family of beta density
    + what would be a reasonable choice for $(a, b)$?
  + methods to determine hyperparameter $\lambda \to$ empirical estimate $\hat{\lambda}$
    + Bayesian Hierarchical Model (BHM)
    + Empirical Bayes (EB) method
  + using $p(\theta) = p_1(\theta | \hat{\lambda})$ as the prior for $\theta$
  + criticizing: the prior density $p(\theta)$ obtained via the data $Y$

+ Bayesian hierarchical Model (EBM)
  + assigning another prior distribution for $\lambda$:
    + $\lambda \sim p_2(\lambda)$
    + $p_2(\cdot)$: a probability density defined over $\Lambda$
  + $\therefore\; p(\theta) = \int p_1(\theta|\lambda) p_2(\lambda)d\lambda$: a more flexible prior for $\theta$
  + forming a hierarchical stages for the complete Bayes model

    \[ Y|\theta, \lambda \sim p(Y|\theta) \implies \theta|\lambda \sim p_1(\theta|\lambda) \quad \&\quad \lambda \sim p_2(\lambda)\]

  + often choosing $p_2(\cdot)$ suitable to obtain estimators w/ good frequentist properties
  + issue:
    + difficult to analytically obtain the posterior distribution of $\theta$ given $Y$
    + inference based on $p(\theta|Y)$ requiring integration both on $\theta$ and $\lambda$
  + solutions: Monte Carlo methods (in particular, Markov Chain Monte Carlo (MCMC) methods) used to obtain samples from the posterior density $p(\theta|Y)$

+ Empirical Bayes (EB) method
  + based on marginal likelihood $m(Y|\lambda) = \int p(Y|\theta) p_1(\theta|\lambda) d\theta$
  + $m(Y|\lambda)$: the marginal likelihood function of $\lambda \implies$ maximizing

    \[ \hat{\lambda} = \hat{\lambda}(Y) = \underset{\lambda \in \Lambda}{\mathrm{argmin}}\; m(Y|\lambda) \]
  
  + alternatively, obtaining a moment based method to 'estimate' $\lambda$
    + $\lambda$: a $q$-dimensional parameter
    + using a set of $q$ suitable moments of $Y$ w.r.t. the marginal density $m(Y|\lambda)$
    + equate moments to the corresponding $q$ empirical moments of the data vector $Y = (y_1, y_2, \dots, y_n)$


## 2. Point Estimation

+ Assumptions & Notations
  + $p(Y|\theta)$: the joint (conditional) density of the data vector $Y = (y_1, y_2, \dots, y_n)$, given a parameter vector $\theta$
  + iid $y_1, y_2, \dots, y_n \implies p(Y|\theta) = \prod_{i=1}^n p(y_i|\theta)$
  + iid for $y_i, i=1, 2, \dots, n$ not required to obtain the posterior estimate of $\theta$
  + $p(\theta)$: prior density; i.e., $\theta \sim p(\theta)$ determined possibly by hierarchical prior or an EB method

+ Statistical inference about $\theta$
  + obtained from the posterior kernel
  + posterior density obtained via prior density w/ Bayes rule
  + prior density: $p(\theta | Y) = K(\theta; Y)/ b(Y)$ where $b(Y) = \int K(\theta; Y) d\theta$
  + the normalizing constant $b(Y)$ not required to obtain posterior estimate (via Monte Carlo method)

+ Modeling the point estimate
  + goal: estimating a special function $\eta = \eta(\theta)$
  + $\theta = (\theta_1, \theta_2, \dots, \theta_m)$: a $m$-dimensional parameter vector
  + various estimators
    + $\eta = \eta(\theta) = \theta_1$: first component of $\theta$: 
    + $\eta = (\theta_1, \theta_2 - \theta_1^2)$
    + $\eta = \eta(\theta) = \theta$: entire parameter vector $\theta$
  + point of estimates of $\eta$: mean, median, or mode of the posterior distribution of $\eta$ given $Y$
  + _posterior mean of $\eta$_

    \[ \hat{\eta}(Y) = E[\eta | Y] = E[\eta(\theta) | Y] = \frac{\int \eta(\theta) K(\theta; Y) d\theta}{\int K(\theta; Y) d\theta} \tag{5} \]

    + $K(\theta; Y)$: the kernel of the posterior density

+ Bayes estimator
  + the optimal (Bayes) estimator of $\eta$: the posterior mean of $\eta$ w/ minimized squared error loss

    \[ E[\| \eta - \hat{\eta} \|^2] \leq E[\| \eta - T(Y)\|^2] \]

    + for any other estimator $T(Y)$ of $\eta$
    + the expectation taken 
      + w.r.t. both $Y$ and $\theta$
      + w.r.t the joint density of $(Y, \theta)$ and 
    + $\| \cdot \|$: the Euclidean norm
  + Bayes estimator exists for more general (convex) loss functions, such as weighted squared error loss, asymmetric loss, etc.
  + Bayes estimator expressed in close form as suitable integrals of the posterior kernel
  + advantage: obtained in a straightforward way and expressed in closed form
  + road-block in practice: high-dimensional integrals not an easy task in most situation
  + solution: advent of modern computing


## 3. Hypothesis Testing

+ Modeling of hypothesis testing
  + Assumptions & Notations
    + $\Theta$: the entire parameter space
    + hypothesis: two competing hypotheses
      + $H_0: \theta \in \Theta_0$ vs. $H_a: \theta \in \Theta_a$
      + $\Theta_0 \cap \Theta_a = \emptyset$ and $\Theta_0 \cup \Theta_a = \Theta$
  + comparing two hypotheses $\iff$ comparing the posterior probabilities of the null set $\Theta_0$ and the alternative set $\Theta_a$
  + deciding which hypothesis w/ a larger probability: reject $H_0 \iff \Pr(\theta \in \Theta_0 | Y) < \Pr(\theta \in \Theta_a | Y)$
  + the posterior probabilities

    \[ \Pr(H_j|Y) = \Pr(\theta \in \Theta_j | Y) = \frac{\int_{\Theta_j} K(\theta; X) d\theta}{\int_{\Theta} K(\theta; Y) d\theta} \quad\text{ for } j = 0, a  \tag{6} \]

  + comparing the numerators to make a decision

    \[ \text{Reject } H_0 \iff \int_{\Theta_0} K(\theta; Y) d\theta < \int_{\Theta_a} K(\theta; Y) d\theta \tag{7} \]

  + carefully constructing the prior distribution especially when one of the hypotheses is a singleton set or containing a lower dimensional plane $\implies$ the prior distribution allowing a positive probability of the null set $\Theta_0$
  + i.e., a prior distribution ensuring $\Pr(\theta \in \Theta_j) > 0$ for $j=0, a$
  + using a prior distribution which assures  $\Pr(\theta \in \Theta_0) = \Pr(\theta \in \Theta_a)$ unless $\exists$ any substantial prior information about the two hypotheses
  + assume a prior w/ equal-probable hypotheses before observing any data
  + example: vitamin C
    + $H_0: \theta = 0.5$ vs. $H_a: \theta > 0.5$
    + $\Theta = [0, 1], \Theta_0 = \{0.5\}$ and $\Theta_a = (0.5, 1]$
    + singleton set: $\Theta_0 = \{0.5\}$
    + lower dimensional plane: $\Theta_0 = \{(\theta_1, \theta_2) \in [0, 1]^2: \theta_1 = \theta_2}

+ Bayesian factor (BF)
  + def: the ratio of posterior odds to prior odds
  + used to choose btw two hypotheses
  + the BF of $H_a$ to $H_0$

    \[\begin{align*}
      BF(Y) &= BF(a|0) = \frac{\Pr(\theta \in \Theta_a | Y)/\Pr(\theta \in \Theta_0 | Y)}{\Pr(\theta \in \Theta_a)/\Pr(\theta \in \Theta_0)} \\\\
       &= \frac{\int_{\Theta_a} K(\theta; Y) d\theta}{\int_{\Theta_0} K(\theta; Y) d\theta} \cdot \frac{\int_{\Theta_0} k(\theta)d\theta}{\int_{\Theta_a} k(\theta) d\theta} \tag{8}
    \end{align*}\]

  + rule to choose btw the two hypotheses

    \[\text{Reject } H_0 \iff \log\left(BF(Y)\right) > 0  \tag{9}\]

  + equal-probable prior: $\Pr(\theta \in \Theta_0) = \Pr(\theta \in \Theta_a)$ or $\int_{\Theta_0} k(\theta) d\theta = \int_{\Theta_a} k(\theta) d\theta$

+ Errors of hypothesis testing
  + Type I error rate: traditional frequentist methods
    + constructed to maintain a specific (but arbitrary) level of significance $\alpha$; e.g., $\alpha = 0.05$
    + $T(Y)$: a test statistic
    + rule: reject $H_0$ if $T(Y) > T_0 \implies$ the cut-off value $T_0$ chosen $\to$ type I error rate

      \[\inf_{\theta \in \Theta_0} \Pr\left(T(Y) > T_0 | \theta\right) \leq \alpha\]

      + $\Pr(T(Y) > T_0 | \theta)$ computed using the conditional joint density $p(Y|\theta)$
  + Type II error rate: Bayesian tests
    + not generally constructed to maintain a specific value of type I error rate
    + type II error rate: (further check required)

      \[\inf_{\theta \in \Theta_a} \Pr(T(Y) \leq T_0 | \theta) \leq \alpha \;\;\forall \theta \in \Theta_a\]

    + using either posterior odds or the BF as a test statistic
      + example:
        + defining $T(Y) = \log(BF(Y))$ the test statistic $\to$ find $T_0$ to satisfy $\Pr(T(Y) > T_0 | \theta) \leq \alpha \;\;\forall\, \theta\in \Theta_0$
        + modify rule (9): reject $H_0 \iff \log(BF(Y)) > T_0$
      + choice of $T_0$ might unnecessarily inflate the type II error rate
    + alternative test statistic
      + simply report posterior probabilities $\Pr(\theta \in \Theta_j | Y)$ for $j = 0, a$
      + researcher decides a cut-off value, e.g., $p_0 \to H_0$ rejected $\iff \Pr(\theta \in \Theta_a | Y) > p_0$
      + example:
        + instead of $p = 0.5$ as default value, using $p_0 = 0.8$
        + arbitrary choice of $p_0$ but so for the significance $\alpha$ (or a cut-off value of 0.05 for the $p$-value)
  + Bayesian type I error rate

    \[ BE_1(T_0) = \Pr(T(Y) > T_0 | \theta \in \Theta_0) \]

    + frequentist method: used to select the cut-off value $T_0 \to \Pr(T(Y) > T_0 | \theta) < \alpha \;\;\forall\, \theta \in \Theta_0 \implies$ the same $T_0 \to BE_1(T_0) \leq \alpha \;\;\forall\; \text{ prior } p(\theta)$
  + Bayesian type II error rate

    \[ BE_2(T_0) = \Pr(T(Y) \leq T_0 | \theta \in \Theta_a) \]

  + $T_0 \nearrow \implies BE_1(T_0) \searrow \;\&\; BE_2(T_0) \nearrow$
  + controlling both types of Bayesian errors: finding a $T_0$ to minimize the the total weight error, i.e., determining

    \[\hat{T}_0 = \arg\min TWE(T_0)\]

    + $TWE(T_)) = w_1 BE_1(T_0) + w_2 BE_2(T_0)$
    + $w_1, w_1 \geq 0$: some suitable nonnegative weights
    + $w_2 = 0$: controlling only type I error rate
    + $w_2 >> 0$: controlling the (Bayesian) power $1 - BE_2(T_0)$
  + sample size: $\hat{T}_0$ fixed $\implies$ optimal (minimum) sample size $\to$ $BE_1(\hat{T}_0) + BE_2(\hat{T}_0) \leq \alpha$


## 4. Region Estimation

+ Region estimate
  + credible set: Bayesian tests
    + a subset $R(Y)$ said a $100(1 - \alpha)\%$ _credible set_ for $\theta$ w/ a given value $\alpha \in (0, 1)$

      \[ R = R(Y) \to \Pr(\theta \in R(Y) | Y) \geq 1 - \alpha \]

    + $R = R(Y)$ guarantees that the probability that $\theta$ in $R(Y)$ is at least $1 - \alpha$
  + confidence set: traditional frequentist tests
    + a subset $C(Y)$ said a $100(1-\alpha)\%$ _confidence set_ for $\theta$

      \[ C = C(Y) = \Pr(\theta \in C(Y) | Y) \geq 1 - \alpha \;\;\forall \theta \in \Theta\]

    + $C(Y)$ merely suggests that if the method of computing the confidence set is repeated many times then at least $1 - \alpha$ proportion of those confidence sets would contain $\theta$
    + an observed data vector $Y \to$ the chance that a confidence set $C(Y)$ contains $\theta$ is either 0 or 1
  
+ Highest Probability Density (HPD) region
  + given a specific level $1-\alpha$ of credible set $R(Y)$
  + _highest probability density_ (HPD) region: the 'smallest' set or region to maintain the given level
  + $R(Y)$ as a HPD region of level $1 - \alpha$, if

    \[ R(Y) = \{\theta \in \Theta: K(\theta; Y) > K_0(Y)\} \tag{10} \]

    + $K_0(Y) > 0 \to \Pr(\theta \in R(Y) | Y) \geq 1 - \alpha$
    + $K(\theta; Y)$: the posterior kernel function
  + in practice, not straightforward to compute the HPD region $R(Y)$, but numerical method
  + $\eta = \eta(\theta) \in \mathbb{R}$ \to$ the HPD region for $\eta$ may consist of a union of intervals
  + e.g., the posterior density as a bimodal density $\to$ the HPD may be the union of two intervals, each centered around the two modes
  + the posterior density of a real-valued parameter $\eta$ as a unimodal $\to$ the HPD region as an interval of the form

    \[ R(Y) = (a(Y), b(Y)) \subseteq \Theta, \;\; -\infty \leq a(Y) < b(Y) \leq \infty \]


## 5. Numerical Integration Methods

+ Integrals for posterior inference
  + issue: almost any posterior inference based on $K(\theta; Y) \to$ high-dimensional integration w/ $\Theta \in \mathbb{R}^m$, where $\mathbb{R}^m$ as a Euclidean
  space
  + examples:
    + point estimate (Eq.5)
    + posterior probability for hypothesis testing (Eq.6)
    + HPD region (Eq.10)
  + ageneric function $\mathfrak{g}(\theta)$ of the parameter $\theta$, the posterior inference requiring the computation of the integral

    \[ I = I(Y) \int \mathfrak{g}(\theta) K(\theta; Y) d\theta \tag{11} \]

    + $\ {g}(\theta) = 1$: denominator of $\int K(\theta|Y)d\theta$ in (Eq.5) and $\int_{\Theta_j} K(\theta; y) d\theta$ in (Eq.6)
    + $\mathfrak{g}(\theta) = \eta(\theta)$: the numerator of $\int \eta(\theta) K(\theta; Y) d\theta$ in (Eq.5)
    + $\mathfrak{g}(\theta) = I_{\Theta_j}(\theta)$: the numerator of $\int_{\Theta_j} K(\theta; Y) d\theta$ in (Eq.6)
  
+ Numerical methods for integrals
  + numerical approaches
    + classical (deterministic) numerical methods
      + applied when $m < 5$
      + performing well in practice provided the function $\mathfrak{g}(\theta)$ and $K(\theta; Y) \gets$ certain smoothness condition
      + order of accuracy: $O(N^{-2/m})$
      + rate of convergence depending on $m$ (curse of dimensionality)
    + stochastic integration methods
      + applied when $m \geq 5$
      + known as Monte Carlo (MC) methods
      + order of accuracy: $O(^{-1/2})$
      + rate of convergence not depending on $m$
  + performance: deterministic >> stochastic


### 5.1 Deterministic Methods

+ Fundamentals of deterministic numerical methods
  + computing integrals usually depending on some adaptation of the quadratic rules
  + goal: computing the integral

    \[ I = \int_a^b h(\theta) d \theta \]

    + $h(\theta) = \mathfrak{g}(\theta) K(\theta;Y)$
    + suppressed the dependence on the data $Y$
  + task: finding suitable weights $w_i$'s and ordered knot points $\theta_j$'s $\to$

    \[ I = \int_a^b h(\theta)d\theta \approx I_N = \sum_{j=1}^N w_j h(\theta_j) \tag{12} \]

  + $\exists\; N \to \infty: |I - I_n| \to 0$
  + different choices of the weights and the knot points $\to$ different integration rules
  + categories of the numerical quadrature rules
    + Newton-Cotes rules
      + based on equally spaced knot points
      + e.g., $\theta_j - \theta_{j-1} = (b-a)/N$
    + Gaussian quadrature rules
      + based on knot points obtained via the roots of the orthogonal polynomials
      + more accurate results, especially $h(\cdot)$ approximates reasonably well by a sequence of polynomials

+ Newton-Cotes rules
  + simplest approaches
    + trapezoidal rule: a two-point rule
    + mid-point rule: another two point rule
    + Simpson's rule: a three-point rule
  + general rule:
    + a $(2k - 1)$-point (or $2k$-point) rule w/ $N$ equally spaced knots w/ suitably chosen weights
    + $\exists\; \theta^\ast \in (a, b) \to |I - I_n| = C_k (b - a)^{2k+1} b^{2k}(\theta^\ast) / N^{2k}$
    + $C_K > 0$: a constant
  + exact result: polynomials of degree $N$ ($N = 2k+1$) or degree $(N-1)$ ($N=2k$) $\to$ N knots
  
+ Gaussian quadrature rules
  + not only suitably choosing the weights but also the knots at which the function $h(\cdot)$ evaluated
  + exact result: polynomials of degree $2N -1$ or less $\to N$ knots
  + $K(\theta; Y)$ as weight functions: abundant different rules
    + Davis, P. J., and Rabinowitz, P. (1984) Methods of Numerical Integration. 2nd Edition. Academic Press, New York.
    + Piessens, R., de Doncker-Kapenga, E., Uberhuber, C., and Kahaner, D. (1983) QUADPACK, A Subroutine Package for Automatic Integration. Springer-Verlag, Berlin.

+ Multi-dimensional numerical integration rules
  + multidimensional trapezoidal rule

    \[ I = I_N + O(N^{2/m}) \]

    + $N$: the number of knots at which evaluating $h(\theta)$Math Symbols List
    + $\exists\; \theta = (\theta_1, \dots, \theta_m)$, a $m$-dimensional vector
  + multidimensional Simpson's rule: $O(N^{-4/m})$
  + Monte Carlo methods: $O_p(N^{-1/2})$
  + accuracy of integration: $m \nearrow \;\;\to$ accuracy $\searrow$

+ Summary of deterministic numerical integration methods  
  + very accurate but declined rapidly as $m \nearrow$
  + invalid theoretical error estimate
    + a complicated parameter space
    + insufficiently smooth function
  + [Alan Genz Software - R, MATLAB, Fortan](https://bit.ly/2xuo9JE)


### 5.2 Monte Carlo Methods

+ MC statistical methods
  + used in statistics and various fields to solve various problems
  + procedure
    + generating pseudo-random numbers
    + observing that sample analogue of the numbers converges to their population versions
  + useful for obtaining numerical solutions to problems which are too complicated to solve analytically or by using deterministic methods
  + relying on two celebrated results in Statistics
    + The (Strong or Weak) _Law of Large Numbers_ (LLN)
    + The _Central Limit  Theorem_ (CLT)
  + generating $\theta^{(l)} \stackrel{iid}{\sim} p(\theta | Y), \;\; l=1,2,\cdots,N$ by using only $K(\theta; Y)$
  + w/ (Strong/Weak) LLN: as $N \to \infty$

    \[ \overline{\mathfrak{g}}_N = \frac{1}{N} \sum_{l=1}^N \mathfrak{g}(\theta^{(l)}) \xrightarrow{p} E[\mathfrak{g}(\theta) | Y] = \int \mathfrak{g}(\theta) p(\theta | Y) d\theta \tag{13} \]

    + $p(\theta | Y)$: the posterior density
    + $E[\mathfrak{g}(\theta) | Y] < \infty$
  + avoid computing the numerator and denominator
  + generating random samples by using only the kernel of the posterior density
  + $N \nearrow  \implies \overline{\mathfrak{g}}_N \to E[\mathfrak{g}(\theta) | Y]$; $\overline{\mathfrak{g}}_N$ = sample mean, $E[\mathfrak{g}(\theta)$: population mean
  + almost no smoothness condition required on the function $\mathfrak{g}(\cdot)$ to apply the MC method

+ Sample size
  + applying CLT to determine the least approximate value of $N$
  + as $N \to \infty$

    \[ \sqrt{N} (\overline{\mathfrak{g}}_N - E[\mathfrak{g}(\theta) | Y]) \sim N\left(0, \sum_{\mathfrak{g}}\right) \tag{14} \]

  + the posterior variance (matrix) of $\mathfrak{g}(\theta)$ as $E[\| \mathfrak{g}(\theta)\|^2 | Y] < \infty$

    \[ \sum_{\mathfrak{g}} = E\big[\left(\mathfrak{g}(\theta) - E[\mathfrak{g}(\theta) | Y]\right)\cdot\left(\mathfrak{g}(\theta) - E[\mathfrak{g}(\theta)|Y]\right)^\prime | Y\big] \]

  + $\overline{\mathfrak{g}}_N = E[\mathfrak{g}(\theta) | Y] + O_p(N^{-1/2})$ and (stochastic) error bound: not depend on the dimension $m$ of $\theta$
  + a real-valued function, $\mathfrak{g}(\cdot) \xrightarrow{CTL}$ approximate value of $N$
    + $\exists\; \epsilon > 0$, what $N \to$ 95% probability $\mathfrak{g}(\theta) = E[\mathfrak{g}(\theta) | Y] \pm \epsilon$
    + using CLT, the 95% CI for $E[\mathfrak{g}(\theta) | Y] = \overline{\mathfrak{g}}_N \pm 1.96 \times \sigma_{\mathfrak{g}} / \sqrt{N}$
    + find $N$: 

      \[ \frac{1.96 \times \sigma_{\mathfrak{g}}}{\sqrt{N}} \leq \epsilon \longrightarrow N \geq \frac{(1.96)^2 \times \sigma_{\mathfrak{g}}^2}{\epsilon^2} \]

    + estimating the sample standard deviation

      \[ \sigma_{\mathfrak{g}}^2 \text{ unknown } \quad\xrightarrow{\text{estimate}}\quad \hat{\sigma}_{\mathfrak{g}}^2 = \frac{\sum_{l=1}^N \left(\mathfrak{g}(\theta^{(l)}) - \overline{\mathfrak{g}}_N \right)^2}{N-1} \]

  + Summary: an accuracy of $\epsilon > 0$ with 95% CI $\implies$

    \[ N \geq \frac{4\hat{\sigma}_{\mathfrak{g}}^2}{\epsilon^2} \]
  
  + examples:
    + $\sigma_{\mathfrak{g}}^2 = 1, \epsilon \approx 0.1 \implies N = 400$
    + $\sigma_{\mathfrak{g}}^2 = 1, \epsilon \approx 0.01 \implies N = 40,000$

  + generating a very large number of samples from the posterior distribution to obtain reasonable accuracy of the posterior means (or generalized moments)

+ Performance of MC methods
  + success of the MC method of integration: 
    + the resulting estimate ($\overline{\mathfrak{g}}_N$) of the posterior mean not fixed
    + depending on particular random number generating mechanism (e.g., seed)
  + error bound of integration methods:
    + MC integration: probabilistic
    + deterministic integration: fixed
  + MC methods
    + hardly improved
    + inferior to the deterministic integration methods as $m \leq 4$ and sufficiently smooth
  + deterministic methods more sufficient
    + smooth function $\mathfrak{g}(\theta)$
    + little dimension $m$ of $\theta$
  + MC methods more advanced
    + non-smooth $\mathfrak{g}(\cdot)$
    + complicated parameter space

+ Importance sampling
  + using another density to generate samples and then useing a weighted sample mean to approximate the posterior mean instead of $K(\theta; Y)$

    \[ \int \mathfrak{g}(\theta) K(\theta; Y) d\theta = \int \mathfrak{g}(\theta) \frac{K(\theta; Y)}{q(\theta)} q(\theta) d\theta = \int \mathfrak{g}(\theta) w(\theta) q(\theta) d\theta \tag{15} \]

    + $q(\theta)$: the _importance proposal_ density $\to \{ \theta \in \Theta: K(\theta; Y) > 0 \} \subseteq \{ \theta \in \Theta: q(\theta) > 0 \}$
    + $w(\theta)$: the _importance weight_ function
  + algorithm to estimate $I = I(Y)$
    1. generate $\theta^{(l)} \stackrel{iid}{\sim} q(\theta)$ for $l = 1, 2, \cdots, N$
    1. compute the importance weights $w^{(l)} = w(\theta^{(l)})$ for $l = 1, 2, \cdots, N$
    1. compute $\overline{I}_N = \frac{1}{N} \sum_{l=1}^N \mathfrak{g}(\theta^{(l)}w^{(l)})$
  + using LLN, $N \to \infty \implies \overline{I}_N \xrightarrow{p} I$
  + estimating the posterior mean

    \[ E[\mathfrak{g}\theta)|Y] \approx \overline{\mathfrak{g}}_N = \frac{\sum_{l=1}^N \mathfrak{g}(\theta^{(l)} w^{(l)})}{\sum_{l=1}^N w^{(l)}} \implies \frac{\sum_{l=1}^N w^{(l)}}{N} \xrightarrow[\text{in probability}]{\text{converge}} \int K(\theta; Y) d\theta \]

+ Markov Chain Monte Carlo (MCMC) methods
  + to resolve the choice of $q(\theta)$, in particular, high dimensional $\theta$ 
  + used to generate (dependent) samples from the posterior distribution, only using the posterior kernel $K(\theta; Y)$
  + based on sampling from the path of a (discrete time) Markov Chain $\{\theta^{(l)}: l = 1, 2, \dots\}$
  + the posterior distribution: stationary distribution of $\{\theta^{(l)}: l = 1, 2, \dots\}$
  + creating a suitable transition kernel of a Markov Chain
    + the Metropolis algorithm: most popular approaches
    + the Metropolis-Hasting algorithm: generalized version
  + generated samples
    + MCMC method: dependent
    + MC methods: independent sample
  + MCMC samples
    + not directly obtained from the posterior distribution
    + obtained in an asymptotic sense $\to$ discarding the first thousand samples
    + $\{\theta^{(l)}: l = 1, 2, \dots\} \to \{\theta^{(l)}: l = B+1, B+2, \dots\, B > 1\}$
  + estimating $\eta = \eta(\theta)$ based on samples $\theta^{(l)} \implies$ computing $\overline{\eta} = \sum_{l=B+1}^N \eta^{(l)} / (N-B) \approx E[\eta|Y], \exists\; N >> B >> 1$


## 6. Examples

+ Example: vitamin C cure within a week (cont.)
  + the posterior density: $p(Y|\theta) = \mathcal{L}(\theta; Y) = \mathcal{L}(\theta | s) = \theta^s (1 - \theta)^{n-s}$
  + the posterior kernel function: $K(\theta; Y) = K(\theta|s) = \theta^s (1-\theta)^{n-s} k(\theta)$
  + a prior kernel function: $k(\theta) \in [0, 1]$
  + task: estimating the odds ratio, $\rho = (\frac{\theta}{1-\theta})$, or the log-odds ratio, $\eta = \log \rho$
  + computing the posterior estimator: $\exists\; k(\theta) \to$

    \[ \hat{\eta} = \frac{\int_0^1 \log(\frac{\theta}{1-\theta})\theta^s(1-\theta)^{n-s} k(\theta) d\theta}{\int_0^1 \theta^s(1-\theta)^{n-s} k(\theta) d\theta} \tag{16} \]

  + choice of kernels
    + beta prior: $k(\theta) = \theta^{a-1}(1-\theta)^{b-1}, \; a, b >0$
      + Uniform prior: a = b = 1
      + Jeffery's prior: a = b = 0.5
      + Haldane's prior: a = b = 0; checking: valid w/ $s(n-s) > 0$, otherwise, improper
    + Zellner's prior: $k(\theta) = \theta^\theta (1-\theta)^{1-\theta}$
  + survey: 15 out of 20 randomly selected patients reported that common cold was cured using vitamin C within a week $\to n = 20, s = 15$
  + the response: $y_i \in \{0, 1\}$
  + patient level predictor variables: gender, age, etc. $\to z_i$, a vector
  + using a regression model to account for subjective level heterogeneity $\to \theta_i = \Pr(x_i = 1|z_i) = \theta(z_i), \; i = 1, \cdots, n$
  + parametric models for the regression function $\theta(\cdot)$
    + logistic regression model: $\theta(z) = (1 + \exp(-\beta^\prime z))^{-1}$
    + probit regression model: $\theta(z) = \Theta(\beta^\prime z)$
    + general regression model: $\theta(z) = F_0(\beta^\prime z), \;\exists\; F(\cdot) \in \mathbb{R}$
  + express the framework as BHM

    \[ x_i|z_i \sim Ber(\theta_i) \to \theta_i = F_0(\beta^\prime z_i) \to \beta \sim N(0, c_oI)\]

    + $c_0 > 0$: a large constant
    + $I$: identity matrix



