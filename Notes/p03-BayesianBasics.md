# Basics of Bayesian Methods

Author: Sujit Ghosh

Year: 2010

[Origin: Chapter 3, pp 155-178](https://www.researchgate.net/profile/Sujit_Ghosh4/publication/45283465_Basics_of_Bayesian_Methods/links/55cce51208ae1141f6b9e8e0/Basics-of-Bayesian-Methods.pdf?_sg%5B0%5D=RYx6EbikVeUbU3ozrXtkqLCXVIvQ0o9eDAKjgFrlujf_APZQowfysbsEnxzlzNaAGSL_YAqpV3FISn1Ucnub-Q.zj6VTWH_eNFhLuRj667jIXPAS4C1jYg7-5hlP55xHIjoJgAlVjaZs2-2jKFIcdC5pA8X7phiUtuf1MWdW8vSzw&_sg%5B1%5D=ERHR_2NWgCe1_zXptdBuyYhmnsX2-0Fj7q-wgrhFbPoQUwwodDpYJCWdsx0A3G8tO_0vbrUSMJNB3N4vClwDV5zDN7rslAQFGr2vFe7JtNOh.zj6VTWH_eNFhLuRj667jIXPAS4C1jYg7-5hlP55xHIjoJgAlVjaZs2-2jKFIcdC5pA8X7phiUtuf1MWdW8vSzw&_sg%5B2%5D=UVSglZoLJvV38KMGeYnGO7ba8kwqYXxzjS1aeHAmZhGOi0LeelZIl7p2M23nf64xGDSJGlJXk6HgW0g.kNxpVr303YvmrXDjZyu-m_IYu_Hhy1anVHmfDHFcriS-WmXE2cPPICYY325bbN5RVkkfZt3EpUgIdd0N70d1Pw&_iepl=)



Book: Bang H., Zhou X., van Epps H., Mazumdar M. (eds) [Statistical Methods in Molecular Biology](http://booksdl.org/get.php?md5=ffb73f3149241e26f224462391bdb64b&key=DMCZ35DZ93WB4E14&mirr=1). Methods in Molecular Biology (Methods and Protocols), vol 620. Humana Press, Totowa, NJ


## 1. Introduction

+ Bayesian inference method
  + providing a logical framework to utilize all available sources of information in making a decision when analyzing data
  + prior knowledge: experience, expert pinion, or collected data in previous studies w/ similar protocol

+ Subjectivity
  + Bayesian statistical methods criticized as being subjective
  + almost any scientific knowledge and theories at best: subjective in nature
  + S. Press and M. Tanur, [Subjectivity of Scientist and the Bayesian Approach](http://booksdl.org/get.php?md5=c94435049cbb7a46c6344e4e6e33739a&key=C3C2V5CV3735F9RE&mirr=1), 2001
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

+ Byes theorem
  + scalar form
    + $p(y|\theta)$: the sampling density of $y \in \mathbb{Y}$ for a given value $\theta \in \Theta$
    + $p(\theta)$: the prior density of $\theta$
    + the conditional density of $\theta$ given $y$

      \[ p(\theta|y) = \frac{p(y|\theta) p(\theta)}{\int p(y|theta)p(\theta)d\theta} = \frac{p(y|\theta)p(\theta)}{m(y)} \tag{1} \]

    + $p(\theta|y)$: the posterior density of $\theta$ given y
    + $m(y) = \int p(y|\theta) p(\theta)$: marginal density of $y$

  + vector form
    + $Y = (y_1, y_2, \cdots, y_n$)$:  a vector of response
    + $p(Y|\theta)$: the joint density of $Y$ given $\theta$
    + the conditional density of $\theta$ given $Y$

      \[ p(\theta|y) = \frac{p(y|\theta) p(\theta)}{\int p(Y|theta)p(\theta)d\theta} = \frac{p(Y|\theta)p(\theta)}{m(y)} \]

  + independently and identically distributed (iid) observations
    + $y_i|\theta \stackrel{\text{iid}}{\sim} p(y|\theta),\; \forall i=1, \dots, n \implies$ $f(Y|\theta) = \prod_{i=1}^n p(y_i | \theta)$
    + More general, the conditional density $p_i(y_1, \dots, y_{i-1}, \theta)$ of $y_i$ given $y_1, \dots, y_{i-1}$ and $\theta$ for $i=2, 3, \dots, n \implies f(Y|\theta) = p_1(x_1|\theta) \prod_{i=1}^n p(x_i|\theta)$
     vector form: replacing $p(y|\theta)$ by $p(Y|\theta)$ fro Eq.(1)

+ Bayesian Model
  + an observed data set $Y$ consisting two quantities
    + $p(Y|\theta)$: sampling density
    + $p(\theta)$: prior density
  + summary

    \[\begin{array}{ll}
      \text{prior density for } \theta: & p(\theta) \\
      \text{sampling density of } y \text{ given } \theta: & p(y|\theta) \\
      \text{marginal density of } y: & m(x) = \int p(y|\theta)p(\theta) d\theta \\
      \text{posterior density of } \theta \text{ given } y: & p(\theta | y) = p(y|\theta)/m(y)
    \end{array}\]
  
+ Marginal density $m(y)$
  + discrete version: replacing integration by summation
  + the marginal density and the posterior density w.r.t a $\sigma$-finite measure; e.g., Lebesgue measure or counting measure
  + solution: suppressing the domain of integration of $\Theta$ while defining $m(y)$ by assuming  that $p(\theta) = 0\;\forall\; \theta \notin \Theta$

+ Kernel functions
  + prior kernel function
    + $p(\theta) = C \cdot k(\theta)$$
      + $C = (\int k(\theta)d\theta)^{-1} > 0$: normalizing constant
      + $k(\cdot)$: a non-negative function defined on the parameter space $\Theta$
    + examples 
      + $\theta^{5.2} (1-\theta)^2.5 I_{(0, 1)}(\theta)$: a kernel function of beta distribution w/ shape parameters 6.2 and 3.5
      + $\theta^5e^{-\theta} I_{(0, \infty)}(\theta)$: a kernel function of gamma distribution w/ parameters 6 and 1
      + indicator function of the set $A$

        \[I_A(\theta) = \begin{cases} 1 & \theta \in A \\ 0 & \theta \notin A \end{cases}\]

  + likelihood function
    + satisfied

      \[ p(Y|\theta) =  C(Y)\mathcal{L}(\theta; Y) \]

      + $C(Y) > 0$: a function of data $Y$ only
      + $\mathcal{L}(\theta; Y)$: the likelihood function of $\theta$
    + example
      + $y_i|\theta \stackrel{\text{iid}}{\sim} N(\theta_1, \theta_2)$ where $\theta \in (\theta_1, \theta_2) \in \Theta = \mathbb{R} \times (0, \infty)$
      + $y_i$ normal distribution w/ mean $\theta_1$ and variance $\theta_2$
      + posterior density: $p(y|\theta) = (2\pi \theta_2)^{-1/2} \exp\left(-(x - \theta_1)^2/2\theta_2\right)$
      + likelihood function: $\mathcal{L}(\theta; Y) = \theta_2^{-n/2} \exp\left(-\sum_{i=1}^n (y_i - \theta_1)^2/2\theta_2 \right)$
  + posterior kernel function

    \[ K(\theta; Y) = \mathcal{L}(\theta; Y) k(\theta) \tag{2} \]

    + $\mathcal{L}(\theta, Y): the likelihood function of $\theta$
    + $k(\theta)$: a prior kernel function of $\theta$

+ The posterior density
  + representation w/ kernel functions

    \[ p(\theta|Y) = \frac{K(\theta; Y)}{\int K(\theta; Y) d\theta} = \frac{\mathcal{L}(\theta; Y) k(\theta)}{\int \mathcal{L}(\theta; Y) k(\theta) d\theta} \tag{3} \]

  + Bayesian Mantra: posterior is proportional to likelihood times prior kernel

    \[p(\theta|Y) \propto \mathcal{L}(\theta; Y) k(\theta) \]0

+ Improper prior
  + def: prior kernel function $k(\theta) \geq 0$ for which $\int C \cdot k(\theta) d\theta  = \infty$
  + determined only up to a constant: $\int k(\theta) d\theta \implies \int C \cdot k(\theta) = \infty \;\forall\; C > 0$
  + not an improper prior (probability) distribution as $p(\theta)$ no longer a (probability) density function
  + checking the posterior still proper for almost all data if the improper prior used
  + using improper prior $p(\theta) = C \cdot k(\theta) \implies$ verifying the posterior kernel finitely integrable $\int K(\theta; Y) d\theta < \infty$ for almost all $Y$
  + Applying Fubini's theorem (on interchanging order of integration):
    + improper prior $\iff$ improper marginal
    + $\int k(\theta) d\theta = \infty \iff  \int m(Y) dY = \infty$
  + __Lemma__: If the likelihood function is bounded below, i.e., $\inf_\theta \mathcal{L}(\theta; Y) \geq \mathcal{L}_0(Y)$ for some $\mathcal{L}_0(Y) >0$, then any improper prior leads to an improper posterior

+ Example: Does vitamin C cure common cold?
  + a randomly chosen group of patients suffering from common col d took (same amount of) vitamin C for 1 week and response whether or not vitamin C cured common cold immediately following the week
  + Notations & Assumptions
    + $y$: response w/ 1 as common cold cured and 0 otherwise
    + $\theta$ = Pr[common cold cured within a week]: the parameter of interest
    + outcome of a given patient: Bernoulli distribution $p(y|\theta) = \theta^y(1-\theta)^{1-y}$ w/ $x = 0, 1$ and $\theta \in [0, 1]$
    + $n$ i.i.d. observations: $y_i \in [0, 1]$
    + $Y = (y_1, y_2, \cdots, y_n)$: the response vector
    + $s$: the total number of patents (out of n) cured within a week on taking vitamin C
  + the joint density of the response vector $Y$: $p(Y|\theta) = \prod_{i=1}^n p(y_i|\theta) = \theta^s(1-\theta)^{n-s}$
  + $s \sim Bin(n, \theta)$, binomial distribution w/ density function $p(s|\theta) = \begin{pmatrix} n \\ s \end{pmatrix} \theta^s (1-\theta){n-s}, \text{ for } s=0, 1, \cdots, n$
  + the posterior density w/ any prior density $p(\theta)$

    \[p(\theta|Y) = \frac{\theta^s(1-\theta)^{n-s} p(\theta)}{\int_0^1 \theta^s (1-\theta)^{n-s} p(\theta) d\theta} \tag{4}\]
  + remark
    + $s = \sum_i y_i$: the _sufficient statistic_; the posterior distribution $p(\theta|Y)$ depends on the data $Y$ only through $s$
    + $p(\theta|Y) = p(s|\theta) \implies$ the conditional density of $\theta$ given only the sum $s$ (and sample size $n$)
    + applying Fisher-Neyman factorization theorem: $f(Y|\theta)$ = the joint density of $X$ given $\theta$ and $S = S(Y)$ as a sufficient statistic (vector) $\implies p(\theta | X) = p(\theta|S)$ for any prior density $p(\theta)$

+ Conjugate family of prior
  + prior distribution: $p(\theta) = C \theta^{a-1}(1-\theta)^{b-1}$ w/ $C = C(a, b)$ and $a > 0, b > 0$ unknown quantities
  + posterior distribution: $C^\ast \theta^{a^\ast -1} (1-\theta)^{b^\ast - 1}$ w/ $C^\ast = (a^\ast, b^\ast), \; a^\ast = (a+s), b^\ast = (b + s)$
  + $C(a, b) = \Gamma(a+b)/\Gamma(a)\Gamma(b)$,  where gamma function $\Gamma(a) = \int_0^\infty t^{a-1}e^{-t} dt$
  + the prior $\theta \sim Beta(a, b) \implies$ the posterior $\theta|Y \sim Beta(a^\ast, b^\ast)$
  + remark:
    + conjugate family: prior density and its leading posterior densities belonging to the same family
    + example: binomial sampling density
      + $s|\theta \sim Bin(n, \theta) \in \{Beta(a, b): a > 0, b>0\}$ form a conjugate family of beta densities
      + the choice of conjugate family not unique
      + $\{C(a, b)\theta^{a-1} (1-\theta)^{b-1}\p_0(\theta}: a>0, b>0\}$ forms a conjugate family
        + $p_0(\cdot)$to be any nonnegative continuous function defined on $[0, 1]$
        + normalizing constant: $C(a, b) = \left(\int_0^1 \theta^{a-1}(1-\theta)^{b-1} p_0(\theta) d\theta\right)^{-1}$
  + exponential family of densities: sampling density w/ a sufficient statistic of constant dimension always finds a conjugate family of prior desnities
  + _natural conjugate family_:
    + constructing a class of such family for i.i.d. observations obtaining from a sampling density $p(y|\theta)$ for $y \in \mathbb{Y}$ and $\theta \in \Theta$
    + by simply defining the family of prior densities as $\{\prod_{j=1}^m p(y_j^0|\theta): y_j^0 \in \mathbb{X}, m = m_0, m_0 + 1, \dots\}$
    + provided $\exists m_0 \in \mathbb{N} \ni \int \prod_{j=1}^{m_0} p(x_j^0 | \theta) d\theta < \infty$
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
  + Haldane's prior: improper prior $p(\theta) = [\theta(1-\theta)]^{-1$
  + uniform prior: same as $Beta(a=1, b=1)$ w/ $\theta$ using binomial sampling density
  + posterior inference about $\theta$ or odds ratio $\rho = \theta/(1-\theta)$ or log-odds ratio $\eta = \log \rho$ relative insensitive to all previous priors


### 1.2 Hierarchical and Empirical Approaches





## 2. Point Estimation






## 3. Hypothesis Testing






## 4. Region Estimation





## 5. Numerical Integration Methods





### 5.1 Deterministic Methods





### 5.2 Monte Carlo Methods





## 6. Examples




