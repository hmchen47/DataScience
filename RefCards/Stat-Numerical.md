# Statistics: Numerical Approaches

## Overview

+ [Computation and Bayesian approach](../Notes/p01-Bayesian.md#319-computational-issues)
  + full probability modeling
    + a.k.a. the Bayesian approach
    + applying probability theory to a model derived from substantive knowledge
    + able to deal w/ realistically complex situations
  + computational difficulties: <br/>w/ the specific problem being to carry out the integrations necessary to obtain the posterior distributions of quantities of interest in situations
    + non-standard prior distributions used
    + additional nuisance parameters in the model
  + solution: Markov Chain Monte Carlo (MCMC) methods




## Numerical Methods

+ [Un-pure Bayesian method](../Notes/p03-BayesianBasics.md#12-hierarchical-and-empirical-approaches)
  + issue: difficult to completely specify the prior density based on available expert opinion or background scientific information
  + utilizing hyperparametes to resolve
    + $\{p_1(\theta|\lambda): \lambda \in \Lambda\}$: a class of prior densities
    + $\lambda$: hyper-parameter
    + $p_1(\cdot | \cdot)$: a conditional density of $\theta$ given $\lambda$
  + methods to determine hyperparameter $\lambda \to$ empirical estimate $\hat{\lambda}$
    + Bayesian Hierarchical Model (BHM)
    + Empirical Bayes (EB) method
  + using $p(\theta) = p_1(\theta | \hat{\lambda})$ as the prior for $\theta$
  + criticizing: the prior density $p(\theta)$ obtained via the data $Y$

+ [Numerical methods](../Notes/p03-BayesianBasics.md#11-the-bayes-rule)
  + computing posterior summary estimates
  + binomial sampling density:
    + Zellner's prior: $p(\theta) = C\theta^\theta (1-\theta)^{1-\theta}$ for $\theta \in (0, 1)$ and C = 1.61857
    + Jeffery's prior: $Beta(a = 0.5, b = 0.5)$
  + Haldane's prior: improper prior $p(\theta) = [\theta(1-\theta)]^{-1}$
  + uniform prior: same as $Beta(a=1, b=1)$ w/ $\theta$ using binomial sampling density
  + posterior inference about $\theta$ or odds ratio $\rho = \theta/(1-\theta)$ or log-odds ratio $\eta = \log \rho$ relative insensitive to all previous priors

+ [Simulation methods for posterior distribution](../Notes/p04a-Bayesian.md#1229--calculating-the-posterior-distribution)
  + general procedure
    + drawing $\theta^1, \dots, \theta^B \sim p(\theta \,|\, \mathcal{D}_n)$
    + generating a histogram of $\theta^1, \dots, \theta^B$ approximates the posterior density $p(\theta \,|\, \mathcal{D}_n)$
  + approximation to the posterior mean: $\overline{\theta}_n = \mathbb{E}(\theta \,|\, \mathcal{D}_n) = B^{-1}\sum_{j=1}^B \theta^j$
  + approximation to posterior $(1-\alpha)$ interval: $(\theta_{\alpha/2}, \theta_{1 - \alpha/2})$ w/ $\theta_{\alpha/2}$ as the $\alpha/2$ sample quantile of $\theta^1, \dots, \theta^B$
  + $\exists\; \theta^1, \dots, \theta^B$ from $p(\theta | \mathcal{D}_n)$ and $\tau^i = g(\theta^i) \,\forall\, i=1, \dots, B \implies \tau^1, \dots, \tau^B$ as a sample from $p(\tau \,|\, \mathcal{D}_n)$

+ [Methods for obtaining simulated values from the posterior](../Notes/p04a-Bayesian.md#1229--calculating-the-posterior-distribution)
  + stochastic simulation methods
    + based on random sampling
    + typical approaches
      + Monte Carlo integration
      + importance sampling
      + Markov chain Monte Carlo (MCMC)
  + variational inference
    + based on deterministic approximation and numerical optimization
    + applied to wide range problems
    + very weak theoretical guarantees

+ [Simulation approach for distribution](../Notes/p04b-Bayesian.md#125-simulation-methods-for-bayesian-computation)
  + drawing sample $X$ from a distribution $F$
    + $F(X)$: uniform distribution over the interval (0, 1)
    + a basic strategy to sample $U \sim \text{Uniform}(0, 1) \to X = F^{-1}(U)$



## Numerical for Integrals

+ [Integrals for posterior inference](../Notes/p03-BayesianBasics.md#5-numerical-integration-methods)
  + issue: almost any posterior inference based on $K(\theta; Y) \to$ high-dimensional integration w/ $\Theta \in \mathbb{R}^m$, where $\mathbb{R}^m$ as a Euclidean
  space
  + ageneric function $\mathfrak{g}(\theta)$ of the parameter $\theta$, the posterior inference requiring the computation of the integral

    \[ I = I(Y) \int \mathfrak{g}(\theta) K(\theta; Y) d\theta \tag{11} \]

+ [Numerical methods for integrals](../Notes/p03-BayesianBasics.md#5-numerical-integration-methods)
  + numerical approaches
    + classical (deterministic) numerical methods
      + applied when $m < 5$
      + order of accuracy: $O(N^{-2/m})$
      + rate of convergence depending on $m$ (curse of dimensionality)
    + stochastic integration methods
      + applied when $m \geq 5$
      + known as Monte Carlo (MC) methods
      + order of accuracy: $O(^{-1/2})$
      + rate of convergence not depending on $m$
  + performance: deterministic >> stochastic

+ [Basic simulation approach to estimate integral](../Notes/p04b-Bayesian.md#125-simulation-methods-for-bayesian-computation)
  + estimating the integral $\int_0^1 h(x) \,dx$ for some complicated function $h$
  + drawing $N$ samples $X_i \sim \text{Uniform}(01, 1)$
  + estimating the integral

    \[ \int_0^1 h(x) \,dx \approx \frac{1}{N} \sum_{i=1}^N h(X_i) \]

  + converged to the desired integral by the law of large numbers




## Bayesian Hierarchical Model

+ [Bayesian hierarchical Model](../Notes/p03-BayesianBasics.md#12-hierarchical-and-empirical-approaches) (EBM)
  + assigning another prior distribution for $\lambda$:
    + $\lambda \sim p_2(\lambda)$
    + $p_2(\cdot)$: a probability density defined over $\Lambda$
  + $\therefore\; p(\theta) = \int p_1(\theta|\lambda) p_2(\lambda)d\lambda$: a more flexible prior for $\theta$
  + forming a hierarchical stages for the complete Bayes model

    \[ Y|\theta, \lambda \sim p(Y|\theta) \implies \theta|\lambda \sim p_1(\theta|\lambda) \quad \&\quad \lambda \sim p_2(\lambda)\]

  + often choosing $p_2(\cdot)$ suitable to obtain estimators w/ good frequentist properties
  + issue: inference based on $p(\theta|Y)$ requiring integration both on $\theta$ and $\lambda$
  + solutions: Monte Carlo methods (in particular, Markov Chain Monte Carlo (MCMC) methods) used to obtain samples from the posterior density $p(\theta|Y)$



## Empirical Bayes Method

+ [Empirical Bayes (EB) method](../Notes/p03-BayesianBasics.md#12-hierarchical-and-empirical-approaches)
  + based on marginal likelihood $m(Y|\lambda) = \int p(Y|\theta) p_1(\theta|\lambda) d\theta$
  + $m(Y|\lambda)$: the marginal likelihood function of $\lambda \implies$ maximizing

    \[ \hat{\lambda} = \hat{\lambda}(Y) = \underset{\lambda \in \Lambda}{\mathrm{argmin}}\; m(Y|\lambda) \]
  
  + alternatively, obtaining a moment based method to 'estimate' $\lambda$
    + $\lambda$: a $q$-dimensional parameter
    + using a set of $q$ suitable moments of $Y$ w.r.t. the marginal density $m(Y|\lambda)$
    + equate moments to the corresponding $q$ empirical moments of the data vector $Y = (y_1, y_2, \dots, y_n)$

+ [Simulation methods for Bayesian inference](../Notes/p04b-Bayesian.md#125-simulation-methods-for-bayesian-computation)
  + the posterior density

    \[ \pi(\theta \,|\, \mathcal{D}_n) = \frac{\mathcal{L}_n(\theta)\pi(\theta)}{c} \]

    + $\mathcal{L}_n(\theta)$: the likelihood function
    + the normalizing constant $c = \int \mathcal{L}_n(\theta) \pi(\theta) \,d\theta$
  + the posterior mean
  
    \[ \overline{\theta} = \int \theta \pi(\theta \,|\, \mathcal{D}_n) \,d\theta = \frac{1}{c} \int \theta\mathcal{L}_n(\theta) \pi(\theta) \,d\theta \]

  + the marginal posterior density for $\theta_i$

    \[ \pi(\theta_i \,|\, \mathcal{D}_n) = \int\int\cdots\int \pi(\theta_1,  \cdots, \theta_d) \,d\theta_1 \cdots d\theta_{i-1} d\theta_{i+1} \cdots d\theta_s \]




## Deterministic Methods

+ [Fundamentals of deterministic numerical methods](../Notes/p03-BayesianBasics.md#51-deterministic-methods)
  + computing integrals usually depending on some adaptation of the quadratic rules
  + goal: computing the integral

    \[ I = \int_a^b h(\theta) d \theta \]

  + task: finding suitable weights $w_i$'s and ordered knot points $\theta_j$'s $\to$

    \[ I = \int_a^b h(\theta)d\theta \approx I_N = \sum_{j=1}^N w_j h(\theta_j) \tag{12} \]

  + $\exists\; N \to \infty: |I - I_n| \to 0$
  + different choices of the weights and the knot points $\to$ different integration rules
  + categories of the numerical quadrature rules
    + Newton-Cotes rules: based on equally spaced knot points
    + Gaussian quadrature rules
      + based on knot points obtained via the roots of the orthogonal polynomials
      + more accurate results, especially $h(\cdot)$ approximates reasonably well by a sequence of polynomials

+ [Newton-Cotes rules](../Notes/p03-BayesianBasics.md#51-deterministic-methods)
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

+ Multi-dimensional numerical integration rules
  + multidimensional trapezoidal rule

    \[ I = I_N + O(N^{2/m}) \]

  + multidimensional Simpson's rule: $O(N^{-4/m})$
  + Monte Carlo methods: $O_p(N^{-1/2})$
  + accuracy of integration: $m \nearrow \;\;\to$ accuracy $\searrow$

+ [Summary of deterministic numerical integration methods ] (../Notes/p03-BayesianBasics.md#51-deterministic-methods)
  + very accurate but declined rapidly as $m \nearrow$
  + invalid theoretical error estimate
    + a complicated parameter space
    + insufficiently smooth function



## Importance Sampling

+ [Importance sampling](../Notes/p03-BayesianBasics.md#52-monte-carlo-methods)
  + using another density to generate samples and then useing a weighted sample mean to approximate the posterior mean instead of $K(\theta; Y)$

    \[ \int \mathfrak{g}(\theta) K(\theta; Y) d\theta = \int \mathfrak{g}(\theta) \frac{K(\theta; Y)}{q(\theta)} q(\theta) d\theta = \int \mathfrak{g}(\theta) w(\theta) q(\theta) d\theta \tag{15} \]

    + $q(\theta)$: the _importance proposal_ density $\to \{ \theta \in \Theta: K(\theta; Y) > 0 \} \subseteq \{ \theta \in \Theta: q(\theta) > 0 \}$
    + $w(\theta)$: the _importance weight_ function
  + algorithm to estimate $I = I(Y)$
    1. generate $\theta^{(l)} \stackrel{iid}{\sim} q(\theta)$ for $l = 1, 2, \cdots, N$
    1. compute the importance weights $w^{(l)} = w(\theta^{(l)})$ for $l = 1, 2, \cdots, N$
    1. compute $\overline{I}_N = \frac{1}{N} \sum_{l=1}^N \mathfrak{g}(\theta^{(l)}w^{(l)})$
  + estimating the posterior mean

    \[ E[\mathfrak{g}\theta)|Y] \approx \overline{\mathfrak{g}}_N = \frac{\sum_{l=1}^N \mathfrak{g}(\theta^{(l)} w^{(l)})}{\sum_{l=1}^N w^{(l)}} \implies \frac{\sum_{l=1}^N w^{(l)}}{N} \xrightarrow[\text{in probability}]{\text{converge}} \int K(\theta; Y) d\theta \]

+ [Modeling of importance sampling](../Notes/p04b-Bayesian.md#1252-importance-sampling)
  + $\exists\; g$ a probability density w/ known distribution
  + the integral

    \[ I = \int h(x) f(x) dx = \int \frac{h(x) f(x)}{g(x)} g(x) dx = \mathbb{E}_g (W) \]

    + $W = h(X)f(X)/g(X)$
    + $\mathbb{E}_g(W)$: the expectation w.r.t $g$

  + importance sampling:
    + simulate $X_1, \dots, X_N \sim g$ 
    + estimate $I$ by the sample average

      \[ \widehat{I} = \frac{1}{N} \sum_{i=1}^N Y_i = \frac{1}{N} \sum_{i=1}^N \frac{h(X_i)f(X_i)}{g(X_i)} \]

  + by the law of large number: $\widehat{I} \xrightarrow{P} I$

+ [Guideline to importance sampling](../Notes/p04b-Bayesian.md#1252-importance-sampling)
  + summary: a good choice for an importance sampling density $g$ should be similar to $f$ but w/ thicker tails
  + __Theorem__: the choice of $g$ that minimizes the variance of $\widehat{I}$ is

    \[ g^*(x) = \frac{|h(x)| f(x)}{\int |h(s)| f(s) ds} \]

+ [Modeling importance sampling in Bayesian inference](../Notes/p04b-Bayesian.md#1252-importance-sampling)
  + $\theta_1, \dots, \theta_N$ as a sample from $g$, the posterior mean

    \[ \mathbb{E}[\theta \,|\, X_1, \dots, X_n] \approx \frac{\frac{1}{N} \sum_{i=1}^N h_1(\theta_i)}{\frac{1}{N} \sum_{i=1}^N h_2(\theta_i)} \]

  + very difficult to choose a good importance sampler $g$, especially in high dimensions




## Monte Carlo Methods

+ [Monte Carlo methods](../Notes/p01-Bayesian.md#3192-markov-chain-monte-carlo-methods)
  + a toolkit of techniques aiming on evaluating integrals or sums by simulation rather than exact or approximate algebraic analysis
  + a.k.a. probabilistic sensitivity analysis
  + the simulated quantities passed into a standard spreadsheet and the resulting distributions of the outputs of the spreadsheet reflecting the uncertainty about the inputs
  + used for Bayesian analysis provided the prior/current posterior distribution of concern is a member of a known family
  + conjugate Bayesian analysis: possible to derive such a posterior distribution algebraically
  + used to find tail areas or more usefully to find the distribution of complex functions of one or more unknown quantities as in the probabilistic sensitivity analysis

+ [MC statistical methods](../Notes/p03-BayesianBasics.md#52-monte-carlo-methods)
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

  + $N \nearrow  \implies \overline{\mathfrak{g}}_N \to E[\mathfrak{g}(\theta) | Y]$; $\overline{\mathfrak{g}}_N$ = sample mean, $E[\mathfrak{g}(\theta)$: population mean
  + almost no smoothness condition required on the function $\mathfrak{g}(\cdot)$ to apply the MC method

+ [Sample size](../Notes/p03-BayesianBasics.md#52-monte-carlo-methods)
  + applying CLT to determine the least approximate value of $N$
  + as $N \to \infty$

    \[ \sqrt{N} (\overline{\mathfrak{g}}_N - E[\mathfrak{g}(\theta) | Y]) \sim N\left(0, \sum_{\mathfrak{g}}\right) \tag{14} \]

  + $\overline{\mathfrak{g}}_N = E[\mathfrak{g}(\theta) | Y] + O_p(N^{-1/2})$ and (stochastic) error bound: not depend on the dimension $m$ of $\theta$
  + Summary: an accuracy of $\epsilon > 0$ with 95% CI $\implies$

    \[ N \geq \frac{4\hat{\sigma}_{\mathfrak{g}}^2}{\epsilon^2} \]
  
+ [Performance of MC methods](../Notes/p03-BayesianBasics.md#52-monte-carlo-methods)
  + error bound of integration methods:
    + MC integration: probabilistic
    + deterministic integration: fixed
  + MC methods more advanced
    + non-smooth $\mathfrak{g}(\cdot)$
    + complicated parameter space

+ [Numerical method for integral](../Notes/p04b-Bayesian.md#1251-basic-monte-carlo-integration)
  + $\exists\, \text{ a function } h$, to evaluate the integral

    \[ I = \int_a^b h(x) dx \]

  + numerical techniques for evaluating $I$
    + Simpson's rule
    + the trapezoidal rule
    + Gaussian quadrature
    + Monte Carlo integration
  
+ [Basic Monte Carlo integration method](../Notes/p04b-Bayesian.md#1251-basic-monte-carlo-integration)
  + approximating $I$ which is notable for its simplicity, generality and scalability
  + $\exists\; w(x) - h(x) (b-a), f(x) = 1/(b-a) \ni$
  
    \[ I = \int_a^b h(x) dx = \int_a^b w(x) f(x) dx \]

  + generating $X_1, \dots, X_N \sim \text{Uniform}(a, b) \implies$ the the law of large numbers

    \[ \widehat{I} \equiv \frac{1}{N} \sum_{i=1}^N w(X_i) \xrightarrow{P} \mathbb{E}(w(X)) = I \]

  + the standard error of the estimate

    \[ \widehat{se} = \frac{s}{\sqrt{N}}, \qquad s^2 = \frac{\sum_{i=1}^N (\widehat{Y}_i - \widehat{I})^2}{N - 1} \;\;\text{ and }\;\; Y_i = w(X_i) \]

  + region estimate: $(1 - \alpha)$ confidence interval = $\widehat{I} \pm z_{\alpha/2} \cdot \widehat{se}$
  + $N \nearrow  \implies CI \searrow$

+ [Issue on basic Monte Carlo method](../Notes/p04b-Bayesian.md#1252-importance-sampling)
  + issue: no guarantee $\pi(\theta | \mathcal{D}_n)$ w/ a known distribution
  + solution: importance sampling - a generalization of basic Monte Carlo



## Monte Carlo Method for Integral

+ [Generalized Monte Carlo integration methods](../Notes/p04b-Bayesian.md#1251-basic-monte-carlo-integration)
  + the integral w/ $f(x)$ a probability density function

    \[ I = \int_a^b h(x) f(x) dx \]

  + special case: $f \sim \text{Uniform}(a, b)$
  + drawing $X_1, \dots, X_N \sim f$

    \[ \widehat{I} := \frac{1}{N} \sum_{i=1}^N h(X_i) \]



## Monte Carlo Method for Gaussian Distribution

+ [Numerical methods for Gaussian density](../Notes/p04b-Bayesian.md#1251-basic-monte-carlo-integration)
  + the standard normal PDF

    \[ f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \]

  + computing the CDF at some point $x$

    \[ I = \int_{-\infty}^x f(s) ds = \Phi(x) \implies I = \int h(s) f(s) ds, \quad h(s) = \begin{cases} 1 & s < x \\ 0 & s \geq x  \end{cases} \]

  + generating $X_1, \dots, X_N \sim N(0, 1)$ and 

    \[ \widehat{I} = \frac{1}{N} \sum_i h(X_i) = \frac{\text{number of observations w/ their values} \leq x}{N} \]



## Monte Carlo Method for Two Binomials

+ [Bayesian inference for two binomial](../Notes/p04b-Bayesian.md#1251-basic-monte-carlo-integration)
  + freqentist analysis
    + $X \sim \text{Binomial}(n, p_1), \,Y \sim \text{Binomial}(m, p_2)$
    + task: to estimate $\delta = p_2 - p_1$
    + the MLE: $\widehat{\delta} = \widehat{p}_1 - \widehat{p}_2 = (Y/m) - (X/n)$
    + the standard error $\widehat{se}$ using the delta method

      \[ \widehat{se} = \sqrt{\frac{\widehat{p}_1 (1 - \widehat{p}_1)}{n} + \frac{\widehat{p}_2 (1 - \widehat{p}_2)}{m}} \]

    + 95% confidence interval: $\widehat{\delta} \pm 2 \cdot \widehat{se}$
  + Bayesian analysis
    + the prior: $\pi(p_1, p_2) = \pi(p_1) \pi(p_2) = 1 \implies$ a flat prior on $(p_1, p_2)$
    + the posterior

      \[ \pi(p_1, p_2 \,|\, X, Y) \propto p_1^X(1 - p_1)^{n - X} p_2^Y (1 - p_2)^{m-Y} \]

    + the posterior mean of $\delta$

      \[ \overline{\delta} = \int_0^1\int_0^1 \delta(p_1, p_2) \pi(p_1, p_2 \,|\, X, Y)\,dp_1 dp_2 = \int_0^1\int_0^1 (p_2 - p_1) \pi(p_1, p_2 \,|\, X, Y) \,dp_1 dp_2 \]

    + obtaining the posterior CDF w/ $A = \{ (p_1, p_2): p_2 - p_1 \leq c \}$

      \[ F(c \,|\, X, Y) = \mathbb{P}(\delta \leq c \,|\, X, Y) = \int_A \pi(p_1, p_2 \,|\, X, Y) \]

    + then differentiate $F \to$ too complicated
  + simulation approach
    + $\pi(p_1, p_2 \,|\, X, Y) = \pi(p_1 \,|\, X) \pi(p_2 \,|\, Y) \implies p_1, p_2$ independent under the posterior distribution
    + simulate $(P_1^{(1)}, P_2^{(1)}), \dots, (P_1^{(N)}, P_2^{(N)})$ from the posterior by drawing ($i = 1,  \dots, N$)

      \[\begin{align*}
        P_1^{(i)} \sim \text{Beta}(X + 1, n - X + 1) \\
        P_2^{(i)} \sim \text{Beta}(Y + 1, n - Y + 1)
      \end{align*}\]

    + $\delta^{(i)} = P_2^{(i)} - P_1{(i)} \implies$

      \[ \overline{\delta} \approx \frac{1}{N} \sum_{i=1}^N \delta^{(i)} \]

    + 95% posterior interval for $\delta$:
      + sorting the simulated values
      + finding the .025 and .975 quantile
    + the posterior density $f(\delta \,|\, X, Y)$
      + applying density estimation techniques to $\delta^{(1)}, \dots, \delta^{(N)}$
      + plotting a histogram



## Monte Carlo Method w/ Multiparameters

+ Bayesian inference for multiple parameters
  + task: to estimate the dose at which the animals w/ 50% chance of dying, LD50
  
    \[ \delta = x_j^*  \qquad j^* = \min\{ j: p_j \geq 1/2 \} \]

  + $\delta$ implicitly just a complicated function of $p_1, \dots, p_{10} \to \delta = g(p_1, \dots, p_{10})$
  + known $(p_1, \dots, p_{10}) \implies$ find $\delta$
  + the posterior mean of $\delta$ w/ $A = \{ (p_1, \dots, p_{10}): p_1 \leq \cdots \leq p_{10}) \}$

    \[ \int\int\cdots\int_A g(p_1, \dots, p_{10})\pi(p_1, \dots, p_{10} \,|\, Y_1, \dots, Y_{10}) \,dp_1 dp_2 \cdots dp_{10} \]

  + the posterior CDF of $\delta$ w/ $B = A \cup \{ (p_1, \dots, p_{10}): g(p_1, \dots, p_{10}) \leq c \}$

    \[\begin{align*}
      F(c \,|\, Y_1, \dots, Y_{10}) &= \mathbb{P}(\delta \leq c \,|\, Y_1, \dots, Y_{10}) \\
      &= \int\int \cdots\int_B \pi(p_1, \cdots, p_{10} \,|\, Y_1, \dots, Y_{10}) dp_1 dp_2 \cdots dp_{10}
    \end{align*}\]

  + simulation procedure
    + taking a flat prior truncated over $A$
    + each $P_i \sim$  Beta distribution
    + drawing from the posterior
      + drawing $P_i \sim \text{Beta}(Y_i + 1, \,n - Y_i +1), \;i = 1, \dots, 10$
      + if $P_1 \leq P_2 \leq \dots \leq P_{10}$ keeping this draw.  Otherwise, throw it away and draw again until getting one to keep
      + let $\delta = x_j^\ast$ w/ $j^\ast = \min \{ j: P_j > 1/2 \}$
    + repeat $N$ times to get $\delta{(1)}, \dots, \delta^{(N)}$ and take

      \[ \mathbb{E}(\delta \,|\, Y_1, \dots, Y_{10}) \approx \frac{1}{N} \sum_{i=1}^N \delta^{(i)} \]

    + estimate the probability mass function as $\delta$ a discrete variable

      \[ \mathbb{P}(\delta = x_j \,|\, Y_1, \dots, Y_{10}) \approx \frac{1}{N} \sum_{i=1}^N I(\delta^{(i)} = x_j) \]





## Markov Chain Monte Carlo Methods

+ [Non-standard distributions and Bayesian approach](../Notes/p01-Bayesian.md#3192-markov-chain-monte-carlo-methods)
  + non-conjugate distribution or nuisance parameters
    + more complex Bayesian analysis
    + not possible to derive the posterior distribution in an algebraic form
  + Markov chain Monte Carlo methods: developed as a remarkably effective means of sampling from the posterior distribution of interest w/o knowing its algebraic form

+ [Essential components of MCMC methods](../Notes/p01-Bayesian.md#3192-markov-chain-monte-carlo-methods)
  + _replacing analytic methods by simulation_
  + _sampling from the posterior distribution_
  + _starting the simulation_
  + _checking convergence_

+ [Markov Chain Monte Carlo (MCMC) methods](../Notes/p03-BayesianBasics.md#52-monte-carlo-methods)
  + to resolve the choice of $q(\theta)$, in particular, high dimensional $\theta$ 
  + used to generate (dependent) samples from the posterior distribution, only using the posterior kernel $K(\theta; Y)$
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

+ [Markov chain Monte Carlo methods](../Notes/p04b-Bayesian.md#1253-markov-chain-monte-carlo-mcmc)
  + constructing a Markov chain $X_1, X_2, \dots$ w/ $f$ as their stationary distribution
  + problem: estimating the integral $I = \int h(x) f(x) dx$
  + under certain conditions, the integral following

    \[ \frac{1}{N} \sum_{i=1}^N h(X_i) \xrightarrow{P} \mathbb{E}_f(h(X)) = I \]

+ [Interpretation of Markov chain](../Notes/p04b-Bayesian.md#1254-why-it-works)
  + a Markov chain w/ the transition kernel $p(y|x)$ = the probability of transiting from $x$ to $y$
  + $f(x) = \int f(y) p(x|y) dy \implies f$ is a stationary distribution for the Markov chain
  + $f$ as a stationary distribution for a Markov chain $\implies$ the data from a sample run of the Markov chain approximating the distribution $f$
  + designing a Markov chain w/ stationary distribution $f \implies$ running the Markov chain and using the resulting data as if it were a sample from $f$




## Metropolis-Hastings Algorithm

+ [Metropolis-Hastings algorithm](../Notes/p04b-Bayesian.md#1253-markov-chain-monte-carlo-mcmc)
  + __algorithm__

    <div style="padding-left: 1em; padding-top: 0.2em;">choose $X_0$ arbitrary </div>
    <div style="padding-left: 1em; padding-top: 0.2em;">Given $X_0, X_1, \dots, X_i$, generate $X_{i+1}$ as following: </div>
      <div style="padding-left: 2em; padding-top: 0.2em;">1. generate a proposal or candidate value $Y \sim q(y | X_i)$</div>
      <div style="padding-left: 2em; padding-top: 0.2em;">2. Evaluate $r \equiv r(X_i, Y)$ where</div>
      <div style="text-align: center; padding-top: 0.2em;">$r(x, y) = \min \{ \frac{f(y)}{f(x)}\,\frac{q(x|y)}{q(y|x)}, \; 1 \}$</div>
      <div style="padding-left: 2em; padding-top: 0.2em;">3. Set</div>
      <div style="text-align: center; padding-top: 0.2em;">$X_{i+1} = \begin{cases} Y & \text{with probability } r \\ X_i & \text{with probability } 1 - r \end{cases}$</div><br/>

  + simple way to execute step 3: generate $U \sim \text{Uniform}(0, 1)$
  + common choice: 
    + $q(Y|x) \sim N(x, b^2), \,b >0 \ni$ the proposal drawing from a normal, centered at the current value
    + the proposal density $q$ symmetric $\implies q(y|x) = q(x|y)$ and

      \[ r = \min\left\{ \frac\left{f(Y)}{f(X_i)}, \,1 \right\} \]

  + constructed $X_0, X_1, \dots \to$ Markov chain
  + the chain mixing well: the sample from the Markov chain starts look like the target distribution $f$ quickly
    + tuning parameter ($b$) $\implies$ the efficiency and mixing-well of the chain



