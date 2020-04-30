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




