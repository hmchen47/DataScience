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

+ [Numerical methods](../Notes/p03-BayesianBasics.md#11-the-bayes-rule)
  + computing posterior summary estimates
  + binomial sampling density:
    + Zellner's prior: $p(\theta) = C\theta^\theta (1-\theta)^{1-\theta}$ for $\theta \in (0, 1)$ and C = 1.61857
    + Jeffery's prior: $Beta(a = 0.5, b = 0.5)$
  + Haldane's prior: improper prior $p(\theta) = [\theta(1-\theta)]^{-1}$
  + uniform prior: same as $Beta(a=1, b=1)$ w/ $\theta$ using binomial sampling density
  + posterior inference about $\theta$ or odds ratio $\rho = \theta/(1-\theta)$ or log-odds ratio $\eta = \log \rho$ relative insensitive to all previous priors

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





## Monte Carlo Methods

+ [Monte Carlo methods](../Notes/p01-Bayesian.md#3192-markov-chain-monte-carlo-methods)
  + a toolkit of techniques aiming on evaluating integrals or sums by simulation rather than exact or approximate algebraic analysis
  + a.k.a. probabilistic sensitivity analysis
  + the simulated quantities passed into a standard spreadsheet and the resulting distributions of the outputs of the spreadsheet reflecting the uncertainty about the inputs
  + used for Bayesian analysis provided the prior/current posterior distribution of concern is a member of a known family
  + conjugate Bayesian analysis: possible to derive such a posterior distribution algebraically
  + used to find tail areas or more usefully to find the distribution of complex functions of one or more unknown quantities as in the probabilistic sensitivity analysis







## Markov Chain Monte Carlo Methods

+ [Non-standard distributions and Bayesian approach](../Notes/p01-Bayesian.md#3192-markov-chain-monte-carlo-methods)
  + non-conjugate distribution or nuisance parameters
    + more complex Bayesian analysis
    + not possible to derive the posterior distribution in an algebraic form
  + Markov chain Monte Carlo methods: developed as a remarkably effective means of sampling from the posterior distribution of interest w/o knowing its algebraic form

+ [Essential components of MCMC methods](../Notes/p01-Bayesian.md#3192-markov-chain-monte-carlo-methods)
  + _replacing analytic methods by simulation_
    + observing some data $y$ to make inference about a parameter $\theta$ of interest
    + the likelihood $p(y | \theta, \psi)$ featuring a set of nuisance parameters $\psi$
    + Bayesian approach
      + a joint prior distribution $p(\theta, \psi)$
      + the joint posterior posterior $p(\theta, \psi | y) \propto p(y|\theta, \psi) p(\theta, \psi)$ 
      + integrating out the nuisance parameters to give the marginal posterior of interest

      \[ p(\theta | y) = \int p(\theta, \psi | y) d\psi \]

    + realistic situations: not a standard form but approximation
    + sampling from the joint posterior $p(\theta, \psi | y)$
    + sample values: $(\theta^{(1)}, \psi^{(1)}), (\theta^{(2)}, \psi^{(2)}), \dots, (\theta^{(j)}, \psi^{(j)}), \dots$
    + creating a smoothed histogram of all the sampled $\theta^{(j)}$ to estimate the shape of the posterior distribution $p(\theta|y)$
    + replacing analytic integration by empirical summaries of sample values
  + _sampling from the posterior distribution_
    + a wealth of theoretical work on ways of sampling from a joint posterior distribution
    + a joint posterior distribution proportional to likelihood $\times$ prior, $p(y|\theta, \psi) p(\theta, \psi)$
    + methods focusing on producing a _Markov chain_
    + Markov chain: the distribution for the next simulated value $(\theta^{(j+1)}, \psi^{(j+1)})$ depends only on the current $(\theta^{(j)}, \psi^{(j)})$
    + the theory of Markov chain: the samples eventually converging into an 'equilibrium distribution'
    + algorithms using $p(y|\theta, \psi) p(\theta, \psi)$ to ensure the equilibrium distribution exactly same as the posterior of interest, including Gibbs sampling and the Metropolis algorithm
  + _starting the simulation_
    + selecting initial values for unknown parameters
    + the choice of initial values: no influence on the eventual samples from the Markov chain
    + reasonable initial values
      + improving convergence
      + avoiding numerical problems
  + _checking convergence_
    + checking convergence of a Markov chain to its equilibrium distribution not straightforward
    + diagnosing lack of convergence simply by observing erratic behavior of the sample values
    + fact: a chain w/ a steady trajectory $\neq$ sampling from the correct posterior distribution
    + stuck in a particular area due to the choice of initial values
    + solution:
      + best to run multiple chains from a adverse set of initial values
      + formal diagnostics to check whether these chains end up
      + to expected chance variability, coming from the same equilibrium distribution which is assumed to be the posterior of interest



