# Chapter 12 Bayesian Inference


Author: H. Liu and L. Wasserman

[Origin](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)

Year: 2014

Related Course: [36-708 Statistical Methods for Machine Learning](http://www.stat.cmu.edu/~larry/=sml/)


## 12.6 Examples Where Bayesian Inference and Frequentist Inference Disagree

+ Bayesian inference
  + Bayes's theorem: a natural way to combine prior information w/ data
  + Bayesian methods: providing no guarantees on long run performance
  + Bayesian methods w/ poor frequency behavior in some cases

+ Normal mean difference
  + sampling distribution: $\mathcal{D}_n = \{ X_1, \dots, X_n \}, \, X \sim N(\mu_i, 1)$
  + the prior: $\pi(\mu_1, \dots, \mu_n)$
  + task: estimate $\pmb{\mu} = (\mu_1, \dots, \mu_n)^T$
  + the posterior of $\pmb{\mu}$
    + multivariate Normal w/ mean $\mathbb{E}(\pmb{\mu} \,|\, \mathcal{D}_n) = (X_1, \dots, X_n)$
    + covariance = the identity matrix
  + $\theta = \sum_{i=1}^n \mu^2_i$
  + $\exists\, c_n \ni C_n = [c_n , \infty) \to \mathbb{P}(\theta \in C_n \,|\, \mathcal{D}_n) = 0.95$
  + frequentist sense

    \[ \mathbb{P}_\mu (\theta \in C_n) \to 0, \,\text{as } n \to \infty \]

  + a sharp difference btw $\mathbb{P}_\mu(\theta \in C_n)$ and $\mathbb{P}(\theta \in \mathcal{D}_n \,|\, \mathcal{D}_n)$





## 12.7 Freedman's Theorem





## 12.8 The Bayes-Frequestist Debate





## 12.9 Summary




