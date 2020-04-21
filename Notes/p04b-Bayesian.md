# Chapter 12 Bayesian Inference


Author: H. Liu and L. Wasserman

[Origin](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)

Year: 2014

Related Course: [36-708 Statistical Methods for Machine Learning](http://www.stat.cmu.edu/~larry/=sml/)


## 12.3 Theoretical Aspects of Bayesian Inference


### 12.3.1 Bayesian Decision Theory

+ Risk of an estimator
  + $\hat{\theta}(X)$:
    + an estimator of a parameter $\theta \in \Theta$
    + $\hat{\theta)$ as a function of the data $X$
  + measuring ht discrepancy btw a parameter $\theta$ and its estimator $\hat{\theta}(X)$ using a loss function $L: \Theta \times \Theta \to \mathbb{R}$
  + the risk of an estimator $\hat{\theta}(X)$

    \[ R(\theta, \hat{\theta}) = \mathbb{E}_\theta(L(\theta, \hat{\theta})) = \int L(\theta, \hat{\theta}(x)) p_\theta(x) dx \]

+ Decision theory from frequentist viewpoint
  + the parameter $\theta$ as a deterministic quantity
  + purpose: finding a minimax estimator $\hat{\theta}$ to minimize the maximum risk, i.e.,

    \[ R_{\max}(\tilde{\theta}) := \sup_{\theta \in \Theta} R(\theta, \,\tilde{\theta}) \]

+ Decision theory from Bayesian viewpoint
  + a random quantity w/ a prior distribution $\pi(\theta)$
  + purpose: finding the estimator $\hat{\theta}(X)$ to minimize the posterior expected loss

    \[ R_{\pi}(\hat{\theta} \,|\, X) = \int_{\Theta} L\left(\theta, \hat{\theta}(X)\right) p(\theta \,|\, X) d\theta \]
  
  + estimator $\hat{\theta}$: a Bayes rule w.r.t. the prior $\pi(\theta)$ if

    \[ R_{\pi}(\hat{\theta}) = \inf_{\tilde{\theta} \in \Theta} R_{\pi}(\tilde{\theta} \,|\, X) \]


+ Bayes rish
  + minimizing the posterior expected loss $\equiv$ minimizing the average risk, Bayes risk

    \[ B_{\pi} = \int R(\theta, \hat{\theta}) \pi(\theta) \,d\theta \]

  + __Theorem__. The Bayes rules minimizes the $B_\pi$

  + __Theorem__. Bayes estimators
    + $L(\theta, \hat{\theta}) = (\theta, \hat{\theta})^2 \implies$ the posterior mean
    + $L(\theta, \hat{\theta}) = |\theta, \hat{\theta}| \implies$ the posterior median
    + $L(\theta, \hat{\theta}) = I(\theta \neq \hat{\theta}) \implies$ the posterior mode



### 12.3.2 Large Sample Properties of Bayes's Procedures




## 12.4 Examples of Bayesian Inference

### 12.4.1 Bayesian Linear Model




### 12.4.2 Hierarchical Models





## 12.5 Simulation Methods for Bayesian Computation





### 12.5.1 Basic Monte Carlo Integration





### 12.5.2 Importance Sampling




### 12.5.3 Markov Chain Monte Carlo (MCMC)




### 12.5.4 Why It Works





### 12.5.5 Different Flavors of MCMC





### 12.5.6 Normalizing Constants







