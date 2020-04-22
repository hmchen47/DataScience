# Chapter 12 Bayesian Inference


Author: H. Liu and L. Wasserman

[Origin](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)

Year: 2014

Related Course: [36-708 Statistical Methods for Machine Learning](http://www.stat.cmu.edu/~larry/=sml/)


## 12.3 Theoretical Aspects of Bayesian Inference


### 12.3.1 Bayesian Decision Theory

+ Risk of an estimator
  + $\widehat{\theta}(X)$:
    + an estimator of a parameter $\theta \in \Theta$
    + $\widehat{\theta}$ as a function of the data $X$
  + measuring ht discrepancy btw a parameter $\theta$ and its estimator $\widehat{\theta}(X)$ using a loss function $L: \Theta \times \Theta \to \mathbb{R}$
  + the risk of an estimator $\widehat{\theta}(X)$

    \[ R(\theta, \widehat{\theta}) = \mathbb{E}_\theta(L(\theta, \widehat{\theta})) = \int L(\theta, \widehat{\theta}(x)) p_\theta(x) dx \]

+ Decision theory from frequentist viewpoint
  + the parameter $\theta$ as a deterministic quantity
  + purpose: finding a minimax estimator $\widehat{\theta}$ to minimize the maximum risk, i.e.,

    \[ R_{\max}(\widetilde{\theta}) := \sup_{\theta \in \Theta} R(\theta, \,\widetilde{\theta}) \]

+ Decision theory from Bayesian viewpoint
  + the parameter $\theta$ as a random quantity w/ a prior distribution $\pi(\theta)$
  + purpose: finding the estimator $\widehat{\theta}(X)$ to minimize the posterior expected loss

    \[ R_{\pi}(\widehat{\theta} \,|\, X) = \int_{\Theta} L\left(\theta, \widehat{\theta}(X)\right) p(\theta \,|\, X) d\theta \]
  
  + estimator $\widehat{\theta}$: a Bayes rule w.r.t. the prior $\pi(\theta)$ if

    \[ R_{\pi}(\widehat{\theta}) = \inf_{\widetilde{\theta} \in \Theta} R_{\pi}(\widehat{\theta} \,|\, X) \]


+ Bayes risk
  + minimizing the posterior expected loss $\equiv$ minimizing the average risk, Bayes risk

    \[ B_{\pi} = \int R(\theta, \widehat{\theta}) \pi(\theta) \,d\theta \]

  + __Theorem__. The Bayes rules minimizes the $B_\pi$

  + __Theorem__. Bayes estimators
    + $L(\theta, \widehat{\theta}) = (\theta, \widehat{\theta})^2 \implies$ the posterior mean
    + $L(\theta, \widehat{\theta}) = |\theta, \widehat{\theta}| \implies$ the posterior median
    + $L(\theta, \widehat{\theta}) = I(\theta \neq \widehat{\theta}) \implies$ the posterior mode



### 12.3.2 Large Sample Properties of Bayes's Procedures

+ Bayesian approach w/ large samples
  + under appropriate conditions
    + the posterior distribution $\approx$ Normal distribution
    + the posterior mean $\approx$ mle
  + __Theorem__.
    + $I(\theta)$: the Fisher information
    + $\widehat{\theta}_n$: the maximum likelihood estimator
    + $\widehat{se} = 1 / \sqrt{n I(\widehat{\theta}_n)}$
    + Under appropriate regularity conditions, the posterior $\approx N(\widehat{\theta}_n, \widehat{se})$, i.e.,

      \[ \int \left|p(\theta \,|\, X_1, \dots, X_n) - N(\theta; \widehat{\theta}, \widehat{se} )\right| \,d\theta \xrightarrow{P} 0 \]

    + $\overline{\theta} - \widehat{\theta} = O_P(1/n)$
    + $z_{\alpha/2}: \alpha/2$-quantile of a standard Gaussian distribution
    + $C_n = [\widehat{\theta}_n - z_{\alpha/2} \, \widehat{se}, \,\widehat{\theta}_n + z_{\alpha/2}\,\widehat{se}]$: the asymptotic frequentist $1-\alpha$ confidence interval

      \[ \mathbb{P}(\theta \in C_n \,|\, \mathcal{D}_n) \to 1 - \alpha \]

  + _Proof_.
    + rigorous proof w van der Vaart (1998)
    + the effect of the prior diminishes as $n \nearrow \;\ni$ $p(\theta \,|\, \mathcal{D}_n) \propto \mathcal{L}_n(\theta)p(\theta) \approx \mathcal{L}_n(\theta)$
    + $l_n(\theta) = \log \mathcal{L}_n(\theta) \implies \log p(\theta\,|\, \mathcal{D}_n) \approx l_n(\theta)$
    + by Taylor expansion around $\widehat{\theta}_n$

      \[\begin{align*}
        l_n(\theta) &\approx l_n(\widehat{\theta}_n) + (\theta - \widehat{\theta}_n) \underbrace{l_n^\prime (\widehat{\theta})}_{=0} + [(\theta - \widehat{\theta}_n)^2/2] l_n^{\prime\prime} (\widehat{\theta}_n) \\
        &= l_n(\widehat{\theta}_n) + [(\theta - \widehat{\theta}_n)^2 / 2] l_n^{\prime\prime} (\widehat{\theta}_n) \\\\
        &\Downarrow (\sigma_n^2 = -1 / l_n^{\prime\prime} (\widehat{\theta}_n) \text{ and exponetiating both sides}) \\\\
        p(\theta \,|\, \mathcal{D}_n) &\propto \exp\left( -\frac{1}{2} \frac{(\theta - \widehat{\theta}_n)^2}{\sigma_n^2} \right)
      \end{align*}\]

    + $\therefore$ the posterior of $\theta \approx N(\widehat{\theta}_n, \sigma_n^2)$
    + $l_i(\theta) = \log p(X_i \,|\, \theta) \implies$

      \[\begin{align*}
        \frac{1}{\sigma_n^2} &= -l^{\prime\prime}(\widehat{\theta_n}) = \sum_{i=1}^n -l_i^{\prime\prime}(\widehat{\theta}_n) = n \left(\frac{1}{n}\right) \sum_{i=1}^n -l_i^{\prime\prime}(\widehat{\theta}_n) \approx n\mathbb{E}_\theta [-l_i^{\prime\prime}(\widehat{\theta}_n)] = nI(\widehat{\theta}_n) \\\\
        & \implies \sigma_n \approx se(\widehat{\theta}_n) \tag*{$\Box$} 
      \end{align*}\]

  + Bayes delta method: $\tau = g(\theta) \implies \tau \,|\, \mathcal{D}_n \approx N(\widehat{\tau}, \widetilde{se}^2), \;\;\widehat{\tau} = g(\widehat{\theta}), \,\widetilde{se} = \widehat{se} \,g'(\widehat{\theta})$



## 12.4 Examples of Bayesian Inference

### 12.4.1 Bayesian Linear Model

+ Gaussian linear regression
  + many frequentist method viewed as the maximum a posterior (MAP) estimator under a Bayesian framework
  + Gaussian linear regression w/ known $\sigma$

    \[ Y = \beta_0 + \sum_{j=1}^d \beta_j X_j = \varepsilon, \quad \varepsilon \sim N(0, \sigma^2) \]

  + sampling distribution: $\mathcal{D}_n = \{ (\pmb{X}_1, Y_1), \dots,  (\pmb{X}_n, Y_n)\}$
  + the conditional likelihood of $\pmb{\beta} = (\beta_0, \beta_1, \dots, \beta_s)^T$

    \[ \mathcal{L}(\pmb{\beta}) = \prod_{i=1}^n p(y_i \,|\, x_i, \pmb{\beta}) \propto \exp\left( -\frac{\sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^d \beta_j x_{ij})^2}{2 \sigma^2} \right) \]

  + a Gaussian prior $\pi_\lambda(\pmb{\beta}) \propto \exp( -\lambda \|\pmb{\beta}\|^2_2 / 2) \implies$ the posterior

    \[ p(\pmb{\beta} \,|\, \mathcal{D}_n) \propto \mathcal{L}(\pmb{\beta}) \pi_\lambda(\pmb{\beta}) \]

  + the MAP estimator, $\widehat{\pmb{\beta}}^{\,MAP}$

    \[ \widehat{\pmb{\beta}}^{\,MAP} = \mathop{\arg\min}_{\pmb{\beta}} p(\pmb{\beta} \,|\, \mathcal{D}_n) = \mathop{\arg\min}_{\pmb{\beta}} \left\{ \sum_{i=1}^n \left(Y_i - \beta_0 - \sum_{j=1}^d \beta_j X_{ij}\right)^2 + \lambda \sigma^2 \|\pmb{\beta}\|_1 \right\} \]

    + exactly the ridge regression w/ the regularization parameter $\lambda' = \lambda \sigma^2$
  + adopting the Laplacian prior $\pi_\lambda(\pmb{\beta}) \propto \exp(-\lambda \|\pmb{\beta}\|_1 / 2) \implies$ the Lasso estimator

    \[ \hat{\pmb{\beta}}^{\,MAP} = \mathop{\arg\min}_{\pmb{\beta}} \left\{ \sum_{i=1}^n \left(Y_i - \beta_0 - \sum_{j=1}^d \beta_j X_{ij}\right)^2 + \lambda \sigma^2 \|\pmb{\beta}\|_1 \right\} \]

  + a complete Bayesian analysis aiming at obtaining the whole posterior distribution $p(\pmb{\beta} \,|\, \mathcal{D}_n)$
  + in general, $p(\pmb{\beta} \,|\, \mathcal{D}_n)$ not having an analytic form and resorting to simulation to approximate the posterior



### 12.4.2 Hierarchical Models





## 12.5 Simulation Methods for Bayesian Computation





### 12.5.1 Basic Monte Carlo Integration





### 12.5.2 Importance Sampling




### 12.5.3 Markov Chain Monte Carlo (MCMC)




### 12.5.4 Why It Works





### 12.5.5 Different Flavors of MCMC





### 12.5.6 Normalizing Constants







