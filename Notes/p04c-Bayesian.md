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

+ Sampling to a Foregone Conclusion
  + random variable: $X \sim N(\theta, 1)$
  + sampling distribution: $\mathcal{D}_N = \{ X_1, \dots, X_N \}$
  + $X_i$: statistics comparing to a new drug to a placebo
  + continuing sampling until $T_n > k, \,T_N = \sqrt{NX_N}, k= 10 \implies$ stop when the drug appears to be much better than the placebo
  + $N$ becoming as a random variable
  + shown that $\mathbb{P}(N < \infty) = 1$
  + shown that the posterior $p(\theta \,|\, X_1, \dots, X_N)$ same as if $N$ had been fixed in advance, i.e., the randomness in $N$ not effecting the posterior
  + approximating to Normal distribution: the prior $\pi(\theta)$ smooth $\implies$ the posterior approximately $\theta \,|\, X_1, \dots, X_N \sim N(\overline(X)_N, 1/N)$
  + region estimate: $C_N = \overline{X}_N \pm 1.96/\sqrt{N} \implies \mathbb{P}(\theta \in C_N \,|\, X_1, \dots, X_N) \approx 0.95$ w/ $0 \notin C_N$
  + stop sampling w/ $T > 10 \ni$

    \[ \overline{X}_N - \frac{1.96}{\sqrt{N}} > \frac{10}{\sqrt{N}} - \frac{1.96}{\sqrt{N}} > 0  \quad\therefore \theta = 0 \ni \mathbb{P}_\theta(\theta \in C_N) = 0 \]

  + sampling to a forgone conclusion
    + the frequentist coverage

      \[ \text{Coverage} = \inf_{\theta} \mathbb{P}_\theta(\theta \in C_N) = 0 \]

    + a serious issue in sequential clinical trials

+ Godambe's example



## 12.7 Freedman's Theorem





## 12.8 The Bayes-Frequestist Debate





## 12.9 Summary




