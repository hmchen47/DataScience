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
  + $p(y|\theta)$: a conditional density of the observation $y$ defined on the _sample space_ $\mathbb{X}$; the _sampling density_ of $y$ given $\theta$
  + $p(\theta)$: the marginal density of $\theta$ to be the _prior density_ of $\theta$ defined on the _parameter space_ $\Theta$
  + actual determination of the prior density = the determination of the sampling density $p(y|\theta)$
  + $\{f(y|\theta): \theta \in \Theta\}$: the parametric class of sampling densities
  + assumption: the specific form of $p(\cdot | \theta)$ completely known once $\theta$ determined
  + assumption: $\Theta \subseteq \mathbb{R}^m$; i.e. finite-dimensional _parametric space_


### 1.1 The Bayes' Rule





### 1.2 Hierarchical and Empirical Approaches





## 2. Point Estimation






## 3. Hypothesis Testing






## 4. Region Estimation





## 5. Numerical Integration Methods





### 5.1 Deterministic Methods





### 5.2 Monte Carlo Methods





## 6. Examples




