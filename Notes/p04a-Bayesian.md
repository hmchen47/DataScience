# Chapter 12 Bayesian Inference


Author: H. Liu and L. Wasserman

[Origin](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)

Year: 2014

Related Course: [36-708 Statistical Methods for Machine Learning](http://www.stat.cmu.edu/~larry/=sml/)


## 12.1 What is Bayesian Inference?

+ Approaches to statistical machine learning
  + _frequentist_ inference
    + probabilities interpreted as long run frequencies
    + goal: to create procedures w/ long run frequency guarantees
  + _Bayesian_ inference
    + probabilities interpreted as subjective degree of belief
    + goal: to state and analyze one's beliefs
  + differences

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr>
        <th style="text-align: left; background-color: #3d64ff; color: #ffffff; width:30%;"> </th>
        <th style="text-align: left; background-color: #3d64ff; color: #ffffff; width:20%;">Frequentist</th>
        <th style="text-align: left; background-color: #3d64ff; color: #ffffff; width:20%;">Bayesian</th>
      </tr>
      </thead>
      <tbody>
      <tr>  <td>Probability is:</td>  <td>limiting relative frequency</td>  <td>degree of belief</td></tr>
      <tr>  <td>Parameter $\theta$ is a:</td>  <td>fixed constant</td>  <td>random variable</td></tr>
      <tr>  <td>Probability statements are about:</td>  <td>procedures</td>  <td>parameters</td></tr>
      <tr>  <td>Frequency guarantees?</td>  <td>yes</td>  <td>no</td></tr>
      </tbody>
    </table>

+ Frequentist approach for interval estimate
  + $\exists\; X_1, \dots, X_n \sim N(\theta, 1)$
  + probability statement about the random interval $C$: confidence interval

    \[ C = \left[\overline{X}_n - \frac{1.96}{\sqrt{n}},\; \overline{X}_n + \frac{1.96}{\sqrt{n}}\right] \quad \implies \quad \mathbb{P}_\theta(\theta \in C) = 0.95 \quad \forall\;  \theta \in \mathbb{R} \]

  + interval: a function of the data $\to$ random
  + parameter $\theta$: fixed, unknown quantity
  + statement: $C$ will trap the true value w/ probability 0.95
  + repeating this experiment many times

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yx567vmm" ismap target="_blank">
        <img src="img/p04-00.png" style="margin: 0.1em;" alt="Repeating the experiment" title="Repeating the experiment" width=550>
      </a>
    </div>
  + finding the interval $C_j$ traps the parameters $\theta_j$, 95 percent of time
  + more precisely

    \[ \mathop{\lim\inf}_{n \to \infty} \frac{1}{n} \sum_{i=1}^n I(\theta_i \in C_i) \geq 0.95 \quad \forall\; \theta_1, \theta_2, \dots \tag{1} \]

+ Bayesian approach for interval estimate
  + unknown parameter $\theta \sim$ prior distribution $\pi(\theta)$ representing ones subjective beliefs about $\theta$
  + observing the data $X_1, \dots, X_n$, the posterior distribution for $\theta$ given the data using Bayes theorem

    \[ \pi(\theta | X_1, \dots, X_n) \propto \mathcal{L}(\theta) \pi(\theta) \tag{2} \]

  + $\mathcal{L}(\theta)$: the likelihood function
  + finding an interval $C \to$

    \[ \int_C \pi(\theta | X_1, \dots, X_n) d\theta = 0.95 \]

  + the degree-of-belief probability statement about $\theta$ given the data

    \[ \mathbb{P}(\theta \in C | X_1, \dots, X_n) = 0.95 \]

  + repeating the experiment many times $\to$ the intervals not trapping the true value 95 percent of the time

+ Summary
  + frequentist inference
    + procedures w/ frequency probability guarantees
    + probability to random interval $C$: a frequentist confidence interval $C$ satisfies

      \[ \inf_{\theta} \mathbb{P}_\theta(\theta \in X) = 1 - \alpha \]

    + $\inf_\theta \mathbb{P}(\theta \in C)$: the coverage of the confidence interval $C$
  + Bayesian inference
    + a method for stating and updating beliefs
    + probability to $\theta$: Bayesian confidence interval $C$ satisfies

      \[ \mathbb{P}(\theta \in C | X_1, \dots, X_N) = 1 - \alpha \]

+ Favors of Bayesian inference
  + subjective Bayesian: probability strictly as personal degrees of belief
  + objective Bayesian: finding prior distributions that formally express ignorance w/ the hope that the resulting posterior is , in some sense, objective
  + empirical Bayesian: estimating the prior distribution from the data
  + frequentist Bayesian: using Bayesian methods only when the resulting posterior has good frequency behavior


## 12.2 Basic Concepts




### 12.2.1 The Mechanics of Bayesian Inference




### 12.2.2 Bayesian Prediction





### 12.2.3 Inference about Functions of Parameters





### 12.2.4 Multiparameter Problems




### 12.2.5 Flat Priors, Improper Priors, and "Noninformative" Priors




### 12.2.6 Conjugate Priors




### 12.2.7 Bayesian Hypothesis Testing




### 12.2.8 Model Comparison and Bayesian Information Criterion




### 12.2.9  Calculating the Posterior Distribution





