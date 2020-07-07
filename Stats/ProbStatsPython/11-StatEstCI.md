# Topic11: Statistics, Parameter Estimation and Confidence Interval
  

## 11.1 Statistics

+ Probability x Statistics
  + probability
    + distribution $\to$ samples
    + $\mu = \sum x p(x) \to$ average of many samples $\sim \mu$
    + $X \ge 0 \to \Pr(X \ge 2 \mu) \le 0.5$ (Markov's inequality)
  + statistics
    + samples $\to$ distribution
    + samples $\to$ parameters $\mu, \sigma$
    + samples $\to$ distribution type, e.g., discrete, monotone

+ Distribution parameters
  + most distribution families determined by <span style="color: magenta;">parameters</span>
    + e.g., $B_p, B_{p, n}, P_\lambda, G_p, U_{a, b}, E_\lambda, N_{\mu, \sigma^2}$
  + generally, any deterministic function of the distribution as a <span style="color: magenta;">parameter</span> or <span style="color: magenta;">property</span>
    + e.g., mean, variance, standard deviation
    + e.g., min, max, values, mode, median

+ Sampling from a distribution
  + distribution
    + discrete: $p$
    + continuous: $f$
  + n independent samples: $X^n \stackrel{\text{def}}{=} X_1, X_2, \dots, X_n \sim p \text{ or } f \quad {\perp \!\!\!\! \perp}$

+ Population
  + population: collection of objects
    + typically many
    + e.g., all UCSD students
  + sample n objects
    + often n $\ll$ population size
    + e.g., pick n students at random
  + deducing population parameter from sample, e.g., average height
  + able to view collection of heights as a distribution, e.g., {153, 178, 165, 153, ...}
  + sampling from population $\to$ sampling from distribution
  + $n \ll$ population size $\to$ roughly ${\perp \!\!\!\! \perp}$

+ Statistic
  + any function of the data observed, e.g., average, maximum, max-min value observed
  + using statistics to infer properties of the distribution or population
    + parameters
    + type of distribution
  + how to do it
  + how well can do it

+ Recall
  + done some of them already
  + WLLN:
    + sample mean $\to \mu$
    + no need to know $p$
  + CLT:
    + normalized mean $to N(0, 1)$
    + no need to know $p$
  + important tools in both probability and statistics
  + revisit both shortly


+ [Original Slides](https://tinyurl.com/y9bunnkc)


### Problem Sets

0. Recall a statistic is a single value calculated from the sample. Which of the following is a statistic?<br/>
  a. sample max<br/>
  b. sample mean<br/>
  c. sample median<br/>
  d. all of the above<br/>

  Ans: d


1. $225$ iPhones go on sale on black Friday, and 100 customers are in line to buy them. If the random number of iPhones that each customer wishes to buy is distributed Poisson with mean 2, approximate the probability that all 100 customers get their desired number of iPhones?

  Ans: 0.96145<br/>
  Explanation: The total iPhone demand may be expressed as a sum $S=X_1+...+X_{100}$, where each $X_i$ is distributed $Poission(2)$, denoting the number of iPhones demanded by the $i$th customer. By the central limit theorem, $S=X_1+...+X_{100}$ is distributed approximately $N(200,200)$. Therefore we may approximate the probability as $\Pr(S\le 225)=\Pr(\frac{S-200}{\sqrt{200}}\le \frac{25}{\sqrt{200}})\approx \Phi(\frac{25}{\sqrt{200}})=0.9615$


2. The number of years a Bulldog lives is a random variable with mean 9 and standard deviation 3, while for Chihuahuas, the mean is 15 and the standard deviation is 4. Approximate the probability the that in a kennel of 100 Bulldogs and 100 Chihuahuas, the average Chihuahua lives at least 7 years longer than the average Bulldog.

  Ans: 0.02275<br/>
  Explanation: Let $B_i$, $C_i$, $i \in \{1,...,100\}$ denote the number of years the $i$th Bulldog, Chihuahua lives respectively, hence $B_i \sim N(9,9),C_i \sim N(15,16)$. Then, by the central limit theorem, the difference in average lifetime, $D= \sum^{100}_{i=1} \frac{C_i - B_i}{100}$ is distributed $N(6,25/100)$. Therefore $\Pr(D\ge 7)$ $=\Pr(\frac{D-6}{\sqrt{25/100}}\ge\frac{1}{\sqrt{25/100}})$ $\approx 1-\Phi(\sqrt{100/25})$ $=1-\Phi(2)=0.0228$



### Lecture Video

<a href="https://tinyurl.com/y9uhpoym" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.2 Mean and Variance Estimation

+ Estimators
  + r.v.'s: $X^n \stackrel{\text{def}}{=} X_1, X_2, \dots, X_n$ independent samples from distribution or a population
  + $p$: unknown distribution or population
  + estimate a distribution parameter $\theta$
    + parameter of $p$
    + wish to estimate
    + e.g., mean example: $\mu, \sigma, X_{\max}$, mode
  + sample
    + $X^n \stackrel{\text{def}}{=} X_1, X_2, \dots, X_n \sim p {\perp \!\!\!\! \perp}$
    + e.g., $X^3 = 5, -2, 6$
  + estimator for parameter $\theta$ as a function
    + function $\widehat{\theta}: \Bbb{R} \to \Bbb{R}$
    + mapping $X^n \to \Bbb{R}$
    + e.g., $\max(X_1, \dots, X_n)$
  + upon observing $X^n$, estimate $\theta$ and defined as $\widehat{\Theta} \stackrel{\text{def}}{=} \widehat{\theta}(X^n)$
    + random variable
    + determined by $X^n$
    + examples
      + $\mu$ - $\widehat{\theta}(X^n): \;\; \frac{X_1 + \cdots + X_n}{n} \quad \frac{\min\{X_i\} + \max\{X_i\}}{2} \quad X_1 \cdot X_2$
      + $\widehat{\Theta}: 3 \quad 2, \quad -10$

+ Observations
  + distribution parameter $\theta$
    + constant
    + e.g., mean: 3.2
  + estimate $\widehat{\Theta} \stackrel{\text{def}}{=} \widehat{\theta}(X^n)$
    + random variable
    + ideally close to $\theta$
  + sample $X^n$ draw $\to$ determining $\widehat{\Theta}$
  + point estimate vs. interval
    + point estimate: single value, e.g., 3.5
    + interval: [3, 4]
  + estimator
    + any function
    + good or bad
  + considering how to
    + come up w/ an estimator?
    + evaluate its performance?

+ Sample $X$
  + applying sample to any parameter $X$
  + property

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">$X$</th>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Sample $X$</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <th>min, $X_{\min}$</th>
        <td style="text-align: center;">$\displaystyle\min_x\{ x: p(x) > 0 \}$</td>
        <th>sample min</th>
        <td style="text-align: center;">$\displaystyle\min_i \{X_i\}$</td>
      </tr>
      <tr>
        <th>max, $X_{\max}$</th>
        <td style="text-align: center;">$\displaystyle\max_x \{x: p(x) > 0\}$</td>
        <th>sample max</th>
        <td style="text-align: center;">$\displaystyle\max_i \{X_i\}$</td>
      </tr>
      <tr>
        <th>mean, $\mu$</th>
        <td style="text-align: center;">$\displaystyle\sum_x x \cdot p(x)$</td>
        <th>sample mean</th>
        <td style="text-align: center;">$\displaystyle\frac 1 n \sum_{i=1}^n X_i$</td>
      </tr>
      </tbody>
    </table>

  + simple estimator: if sample is whole population, exact
  + sometimes works well, even for small samples

+ Estimator evaluation
  + parameter may have several estimators
  + evaluate quality of estimator for a parameter
  + e.g., bias, variance, and mean squared error

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/y7s47zez" ismap target="_blank">
        <img src="img/t11-01.png" style="margin: 0.1em;" alt="Illustrative demo for goodfit, bias and wide variance" title="Illustrative demo for goodfit, bias and wide variance" width=400>
      </a>
    </div>

+ Bias and variance
  + bias: $\widehat{\Theta}$ estimator for $\theta$
    + definition: <span style="color: magenta;">bias</span> of $\widehat{\Theta}$ as the expected overestimate of $\theta$

      \[\text{Bias}_\theta (\widehat{\Theta}) \stackrel{\text{def}}{=} E[\widehat{\Theta} - \theta] = \mu_{\widehat{\Theta}} - \theta \quad\to\quad \text{Bias}(\widehat{\Theta}) \]

    + <span style="color: magenta;">unbiased</span>: estimator w/ 0 bias, i.e., $\mu_{\widehat{\Theta}} = \theta$
    + bias = inequality
  + variance
    + definition: $Var(\widehat{\Theta}) = E[(\widehat{\Theta} - \mu_{\widehat{\Theta}})^2]$
    + unrelated to $\theta$
  + ideally 0 bias and variance
    + 0 bias: mean as $\theta$
    + 0 variance: a constant, always the value of $\theta$
  + typically trade off btw bias and variance

+ Mean squared error
  + single measure for performance of estimator $\widehat{\Theta}$ for $\theta$
  + <span style="color: magenta;">MSE</span> of $\widehat{\Theta}$: expected squared distance from $\theta$

    \[ \text{MSE}_{\theta}(\widehat{\Theta}) \stackrel{\text{def}}{=} E[(\widehat{\Theta} - \theta)^2] \quad\to\quad \text{MSE}(\widehat{\Theta}) \]

  + common in science and engineering, e.g., communication, transportation, and production
  + need to re-evaluate?
    + MSE related to bias and variance

+ Bias-Variance Bromance <br/>
  MSE = $\text{Bias}^2$ + Variance

  \[\begin{align*}
    \text{MSE}(\Theta) &= E[(\Theta - \theta)^2] = E^2[\Theta - \theta] + Var(\Theta - \theta) \\
    &= \text{Bias}^2(\Theta) + Var(\Theta) \\\\
    E[\Theta - \theta] &\stackrel{\text{def}}{=} \text{Bias}(\Theta) \qquad Var(\Theta - \theta) = Var(\Theta)
  \end{align*}\]

+ Mean example
  + unknown distribution or population $p$
  + estimate mean $\mu$
  + n samples: $X_1, X_2, \dots, X_n \sim p \; {\perp \!\!\!\! \perp}$
  + sample mean: $\overline{X} \stackrel{\text{def}}{=} \frac{1}{n} \sum_{i=1}^n X_i$
  + evaluate
    + bias, variance, MSE
    + weak law of large numbers

  + bias
    + sample mean: $\overline{X} \stackrel{\text{def}}{=} \frac{1}{n} \sum_{i=1}^n X_i$
    + expectation

      \[ E[\overline{X}] = E\left[ \frac 1 n \sum_{i=1}^n X_i \right] = \frac 1 n \sum_{i=1}^n E[X_i] = \mu \]

    + bias: 

      \[ \text{Bias}(\overline{X}) = E[\overline{X}] - \mu = \mu - \mu = 0 \]

    + sample mean: unbiased estimator for distribution mean

  + variance

    \[\begin{align*}
      Var(\overline{X}) &= Var\left( \frac{1}{n} \sum_{i=1}^n X_i \right) = \frac{1}{n^2} Var\left( \sum_{i=1}^n X_i \right) \\
      &= \frac{1}{n^2} \sum_{i=1}^n Var(X_i) = \frac{1}{n^2} \sum_{i=1}^n \sigma^2 = \frac{\sigma^2}{n} \\\\
      &\therefore\; \sigma_{\overline{X}} = \frac{\sigma}{\sqrt{n}}
    \end{align*}\]

    + decreasing w/ $n \nearrow$
    + increasing w/ $\sigma \nearrow$
  + experiments

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="url" ismap target="_blank">
        <img src="img/t11-02a.png" style="margin: 0.1em;" alt="Experiment results: mean=0, sample size=5, repetition= 3000" title="Experiment results: mean=0, sample size=5, repetition= 3000" height=150>
        <img src="img/t11-02b.png" style="margin: 0.1em;" alt="Experiment results: mean=0, sample size=50, repetition= 3000" title="Experiment results: mean=0, sample size=50, repetition= 3000" height=150>
        <img src="img/t11-02c.png" style="margin: 0.1em;" alt="Experiment results: mean=0, sample size=5, repetition= 400" title="Experiment results: mean=0, sample size=5, repetition= 400" height=150>
        <img src="img/t11-02d.png" style="margin: 0.1em;" alt="Experiment results: mean=0, sample size=50, repetition= 400" title="Experiment results: mean=0, sample size=50, repetition= 400" height=150>
      </a>
    </div>

  + MSE of sample mean

    \[ \text{MSE}_\mu (\overline{X}) = \text{Bias}_\mu^2(\overline{X}) + Var(\overline{X}) = \frac{\sigma^2}{n} \]

    + increasing w/ $\sigma \nearrow$
    + decreasing w/ $n \nearrow$
    + same estimator works for all distributions
    + accuracy (MSE) independent of population size


+ [Original Slides](https://tinyurl.com/y7s47zez)


### Problem Sets





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.3 Variance Estimation







+ [Original Slides]()


### Problem Sets





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.4 Unbiased Variance Estimation







+ [Original Slides]()


### Problem Sets





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.5 Estimating Standard Deviation







+ [Original Slides]()


### Problem Sets





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.6 Confidence Interval







+ [Original Slides]()


### Problem Sets





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.7 Confidence Interval - Sigma Unknown







+ [Original Slides]()


### Problem Sets





### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## Lecture Notebook 11









## Programming Assignment 11









