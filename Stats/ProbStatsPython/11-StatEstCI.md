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

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 11.2 Mean and Variance Estimation







+ [Original Slides]()


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









