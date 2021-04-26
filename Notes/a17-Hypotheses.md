# Which hypothesis test to perform?

Author: R. Joseph

Date: 2018-12-20

[Original](https://towardsdatascience.com/which-hypothesis-test-to-perform-89d7044d34a1)


## Overview

+ Statistics
  + objective: make references about a population based on information contained a sample
  + parameters: numerical measurements used to characterize population
  + population parameters
    + $\mu$: mean
    + $M$: median
    + $\sigma$: standard deviation
    + $\pi$: proportion

+ Inferential problems
  + formulated as an inference about one of parameters of population
  + categories of inferences
    + __estimate__ the value of the population parameters
    + __test a hypothesis__ about the value of the parameter

+ Summary of the various hypothesis tests and when to use

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 55vw;"
      onclick= "window.open('https://towardsdatascience.com/which-hypothesis-test-to-perform-89d7044d34a1')"
      src    = "https://miro.medium.com/max/875/1*8pSgz0bAlIQ3wlGNJAc-6g.png"
      alt    = "Summary of the hypothesis tests"
      title  = "Summary of the hypothesis tests"
    />
  </figure>



## One sample t-test

+ One sample t-test
  + determining whether the sample mean is statistically different from a known or hypothesized population mean
  + hypotheses: two-tailed one sample t-test
    + __Null Hypothesis:__ the sample mean = the population mean
    + __Alternative Hypothesis:__ the sample mean $\ne$ the population mean

+ Summary of null and alternative hypotheses

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
      onclick= "window.open('https://towardsdatascience.com/which-hypothesis-test-to-perform-89d7044d34a1')"
      src    = "https://miro.medium.com/max/875/1*IQI1DObnx9k2gwnSHGkRCA.png"
      alt    = "Table of types of hypothesis tests and their hypotheses"
      title  = "Table of types of hypothesis tests and their hypotheses"
    />
  </figure>

+ Test Statistic
  + formula

    \[ t_{obs} = \frac{\overline{y} - \mu_0}{s/\sqrt{n}} \]

    + $\mu_0$: proposed constant for population mean
    + $\overline{y}$: sample mean
    + $s$: sample standard deviation
    + $n$: sample size
  + calculated $t$ compared to the critical $t$ value from the $t$ distribution table w/ degrees of freedom $df = n - 1$
  + calculated $t$ value $\implies$ reject the null hypothesis

### Example




