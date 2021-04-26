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

+ Example
  + glaucoma study
    + recorded intraocular pressure values
    + samples of 21 elderly subjects
    + the mean intraocular pressure of the population differs from 14 mm Hg?
  
    <table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 26.5vw;" cellspacing=0 cellpadding=5 border=1 align="center">
      <caption style="font-size: 1.5em; margin: 0.2em;  background-color: #3d64ff; color: #ffffff;">Intraocular Pressure</caption>
      <tbody>
      <tr style="vertical-align:middle">
        <td style="text-align: left;">14.5</td><td style="text-align: left;">12.9</td><td style="text-align: left;">14.0</td><td style="text-align: left;">16.1</td><td style="text-align: left;">12.0</td><td style="text-align: left;">17.5</td><td style="text-align: left;">14.1</td>
      </tr>
      <tr style="vertical-align:middle">
        <td style="text-align: left;">12.9</td><td style="text-align: left;">17.9</td><td style="text-align: left;">12.0</td><td style="text-align: left;">16.4</td><td style="text-align: left;">24.2</td><td style="text-align: left;">12.2</td><td style="text-align: left;">14.4</td>
      </tr>
      <tr style="vertical-align:middle">
        <td style="text-align: left;">17.0</td><td style="text-align: left;">10.0</td><td style="text-align: left;">18.5</td><td style="text-align: left;">20.8</td><td style="text-align: left;">16.2</td><td style="text-align: left;">14.9</td><td style="text-align: left;">19.6</td>
      </tr>
      </tbody>
    </table><br/>

  + step 1: testing method
    + test: one sample t-test for means
    + hypothesis test for means: a t-test
    + unknown the population standard deviation
    + estimate it w/ the sample standard deviations
  + step 2: assumptions
    + the dependent variable must be continuous
    + the observations are independent of one another
    + the dependent variable should be approximately normally distributed
    + the dependent variable should not contain any outliers
  + step 3: hypotheses
    + null hypothesis: the population mean of the intraocular pressure is 14 mm Hg
    + alternative hypothesis: the population mean of the intraocular pressure differs from 14 mm Hh
    + mathematical formula

      \[ H_0: \mu = 14 \hspace{0.5em}\text{vs.}\hspace{0.5em} H_a: \mu \ne 14 \]

  + step 4: calculate the test statistic
    + $y =$ sample mean = 15.623
    + $s =$ sample standard deviation = 3.382
    + $n = 21$
    + $\therefore\; t = 2.199$
    + p-value (two-tailed): the area in both tails of the $t$ distribution

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 25vw;"
        onclick= "window.open('https://towardsdatascience.com/which-hypothesis-test-to-perform-89d7044d34a1')"
        src    = "https://miro.medium.com/max/875/1*CJkTAUWtftyijVnGzEjsJw.png"
        alt    = "p-value w/ wo-tailed hypothesis"
        title  = "p-value w/ wo-tailed hypothesis"
      />
    </figure>

  + step 5: determine p-value and compare w/ the significance level
    + the corresponding p-value for the t statistic: 0.0398
    + p-value:
      + the chance of observing your sample results or more extreme results assuming the null hypothesis is true
      + small chance $\implies$ reject the null hypothesis
    + the probability of observing a sample mean of 15.623 mm Hg or a value more extreme, assuming the true pressure is 14 mm Hg
    + p-value = 0.0398 < 0.05 $\implies$ reject the null hypothesis
    + statement: there is sufficient sample evidence to conclude that the true mean intracoular pressure differs from 14 mm Hg.
  + python snippet

    ```python
    #import libraries
    import pandas as pd
    from scipy import stats
    import os
    os.chdir('C:\\Users\\rohan\\Documents\\Analytics\\Data')

    #import file and apply one sample t test
    a = pd.read_excel('onesamplet.xlsx')
    stats.ttest_1samp(a,14)

    # Ttest_1sampResult(statistic=array([2.19967042]),, pvalue=array([0.03975528]))
    ```



