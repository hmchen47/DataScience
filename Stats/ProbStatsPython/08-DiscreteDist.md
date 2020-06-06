# Topic 8: Discrete Distribution Families


## 8.0 Introduction

+ Distributions
  + discrete: Bernoulli, Binomial, Poisson, Geometric
  + continuous: uniform, exponential, Gaussian

+ Bernoulli distribution
  + simplest non-constant distribution
  + foundation of many others
  + parameters: $\mu, Var, \sigma$
  + repeated experiments
  + Jacob Bernoulli
    + 1655 ~ 1705
    + Theology $\to$ Mathematics
    + main contribution:
      + calculus, integrals
      + "Euler" number
      + $e = \lim_{n \to \infty} (1 + \frac 1 n)^n$
      + $e \to b$
      + ars conjectandi = First law of large numbers
    + mentored brother Johann: Medicine $\to$ Math, Dynasty
  + the simplest distribution
    + simplest: one value, 5, constant, always the same, trivial
    + simplest non-trivial: two values
      + simplest values: 0 and 1
      + Bernoulli coin
  + modeling
    + $B_p\; 0 \le p \le 1$
    + $p(0) = 1-p = \overline(p) = q \quad p(1) = p$
  + unitarity: $p(0) + p(1) = (1-p) + p = 1$
  + $X \sim B_p$:
    + Bernoulli
    + random variable, coin, experiment, trial

+ Who cares about two values?
  + binary version of complex events
    + products
      + 80 good, 20 defective
      + select one: good or not
      + $!\sim B_{.8}$
    + next child will be a boy: $\sim B_{.5}$
  + generalized to more complex variables
    + e.g., patient has one of three diseases
  + repeated trials yield \# successes
    + many important distributions
    + Binomial, Geometric, Poisson, Normal
  
+ Mean
  + $X \sim B_p \quad p(0) = 1-p \quad p(1) = p$
  + $E[X] = \sum p(x) \cdot x = (1-p) \cdot 0 + p \cdot 1$
    + e.g. $X \sim B_{.8} \to E[X] = 0.8$
  + $E[X] = P(X=1)$
  + fraction of times expect to see 1

+ Variance
  + $X \sim B_p \to E[X] = p$
  + variance
    + easy route
    + #0^2 = 0 \quad 1^2 =1 \quad X^2 = X \quad E[X^2]=E[X] = p$
    + $Var(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p) = pq$
  + standard deviation: $\sigma = \sqrt{pq}$
  + $E, Var, \sigma$ for various $p$
    + $B_p$ varying most when $p = \frac12$
    + $p = 0 \to E[X] = 0, \;Var(X) = 0, \;\sigma = 0$
    + $p = 1 \to E[X] = 1, \;Var(1) = 0, \;\sigma = 0$
    + $p = \tfrac12 \to E[X] = \tfrac12, \;Var(X) = \frac14, \;\sigma = \frac12$

+ Independent trials
  + much of $B_p$ importance stems from multiple trials
  + most common: independent ${\perp \!\!\!\! \perp}$
    + $0 \le p \le 1\quad X_1, X_2, X_3 \sim B_p \to {\perp \!\!\!\! \perp}$
    + $q \stackrel{\text{def}}{=} 1-p\; P(110) = p^2q = P(101) = P(011)$
  + generally, $X_1, X_2, \cdot, X_n \sim B_p \to  {\perp \!\!\!\! \perp}$
    + $x^n = x_1, x_2, \cdots, x_n \in \{0, 1\}^n$
    + $n_0$: number of 0's
    + $n_1$: number of 1's
    + $P(x_1, \dots, x_n) = p^{n_1} q^{n_0}$
    + e.g., $P(10101) = p^{n_1} q^{n_0} = p^3 q^2$

+ Typical samples

  <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Distribution</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Typical seq.</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Decscription</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Probability</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">$B_0$</td> <td style="text-align: center;">$0000000000$</td> <td style="text-align: center;">constant 0</td> <td style="text-align: center;">$1^{10} = 1</td> </tr>
    <tr> <td style="text-align: center;">$B_1$</td> <td style="text-align: center;">$1111111111$</td> <td style="text-align: center;">constant 1</td> <td style="text-align: center;">$1^{10} = 1$</td> </tr>
    <tr> <td style="text-align: center;">$B_{0.8}$</td> <td style="text-align: center;">$1110111011$</td> <td style="text-align: center;">80% 1's</td> <td style="text-align: center;">$0.8^8 \cdot 0.2^2$ ${}^{\star}$</td> </tr>
    <tr> <td style="text-align: center;">$B_{0.5}$</td> <td style="text-align: center;">$1011010010$</td> <td style="text-align: center;">50% 1's ${}^{\S}$</td> <td style="text-align: center;">$0.5^{10}$</td> </tr>
    <tr>
      <td colspan="4">${}^\star$: not most probable, most probable: $1, \dots, 1$, unlikely to be seen<br/>${}^\S$: fair coin flip</td>
    </tr>
    </tbody>
  </table>





+ [Original Slides](https://tinyurl.com/ybc73cpq)


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.1 Bernoulli Distribution





+ [Original Slides]()


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.2 Binomial Distribution





+ [Original Slides]()


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.3 Poisson Distribution





+ [Original Slides]()


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.4 Geometric Distribution





+ [Original Slides]()


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.5 Geometric Distribution Example





+ [Original Slides]()


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## Lecture Notebook 8










## Programming Assignment 8








