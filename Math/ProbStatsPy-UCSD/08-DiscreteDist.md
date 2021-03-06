# Topic 8: Discrete Distribution Families


## 8.0 Introduction

+ Distributions
  + discrete: Bernoulli, Binomial, Poisson, Geometric
  + continuous: uniform, exponential, Gaussian

+ Discussion
  + Motivation
  + Applications
  + Formulate
  + Visualize
  + Examples
  + Properties: $\mu \; Var\; \sigma$
  + Python: plot and experiment within NB

+ Show distribution
  + non-negative
  + sum to 1


+ [Original Slides](https://tinyurl.com/ybc73cpq)


### Problem Sets

1. For which value of the parameter $\alpha$ is the function $f(x)=\frac{2(10−x)+\alpha}{100}$ over $\{1,2, \cdots,10\}$ a p.m.f.?<br/>
  a. -1<br/>
  b. 0<br/>
  c. 1<br/>
  d. 2<br/>

  Ans: c<br/>
  Explanation: Following $\sum_{x = 1}^{10} f(x) = 1$, we have $\alpha = 1$


### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.1 Bernoulli Distribution

+ Contents covered w/ Bernoulli distribution
  + simplest non-constant distribution
  + foundation of many others
  + parameters: $\mu, Var, \sigma$
  + repeated experiments

+ Jacob Bernoulli
  + 1655 ~ 1705
  + Theology $\to$ Mathematics
  + main contribution:
    + calculus, integrals
    + "Euler" number: $e = \lim_{n \to \infty} (1 + \frac 1 n)^n$; $e \to b$
    + ars conjectandi = The Art of Conjecture
    + First law of large numbers
  + mentored brother Johann: Medicine $\to$ Math, Dynasty

+ The simplest distribution
  + simplest: one value, constant, always the same, trivial, e.g., 5
  + simplest non-trivial
    + two values
    + simplest values: 0 and 1
    + Bernoulli coin

+ Bernoulli distribution
  + two values: 0 and 1, or failure and success
  + notation: $B_p\quad 0 \le p \le 1$
  + pmf: $p(0) = 1-p = \overline{p} = q \quad p(1) = p$
  + unitary: $p(0) + p(1) = (1-p) + p = 1$
  + $X \sim B_p$:
    + Bernoulli
    + random variable, coin, experiment, trial

+ Characteristics
  + binary version of complex events
    + example
      + products: 80 good, 20 defective
      + select one: good or not
      + $\sim B_{.8}$
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
    + $0^2 = 0 \quad 1^2 =1 \quad X^2 = X \quad E[X^2]=E[X] = p$
    + $Var(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p) = pq$
  + standard deviation: $\sigma = \sqrt{pq}$
  + various $p$
    + $p = 0 \to E[X] = 0, \;Var(X) = 0, \;\sigma = 0$
    + $p = 1 \to E[X] = 1, \;Var(1) = 0, \;\sigma = 0$
    + $p = \tfrac12 \to E[X] = \tfrac12, \;Var(X) = \frac14, \;\sigma = \frac12$
    + $B_p$ varying most when $p = \frac12$

+ Independent trials
  + much of $B_p$ importance stems from multiple trials
  + most common type of Bernoulli distribution: independent ${\perp \!\!\!\! \perp}$
    + $0 \le p \le 1\quad X_1, X_2, X_3 \sim B_p \to {\perp \!\!\!\! \perp}$
    + $q \stackrel{\text{def}}{=} 1-p\quad P(110) = p^2q = P(101) = P(011)$
  + generally, $X_1, X_2, \cdots, X_n \sim B_p \to  {\perp \!\!\!\! \perp}$
    + $x^n = x_1, x_2, \cdots, x_n \in \{0, 1\}^n$
    + $n_0$ = number of 0's; &nbsp;&nbsp;&nbsp;&nbsp;  $n_1$ = number of 1's
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
    <tr> <td style="text-align: center;">$B_0$</td> <td style="text-align: center;">$0000000000$</td> <td style="text-align: center;">constant 0</td> <td style="text-align: center;">$1^{10} = 1$</td> </tr>
    <tr> <td style="text-align: center;">$B_1$</td> <td style="text-align: center;">$1111111111$</td> <td style="text-align: center;">constant 1</td> <td style="text-align: center;">$1^{10} = 1$</td> </tr>
    <tr> <td style="text-align: center;">$B_{0.8}$</td> <td style="text-align: center;">$1110111011$</td> <td style="text-align: center;">80% 1's</td> <td style="text-align: center;">$0.8^8 \cdot 0.2^2$ ${}^{\star}$</td> </tr>
    <tr> <td style="text-align: center;">$B_{0.5}$</td> <td style="text-align: center;">$1011010010$</td> <td style="text-align: center;">50% 1's ${}^{\S}$</td> <td style="text-align: center;">$0.5^{10}$</td> </tr>
    <tr>
      <td colspan="4">${}^\star$: not most probable, most probable: $1, \dots, 1$, unlikely to be seen<br/>${}^\S$: fair coin flip</td>
    </tr>
    </tbody>
  </table>

+ Summary: Bernoulli distribution
  + simplest non-constant distribution
  + notation: $B_p \quad 0 \le p \le 1$
  + typical values
    + 0 and 1
    + $p(1) = p \quad p(0) = 1 - p$
  + properties:
    + $\mu = p$
    + $Var = pq$
    + $\sigma = \sqrt{pq}$
  + foundation of many other distributions


+ [Original Slides](https://tinyurl.com/yb6l3dwa)


### Problem Sets

0. Every random variable distributed over {0, 1} is Bernoulli.<br/>
  a. Yes<br/>
  b. Not necessarily<br/>

  Ans: <span style="color: magenta;">a</span><br/>
  Explanation: Every random variable over {0,1} attains the value 1 with some probability (p) and 0 with the remaining probability $(1-p)$, hence is \(B_p\). So the answer is Yes.


1. $X \sim B_p$ with  $p>0.5$  and  $Var(X)=0.24$. Find<br/>
  a. $p$,<br/>
  b. $E[X]$.<br/>

  Ans: a. (0.6); b. (0.6)<br/>
  Explanation:
    + For a Bernoulli distribution, $E[X^2]=E(X)=p$. Thus $Var(X)$ $=E[X^2]−(E[X])^2$ $=p−p^2=p(1−p)$. Since $0.24$ $=Var(X)$ $=p(1−p)$ and $p \ge 0.5$, we must have $p=0.6$.
    + $E[X]=p=0.6$.


2. Which of the following hold for two Bernoulli variables?<br/>
  a. Independent implies uncorrelated,<br/>
  b. Uncorrelated implies independent.<br/>

  Ans: ab<br/>
  Explanation
    + True. It is trivial.
    + True. Let  $X \sim B_{p_x}, Y \sim B_{p_y}$.

      If $X$ and $Y$ are uncorrelated, 
      
        \[\begin{align*}
          \text{Cov}(X, Y) &= E(XY) - E(X)E(Y) = \sum_{x = 0}^{1} \sum_{y = 0}^{1} xyP(X = x, Y = y) - p_{x}p_{y} \\ 
          &= P(X = 1, Y = 1) - p_{x}p_{y} = P(X = 1 | Y = 1)P(Y = 1) - p_{x}p_{y} \\ 
          &= (P(X = 1 | Y = 1) - p_{x})p_{y} = 0
        \end{align*}\]

      Hence, $P(X=1 \mid Y=1)=p_x=P(X=1)$ and similarly $P(Y=1 \mid X=1)=p_y=P(Y=1)$. From that, we have $P(X = 0 | Y = 1) = \frac{P(Y = 1 | X = 0)P(X = 0)}{P(Y = 1)} =$ $1 - p_{x} = P(X = 0) \implies$ $P(Y = 1 | X = 0) = p_{y} = P(Y = 1)$, and similarly $P(X=1|Y=0)=p_x=P(X=1)$. Thus, $X$ and $Y$ are independent.


3. Consider ten independent $B_{0.3}$ trials. Which of the following is the most probable?<br/>
  a. 0000000000<br/>
  b. 1111111111<br/>
  c. 1110000000<br/>
  d. 0001111111<br/>

  Ans: a<br/>
  Explanation: Under $B_{0.3}$, the probability of sequence with $w$ ones and $n−w$ zeros is $0.3^w \cdot 0.7^{(n−w)}$ $=0.7^n \cdot (3/7)^w$, which decreases with $w$. Hence 0000000000 is the most likely sequence with probability $0.7^{10}$, while 1111111111 is least likely with probability $0.3^{10}$. This is also logical as under $B_{0.3}$, every bit is more likely to be a 0 than a 1.


4. Consider ten independent $B_{0.3}$ trials. Which of the following is the most probable?

  Try to reconcile with the previous question.<br/>
  a. 10 zeros<br/>
  b. 10 ones<br/>
  c. 3 ones and 7 zeros<br/>
  d. 3 zeros and 7 ones.<br/>

  Ans: c<br/>
  Explanation: First, intuitively, for $B_{0.3}$ we expect to see roughly 30% 1's.  Slightly more rigorously, while individually, a sequence with 10 zeros is the most likely among all sequences, there is only one such sequence. When you balance the probability of each sequence with the number of such sequence, you see that observing a sequence with 3 ones and 7 zeros is most likely. We will do this calculation formally when we study binomial distributions in the next section.


5. Bernoulli modifications

  Let $X \sim B_{0.2}$. Find the Bernoulli parameter for the following random variables. Write $−1$ if they are not Bernoulli.<br/>
  a. $X^2$,<br/>
  b. $+\sqrt{X}$,<br/>
  c. $1 − X$,<br/>
  d. $−X$.<br/>

  Ans: a. (0.2); b. (0.2); c. (0.8); d. (-1)<br/>
  Explanation
    + Since $X \in \{0,1\}$, we have $X^2=X$.
    + Since $X \in \{0,1\}$, we have $+\sqrt{X}=X$.
    + $1−X$ takes values in {0,1}, hence is Bernoulli, and $1−X=1 \iff X=0$, which happens with probability 0.8.
    + $−X$ takes values in $\{0,−1\}$, hence is not Bernoulli.


6. Let $X \sim B_{0.4}$, $Y \sim B_{0.2}$, and they are independent. Find the Bernoulli parameter for the following random variables. Write $−1$ if they are not Bernoulli.<br/>
  a. $X \cdot Y$,<br/>
  b. $X^Y$, recall that $0^ 0=1$,<br/>
  c. $|X−Y|$,<br/>
  d. $X+Y$.<br/>

  Ans: a. (0.08); b. (0.88); c. (0.44); d. (-1)<br/>
  Explanation
    + $X \cdot Y$ takes values in $\{0,1\}$ hence is Bernoulli. It is $1 \iff X=Y=1$ which happens with probability $0.4 \cdot 0.2=0.08$.
    + $X^Y$ takes values in $\{0,1\}$, hence is Bernoulli. It is $0 \iff X=0$ and $Y=1$, which happens with probability $0.6 \cdot 0.2=0.12$, hence it is 1 with probability 0.88.
    + $|X−Y|$ takes values in $\{0,1\}$, hence is Bernoulli. It is $1 \iff X \neq Y$, which happens with probability $0.6 \cdot 0.2+0.4 \cdot 0.8=0.44$.
    + $X+Y$ takes values in $\{0,1,2\}$, hence is not Bernoulli.


7. Bernoulli sum

  $X=U+V$, where $U$ and $V$ are independent Bernoulli variables with different expectations but the same variance $0.21$. Find:<br/>
  a. $E(X)$,<br/>
  b. $Var(X)$,<br/>
  c. $\sigma_X$.<br/>

  Ans: a. (1); b. (0.42); c. (0.6481)<br/>
  Explanation
    + Let $U \sim B_p$ and $V \sim B_q$. Since $U$ and $V$ have the same variance, $p \cdot (1−p)=q \cdot (1−q)$, and since $p \neq q$, we must have $q=1−p$. Hence  $E[X]=E[U+V]=E[U]+E[V]=p+q=p+(1−p)=1$.
    + $Var(X)=Var(U)+Var(V)=0.42$.
    + $\sigma_X = \sqrt{Var(X)} = 0.6481$


8. Let $X$ be the number of heads when flipping four coins with heads probabilities 0.3, 0.4, 0.7, and 0.8. Find:<br/>
  a. $P(X=1)$,<br/>
  b. $E(X)$,<br/>
  c. $Var(X)$.<br/>

  Ans: a. (0.1872); b. (2.2); c. (0.82)<br/>
  Explanation
    + $P(X = 1) = 0.3 \cdot 0.6 \cdot 0.3 \cdot 0.2 + 0.7 \cdot 0.4 \cdot 0.3 \cdot 0.2 +$ $0.7 \cdot 0.6 \cdot 0.7 \cdot 0.2 +$ $0.7 \cdot 0.6 \cdot 0.3 \cdot 0.8 =$ $0.1872$
    + $E[X] = 0.3 + 0.4 + 0.7 + 0.8 = 2.2$
    + $Var(X) = 0.21 + 0.24 + 0.21 + 0.16 = 0.82$


9. Light bulbs

  Every light bulb is defective with 2% probability. What is the probability that a package of 8 bulbs will not suffice for a project requiring 7?

  Ans: 0.01034<br/>
  Explanation: Let $X \sim B(0.02,8)$ be the number of defective bulbs in a package. The box will not suffice if there are 2 or more defective bulbs, which happens with probability. $P(X \gt 1) = 1 - P(X = 0) - P(X = 1) =$ $1 - \binom{8}{0} \cdot (1-0.02)^8 - \binom{8}{1} \cdot (1-0.02)^7 \cdot 0.02 = 0.0103.$



### Lecture Video

<a href="https://tinyurl.com/yajddf3n" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.2 Binomial Distribution

+ The Binomial distribution
  + $n$ independent Bernoulli experiments
  + each "success" w/ the same probability $p$
  + "failure" w/ probability $1 - p = \overline{p} = q$
  + $B_{p,n}$ or $B_{n, p}$: distribution of \# successes
    + $B_{n, p}$ more common
    + using $B_{p, n}$ as generalized $B_p$
    + $B_{p, n}$: natural for Poisson Binomial 
  + example: $n$ independent coin flips
    + $P(\text{heads}) = p$
    + $B_{p, n}$: distribution of \# heads
    + e.g., $B_{\frac13, 5} \quad B_{.2, 10}$
  + no confusion: $n \in \Bbb{N}, \;0 \le p \le 1$
  + use $B_{p, n}$ because
    + generalized $B_p$
    + main parameter: $p$
    + extending to Poisson Binomial

+ Applications
  + positive responses to a treatment
  + faulty components
  + rainy days in a month
  + delayed flights

+ Small $n$
  + $n$ independent experiments
  + each w/ success probability $p$
  + failure probability: $q = 1 - p$
  + $b_{p, n}(k)$: probability of $k$ success, $0 \le k \le n$
  + example:
    + $n = 0$: $seq = \Lambda \quad k = 0 \quad b_{p, 0}(k) = 1$
    + $n = 1$: 
      + seq = 0: $k = 0 \quad b_{p, 1}(k) = q$
      + seq = 1: $k = 1 \quad b_{p, 1}(k) = p$
      + $p + q = 1$
    + $n = 2$:
      + seq = 00: $k = 0 \quad b_{p, 2}(k) =q^2$
      + seq${}^\star$ = 01, 10: $k = 1 \quad b_{p, 2}(k) = 2pq$
      + seq = 11: $k = 2 \quad b_{p, 2}(k) = p^2$
      + $p^2 + 2pq + q^2 = (p+q)^2 = 1^2 = 1$
      + ${}^\star$: each sequence w/ probability $pq$

+ General $n$ and $k$
  + $n$ ${\perp \!\!\!\! \perp}$ $B_p$ experiments
    + $k$ successes: success = 1
    + $0 \le k \le n$
  + pmf: $b_{p, n}(k) = p(k \text{ successes}) = \binom n k p^k q^{n-k}$
    + every $k$ successes sequence
      + $n-k$ failures
      + prob. $p^k \cdot q^{n-k}$
    + $\tbinom n k$ such sequences
  + distribution over $n+1$ values, including $0$ and $n$

+ Unitary
  + $0 \le k \le n \quad p(X = k) = b_{p, n}(k) = \binom n k p^k q^{n-k}$
  + Binomial theorem:

    \[ (a + b)^n = \sum_{k=0}^n \binom n k a^k b^{n-k} \]

  + sum to 1?

    \[ \sum_{k=0}^n b_{p, n} (k) = \sum_{k=0}^n p^k q^{n-k} = (p + q)^n = 1^n = 1 \]

+ Simulation: Typical distributions
  + $n= 20 \to b_{p, 20}(k)$
  + coin: $P(\text{heads}) = p \quad$ 20 flips $P(k, \text{ heads})$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yajcj3dk" ismap target="_blank">
      <img src="img/t08-01.png" style="margin: 0.1em;" alt="Typical distributions of head counts w/ 20 coin flips" title="Typical distributions of head counts w/ 20 coin flips" width=450>
    </a>
  </div>

+ Simulation: Typical samples

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yajcj3dk" ismap target="_blank">
      <img src="img/t08-02.png" style="margin: 0.1em;" alt="Typical samples w/ various repeated experiments" title="Typical samples w/ various repeated experiments" width=450>
    </a>
  </div>

+ Example: Multiple choice
  + exam w/ 6 multiple-choice questions, each w/ 4 possible answers
  + each question, student selects one of the 4 answers randomly
  + $X$ = \# correct answer $\sim B_{\frac14, 6}$
  + passing: $\ge 4$ correct answers
  + $P(\text{passing}) = ?$

    \[\begin{align*}
      P(4) &= \tbinom64 \cdot (\tfrac14)^4 \cdot (\tfrac34)^2 \approx 0.0329 \\
      P(5) &= \tbinom65 \cdot (\tfrac14)^5 \cdot (\tfrac34)^1 \approx 0.00439 \\
      P(6) &= \tbinom66 \cdot (\tfrac14)^6 \cdot (\tfrac34)^0 \approx 0.000244 \\
      P(\ge 4) &= P(4) + P(5) + P(6) \approx 0.03759
    \end{align*}\]

+ Interpretation as a Sum
  + $B_{p, n}$: a sum of $n\;B_p$
  + $X_1, \cdots, X_n \sim B_p\quad {\perp \!\!\!\! \perp}$
  + $X \stackrel{\text{def}}{=} \sum_{i=1}^n X_i$
  + $P(X=k) = P(\text{exactly } k \text{ of } X_1, \cdots, X_n \text{ are } 1) = \binom n k p^k q^{n-k} = b_{p, n}(k)$
  + $X \sim B_{p, n}$
  + applied to mean and variance

+ Mean and Variance
  + $X \sim B_{p, n}$
  + $X = \sum_{i=1}^n X_i \qquad X_1, \cdots, X_n \sim B_p \quad{\perp \!\!\!\! \perp}$

    \[\begin{align*}
      E[X] &= E\left[\sum_{i=1}^n X_i\right] \underbrace{=}_{\text{LE}} \sum E[X_i] \underbrace{=}_{B_p} \sum p = np \\\\
      Var(X) &= Var \left(\sum_{i=1}^n X_i \right) \underbrace{=}_{{\perp \!\! \perp}} \sum Var(X_i) \underbrace{=}_{B_p} \sum pq = npq \\\\
      \sigma &= \sqrt{npq}
    \end{align*}\]

+ Example: Multiple choice
  + exam w/ 6 multiple-choice questions, each w/ 4 possible answers
  + for each question, student selects one of the 4 answers randomly
  + $X$ = \# correct answers $\sim B_{\frac14, 6}$
  + mean: $E[X] = np = 6 \cdot \frac14 = 1.5$
  + standard deviation: $\sigma = \sqrt{npq} = \sqrt{6 \cdot \frac14 \cdot \frac34} = \frac{\sqrt{18}}{4}$

+ Example: Voting
  + for simplicity odd \# voters: $2n+1$
  + each equally likely D or R
  + $P(\text{voter makes a difference}) = P(\text{other 2n voters equally split})$

    \[\begin{align*}
      b_{p, n}(k) &= \binom n k p^k q^{n-k} = \binom{2n}{n} \frac{1}{2^n} \cdot \frac{1}{2^n} = \frac{(2n)!}{n! \cdot n! \cdot 2^n \cdot 2^n} \\\\
      &\approx \frac{\sqrt{2\pi \cdot 2n} (\frac{2n}{e})^{2n}}{\left({\sqrt{2\pi n}\left(\frac n e\right)^n}\right)^2 2^{2n}} = \frac{1}{\pi n} \quad(\color{Magenta}{\gg \frac 1 n})
    \end{align*}\]

  + applying Stirling approximation

    \[ n! \approx \sqrt{2\pi n}\left(\frac n e\right)^n \]

+ Poisson binomial
  + generalizing the binomial distribution $n \ge 1$
    + binomial distribution:
      + notation: $B_{p,n}$
      + for $ 1 \le i \le n \quad X_i \sim B_p \quad {\perp \!\!\!\! \perp}$
      + $X = \sum_{i=1}^n X_i$
    + Poisson binomial
      + notation: $PB_{p_1, \dots, p_n}$
      + for $ 1 \le i \le n \quad X_i \sim B_{p_i} \quad {\perp \!\!\!\! \perp}$
      + $X = \sum_{i=1}^n X_i$
  + examples
    + $PB_{\frac14, \frac23} \to X_1 \sim B_{\frac14} \;\; X_2 \sim B_{\frac23} \quad {\perp \!\!\!\! \perp}$<br/><br/>

    <table style="font-family: arial,helvetica,sans-serif; width: 30vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$X_1$</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$X_2$</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">$P$</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$X$</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$P(X)$</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">$0$</td> <td style="text-align: center;">$0$</td> <td style="text-align: center;">$\frac34 \cdot \frac13 = \frac14$</td> <td style="text-align: center;">$0$</td> <td style="text-align: center;">$\frac14$</td>
      </tr>
      <tr>
        <td style="text-align: center;">$0$</td> <td style="text-align: center;">$1$</td> <td style="text-align: center;">$\frac34 \cdot \frac23 = \frac12$</td> <td rowspan="2" style="text-align: center;">$1$</td> <td rowspan="2" style="text-align: center;">$\frac{7}{12}$</td>
      </tr>
      <tr>
        <td style="text-align: center;">$1$</td> <td style="text-align: center;">$0$</td> <td style="text-align: center;">$\frac14 \cdot \frac 13 = \frac{1}{12}$</td>
      </tr>
      <tr>
        <td style="text-align: center;">$1$</td> <td style="text-align: center;">$1$</td> <td style="text-align: center;">$\frac14 \cdot \frac23 = \frac16$</td> <td style="text-align: center;">$2$</td> <td style="text-align: center;">$\frac16$</td>
      </tr>
      </tbody>
    </table>

+ Expectation and Variance
  + $X \sim PB_{p_1, p_2, \cdots, p_n} \qquad X = \sum_{i=1}^n X_i \qquad X_i \sim B_{p_i}$
  
    \[\begin{align*}
      E[X] &= E\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n E[X_i] = \sum_{i=1}^n p_i \\\\
      Var(X) &= Var\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n Var(X_i) = \sum_{i=1}^n p_i(1-p_i)
    \end{align*}\]

  + $p(k)$:
    + no closed form
    + computationally

+ Plot & Sample in Python
  + Notebook: plot distribution, and generating data
  + import libraries

    ```python
    from scipy.stats import binom
    from matplot.pyplot import plot, hist
    ```

  + coin heads w/ prob .3, 50 flips: prob of k heads $\sim B_{.3, 50}$

    ```python
    plot(range(51), binom.pmf(k=range(51), n=50, p=.3))
    ```

  + repeat 1000 times $\to$ histogram

    ```python
    hist(binom.rvs(n=50, p=.3, size=1000))
    ```

+ Example: Coin flips
  + most basic convergence to average: $B(p)$
  + flip $n\; B(p)$ coin, average \# 1's will approach $np$
  + probability of a sequence w/ $k$ 1's and $(n-k)$ 0's is $p^k q^{n-k}$
  + Wolog assume $p > 0.5$, then most likely is $1^n$
  + yet by WLLN w/ probability $\to 1$, roughly $pn$ 1's and $qn$ 0's
  + why do we observe these sequences and not the most likely ones?
  + strength in \#s: \# sequences of a given composition increasing near $\frac12$
  + $pn$ balances \# $x$ probability


+ [Original Slides](https://tinyurl.com/yajcj3dk)


### Problem Sets

0. If you flip a fair coin 10 times and let X be the total number of heads, then Var(X) is<br/>
  a. 1.5<br/>
  b. 2.5<br/>
  c. 3.5<br/>
  d. None of the above<br/>

  Ans: b<br/>
  Explanation: The answer is $Var(X) = np(1 - p) = 2.5.$


1. There are 5 traffic signals between your home and work. Each is red with probability 0.35, independently of all others. Find:

  1) the probability of encountering no red lights,<br/>
    a. 2.26%<br/>
    b. 5.2%<br/>
    c. 11.6%<br/>
    d. 17.5%<br/>

  2) the probability of encountering only red lights,<br/>
    a. 0.03%<br/>
    b. 0.52%<br/>
    c. 1.16%<br/>
    d. 16.4%<br/>

  3) the expected number of red lights you will encounter?<br/>
    a. 0.75 <br/>
    b. 1.42 <br/>
    c. 1.75 <br/>
    d. 2.25 <br/>

  Ans: 1. (c); 2. (b); c. (c)<br/>
  Explanation
    + $(1 - 0.35)^5 = 0.116$
    + $0.35^5=0.0052$
    + The expectation of the sum is the sum of the expectations. $0.35+0.35+0.35+0.35+0.35=1.75$.


2. If every student is independently late with probability 10%, find the probability that in a class of 30 students:

  1) nobody is late,<br/>
  a. 4.2%<br/>
  b. 8.0%<br/>
  c. 17.4%<br/>
  d. 33.3%<br/>

  2) exactly 1 student is late.<br/>
  a. 3.33%<br/>
  b. 5.25%<br/>
  c. 7.75%<br/>
  d. 14.1%<br/>

  Ans: 1. (a); b. (d)<br/>
  Explanation
    + $(1 - 0.1)^{30} = 0.042$
    + $(1 - 0.1)^{29} \times 0.1 \times \binom{30}{1} = 0.141$


3. A coin with heads probability 0.6 is tossed 6 times, calculate the probability of observing:<br/>
  a. exactly two heads,<br/>
  b. at most one tails,<br/>
  c. an even number of heads.<br/>

  Ans: a. (0.13824); b (0.23328); c. (0.5)<br/>
  Explanation
    + The number of heads, $H$ is distributed as $H \sim Binomial(6,0.6)$. Thus the probability here is $P(H=2)=\binom62 0.6^2 \cdot 0.4^4=0.13824$.
    + The number of tails, $T$ is distributed as $T \sim Binomial(6,0.4)$. Thus the probability here is $P(T=0)+P(T=1) = \binom60 0.6^6+\binom61 0.4^1\cdot 0.6^5 = 0.23328$
    + Using the distribution of $H$, the probability here is $P(H=0)+P(H=2)+P(H=4)+P(H=6)$ $=\binom60 0.4^6+\binom62 0.6^2\cdot 0.4^4+\binom64 0.6^4\cdot 0.4^2+\binom66 0.6^6=0.50032$


4. A Binomial distribution $B_{p,n}$, where $p \neq 0$, has the same mean and standard deviation, namely $\mu=σ$.

  Find the mean of $B_{p,n+1}$

  Ans: 1<br/>
  Explanation: Since $B_{p,n}$ has $\sigma= \mu$, we have $np=\sqrt{npq}$, hence $1−p=q=np$, or $p \cdot (n+1)=1$.


5. For $X \sim B_{0.7,10}$, find:<br/>
  a. $E(X)$,<br/>
  b. $Var(X)$,<br/>
  c. $\sigma_X$,<br/>
  d. The most likely outcome of $X$.<br/>

  Ans: a. (7); b. (2.1); c. (1.4491); d.(7)<br/>
  Explanation
    + $E[X]=np=7$.
    + $Var(X)=np(1−p)=2.1$.
    + $\sigma_X = \sqrt{Var(X)} = 1.4491$
    + $P(X = 7) = \underset{x}{\max} P(X = x)$

6. Ten balls are randomly dropped into four urns. Let $X$ be the number of balls dropped into one pre-selected urn. Find:<br/>
  a. $P(X=0)$,<br/>
  b. $P(X=1)$,<br/>
  c. $E(X)$,<br/>
  d. $Var(X)$.<br/>

  Ans: a. (0.05631); b. (0.1877); c. (2.5); d. (1.875)<br/>
  Explanation
    + Clearly $X$ is distributed $B_{1/4,10}$. Hence $P(X=0)=\binom{10}0\cdot(1/4)^0\cdot(3/4)^{10}=(3/4)^{10}=0.056$
    + $P(X = 1) = \binom{10}1 \cdot (1/4)^1 \cdot (3/4)^9=0.188$
    + $E(X) = np = 10\cdot\frac14=\frac52.$
    + $Var(X)= np(1-p) = 10\cdot\frac14\frac34=\frac{30}{16}=\frac{15}{8}$


7. Our TA owns four Porsches, each works  80%  of the time, and two Ferraris, each works  60%  of the time. What is the probability that on a given day, at least half of the Porsches and at least half of the Ferraris work?

  Ans: 0.8172<br/>
  Explanation: Let $Q$ be the number of Porshes that work and $F$ be the number of Ferraris that work.  $P(Q\ge 2)=1-P(Q=0)-P(Q=1)=$ $1-{4\choose 0}0.2^4-{4\choose 1}0.8^1\cdot 0.2^3=$ $0.9728$. $P(F\ge 1)=1-P(F=0)=$ $1-{2\choose 0}0.4^2=0.84$.  Therefore the required probability is $P(Q\ge 2)\cdot P(F\ge 1)=0.817152$.


8. Alice solves every puzzle with probability 0.6, and Bob, with probability 0.5. They are given 7 puzzle and each chooses 5 out of the 7 puzzles randomly and solves them independently. A puzzle is considered solved if at least one of them solves it. What is the probability that all the 7 puzzles are solved?

  Ans: <span style="color: cyan;">0.021p</span><br/>
  Explanation: The probability that all the 7 puzzles are chosen is the probability that Bob chooses the two puzzles Alice did not pick, namely, \(\frac{\binom{5}{3}\binom22}{\binom{7}{5}} = \frac{10}{21}\). Every puzzle they both attempt, they both fail with probability \(0.4\cdot0.5=0.2\), hence at least one solves with probability \(1-0.2=0.8\). It follows that all puzzles are solved with probability \(\frac{10}{21}\cdot 0.6^2 \cdot 0.5^2 \cdot 0.8^3 = 0.0219\).



### Lecture Video

<a href="https://tinyurl.com/yamfgnzo" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.3 Poisson Distribution

+ The Poisson distribution
  + parameter: $\lambda \ge 0$
  + support: $\Bbb{N}$
  + pmf: $P_\lambda (k) = e^{-\lambda} \frac{\lambda^k}{k!}$
  + significance: approximating $B_{p, n}$ for large $n$ and small $p$ so that $np =\lambda$ is moderate

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.statisticshowto.com/poisson-distribution/" ismap target="_blank">
      <img src="https://tinyurl.com/yawo34ad" style="margin: 0.1em;" alt="Poisson distributions, valid only for integers on the horizontal axis. \lambda (also written as μ) is the expected number of event occurrences." title="Poisson distributions, valid only for integers on the horizontal axis. \lambda (also written as μ) is the expected number of event occurrences." width=250>
    </a>
  </div>

+ Applications
  + $P_\lambda$ approximating $B_{p, n}$ for small $p$, large $n$
  + Numerous applications
    + people clicking ad
    + daily store customers
    + responses to spam
    + rare-disease infection
    + gallery purchasing customers
    + flights no shows
    + daily 911 calls
    + typos in a page

+ Small $k$

  <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$\lambda$</th>
      <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$P_{\lambda} (k)$</th>
      <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Main Component Approx.</th>
      <th colspan="4" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">$k$</th>
    </tr>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">0</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">1</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">2</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">3</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td style="text-align: center;">General</td> <td style="text-align: center;">$e^{-\lambda}\frac{\lambda^k}{k!}$</td>
      <td style="text-align: center;">$\frac{\lambda^k}{k!}$</td> <td style="text-align: center;">$\frac{1}{e^\lambda}$</td>
      <td style="text-align: center;">$\frac{\lambda}{e^\lambda}$</td> <td style="text-align: center;">$\frac{\lambda^2}{2e^\lambda}$</td>
      <td style="text-align: center;">$\frac{\lambda^3}{6e^\lambda}$</td>
    </tr>
    <tr>
      <td style="text-align: center;">1</td> <td style="text-align: center;">$\frac{1}{e\cdot k!}$</td>
      <td style="text-align: center;">$\frac{1}{e}$</td> <td style="text-align: center;">$\frac{1}{e}$</td>
      <td style="text-align: center;">$\frac{1}{e}$</td> <td style="text-align: center;">$\frac{1}{2e}$</td>
      <td style="text-align: center;">$\frac{1}{6e}$</td>
    </tr>
    <tr>
      <td style="text-align: center;">2</td> <td style="text-align: center;">$\frac{2^k}{e^2 \cdot k!}$</td>
      <td style="text-align: center;">$\frac{2^k}{e^2}$</td> <td style="text-align: center;">$\frac{1}{e^2}$</td>
      <td style="text-align: center;">$\frac{2}{e^2}$</td> <td style="text-align: center;">$\frac{2}{e^2}$</td>
      <td style="text-align: center;">$\frac{4}{3e^2}$</td>
    </tr>
    <tr>
      <td style="text-align: center;">0</td> <td style="text-align: center;">$\frac{0^k}{k!}$</td>
      <td style="text-align: center;">$0^k$</td> <td style="text-align: center;">1</td>
      <td style="text-align: center;">0</td> <td style="text-align: center;">0</td>
      <td style="text-align: center;">0</td>
    </tr>
    </tbody>
  </table>

+ Binomial approximation
  + $P_\lambda$ approximating $B_{p, n}$ for $\lambda = np$, when $ n \gg 1 \gg p$

    \[\begin{align*}
      B_{p, n}(k) &= \binom n k p^k q^{n-k} \qquad (q = 1 - p \;\&\; p = \frac{\lambda}{n})\\
      &= \binom n k \left(\frac \lambda n\right)^k \left(1 - \frac \lambda n\right)^{n-k} = \dfrac{n^{\underline{k}}}{k!} \cdot \dfrac{\lambda^k}{n^k} \cdot \dfrac{\left(1-\frac{\lambda}{n}\right)^n}{\left(1 - \frac{\lambda}{n}\right)^k}
    \end{align*}\]

  + fix $k$ and $\lambda$, let $n \nearrow$ and $p \searrow$

  + Limit of Binomial

    \[ B_{p, n}(k) = \dfrac{n^{\underline{k}}}{k!} \cdot \dfrac{\lambda^k}{n^k} \cdot \dfrac{\left(1-\frac{\lambda}{n}\right)^n}{\left(1 - \frac{\lambda}{n}\right)^k} \quad \xrightarrow{n \to \infty}\quad e^{-\lambda} \frac{\lambda^k}{k!} \]

    + $\lambda$ and $k$ fixed, $n \to \infty$
      + $\dfrac{\lambda^k}{n^k} = \frac n n \cdot \frac{(n-1)}{n} \cdots \frac{(n-k+1)}{n} \to 1$: fixed \# (k) terms, each $\to 1$
      + $\left(1 - \frac{\lambda}{n}\right)^k \to 1$: fixed \# (k) terms, each $\to 1$
      + $\left(1-\frac{\lambda}{n}\right)^n = \left((1 - \frac 1 n)^{\frac n \lambda} \right)^{\lambda} \to (e^{-1})^\lambda = e^{-\lambda}$: increasing \# terms, each $\to 1 \quad (1 - \frac 1 m)^m \to e^{-1}$

+ Axioms
  + probability: $P_\lambda (k) = e^{-\lambda} \frac{\lambda^k}{k!} \quad k \ge 0$
  + non-negative: $P_\lambda (k) \ge 0$
  + Taylor expansion: $e^\lambda = \sum_{k=0}^\infty \frac{\lambda^k}{k!}$
  + unitary: 

    \[ \sum_{k=0}^\infty P_\lambda (k) = \sum_{k=0}^\infty e^{-\lambda} \frac{\lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^\infty \frac{\lambda^k}{k!} = e^{-\lambda}e^\lambda = 1 \]

+ Comparison of Mean and Variance of Binomial and Poisson distributions
  + $P_\lambda$ approximating $B_{p, n}$ for $\lambda = np$ when $n \gg 1 \gg p$
  + expectation:
    + $B_{p, n}$: $\mu = np$
    + $P_\lambda$: $\mu = \lambda$
  + variance
    + $B_{p, n}$: $Var = npq$
    + $P_\lambda$: $Var = \lambda$

+ Mean and variance of Poisson distribution
  + Expectation: $E[X] = E[X^{\underline{1}}] = \lambda$
  + variance
    + $Var(X) = E[X^2] - (E[X])^2 = \lambda^2 + \lambda - \lambda^2 = \lambda$
    + $E[X(X-1)] = E[X^\underline{2}] = \lambda^2$
    + $E[X^2] = E[X(X-1) + X] = E[X(X-1)] + E[X] = \lambda^2 + \lambda$
  + standard deviation: $\sigma = \sqrt{\lambda} \to$ small relative to the mean

  + Observation

      \[\begin{array}{lcl}
        \frac{d}{d \lambda} \lambda^k = k \lambda^{k-1} = \dfrac{k}{\lambda} \lambda^k &\quad& \dfrac{d^2}{d \lambda^2} \lambda^k = k^{\underline{2}} \lambda^{k-2} = \dfrac{k^{\underline{2}}}{\lambda^2} \lambda^k \\
        \dfrac{d^r}{d \lambda^r} \lambda^k = k^{\underline{r}} \lambda^{k-r} = \dfrac{k^{\underline{r}}}{\lambda^r} \lambda^k &\quad&
        k^{\underline{r}} \lambda^k = \lambda^r \dfrac{d^r}{d\lambda^r} \lambda^k
      \end{array}\]

  + Falling moments
    + $X \sim P_\lambda \qquad k^{\underline{r}} \lambda^k = \lambda^r \frac{d^r}{d\lambda^r} \lambda^k$

      \[\begin{align*}
        E[X^{\underline{r}}] &= \sum_{k=0}^\infty k^{\underline{r}} P_\lambda(k) = \sum_k k^{\underline{r}} e^{-\lambda} \frac{\lambda^k}{k!} 
          = e^{-\lambda} \sum_k k^{\underline{r}} \frac{\lambda^k}{k!} = e^{-\lambda} \sum_k \frac{\lambda^r}{k!} \frac{d^r}{d\lambda^r} e^\lambda \\
        &= e^{-\lambda} \lambda^r \frac{d^r}{d\lambda^r} \sum_k \frac{\lambda^k}{k!} = e^{-\lambda} \lambda^r \frac{d^r}{d\lambda^r} e^\lambda 
          = e^{-\lambda} \lambda^r e^\lambda = \lambda^r
      \end{align*}\]

    + $E[X] = E[X^{\underline{1}}] = \lambda \qquad E[X(X-1)] = E[X^\underline{2}] = \lambda^2$

+ Approximation example
  + factory producing 200 times, each defective w/ probability 1%
  + $P(3 \text{ defective})?$
    + Binomial (precise): $B_{0.01, 200}(3) = \binom{200}{3} (0.01)^3(0.99)^{197} \approx 0.181$
    + Poisson (approximation): $\lambda = 200 \cdot 0.01 = 2 \quad P_2(3) = e^{-2}\frac{2^3}{3!} \approx 0.18$
  + $P(\text{some defective})?$
    + $B_{0.01, 200}(0) = \binom{200}{0} (0.99)^{200} \approx 0.134 \quad B_{0.01, 200}(\ge 1) = 1 - 0.134 \approx 0.866$
    + $P_2(0) = e^{-2} \frac{2^0}{0!} = e^{-2} \approx 0.135 \quad P_2(\ge 1) = 1- 0.135 \approx 0.865$

+ Summary: Poisson distribution
  + pmf: $P_\lambda (k) = e^{-\lambda} \frac{\lambda^k}{k!} \quad \lambda \ge 0 \quad k \ge 0$
  + approximation: $B_{p,n}$ for $\lambda = np$, when $n \gg 1 \gg p$
  + applications: \# of ad clicks, rare disease, production defects
  + properties
    + $\mu = \lambda$
    + $Var = \lambda$
    + $\sigma = \sqrt{\lambda}$


+ [Original Slides](https://tinyurl.com/y9zxs75t)


### Problem Sets

0. The mean and the variance of a Poisson distribution is the same.

  Ans: True


1. Assume a telemarketer's successful sales per hour is a Poisson random variable with $\lambda=2$. What is the probability that the telemarketer makes no sales in 1 hour?<br/>
  a. 13.5%<br/>
  b. 22.5%<br/>
  c. 27.7%<br/>
  d. 31.2%<br/>

  Ans: a<br/>
  Explanation: $P(X = k) = \frac{\lambda^k}{k!}e^{-\lambda}$. With $\lambda = 2$, $P(X = 0) = e^{-2} = 0.135$


2. The expectation of a Poisson random variable and its variance are<br/>
  a. equal<br/>
  b. not equal<br/>

  Ans: a


3. Random variable X is distributed Poisson, and $P(X=2)=P(X=4)$. Find $P(X=3)$.

  Ans: 0.2169<br/>
  Explanation: $P(X=2)=P(X=4) \implies \lambda =2 \sqrt{3}$. Hence $P(X=3)=4\sqrt{3}\cdot e^{-2\sqrt{3}}$.


4. Let $X$ be distributed Poisson with parameter 1. Find $P(X \ge 2 \mid X \le 4)$.

  Ans: 0.2606<br/>
  Explanation: Here $P(X=i)=e^{−1}/i!$. Therefore $P(X\ge 2 \mid X\le 4) =$ $\frac{P(X\ge 2\cap X\le 4)}{P(X\le 4)} =$ $\frac{(1/2+1/6+1/24)e^{-1}}{(1+1+1/2+1/6+1/24)e^{-1}}$, and the answer follows.


5. Assume the number of typo errors on a single page of a book follows Poisson distribution with parameter 1/3. Calculate the probability that on one page there are<br/>
  a. no typo,<br/>
  b. exactly two typos,<br/>
  c. more than one typo?<br/>

  Ans: a. (0.7165); b. (0.03981); c. (0.0446249)<br/>
  Explanation
    + Recall that for a random variable $X \sim Poisson(\lambda)$ its distribution is given by $P(X=i)=e^{-\lambda}\cdot\frac{\lambda^i}{i!}, \, \text{for} \, i>0$. $\lambda=1/3$ here. Thus probability of no typo errors is $P(X=0)=e^{-\lambda}=e^{-1/3}=0.7165313$.
    + Similarly the probability of exactly two typo errors is $P(X=2)=e^{-\lambda}\frac{\lambda^2}{2!}=e^{-1/3}\frac{(1/3)^2}{2!}=0.039807295$
    + Consider the compliment or the probability of at most one typo. This probability is given by $P(X=0)+P(X=1) =$ $e^{-\lambda}+e^{-\lambda}\frac{\lambda}{1!} =$ $e^{-1/3}+e^{-1/3}\frac{(1/3)}{1!}$. Thus our required probability is $1-(e^{-1/3}+e^{-1/3}\frac{(1/3)}{1!})=0.0446249$.


6. If a random variable $X$ follows Poisson distribution with $\lambda = 2.5$, calculate<br/>
  a. $E[X]$<br/>
  b. $E[X^2]$<br/>
  c. $V(X)$<br/>

  Ans: a. (2.5); b. (8.75); c. (2.5)<br/>
  Explanation:
    + You can solve all these parts either applying the fomulas for Poisson distributions or, as we do here, directly. $E(X)=\sum_{i=0}^{\infty}i\cdot P(X=i)$ $=\sum_{i=1}^{\infty}i\cdot e^{-\lambda}\cdot\frac{\lambda^i}{i!}$ $=\lambda\sum_{i=1}^{\infty}e^{-\lambda}\cdot\frac{\lambda^{i-1}}{i-1!}$ $=\lambda\sum_{j=0}^{\infty}e^{-\lambda}\cdot\frac{\lambda^{j}}{j!}$ $=\lambda=2.5$.
    + $E(X^2)=\sum_{i=0}^{\infty}i^2\cdot P(X=i)$ $=\sum_{i=1}^{\infty}i^2\cdot e^{-\lambda}\cdot\frac{\lambda^i}{i!}$ $=\sum_{i=1}^{\infty}(i(i-1)+i)\cdot e^{-\lambda}\cdot\frac{\lambda^i}{i!}$ $=\sum_{i=1}^{\infty}i(i-1)\cdot e^{-\lambda}\cdot\frac{\lambda^i}{i!}+\sum_{i=1}^{\infty}i\cdot e^{-\lambda}\cdot\frac{\lambda^i}{i!}$ $=\lambda^2\sum_{i=2}^{\infty}e^{-\lambda}\cdot\frac{\lambda^{i-2}}{i-2!} +\lambda\sum_{i=1}^{\infty}e^{-\lambda}\cdot\frac{\lambda^{i-1}}{i-1!}$ $=(\lambda^2+\lambda)\sum_{j=0}^{\infty} e^{-\lambda}\cdot\frac{\lambda^{j}}{j!}$ $=\lambda^2+\lambda=8.75.$
    + We know $V(X)=E(X^2)-E(X)^2=\lambda^2+\lambda-\lambda^2=\lambda=2.5$


7. Assume the number of tropical storms making landfall in the Philippines each year follows Poisson distribution with parameter $9$. What is the probability that there are less than $6$ tropical storms making landfall in Philippines in one year?

  Ans: 0.1156<br/>
  Explanation: Follows from summing the corresponding Poisson probabilities under $\lambda=9$.


8. A computer manufacturing company produce chips with defect probability  0.001 . In a package of  2000  chips, denote the number of defective chips by  X . Use Poisson distribution for approximation:<br/>
  a. The Poisson parameter for  X  is:<br/>
  b. $P(X>1)=$?<br/>
  c. $P(X \le 3)=$?<br/>

  Ans: a. (2); b. (0.594); c. (0.8571)<br/>
  Explanation
    + Poisson approximation yields $\lambda=np=2000 \cdot 0.001=2$.
    + Evaluate Poisson probability of $P(X>1)$ with $\lambda=2$.
    + Evaluate Poisson probability of $P(X\le 3)$ with $\lambda = 2$.


9. A vendor sells merchandise through Amazon and Ebay. On Ebay she sells an average rate of 2 items per day, while on Amazon the daily average is 3. Both sales follow a Poisson distribution and are independent of each other. What is the probability that she sells 5 items on a given day?

  Ans: 0.17546<br/>
  Explanation: You can solve this the hard way, namely $P_3(0) \cdot P_2(5)+P_3(1) \cdot P_2(4)+ \cdots +P_3(5) \cdot P_2(0)$.  Or note that the sum of independent $P_{\lambda_1}$ and a  $P_{\lambda_2}$ random variables is a $P_{\lambda_1+\lambda_2}$ random variable, the total number of items the vendor sells is a $P_{2+3}=P_5$ variable and $P_5(5) = \frac{e^{-5} \cdot 5^5}{5!}$.


### Lecture Video

<a href="https://tinyurl.com/y7pv37da" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.4 Geometric Distribution

+ Back to basics
  + independent $B_p$ coin flips: $p(1) = p \quad p(0) = 1 - p \stackrel{\text{def}}{=} q$
  + two derived distributions
    + Binomial: $B_{p, n}$ = \# 1's in flips
    + Geometric: $G_p$ = \# flips till first 1 = time to first success

<table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Flips</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$X$</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:1%;">$\hspace{0.05em}$</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$n$</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">$X_1, \dots, X_n \sim B(p)$</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$p(n)$</th>
  </tr>
  </thead>
  <tbody>
  <tr> <td style="text-align: center;">$101011$</td> <td style="text-align: center;">1</td> <td></td> <td style="text-align: center;">1</td> <td>$X_1 = 1$</td> <td style="text-align: center;">$p$</td> </tr>
  <tr> <td style="text-align: center;">$010111$</td> <td style="text-align: center;">2</td> <td></td> <td style="text-align: center;">2</td> <td>$X_1 = 0\;\; X_2 =1$</td> <td style="text-align: center;">$qp$</td> </tr>
  <tr> <td style="text-align: center;">$001010$</td> <td style="text-align: center;">3</td> <td></td> <td style="text-align: center;">3</td> <td>$X_1 = X_2 = 0 \;\; X_3=1$</td> <td style="text-align: center;">$q^2p$</td> </tr>
  <tr> <td style="text-align: center;">$\cdots$</td> <td style="text-align: center;">$\cdots$</td> <td></td> <td style="text-align: center;">$\cdots$</td> <td>$\cdots$</td> <td style="text-align: center;">$\cdots$</td> </tr>
  <tr> <td style="text-align: center;">$\underbrace{0\dots01}_{n}0$</td> <td style="text-align: center;">n</td> <td></td> <td style="text-align: center;">n</td> <td>$X_1=\cdots=X_{n-1}=0 \;\; X_n = 1$</td> <td style="text-align: center;">$q^{n-1}p$</td> </tr>
  </tbody>
</table>

+ Geometric distribution
  + Notation: $G_p \quad 0 <p \le 1$
  + probability: $p(n) = q^{n-1} p \stackrel{\text{def}}{=} g_p(n) \quad n \ge 1$
  + range: $p \neq 0$, $n$ can be arbitrary high

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/avarkvc" ismap target="_blank">
      <img src="https://tinyurl.com/y7a8jauc" style="margin: 0.1em;" alt="Probability mass function of Geometric distribution" title="Probability mass function of Geometric distribution" width=450>
    </a>
  </div>

+ Example
  + 30 years ago, thief trying to find a matching key
  + trials to hit a target
  + attempts till success
  + or till failure nowadays

+ Axioms
  + probability: $P(n) = pq^{n-1} \quad n \ge 1 \quad q = 1-p$
  + non-negative: $P(n) \ge 0$
  + unitary

    \[ \sum_{n=1}^\infty p(1-p)^{n-1} = p \sum_{i=0}^\infty (1-p)^i = p \cdot \frac{1}{1 - (1-p)} = \frac p p = 1 \]

+ Cumulative Distribution Function (CDF)
  
  $n \in \Bbb{N} \quad X > n \iff X_1 = \cdots = X_n =0$
  
  \[ P(X > n) = P(X_1 = \cdots = X_n = 0) = q^n \]

  \[F(n) = P(X \le n) = 1 - P(X > n) = 1 - q^n \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/avarkvc" ismap target="_blank">
      <img src="https://tinyurl.com/ybmlj5xz" style="margin: 0.1em;" alt="Cumulative distribution function of Geometric distribution" title="Cumulative distribution function of Geometric distribution" width=450>
    </a>
  </div>

+ Expectation via "Right" CDF
  + $x \in \Bbb{N} \quad P_k = P(X = k)$
  + general expectation:

    \[\begin{align*}
      E[X] &= \sum_{k=0}^\infty kP_k = P_1 + 2 P_2 + 3 P_3 + \cdots \\
      &= P(X \ge 1) + P(X \ge 2) + P(X \ge 3) + \cdots
    \end{align*}\]

  + geometric distribution

    \[ E[X] = \sum_{k=1}^\infty P(X \ge k) = \sum_{i=0}^\infty P(X > i) = \sum_{i=0}^\infty \frac{1}{1-q} = \frac{1}{p} \]

+ Variance

  \[\begin{align*}
    E[X(X-1)] &= \sum_{n=1}^\infty n(n-1) \cdot P(X=n) = p\sum_{n=1}^\infty n(n-1)q^{n-1} \\
    &= pq \sum_{n=2}^\infty \frac{d^2}{dq^2} q^n = pq \frac{d^2}{dq^2} \sum_{n=2}^\infty q^n \\
    & = pq \frac{d^2}{dq^2} (\frac{1}{1-q} -1 -q) \hspace{4em}\left( \frac{1}{1-q} \right)^\prime = \frac{1}{(1-q)^2} \\
    &= pq \frac{2}{(1-q)^3} = \frac{2q}{p^2} \hspace{6em}\left( \frac{1}{(1-q)^2} \right)^\prime = \frac{2}{(1-q)^3}\\
  \end{align*}\]

  \[ E[X^2] = E[X(X-1)] + E[X] = \frac{2q}{p^2} + \frac{1}{p} = \frac{2q+p}{p^2} = \frac{1+q}{p^2} \]

  \[ Var(X) = E[X^2] - (E[X])^2 = \frac{1+q}{p^2} - \frac{1}{p^2} = \frac{q}{p^2} \quad\to\quad \sigma = \frac{\sqrt{q}}{p} \]

+ Example: fair coin
  + $X \sim G_{\frac12}$
  + $P(X = k) = g_{0.5}(k) = (\frac12)^{k-1} \cdot \frac12 = \frac{1}{2^k}$
  + $E[X] = \frac1q = 2$
  + $Var(X) = \frac{q}{p^2} = 2$

+ "Even" odds
  + normally, "even ods" means that two events have equal probabilities
  + something a bit more "odd"
  + probability that a Geometric random variable is even

    \[ P(1) = p \quad P(2) = p \cdot q = q \cdot P(1) \]

+ Memoryless
  + Definition: (memoryless) a distribution over $\Bbb{P} = \{1, 2, \dots\}$ is a <span style="color: magenta; font-weight: bold;">memoryless</span> if $\forall\, n \ge 0, m > 1$

    \[ P(X = n+m \mid X > n) = P(X=m) \]

  + e.g., $P(X = 12 \mid X > 10) = P(X=2)$
  + after observing or any number of samples, process behaves as at the start

+ Geometric and Memoryless
  + Geometric $\to$ Memoryless

    \[\begin{align*}
      P(X = n+m \mid X > n) & = \frac{P(X=n+m, X > n)}{P(X > n)} = \frac{P(X=n+m)}{P(X>n)} \\
      &= \frac{p \cdot q^{n+m-1}}{q^n} = p \cdot q^{m-1} = P(X=m)
    \end{align*}\]

    + all geometric distributions are memoryless
  + Memory $\to$ Geometric
    + any discrete memoryless distribution over $\Bbb{P}$ is geometric
    + $p \stackrel{\text{def}}{=} P(X=1) \quad q \stackrel{\text{def}}{=} 1 - p = P(X > 1)$

    \[\begin{align*}
      \forall\, n \ge 1, \;\; P(X=n+1) &= P(X>1 \wedge X = n+1) \\
      &= P(X > 1) \cdot P(X=n+1 \mid X > 1) = q \cdot P(X=n)
    \end{align*}\]

    + Hence, $P(X=2) = qp, \;\;P(X=3) = q^2p, \dots$
    + $P(X=n) = q^{n-1}q \to$ Geometric

+ $r$ successes
  + Geometric: $P(X=n) = P(\text{first success at n'th trial})$
  + generalized geometric: $n \ge r$

    \[\begin{align*}
      P(\text{r'th success at n'th trial}) &= P(r-1 \text{ successes in } n-1 \text{ trial}) \cdot P(\text{n'th trial is success})\\
       &= b_{n-1, p}(r-1) \cdot p = \binom{n-1}{r-1} p^{r-1} q^{n-r} p \\
      &= \binom{n-1}{r-1} p^r q^{n-r}
    \end{align*}\]

  + special case: $r = 1 \to pq^{n-1} = g_p(n)$
  + a.k.a. negative binomial distribution

+ Summary
  + pmf: $P(n) = p q^{n-1} \quad n \ge 1 \quad q = 1 - p$
  + memoryless
  + mean: $E[X] = \frac 1 p$
  + variance: $Var(X) = \frac{q}{p^2}$
  + standard deviation: $\sigma = \frac{\sqrt{q}}{p}$
  + $r$ successes: $P(\text{r'th success at n'th trial}) = \binom{n-1}{r-1} p^r q^{n-r}$


+ [Original Slides](https://tinyurl.com/y73cv4pe)


### Problem Sets

0. Which of the following distributions is memoryless?<br/>
  a. Poisson<br/>
  b. Geometric<br/>
  c. Both<br/>
  d. Neither<br/>

  Ans: b<br/>
  Explanation: Only the geometric distribution.


Several of the following questions ask about the number of experiments performed till a certain outcome is observed. Unless otherwise stated, include the final experiment (where the outcome is observed) in the count. For example, the number of  coin tosses till observing a heads in the sequence t, t, h, is 3.


1. A die is rolled until the number 1 turns up. The expected number of rolls is<br/>
  a. 2,<br/>
  b. 4,<br/>
  c. 6,<br/>
  d. 8.<br/>

  Ans: c<br/>
  Explanation: $E(X) = \frac{1}{p} = 6$


2. A pair of dice are repeatedly rolled till the two sum to $\ge 10$. For example (6,3), (2,4), (5,5), stopping after three pair rolls. The expected number of times the pair is rolled is:<br/>
  a. 2,<br/>
  b. 4,<br/>
  c. 6,<br/>
  d. 8.<br/>

  Ans: c<br/>
  Explanation: There are 6 outcomes where a pair of dice sums to at least 10: (4,6), (5,5), (5,6), (6,4), (6,5), (6,6).  Hence the probability of this large sum is 6/36=1/6.  The number of times we rol the pair till we observe $\ge 10$, is distributed $G_{1/6}$.  The expected number is 6.  


3. A $G_p$ random variable is odd with probability<br/>
  a. $\frac{1−p}{2−p}$,<br/>
  b. $\frac{p}{2−p}$,<br/>
  c. $\frac{1}{2−p}$,<br/>
  d. $p+(1−p)^2 \cdot p$.<br/>

  Ans: c<br/>
  Hints:
    + There are two natural ways to find the probability that $X \sim G_p$ is odd. You may want to try both. One is "brute force", by adding the probabilities that $X=1,3,5, \dots$. The other is by relating $P(X \text{ is even})$ to $P(X \text{ is odd})$.
    + For the brute-force way, recall that $1+q+q^2+ \cdots=\frac{1}{1−q}$. If you use the second method, first show that $P(X\text{ is even})=(1-p)\cdot P(X\text{ is odd})$.

  Explanation: There are two natural ways to find the probability that $X \sim G_p$ is odd.
    + The first is "brute force". Recall that $1+q+q^2+ \cdots =$ $\frac1{1-q}$. Hence, $P(X \text{ is odd})$ $=P(X=1)+P(X=3)+\ldots$ $= p+{\overline p}^2\cdot p+{\overline p}^4\cdot p+\ldots$ $= \frac{p}{1-{\overline p}^2}=$ $\frac{p}{1-(1-p)^2}=$ $\frac{p}{2p-p^2}=\frac1{2-p}$
    + The second method is by relating $(P(X \text{ is even})$, $P(X\text{ is even})=$ $P(X\text{ is even}\cap X\gt 1)=$ $P(X\gt 1)\cdot P(X \text{ is even}\mid {X \gt 1})=$ $P(X \gt 1)\cdot P(X \text{ is odd})$. $X$  is even or odd, hence $1=P(X\text{ is odd})+(1-p)\cdot P(X\text{ is odd})=$ $(2-p)\cdot P(X\text{ is odd})$.  Hence, $P(X\text{ is odd})=\frac1{2-p}$
    + [Probability  X  is odd in a geometric distribution - StackExchange](https://tinyurl.com/y6987gfj)
    + [ the probability that a geometric random variable X is an even number - StackExchange](https://tinyurl.com/y4dbaau2)
    + [Probability that a geometric random variable is even - StackExchange](https://tinyurl.com/y57k6do5)


4. Find the expected number of coin tosses till the third heads appears, (e.g., for  h,t,h,t,h , five coins were tossed).

  Ans: 6<br/>
  Explanation: For $1 \le i \le 3$, let $X_i$ be the number of tosses between the $i−1$th and $i$th heads. For example, for t,h,t,t,h,h, then $X_1=2$, $X_2=3$, and $X_3=1$. Each $X_i$ is a distributed $G_{1/2}$, hence has expectation 2. The number of coin tosses till the third head appears is $X_1+X_2+X_3$, and by the linearity of expectations, $E(X_1+X_2+X_3)=E(X_1)+E(X_2)+E(X_3)=6$


5. $X$ is the random number of times a coin with heads probability $1/4$ is tossed till the first heads appears, find:<br/>
  a. $E(X)$,<br/>
  b. $E(X^2)$,<br/>
  c. $V(X)$,<br/>
  d. $\sigma_X$,<br/>
  e. $P(X \le 10)$,<br/>
  f. $P(X > 5)$.<br/>

  Ans: a. (4); b. (28); c. (12); d. (3.4641); e. (0.9437); f. (0.2373)<br/>
  Explanation
    + $E(X) = \frac{1}{p} = 4$
    + $E(X^2) = V(X) + E^2(X) = 28$
    + $V(X) = \frac{1-p}{p^2} = 12$
    + $\sigma_X = \sqrt{V(X)} = 3.4614$
    + $P(X\le 10) = \sum_{i = 0}^{9} pq^i = 0.9437$
    + $P(X \gt 5) = \sum_{i = 6}^{\infty} pq^i = 1 - \sum_{i = 0}^{4} qp^i = 0.2373$


6. Two coins with heads probabilities $1/3$ and $1/4$ are alternately tossed, starting with the $1/3$ coin, until one of them turns up heads. Let $X$ denote the total number of tosses, including the last. Find:<br/>
  a. $P(X=5)$,<br/>
  b. $P(X \text{ is odd})$,<br/>
  c. $E(X)$.<br/>

  Ans: a. (1/12); b. (2/3); c. (10/3)<br/>
  Explanation
    + $P(X=5)=\frac23\cdot\frac34\cdot\frac23\cdot\frac34\cdot\frac13=\frac1{12}$
    + Similar to Problem 3, this can be done in two ways. Brute force or relating two probabilities. For the brute force, $P(X\text{ is odd})=P(X=1)+P(X=3)+\ldots=$ $\frac13+\frac23\cdot\frac34\cdot\frac13+\cdots$ $=\frac13+\frac12\cdot\frac13+(\frac12)^2\cdot\frac13+\ldots=\frac13\cdot(1+(\frac12)^2+(\frac12)^3+\ldots)$ $=\frac13\cdot\frac{1}{1-\frac12}=\frac23$. Alternatively, $P(X\text{ is odd})=P(X=1)+P(X\text{ is odd}\cap X\ge 3)$ $=P(X=1)+P(X\ge 3)\cdot P(X\text{ is odd}\mid X\ge 3)$ $=P(X=1)+P(X\ge 3)\cdot P(X\text{ is odd})$ $=\frac13+\frac23\cdot\frac34\cdot P(X\text{ is odd})$.  Hence $12⋅P(X \text{ is odd})=\frac13$, or $P(X \text{ is odd})=\frac23$,
    + $E(X) = \frac13\cdot1 + \frac23 \cdot \frac14 \cdot2+ \sum^{\infty}_{i=3} i \cdot P(X = i)$ $= \frac13\cdot1 + \frac23 \cdot \frac14 \cdot2+ \sum^{\infty}_{i=1} (i + 2) \cdot P(X = i + 2, X > 2)$ $= \frac13\cdot1 + \frac23 \cdot \frac14 \cdot2+ \sum^{\infty}_{i=1} (i + 2) \cdot P(X = i + 2 | X > 2) \cdot P(X > 2)$ $= \frac13\cdot1 + \frac23 \cdot \frac14 \cdot2+ \frac23 \cdot\ \frac34 \cdot \sum^{\infty}_{i=1} (i + 2) \cdot P(X = i)$ $= \frac13\cdot1 + \frac23 \cdot \frac14 \cdot2+ \frac23 \cdot\ \frac34 \cdot(E(X)+2)$. Hence $E[X]⋅(1−\frac12)= \frac13 + \frac13 + 1= \frac53$. And therefore $E[X]=\frac{10}{3}$.



### Lecture Video

<a href="https://tinyurl.com/ya89ejfl" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>



## 8.5 Geometric Distribution Example

+ Startup statistics
  + P(startup success) = 20%, independent of previous attempts
  + Expected \# startups till first success

    \[ X \sim G_{0.2} \qquad E[X] = \frac{1}{.2} = 5 \]

  + Home grown entrepreneur
  + dad will fund up to three startups $P(\text{success})?$
    + one of first three startups succeeds
    + $P(X \le 3) = F(3) = 1 - (0.8)^3 \approx 0.49$

  + Cry uncle
    + even wealthier uncle funds next three startups (4, 5, 6)
    + $P(\text{success w/ uncle if dad's help did not suffice})?$

      \[\begin{align*}
        P(C \in \{4, 5, 6\} \mid X > 3) &= P(4 \mid X > 3) + P(5 \mid X > 3) + P(6 \mid X > 3) \\
        &= P(1) + P(2) + P(3) = P(X \le 3) \approx 49\%
      \end{align*}\]

    + $P(\text{success w/ uncle})?$
      + 1, 2 ,3 failed but one of 4, 5, 6 succeeded

      \[\begin{align*}
        P(3 < X \le 6) &= P(X > 3 \cap X  \le 6) = P(X > 3) \cdot P(X \le 6 \mid X > 3) \\
          &= (0.8)^3 \cdot 0.49 \approx 25\% \hspace{4em} \left(P(X_1, X_2, X_3 \text{ failed}) = q^3\right)\\
        P(3 < X \le 6) &= F(6) - F(3) = (1-0.8^6) - (1 - 0.8^3)\\
          &= 0.8^3 - 0.8^6 \approx 25\%
      \end{align*}\]

+ Foreign-born entrepreneur
  + $X$: time to first success w/ $p = 0.2$
  + $r^X$: fraction of company you keep w/ $r = 0.5$

    \[\begin{align*}
      E[r^X] &= \sum_{k=1}^\infty r^k P(X=k) = \sum_{k=1}^\infty p q^{k-1} r^k = pr \sum_{i=0}^\infty (qr)^i \\
        &= \frac{pr}{1-qr} = \frac{0.2 \cdot 0.5}{1 - 0.8 \cdot 0.5} = \frac{0.1}{1-0.4} = \frac{0.1}{0.6} \approx 16.67\%
    \end{align*}\]

+ Coupon collector problem
  + pre-coupon
    + $n$ coupons
    + each item containing one coupon selected uniformly
    + collecting all coupons, get a prize
    + how many items need to buy to collect all?
  + expectation
    + $X$: \# items to collect all coupons (coupon 1, 2, 3)
    + $n = 3 \quad$ items: 1 2 3 4 5 6 7 $\hspace{1em}X=7 \quad E[X]?$

      \[ \text{Coupon }\hspace{1em} \underbrace{2}_{X_1 = 1}\hspace{1em} \hspace{1em}\underbrace{2\; 3}_{X_2 = 3-1=2}\hspace{1em} \hspace{1em}\underbrace{2\; 3\; 3\; 1}_{X_3 = 7-3=4} \] 

    + $X_i$: \# items to get $i^{th}$ coupon after getting $i - 1$ coupons
    + $X = X_1 + X_2 + X_3 \to X_1 = 1, X_2 \sim G_{2/3}, X_3 \sim G_{1/3} \;\;{\perp \!\!\!\! \perp}$
    + $X_1 = 1, X_2 = 2, X_3 = 4$

+ General $n$
  + $X_i \sim G\left(\frac{n - (i-1)}{n}\right) = G\left( \frac{n-i+1}{n} \right)$
  + mean: $E[X_i] = \frac{n}{n-i+1}$
  + collected sum: $X = \sum_{i=1}^n X_i$
  + expected of collected sum:

    \[\begin{align*}
      E[X] &= \sum_{i=1}^n E[X_i] = \sum_{i=1}^n \frac{n}{n-i+1} = \frac n n + \frac{n}{n-1} + \frac{n}{n-2} + \cdots + \frac n 1 \\
        &= n(\frac11 + \frac12 + \cdots + \frac 1 n) = nH_n \approx n \ln n + 0.577n
    \end{align*}\]

  + Harmonic sum: $H_n = \frac11 + \frac12 + \cdots + \frac 1 n \to \ln n + 0.577$

    \[\begin{align*}
      H_n &> \int_1^{n+1} \frac 1 x dx = \left. \ln x \right|_1^{n+1} = \ln(n+1) \\
      H_n &\le 1 + \int_1^n \frac 1 x dx = 1 + \left. \ln x \right|_1^n = 1 + \ln n \\
      H_n &\to \ln n + 0.577
    \end{align*}\]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/psh9xdx" ismap target="_blank">
        <img src="https://tinyurl.com/y8hswat4" style="margin: 0.1em;" alt="Illustration of the integral test." title="Illustration of the integral test." width=200>
      </a>
    </div>

+ Variance
  + $X \sim G(P) \quad Var(X) = \frac{1 - p}{p^2} \le \frac{1}{p^2}$

  \[\begin{align*}
    Var(X) &= Var\left( \sum_{i=1}^n X_i \right) = \sum_{i=1}^n Var(X_i) \le \sum_{i=1}^n \frac{1}{\left( \frac{n-i+1}{n} \right)^2} \\
    &= n^2 \left( \frac{1}{n^2} + \frac{1}{(n-1)^2} + \cdots + \frac{1}{1^2} \right) \le \frac{\pi^2}{6} n^2 \\\\
    \sigma &\le \frac{\pi}{\sqrt{6}} n
  \end{align*}\]


+ [Original Slides](https://tinyurl.com/y86jwhb3)


### Problem Sets

0. If X and Y are two independent geometric random variables, then X+Y also is also geometric.

  Ans: False<br/>
  Explanation: False. For example, if (X) and (Y) have the same success probability, (X+Y) will follow a negative-binomial distribution.


1. In a basketball shooting workout, a player keeps shooting until she makes 10 baskets. Suppose the probability that she makes any given shot is 0.7, and let $X$ be the total number of shots she takes. Calculate:<br/>
  a. $E[X]$,<br/>
  b. $V(X)$.<br/>

  Ans: a. (100/7); b. (300/49)<br/>
  Explanation
    + Let $X_i$ be the random variable indicating the number of shots between the $(i−1)^{th}$ and  ith  shots $(i \in \{1,...,10\})$. Then, the total number of shots, $T = \sum^{10}_{i=1} X_i$. Using the fact that here each of the random variables $X_i \sim Geometric(0.7)$, and that for a geometric distribution with parameter $p, E[Xi]=1/p, E[X_i]=1/0.7=10/7$. Further, by linearity of expectation $E[T]= \sum^{10}_{i=1} E[X_i]=100/7$.
    + Using that $X_i \sim Geometric(0.7)$, and that for a geometric distribution with parameter $p, V(X_i)=(1−p)/p2, V(X_i)=0.3/0.49=30/49$. Here, each of the $X_i$, $(i \in \{1,...,10\})$, are also independent. Thus $V(T)= \sum^{10}_{i=1} V(X_i)=300/49$.


2. A production line has a 5% defective rate, and its products are inspected one-by-one until the first defect is found. Given that the first 10 inspections do not find any defect, what is the probability that the number of inspections is at most 20?

  Ans: $1 - 0.95^{10} = 04013$<br/>
  Explanation: Let $D$ be the event of interest here. Further let  E  denote the event that any 10 consecutive inspections find a defect. Since the inspections here are independent, the required probability $P(D)=P(E)$.  Now if $\overline{E}$ denotes the compliment of event $E$, $P(\overline{E})=1−P(E)=(1−0.05)^{10}$ since in $\overline{E}$ we require that no defective item be discovered in each of the 10 inspections. Thus $P(D)=P(E)=1−0.95^{10}$.


3. A bag contains $K$ blue balls and $N−K$ red balls. Find the expected number of blue balls observed when $n$ balls are randomly drawn. Does the answer depend on whether the selection is with or without replacement?<br/>
  a. $n\frac{K}{N}$<br/>
  b. $(n-1)\frac{K}{N}$<br/>
  c. $(n-1)\frac{K-1}{N-1}$<br/>
  d. $(n)\frac{K-1}{N-1}$<br/>

  Ans: <span style="color: magenta;">a</span><br/>
  Explanation: 
    + Without replacement, the expectation is 
      \[\begin{align*}
        \sum_{k=0}^n k\frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}} &= \frac{K}{\binom{N}{n}} \sum_{k=1}^{n} \binom{K-1}{k-1}\binom{N-K}{n-k} = \frac{K}{\binom{N}{n}} \underbrace{\sum_{k=0}^{n-1} \binom{K-1}{k}\binom{N-K}{n-1-k}}_{\substack{\text{# of ways to choose } n - 1 \\ \text{ balls} \text{out of } N - 1 \text{ balls}}} \\
        &= \frac{K}{\binom{N}{n}} \binom{N-1}{n-1} = n\frac{K}{N}
      \end{align*}\]

    + With replacement, the expectation is trivial, which is $n\frac K N$.
    + Hence, the answer does not depend on whether the selection is with or without replacement.


4. A bag contains $6$ blue balls and $9$ red balls, if $5$ balls are randomly picked from the bag with replacement, what is the most likely number of blue balls that will be picked?

  Ans: 2<br/>
  Explanation: Intuitively, it is most likely to get 2 blue balls and 3 red balls. Let $X$ be the number of blue balls. $P(X = k) = \frac{\binom{6}{k} \binom{9}{5 - k}}{\binom{15}{5}}$, and we can show that it reaches its maximum when $k=2$.



### Lecture Video

<a href="https://tinyurl.com/y7bujhby" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 8

+ [Original Lecture NB](https://tinyurl.com/y9gvbznk)

+ [Local Lecture NB](img/Topic08_Lecture.ipynb)

+ [Local Lecture Python code](img/Topic08_Lecture.py)

+ `scipy.stats.xxx` class: xxx = bernoulli, binom, poisson, geom<br/><br/>

  <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Return Value(s)</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td style="font-weight: bold;">rvs(<code>args1</code>, loc=0, size=1, random_state=None)</td>
      <td>Random variates.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">pmf(<code>args2</code>, loc=0)</td>
      <td>Probability mass function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">logpmf(<code>args2</code>, loc=0)</td>
      <td>Log of the probability mass function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">cdf(<code>args2</code>, loc=0)</td>
      <td>Cumulative distribution function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">logcdf(<code>args2</code>, loc=0)</td>
      <td>Log of the cumulative distribution function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">sf(<code>args2</code>, loc=0)</td>
      <td>Survival function (also defined as <code>1 - cdf</code>, but <i>sf</i> is sometimes more accurate).</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">logsf(<code>args2</code>, loc=0)</td>
      <td>Log of the survival function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">ppf(<code>args3</code>, loc=0)</td>
      <td>Percent point function (inverse of <code>cdf</code> — percentiles).</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">isf(<code>args3</code>, loc=0)</td>
      <td>Inverse survival function (inverse of <code>sf</code>).</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">stats(<code>args1</code>, loc=0, moments='mv')</td>
      <td>Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">entropy(<code>args1</code>, loc=0)</td>
      <td>(Differential) entropy of the RV.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">expect(func, args=(<code>args1</code>,), loc=0, lb=None, ub=None, conditional=False)</td>
      <td>Expected value of a function (of one argument) with respect to the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">median(<code>args1</code>, loc=0)</td>
      <td>Median of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">mean(<code>args1</code>, loc=0)</td>
      <td>Mean of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">var(<code>args1</code>, loc=0)</td>
      <td>Variance of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">std(<code>args1</code>, loc=0)</td>
      <td>Standard deviation of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">interval(alpha, <code>args1</code>, loc=0)</td>
      <td>Endpoints of the range that contains alpha percent of the distribution</td>
    </tr>
    <tr>
      <td colspan="2">
        <a href="https://tinyurl.com/y72ffzu9">bernoulli</a>: <code>args1</code> = <code>p</code>, <code>args2</code> = <code>k, p</code>, <code>args3</code> = <code>q, p</code><br/><br/>
        <a href="https://tinyurl.com/y8fdjfpy">binom</a>: <code>args1</code> = <code>n, p</code>, <code>args2</code> = <code>k, n, p</code>, <code>args3</code> = <code>q, n, p</code><br/><br/>
        <a href="https://tinyurl.com/ybx8l3ot">poisson</a>: <code>args1</code> = <code>mu</code>, <code>args2</code> = <code>k, mu</code>, <code>args3</code> = <code>q, mu</code><br/><br/>
        <a href="https://tinyurl.com/y9bbkpnt">geom</a>: <code>args1</code> = <code>p</code>, <code>args2</code> = <code>k, p</code>, <code>args3</code> = <code>q, p</code>
      </td>
    </tr>
    </tbody>
  </table>

+ [`numpy.linspace`](https://tinyurl.com/yc4ne47x)
  + Syntax: `numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
  + Docstring
    + Return evenly spaced numbers over a specified interval.
    + Returns num evenly spaced samples, calculated over the interval [start, stop].
    + The endpoint of the interval can optionally be excluded.

+ [`numpy.histogram`](https://tinyurl.com/y4exrjuz)
  + Syntax: `numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)`
  + Docstring: Compute the histogram of a set of data.

+ [`numpy.histogram_bin_edges`](https://tinyurl.com/y5ss3l6l)
  + Syntax: `numpy.histogram_bin_edges(a, bins=10, range=None, weights=None)`
  + Docstring: Function to calculate only the edges of the bins used by the histogram function.

+ [`Series.value_counts`](https://tinyurl.com/y5hno3mz)
  + Syntax: `Series.value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True)`
  + Docstring
    + Return a Series containing counts of unique values.
    + The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.



## Programming Assignment 8

### Introduction

In this assignment you will learn to use functions to access, query, and gain simple insights from datasets.


### Dataset

The Titanic was a British passenger liner that collided with an iceberg and sank in its maiden voyage. Tragically, of the 2,200 passengers onboard, only 800 survived. We will use the [Titanic:Machine Learning from disaster](https://www.kaggle.com/c/titanic/data) dataset to estimate survival probabilies associated with various passengers. We will use only the train.csv file.


### Some useful functions

The following functions outlined in pandas library may prove helpful.

+ `pandas.read_csv()` $\to$ Reads the .csv file
+ `pandas.value_counts()` $\to$ Returns count of unique values


### Questions

Using the dataset, estimate the following.

1. Survival probability for a passenger.

  Ans: 0.3838<br/>
  
  ```python
  import pandas as pd
  import numpy as np

  data = pd.read_csv('./Titanic/train.csv')

  print(data['Survived'].value_counts(normalize=True))
  ```


2. Survival probability for a female passenger.

  Ans: 0.742038<br/>

  ```python
  import pandas as pd
  import numpy as np

  data = pd.read_csv('./Titanic/train.csv')

  print(data[data['Sex'] == 'female'].groupby(['Sex'])['Survived'].value_counts()\
      / data[data['Sex'] == 'female']['Survived'].count())
  ```


3. Probability that a person is a female and survives.

  Ans: 0.2615<br/>

  ```python
  import pandas as pd
  import numpy as np

  data = pd.read_csv('./Titanic/train.csv')

  print(data.groupby(['Sex'])['Survived'].value_counts() / data['Survived'].count())
  ```


4. It is more probable for a person from the first class to survive as compared to the second or third class. (True/False)

  Ans: True





