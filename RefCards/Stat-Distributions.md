# Statistics: Distributions


## Bernoulli Distribution

+ [Bernoulli distribution](/Notes/p01-Bayesian.md#361-binary-data-with-a-discrete-prior-distribution): for a single Bernoulli trial w/ outcome 0 or 1, the likelihood for each possible value for $\theta$

  \[ p(y | \theta_j) = \theta_j^y (1 - \theta_j)^{1-y}  \qquad \text{ where } \quad
    p(y | \theta_j) = \begin{cases} \theta_j & \text{ if } y = 1 \\ 1 - \theta_j & \text{ if } y = 0 \end{cases}
  \]

+ [Bernoulli distribution](../Stats/ProbStatsPython/08-DiscreteDist.md#81-bernoulli-distribution)
  + notation: $B_p\quad 0 \le p \le 1$
  + pmf: $p(0) = 1-p = \overline{p} = q \quad p(1) = p$
  + unitary: $p(0) + p(1) = (1-p) + p = 1$
  + $X \sim B_p$

+ [Characteristics](../Stats/ProbStatsPython/08-DiscreteDist.md#81-bernoulli-distribution)
  + binary version of complex events
  + repeated trials yield \# successes
  
+ [Mean and Variance](../Stats/ProbStatsPython/08-DiscreteDist.md#81-bernoulli-distribution)
  + mean: $E[X] = \sum p(x) \cdot x = (1-p) \cdot 0 + p \cdot 1$
  + variance: $Var(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p) = pq$
  + standard deviation: $\sigma = \sqrt{pq}$
  + various $p$
    + $p = 0 \to E[X] = 0, \;Var(X) = 0, \;\sigma = 0$
    + $p = 1 \to E[X] = 1, \;Var(1) = 0, \;\sigma = 0$
    + $p = \tfrac12 \to E[X] = \tfrac12, \;Var(X) = \frac14, \;\sigma = \frac12$
    + $B_p$ varying most when $p = \frac12$

+ [Independent trials](../Stats/ProbStatsPython/08-DiscreteDist.md#81-bernoulli-distribution)
  + most common type of Bernoulli distribution: independent ${\perp \!\!\!\! \perp}$
  + generally, $X_1, X_2, \cdots, X_n \sim B_p \to  {\perp \!\!\!\! \perp}$
    + $x^n = x_1, x_2, \cdots, x_n \in \{0, 1\}^n$
    + $n_0$ = number of 0's; &nbsp;&nbsp;&nbsp;&nbsp;  $n_1$ = number of 1's
    + $\Pr(x_1, \dots, x_n) = p^{n_1} q^{n_0}$



## Binomial Distribution

+ [Binomial distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + $Y$: a discrete binomial variable w/ the sampling distribution of the total number of 'successes' in $n$ independent Bernoulli trials
  + $\theta$: the probability of success in each Bernoulli trial
  + $\theta^y (1 - \theta)^{n-y}$: the likelihood, the probability for a specific sequence of $n-y$ 'failure' and $y$ 'successes', $\begin{pmatrix} n \\ y \end{pmatrix}$ sequences
  + $Y \sim Bin(n, \theta)$: a binomial distribution w/ properties

    \[\begin{align*}
      p(y | n, \theta) & = \begin{pmatrix} n \\ y \end{pmatrix} \theta^y (1-\theta)^{n-y}, \qquad y = 0, 1, \dots, n \tag{Bin.prob} \\
      E(Y | n, \theta) &  = n \theta \tag{Bin.mean} \\
      Var(Y | n, \theta) &= n \theta (1-\theta) \tag{Bin.var}
    \end{align*}\]

  + $Y \sim Bern(\theta)$: a Bernoulli distribution, a binomial w/ $n=1$

+ [The Binomial distribution](../Stats/ProbStatsPython/08-DiscreteDist.md#82-binomial-distribution)
  + "failure" w/ probability $1 - p = \overline{p} = q$
  + notation: $B_{p,n}$ or $B_{n, p}$: distribution of \# successes
    + $B_{n, p}$ more common
    + using $B_{p, n}$ as generalized $B_p$
    + $B_{p, n}$: natural for Poisson Binomial
  + use $B_{p, n}$ because
    + generalized $B_p$
    + main parameter: $p$
    + extending to Poisson Binomial

+ [General $n$ and $k$](../Stats/ProbStatsPython/08-DiscreteDist.md#82-binomial-distribution)
  + $n$ ${\perp \!\!\!\! \perp}$ $B_p$ experiments
  + $b_{p, n}(k) = p(k \text{ successes}) = \binom n k p^k q^{n-k}$

+ [Unitary](../Stats/ProbStatsPython/08-DiscreteDist.md#82-binomial-distribution)

    \[ \sum_{k=0}^n b_{p, n} (k) = \sum_{k=0}^n p^k q^{n-k} = (p + q)^n = 1^n = 1 \]

+ [Interpretation as a Sum](../Stats/ProbStatsPython/08-DiscreteDist.md#82-binomial-distribution)
  + $X_1, \cdots, X_n \sim B_p\quad {\perp \!\!\!\! \perp}$
  + $X \stackrel{\text{def}}{=} \sum_{i=1}^n X_i$
  + $\Pr(X=k) = \Pr(\text{exactly } k \text{ of } X_1, \cdots, X_n \text{ are } 1) = \binom n k p^k q^{n-k} = b_{p, n}(k)$

+ [Mean and Variance](../Stats/ProbStatsPython/08-DiscreteDist.md#82-binomial-distribution)

    \[\begin{align*}
      E[X] &= E\left[\sum_{i=1}^n X_i\right] \underbrace{=}_{\text{LE}} \sum E[X_i] \underbrace{=}_{B_p} \sum p = np \\\\
      Var(X) &= Var \left(\sum_{i=1}^n X_i \right) \underbrace{=}_{{\perp \!\! \perp}} \sum Var(X_i) \underbrace{=}_{B_p} \sum pq = npq \\\\
      \sigma &= \sqrt{npq}
    \end{align*}\]



## Beta Distribution

+ [Beta distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + a flexible and mathematically convenient class for quantities contained to lie btw 0 and 1
  + form: $Y \sim Beta(a, b)$
  + properties

    \[\begin{align*}
      p(y | a, b) &= \frac{\Gamma (a, b)}{\Gamma (a) \Gamma (b)} y^{a-1} (1-y)^{b-1}, \quad y \in (0, 1) \tag{Beta.prob}\\
      E(Y|a, b) &= \frac{a}{a+b} \tag{Beta.mean}\\
      Var(Y|a, b) &= \frac{ab}{(a+b)^2(a+b+1)} \tag{Beta.var}
    \end{align*}\]

    where $\Gamma(a) = (a-1)!$


## Beta-Binomial Distribution

+ [The Beta-Binomial distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the Beta distribution as a conjugate distribution of the binomial distribution
  + an analytically tractable compound distribution
  + $\theta$ parameter in the binomial distribution as being randomly draw from a beta distribution
  
    \[ X \sim Bin(n, \theta) \implies p(X=k | p, n) = L(p | k) = \begin{pmatrix} n \\ k \end{pmatrix} \theta^k (1-\theta)^{n-k} \]

  + $\theta$: a random variable w/ a beta distribution

    \[ p(\theta | a, b) = Beta(a, b) = \frac{\theta^{a-1} (1-\theta)^{b-1}}{B(a, b)} \qquad \text{ for } 0 \leq \theta \leq 1 \]

    + $B(a, b) = \Gamma(a) \Gamma(b) / \Gamma(a+b)$

  + the compound distribution

    \[\begin{align*}
      p(k | n, a, b) 
        &= \int_0^1 \underbrace{L(\theta | k)}_{\text{binomial}} \cdot \underbrace{p(\theta | a, b)}_{\text{beta}} d\theta \\
        &= \begin{pmatrix} n \\ k \end{pmatrix} \frac{1}{B(a, b)} \int_0^1 \theta^{k+a-1} (1-\theta)^{n-k+b-1} d\theta
        = \begin{pmatrix} n \\ k \end{pmatrix} \frac{B(k+a, n-k+b)}{B(a, b)} \\\\
        &= \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)} \frac{\Gamma(k+a)\Gamma(n-k+b)}{\Gamma(n+a+b)} \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
    \end{align*}\]



## Poisson Distribution

+ Binomial approximation
  + $P_\lambda$ approximating $B_{p, n}$ for $\lambda = np$, when $ n \gg 1 \gg p$

    \[ B_{p, n}(k) &= \binom n k p^k q^{n-k} = \dfrac{n^{\underline{k}}}{k!} \cdot \dfrac{\lambda^k}{n^k} \cdot \dfrac{\left(1-\frac{\lambda}{n}\right)^n}{\left(1 - \frac{\lambda}{n}\right)^k} \]

+ Axioms
  + probability: $P_\lambda (k) = e^{-\lambda} \frac{\lambda^k}{k!} \quad k \ge 0$
  + non-negative: $P_\lambda (k) \ge 0$
  + Taylor expansion: $e^\lambda = \sum_{k=0}^\infty \frac{\lambda^k}{k!}$
  + unitary:

    \[ \sum_{k=0}^\infty P_\lambda (k) = \sum_{k=0}^\infty e^{-\lambda} \frac{\lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^\infty \frac{\lambda^k}{k!} = e^{-\lambda}e^\lambda = 1 \]

+ Mean and variance of Poisson distribution
  + Expectation: $E[X] = E[X^{\underline{1}}] = \lambda$
  + variance: $Var(X) = E[X^2] - (E[X])^2 = \lambda^2 + \lambda - \lambda^2 = \lambda$
  + standard deviation: $\sigma = \sqrt{\lambda} \to$ small relative to the mean

+ Summary: Poisson distribution
  + pmf: $P_\lambda (k) = e^{-\lambda} \frac{\lambda^k}{k!} \quad \lambda \ge 0 \quad k \ge 0$
  + approximation: $B_{p,n}$ for $\lambda = np$, when $n \gg 1 \gg p$
  + applications: \# of ad clicks, rare disease, production defects
  + properties
    + $\mu = \lambda$
    + $Var = \lambda$
    + $\sigma = \sqrt{\lambda}$





## Geometric Distribution

+ [Back to basics](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + independent $B_p$ coin flips: $p(1) = p \quad p(0) = 1 - p \stackrel{\text{def}}{=} q$
  + two derived distributions
    + Binomial: $B_{p, n}$ = \# 1's in flips
    + Geometric: $G_p$ = \# flips till first 1 = time to first success

+ [Geometric distribution](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + Notation: $G_p \quad 0 < p \le 1$
  + probability: $p(n) = q^{n-1} p \stackrel{\text{def}}{=} g_p(n) \quad n \ge 1$
  + range: $p \neq 0$, $n$ can be arbitrary high

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/avarkvc" ismap target="_blank">
      <img src="https://tinyurl.com/y7a8jauc" style="margin: 0.1em;" alt="Probability mass function of Geometric distribution" title="Probability mass function of Geometric distribution" width=450>
    </a>
  </div>

+ [Axioms](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + probability: $\Pr(n) = pq^{n-1} \quad n \ge 1 \quad q = 1-p$
  + non-negative: $\Pr(n) \ge 0$
  + unitary

    \[ \sum_{n=1}^\infty p(1-p)^{n-1} = p \sum_{i=0}^\infty (1-p)^i = p \cdot \frac{1}{1 - (1-p)} = \frac p p = 1 \]

+ [Cumulative Distribution Function (CDF)](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  
  $n \in \Bbb{N} \quad X > n \iff X_1 = \cdots = X_n =0$
  
  \[ \Pr(X > n) = \Pr(X_1 = \cdots = X_n = 0) = q^n \]

  \[F(n) = \Pr(X \le n) = 1 - \Pr(X > n) = 1 - q^n \]

+ [Expectation via "Right" CDF](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + geometric distribution

    \[ E[X] = \sum_{k=1}^\infty \Pr(X \ge k) = \sum_{i=0}^\infty \Pr(X > i) = \sum_{i=0}^\infty \frac{1}{1-q} = \frac{1}{p} \]

+ [Variance](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)

  \[ E[X^2] = E[X(X-1)] + E[X] = \frac{2q}{p^2} + \frac{1}{p} = \frac{2q+p}{p^2} = \frac{1+q}{p^2} \]

  \[ Var(X) = E[X^2] - (E[X])^2 = \frac{1+q}{p^2} - \frac{1}{p^2} = \frac{q}{p^2} \quad\to\quad \sigma = \frac{\sqrt{q}}{p} \]

+ ["Even" odds](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + normally, "even ods" means that two events have equal probabilities
  + something a bit more "odd"
  + probability that a Geometric random variable is even

    \[ \Pr(1) = p \quad \Pr(2) = p \cdot q = q \cdot \Pr(1) \]

+ [Memoryless](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + Definition: (memoryless) a distribution over $\Bbb{P} = \{1, 2, \dots\}$ is a <span style="color: magenta; font-weight: bold;">memoryless</span> if $\forall\, n \ge 0, m > 1$

    \[ \Pr(X = n+m \mid X > n) = \Pr(X=m) \]

  + after observing or any number of samples, process behaves as at the start

+ [Geometric and Memoryless](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + Geometric $\to$ Memoryless
    + all geometric distributions are memoryless
  + Memory $\to$ Geometric
    + any discrete memoryless distribution over $\Bbb{P}$ is geometric

+ [$r$ successes](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + generalized geometric: $n \ge r$

    \[ \Pr(\text{r'th success at n'th trial}) = \binom{n-1}{r-1} p^r q^{n-r} \]

  + special case: $r = 1 \to pq^{n-1} = g_p(n)$
  + a.k.a. negative binomial distribution

+ [Summary](../Stats/ProbStatsPython/08-DiscreteDist.md#84-geometric-distribution)
  + pmf: $\Pr(n) = p q^{n-1} \quad n \ge 1 \quad q = 1 - p$
  + memoryless
  + mean: $E[X] = \frac 1 p$
  + variance: $Var(X) = \frac{q}{p^2}$
  + standard deviation: $\sigma = \frac{\sqrt{q}}{p}$
  + $r$ successes: $\Pr(\text{r'th success at n'th trial}) = \binom{n-1}{r-1} p^r q^{n-r}$

+ [Coupon collector problem](../Stats/ProbStatsPython/08-DiscreteDist.md#85-geometric-distribution-example)
  + expectation
    + $X_i$: \# items to get $i^{th}$ coupon after getting $i - 1$ coupons
    + $X = X_1 + X_2 + X_3 \to X_1 = 1, X_2 \sim G_{2/3}, X_3 \sim G_{1/3} \;\;{\perp \!\!\!\! \perp}$
    + $X_1 = 1, X_2 = 2, X_3 = 4$

  + General $n$
    + $X_i \sim G\left(\frac{n - (i-1)}{n}\right) = G\left( \frac{n-i+1}{n} \right)$
    + mean: $E[X_i] = \frac{n}{n-i+1}$
    + collected sum: $X = \sum_{i=1}^n X_i$
    + expected of collected sum:

      \[ E[X] = \sum_{i=1}^n E[X_i] =  n(\frac11 + \frac12 + \cdots + \frac 1 n) = nH_n \approx n \ln n + 0.577n \]

    + Harmonic sum: $H_n = \frac11 + \frac12 + \cdots + \frac 1 n \to \ln n + 0.577$

  + Variance
    + $X \sim G(P) \quad Var(X) = \frac{1 - p}{p^2} \le \frac{1}{p^2}$

    \[\begin{align*}
      Var(X) &= n^2 \left( \frac{1}{n^2} + \frac{1}{(n-1)^2} + \cdots + \frac{1}{1^2} \right) \le \frac{\pi^2}{6} n^2 \\\\
      \sigma &\le \frac{\pi}{\sqrt{6}} n
    \end{align*}\]





