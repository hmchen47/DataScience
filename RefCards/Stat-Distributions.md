# Statistics: Distributions


## Bernoulli Distribution

+ [Bernoulli distribution](/Notes/p01-Bayesian.md#361-binary-data-with-a-discrete-prior-distribution): for a single Bernoulli trial w/ outcome 0 or 1, the likelihood for each possible value for $\theta$

  \[ p(y | \theta_j) = \theta_j^y (1 - \theta_j)^{1-y}  \qquad \text{ where } \quad
    p(y | \theta_j) = \begin{cases} \theta_j & \text{ if } y = 1 \\ 1 - \theta_j & \text{ if } y = 0 \end{cases}
  \]


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



## Uniform Distribution

+ [Uniform distribution](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)
  + __Definition__: (uniform) for $a < b$, the <span style="color: cyan; font-weight: bold;">uniform</span> distribution, $U_{[a, b]}$, is constant inside $[a. b]$ and 0 outside

    \[ f(x) = \begin{cases} c & x \in [a, b] \quad \text{equally likely} \\ 0 & x \notin [a, b] \quad \text{never happen} \end{cases} \]

  + unitary:
    + $1 = \int_{-\infty}^\infty f(x)\, dx = c(b-a)$
    + $c = \frac{1}{b-a}$

+ [Cumulative distribution function](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)

  \[ F(x) = \int_{-\infty}^x f(u) du = \begin{cases}
    \int_{-\infty}^x 0\, du = 0 & x \le a \\\\ F(a) + \int_a^x \frac{1}{b-a} \,du = \frac{x-a}{b-a} & a \le x \le b \\\\ F(b) + \int_b^x 0\, du = 1 & x \ge b
  \end{cases}\]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yata7tbx" ismap target="_blank">
      <img src="../Stats/ProbStatsPython/img/t09-04.png" style="margin: 0.1em;" alt="PDF and CDF of Uniform Distribution" title="PDF and CDF of Uniform Distribution" width=250>
    </a>
  </div>

+ [Interval probabilities](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)
  
  for $a \le \alpha \le \beta \le b$

    <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Interval</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Probability</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td>$(\alpha, \beta]$</td>
        <td>$F(\beta) - F(\alpha) = \frac{\beta - a}{b-a} - \frac{\alpha - a}{b - a} = \frac{\beta - \alpha}{b - a}$</td>
      </tr>
      <tr>
        <td>$[\beta, \infty)$</td>
        <td>$F(\infty) - F(\beta) = 1 - \frac{\beta - a}{b-a} = \frac{b - \beta}{b - a}$</td>
      </tr>
      <tr>
        <td>$\{\alpha\}$</td>
        <td>$F(\alpha) - F(\alpha) = 0$</td>
      </tr>
      </tbody>
    </table>

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yata7tbx" ismap target="_blank">
      <img src="../Stats/ProbStatsPython/img/t09-05.png" style="margin: 0.1em;" alt="Illustration of interval probabilities" title="Illustration of interval probabilities" width=200>
    </a>
  </div>

+ [Expectation and variance](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)
  + pdf: $X \sim U_{[0, 1]}$ first, $f(x) =1,\; 0 \le x \le 1$
  + mean: $E[X] = \frac12$ by symmetry
  + $E[X^2] = \frac13$
  + variance: $Var(X) = \frac{1}{12}$
  + standard deviation: $\sigma = \frac{1}{2\sqrt{3}} \approx 0.29$

+ [Translation and scaling](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)
  + pdf: $Y = aX + b \stackrel{\text{def}}{=} g(X)$

    \[ f_Y(y) = \left.\frac{f_X(x)}{|g^\prime(x)|}\right|_{x = g^{-1}(y)} = \frac 1 a \]

    + or: equal-length interval map to equal-length intervals

+ [General $\mu$ and $\sigma$](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)
  + notation: $Y \sim U_{[a, b]}$
  + pdf: $Y = (b-a)X + a$
  + mean: $E[Y] = (b-a) E[X] + a = \frac{b-a}{2} + a = \frac{a+b}{2}$
  + variance: $Var(Y) = Var((b-a)X + a) = (b-a)^2 Var(X) = \frac{(b-a)^2}{12}$
  + standard deviation: $\sigma = \frac{b-a}{2\sqrt{3}} \approx 0.29(b-a)$

+ [Uniform Distributions](../Stats/ProbStatsPython/09-ContDist.md#93-uniform-distribution)
  + notation: $U_{[a, b]} \quad a < b$
  + pdf:

    \[ f(x) = \begin{cases} \frac{1}{b-a} & X \in [a, b] \\ 0 & x \notin [a, b] \end{cases} \]

  + CDF

    \[ F(X) = \begin{cases} 0 & x \le a \\ \frac{x-a}{b-a} & a \le x \le b \\ 1 & x \ge b \end{cases} \]

  + parameters
    + $\mu = \frac{a+b}{2}$
    + $Var = \frac{(b-a)^2}{12}$
    + $\sigma = \frac{b-a}{2\sqrt{3}}$




## Exponential Distribution

+ Exponential distribution
  + pdf: $\lambda > 0$

    \[ f_\lambda(x) = \begin{cases} \lambda e^{-\lambda x} & x \ge 0 \\ 0 & x < 0 \end{cases} \]

  + cdf:

    \[\begin{align*}
      \Pr(X > x) &= \begin{cases} \int_x^\infty \lambda e^{-\lambda u}\,du = \left. -e^{-\lambda u} \right|_x^\infty = e^{-\lambda x} & x \ge 0 \\ 1 & x \le 0 \end{cases} \\\\
      F(x) = \Pr(X \le x) &= \begin{cases} 1 - \Pr(X > x) = 1 - e^{-\lambda x} & x \ge 0 \\ 0 & x \le 0 \end{cases}
    \end{align*}\]

+ Expectation and variance
  + mean

    \[ E[X] = \int_0^\infty x \lambda e^{-\lambda x}\,dx  = \frac{1}{\lambda} \]

  + variance

    \[\begin{align*}
      E[X^2] &= \frac{2}{\lambda^2} \\
      Var(X) &=  \frac{1}{\lambda^2} 
    \end{align*}\]

+ Memoryless
  + Exponential distribution: $X \sim f_\lambda \quad a, b \ge 0$

    \[\begin{align*}
      \Pr(X \ge a + b \mid X \ge a) &= \Pr(X \ge b) \\
      \Pr(X < a + b \mid x \ge a) &= \Pr(X < b)
    \end{align*}\]

  + pdf

    \[f(X = a + b \mid X \ge a) = f(X = b) \]

  + Summary: exponential
    + pdf

      \[ f_\lambda(x) = \begin{cases} \lambda e^{-\lambda x} & x \ge 0 \\ 0 & x \le 0 \end{cases} \]

    + cdf

      \[ F(x) = \begin{cases} 1 - e^{-\lambda x} & x \ge 0 \\ 0 & x \le 0 \end{cases} \]

    + properties
      + mean: $E[X] = \frac{1}{\lambda}$
      + variance: $Var(X) = \frac{1}{\lambda^2}$
      + standard deviation: $\sigma = \frac{1}{\lambda}$
      + memoryless




## Gaussian Distribution

+ [Normal distribution](../Stats/ProbStatsPython/09-ContDist.md#95-gaussian-distribution)
  + notation: $X \sim N(\mu, \sigma^2)$
  + pdf: $f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
  + cdf:
    + $\Phi(x) \triangleq F(x) = \frac{1}{2\pi} \int_{-\infty}^x \exp(-\frac{x^2}{2})\, dx$
    + no known formula
    + instead use table or computer

+ [Linear transformations](../Stats/ProbStatsPython/09-ContDist.md#95-gaussian-distribution)
  + linear transformation of normal distributions are normal
    + $X \sim N(\mu, \sigma^2)$
    + pdf: $f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
    + affine: $Y = aX +b$
    + $\forall\,$ r.v.: $\mu_Y = a \mu_X + b \quad \sigma_Y = a \sigma_X$
  + variable transformation: show normal $Y \sim N\left(a\mu+b, (a\sigma)^2\right)$

+ [Standard Normal distribution](../Stats/ProbStatsPython/09-ContDist.md#95-gaussian-distribution)
  + w/o loss of generality considering $X \sim N(0, 1)$

    \[ f(x) = \frac{1}{2\pi} e^{-\frac{x^2}{2}} \]
  + mean: symmetry at $E[X] = 0$

  + variance

    \[ Var(x) = E[X^2] - (E[X])^2 = 1 - 0 = 1 \]

+ [The z table](../Stats/ProbStatsPython/09-ContDist.md#96-gaussian-distribution---probabilities)
  + distribution: $X \sim N(0, 1)$

    \[\begin{align*}
      \Pr(X \le a) &= \Phi(a) \\
      \Pr(X \ge a) &= 1 - \Phi(a) \\
      \Pr(a \le X \le b) &= \Phi(b) - \Phi(a)
    \end{align*}\]

  + negative values
    + $a > 0$
    + $\Phi(-a) = \Pr(X \le -a) = \Pr(X \ge a) = 1 - \Pr(X \le a) = 1 - \Phi(a)$

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="https://tinyurl.com/yaef2n9n" ismap target="_blank">
          <img src="../Stats/ProbStatsPython/img/t09-08.png" style="margin: 0.1em;" alt="Diagrams of various interpretations w/ negative values" title="Diagrams of various interpretations w/ negative values" width=550>
        </a>
      </div>

    + $\Pr(x \le -a) = \Phi(-a) = 1 - \Phi(a)$
    + $\Pr(X \ge -a) = 1 - \Phi(-a) = \Phi(a)$
    + $\Pr(-a \le X \le b) = \Phi(b) - \Phi(-a) = \Phi(b) - (1 - \Phi(a)) = \Phi(a) + \Phi(b) - 1$

+ [General normal distribution](../Stats/ProbStatsPython/09-ContDist.md#96-gaussian-distribution---probabilities)
  + distribution: $X \sim N(\mu, \sigma^2)$
  + $\Pr(a \le X \le b) = \Pr\left.(\frac{a - \mu}{\sigma} \le Z \le \frac{b - \mu}{\sigma}\right)$
  + standardized version of $X$ = Z score: $Z = \frac{X- \mu}{\sigma} \sim N(0, 1)$

+ [68 - 95 - 99.7 Rule](../Stats/ProbStatsPython/09-ContDist.md#96-gaussian-distribution---probabilities)

  \[ \Pr(\mu - \alpha \sigma \le X \le \mu + \alpha \sigma) = \Pr(-\alpha \le Z \le \alpha) = 2\Phi(\alpha) - 1 \]

  <table style="font-family: arial,helvetica,sans-serif; width: 30vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">$\alpha$</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">$\Pr(|X - \mu| \le \alpha \sigma)$</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">1</td> <td style="text-align: center;">2(0.8413) -1 = 0.682</td> </tr>
    <tr> <td style="text-align: center;">2</td> <td style="text-align: center;">2(0.9772) - 1 = 0.9544</td> </tr>
    <tr> <td style="text-align: center;">3</td> <td style="text-align: center;">2(0.9987) - 1 = 0.9974</td> </tr>
    </tbody>
  </table>

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yba6p7dj" ismap target="_blank">
      <img src="https://tinyurl.com/ycg35sgg" style="margin: 0.1em;" alt="Probabilities and standard deviations" title="Probabilities and standard deviations" height=200>
    </a>
  </div>

+ [Normal approximation of Binomial distribution](../Stats/ProbStatsPython/09-ContDist.md#96-gaussian-distribution---probabilities)
  + Binomial: $X \sim B_{n, p} \qquad \mu = np \quad \sigma = \sqrt{npq}$
  + Normal approximation: $Y \sim N(np, npq) \qquad \Pr(X = k) \approx \Pr(k - \frac12 \le Y \le k + \frac12)$


+ [Summary](../Stats/ProbStatsPython/09-ContDist.md#95-gaussian-distribution)
  + notation: $X \sim N(\mu, \sigma)$
  + pdf: $f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{\frac{-(x-\mu)^2}{2\sigma^2}} \quad -\infty < x < \infty$
  + cdf: $\Phi(x) \triangleq F(x) = \frac{1}{2\pi} \int_{-\infty}^x \exp(-\frac{x^2}{2})\, dx$
  + mean: $E[X] = \mu$
  + variance: $Var = \sigma^2$
  + standard deviation: $\sigma = \sigma$
  + very common in nature

