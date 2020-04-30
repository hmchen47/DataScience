# Statistics: Distributions


## Relationship of Distribution

+ [The common distributions](http://www.stat.rice.edu/~dobelman/courses/texts/Distributions.Chart.C&B.pdf)

+ [Interrelationships among discrete distribution](http://www.stat.rice.edu/~dobelman/courses/texts/Distributions.Discrete.Kendall.jpg)

+ [Univariate Distribution Relationships](http://www.stat.rice.edu/~dobelman/courses/texts/leemis.distributions.2008amstat.pdf)

+ [Tables, Cheatsheets and Reference](http://www.stat.rice.edu/~dobelman/courses/statmisc.html#tables)


## Tables of Common Continuous Distributions

<table style="font-family: arial,helvetica,sans-serif; width: 60vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
  <thead>
  <tr>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Distribution</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Probability  Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Mean</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Variance</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Moment-Generating Function</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>Binomial</td>
    <td style="text-align: center; height: 1.3em;">$p(k) = \binom{n}{k} p^k (1-p)^{n-k}$<div style="padding-top: 0.8em;">$\hspace{3.0em}k = 0, 1, \dots, n$</div></td>
    <td style="text-align: center; height: 1.3em;">$np$</td>
    <td style="text-align: center; height: 1.3em;">$np(1-p)$</td>
    <td style="text-align: center; height: 1.3em;">$\left(p\,e^t + (1-p)\right)^n$</td>
  </tr>
  <tr>
    <td>Geometric</td>
    <td style="text-align: center; height: 1.3em;">$ p(k) = p(1-p)^{1k - 1}$<div style="padding-top: 0.8em;">$\hspace{3.0em}k = 1, 2, \dots$</div></td>
    <td style="text-align: center; height: 1.3em;">$ \frac{1}{p} $</td>
    <td style="text-align: center; height: 1.3em;">$ \frac{1-p}{p^2} $</td>
    <td style="text-align: center; height: 1.3em;">$ \frac{p\,e^t}{1 - (1-p)e^t} $</td>
  </tr>
  <tr>
    <td>Hypergeometric</td>
    <td style="text-align: center; height: 1.3em;">$p(k) = \frac{\binom{r}{k}\binom{N-r}{n-k}}{\binom{N}{n}}$<div style="padding-top: 0.8em;">$y = \begin{cases} 0, 1, \dots, n & \text{if } n \leq r \\ 0, 1, \dots, r & \text{if } n > r \end{cases}$</div></td>
    <td style="text-align: center; height: 1.3em;">$\frac{nr}{N} $</td>
    <td style="text-align: center; height: 1.3em;">$n\left(\frac{r}{N} \right) \left(\frac{N - r}{N} \right) \left(\frac{N - n}{N-1} \right) $</td>
    <td style="text-align: center; height: 1.3em;"></td>
  </tr>
  <tr>
    <td>Poisson</td>
    <td style="text-align: center; height: 1.3em;">$p(k) = \frac{\lambda^k\,e^{-\lambda}}{k!}, \;k = 0, 1, \dots$</td>
    <td style="text-align: center; height: 1.3em;">$\lambda$</td>
    <td style="text-align: center; height: 1.3em;">$\lambda$</td>
    <td style="text-align: center; height: 1.3em;">$\exp\left(\lambda(e^t - 1)\right)$</td>
  </tr>
  <tr>
    <td>Negative binomial</td>
    <td style="text-align: center; height: 1.3em;">$p(k) = \binom{k-1}{r-1} p^r (1-9)^{k-r}$<div style="padding-top: 0.8em;">$\hspace{3.0em}k = r, r+1, \dots$</div></td>
    <td style="text-align: center; height: 1.3em;">$\frac{r}{p}$</td>
    <td style="text-align: center; height: 1.3em;">$\frac{r(1-p)}{p^2}$</td>
    <td style="text-align: center; height: 1.3em;">$\left( \frac{p\,e^t}{1 - (1-p)e^t} \right)$</td>
  </tr>
  </tbody>
</table>

## Tables of Common Continuous Distributions

<table style="font-family: arial,helvetica,sans-serif; width: 60vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
  <thead>
  <tr>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Distribution</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Probability  Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Mean</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Variance</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Moment-Generating Function</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>Uniform</td>
    <td style="text-align: center; height: 1.3em;">$f(x) = \frac{1}{\theta_2 - \theta_1}; \quad \theta_1 \leq x \leq \theta_2$</td>
    <td style="text-align: center; height: 1.3em;">$\frac{\theta_1 + \theta_2}{2}$</td>
    <td style="text-align: center; height: 1.3em;">$\frac{(\theta_2 - \theta_1)^2}{12}$</td>
    <td style="text-align: center; height: 1.3em;">$\frac{e^{t\theta_2} - e^{t\theta_1}}{t(\theta_2 -\theta_1)}$</td>
  </tr>
  <tr>
    <td>Normal</td>
    <td style="text-align: center; height: 1.3em;">$f(x) = \frac{1}{\sigma\sqrt{e\pi}}\exp\left( -\frac{1}{2\sigma^2} (x - \mu)^2 \right)$<div style="padding-top: 0.8em;">$\hspace{5.0em}-\infty < x < \infty$</div></td>
    <td style="text-align: center; height: 1.3em;">$\mu $</td>
    <td style="text-align: center; height: 1.3em;">$\sigma^2 $</td>
    <td style="text-align: center; height: 1.3em;">$\exp\left(\mu t + \frac{t^2\sigma^2}{2}\right)$</td>
  </tr>
  <tr>
    <td>Exponential</td>
    <td style="text-align: center; height: 1.3em;">$f(x) = \frac{1}{\beta} e^{-x/\beta}, \quad\beta > 0, 0 < x < \infty$</td>
    <td style="text-align: center; height: 1.3em;">$\beta$</td>
    <td style="text-align: center; height: 1.3em;">$\beta62$</td>
    <td style="text-align: center; height: 1.3em;">$(1 - \beta)^{-1}$</td>
  </tr>
  <tr>
    <td>Gamma</td>
    <td style="text-align: center; height: 1.3em;">$f(x) = \left(\frac{1}{\Gamma(\alpha)} \beta^\alpha \right) y^{\alpha-1} e^{-x/\beta}$<div style="padding-top: 0.8em;">$\hspace{5.0em}0 < y < \infty$</div></td>
    <td style="text-align: center; height: 1.3em;">$\alpha\beta$</td>
    <td style="text-align: center; height: 1.3em;">$\alpha\beta^2$</td>
    <td style="text-align: center; height: 1.3em;">$(1-\beta t)^{-\alpha}$</td>
  </tr>
  <tr>
    <td>Chi-square</td>
    <td style="text-align: center; height: 1.3em;">$f(x) = \frac{y^{\nu/2)-1}e^{-x/2}}{2^{\nu/2} \Gamma(\nu/2)}, \;y \neq 0$</td>
    <td style="text-align: center; height: 1.3em;">$\nu$</td>
    <td style="text-align: center; height: 1.3em;">$2\nu$</td>
    <td style="text-align: center; height: 1.3em;">$(1-2t)^{-\nu/2}$</td>
  </tr>
  <tr>
    <td>Beta</td>
    <td style="text-align: center; height: 1.3em;">$f(x) = \frac{\Gamma(\alpha) + \Gamma(\beta)}{\Gamma(\alpha) \Gamma(\beta)} y^{\alpha-1}(1-y)^{\beta-1}$<div style="padding-top: 0.8em;">$\hspace{5.0em}0 < y < 1$</div></td>
    <td style="text-align: center; height: 1.3em;">$\frac{\alpha}{\alpha+\beta}$</td>
    <td style="text-align: center; height: 1.3em;">$\frac{\alpha\beta}{(\alpha + \beta)^2(\alpha+\beta+1)}$</td>
    <td style="text-align: center; height: 1.3em;">$\not\exists$ in closed form</td>
  </tr>
  </tbody>
</table>


## Bernoulli Distribution

+ [Bernoulli distribution](/Notes/p01-Bayesian.md#361-binary-data-with-a-discrete-prior-distribution): for a single Bernoulli trial w/ outcome 0 or 1, the likelihood for each possible value for $\theta$

  \[ p(y | \theta_j) = \theta_j^y (1 - \theta_j)^{1-y}  \hspace{5.0em} \text{ where } \quad
    p(y | \theta_j) = \begin{cases} \theta_j & \text{ if } y = 1 \\ 1 - \theta_j & \text{ if } y = 0 \end{cases}
  \]


## Binomial Distribution

+ [Binomial distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + $Y$: a discrete binomial variable w/ the sampling distribution of the total number of 'successes' in $n$ independent Bernoulli trials
  + $\theta$: the probability of success in each Bernoulli trial
  + $\theta^y (1 - \theta)^{n-y}$: the likelihood, the probability for a specific sequence of $n-y$ 'failure' and $y$ 'successes', $\begin{pmatrix} n \\ y \end{pmatrix}$ sequences
  + $Y \sim Bin(n, \theta)$: a binomial distribution w/ properties

    \[\begin{align*}
      p(y | n, \theta) & = \begin{pmatrix} n \\ y \end{pmatrix} \theta^y (1-\theta)^{n-y}, \hspace{5.0em} y = 0, 1, \dots, n \tag{Bin.prob} \\
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

+ [Beta function](https://www.statlect.com/mathematical-tools/beta-function)
  + __Definition__: The __Beta function__ is a function $B: \mathbb{R}^2_+ \to \mathbb{R}$

    \[ B(x, y) = \frac{\Gamma(x) \Gamma(y)}{\Gamma(x+y)} \]

    where $\Gamma(\;)$ is the Gamma function
  + Integral btw zero and infinity

    \[ B(x, y)  = \int_0^\infty t^{x-1} (1+t)^{-x-y} dt \]

  + Integral btw zero and one

    \[ B(x, y) = \int_0^1 t^{x-1} (1-t)^{y-1} dt \]

  + Incomplete Beta function: replacing upper bound of integration ($t = 1$) w/ a variable ($t = z \leq 1$)

    \[ B(z, x, y) = \int_0^z t^{x-1} (1 - t)^{y-1} dt \]

  + the mean of Beta distribution $Beta(a, b)$: $a / (a+b)$


## Beta-Binomial Distribution

+ [The Beta-Binomial distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the Beta distribution as a conjugate distribution of the binomial distribution
  + an analytically tractable compound distribution
  + $\theta$ parameter in the binomial distribution as being randomly draw from a beta distribution
  
    \[ X \sim Bin(n, \theta) \implies p(X=k | p, n) = L(p | k) = \begin{pmatrix} n \\ k \end{pmatrix} \theta^k (1-\theta)^{n-k} \]

  + $\theta$: a random variable w/ a beta distribution

    \[ p(\theta | a, b) = Beta(a, b) = \frac{\theta^{a-1} (1-\theta)^{b-1}}{B(a, b)} \hspace{5.0em} \text{ for } 0 \leq \theta \leq 1 \]

    + $B(a, b) = \Gamma(a) \Gamma(b) / \Gamma(a+b)$

  + the compound distribution

    \[\begin{align*}
      p(k | n, a, b) 
        &= \int_0^1 \underbrace{L(\theta | k)}_{\text{binomial}} \cdot \underbrace{p(\theta | a, b)}_{\text{beta}} d\theta \\
        &= \begin{pmatrix} n \\ k \end{pmatrix} \frac{1}{B(a, b)} \int_0^1 \theta^{k+a-1} (1-\theta)^{n-k+b-1} d\theta
        = \begin{pmatrix} n \\ k \end{pmatrix} \frac{B(k+a, n-k+b)}{B(a, b)} \\\\
        &= \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)} \frac{\Gamma(k+a)\Gamma(n-k+b)}{\Gamma(n+a+b)} \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
    \end{align*}\]


## Dirichlet distribution

+ [Dirichlet distribution](../Notes/p04a-Bayesian.md#1202-probability-and-statistics)
  + a generalization of the Beta distribution
    + 2-dim Dirichlet distribution = the Beta distribution
    + let $q = (q_1, q_2)$, and $q \sim \text{Dirichlet}(\alpha_1, \alpha_2) \implies$

      \[ q_1 \sim \text{Beta}(\alpha_1, \alpha_2)\quad\text{and}\quad q_2 = 1 - q_1 \]

  + more generally, the marginals of the Dirichlet distribution are also beta distribution.

    \[ q \sim \text{Dirichlet}(\alpha_1, \dots. \alpha_J) \;\implies\; q_i \sim \text{Beta}(\alpha_j,\; \sum_{i \neq j} \alpha_i) \]

  + the density of the Dirichlet distribution in the most convenient way

    \[ p(q\,|\,\alpha) = \frac{\Gamma(\alpha_1 + \cdots + \alpha_J)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_J)} \prod_{j=1}^J q_j^{\alpha_j - 1} \hspace{5.0em} (q_j \geq 0; \quad \sum_j q_j = 1) \]

+ [The Dirichlet distribution for $K$ outcomes](../Notes/p04a-Bayesian.md#1202-probability-and-statistics)
  + the exponential family distribution on the $K-1$ dimensional probability simplex
  + the parameters of the model: $\pmb{\alpha} = (\alpha_1, \dots, \alpha_K)^T \in \mathbb{R}_+^K$, a non-negative vector of scaling coefficients
  + probability simplex defined as

    \[ \Delta_k = \left\{\pmb{\theta} = (\theta_1, \dots, \theta_K)^T \in \mathbb{R}^K \,|\, \theta_i \geq 0\; \forall i, \sum_{i=1}^K \theta_i = 1 \right\} \]

  + the probability density of Dirichlet distribution

    \[ \pi_{\pmb{\alpha}}(\pmb{\theta}) = \frac{\Gamma(\sum_{j=1}^K \alpha_j)}{\prod_{j=1}^K \Gamma(\alpha_j)} \prod_{j=1}^K \theta_j^{\alpha_j -1} \]

  + the mean of a Dirichlet distribution $\pi_\alpha (\pmb{\alpha})$

    \[ \mathbb{E}(\pmb{\theta}) = \left( \frac{\alpha_1}{\sum_{i=1}^K \alpha_i}, \dots, \frac{\alpha_K}{\sum_{i=1}^K \alpha_i} \right)^T \]


## Wishart distribution

+ [The Wishart distribution](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + a multidiemsional analogue of the Gamma distribution
  + a distribution over symmetric positive semi-definite $d \times d$ matrices $\mathbf{W}$
  + the density

    \[ \pi_{\nu_0, \mathbf{S}_0}(\mathbf{W}) \propto |\mathbf{W}|^{(\nu_0 + d + 1)/2} \exp\left( -\frac{1}{2} \text{tr}(\mathbf{S}_0^{-1} \mathbf{W}) \right) \]

    + $\nu_0$: the degrees of freedom
    + $\mathbf{S}_0$: the positive-definite matrix
+ $\mathbf{W}^{-1} \sim \text{Wishart}(\nu_0, \mathbf{S}_0) \implies \mathbf{W} \sim$ inverse Wishart distribution
+ the density of the inverse Wishart distribution

  \[ \pi_{\nu_0, \mathbf{S}_0}(\mathbf{W}) \propto |\mathbf{W}|^{-(\nu_0+d+1)/2} \exp \left( -\frac{1}{2} \text{tr}(\mathbf{S}_0 \mathbf{W}^{-1}) \right) \]


## Pareto distribution

+ [Pareto distribution](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + the standard power-law distribution
  + $\theta \sim \text{Pareto}(\nu_0,\, k)$, the survival function

    \[ \mathbb{P}(\theta \geq t) = \left( \frac{t}{\nu_0} \right)^{-k}, \quad t \geq \nu_0 \]

    + $k$: the rate of decay
    + $\nu_0$: the support of the distribution
  + the density

    \[ \pi_{k, \nu_0}(\theta) = \begin{cases}
      k \nu_0^k/\theta^{k+1} & \theta \geq \nu_0 \\
      0 & \text{otherwise}
    \end{cases}\]