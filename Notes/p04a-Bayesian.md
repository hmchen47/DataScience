# Chapter 12 Bayesian Inference


Author: H. Liu and L. Wasserman

[Origin](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)

Year: 2014

Related Course: [36-708 Statistical Methods for Machine Learning](http://www.stat.cmu.edu/~larry/=sml/)


## 12.0 Mathematical Tools

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


+ [Dirichlet distribution](https://stephens999.github.io/fiveMinuteStats/dirichlet.html)
  + a generalization of the Beta distribution
    + 2-dim Dirichlet distribution = the Beta distribution
    + let $q = (q_1,, q_2)$, and $q \sim Dirichlet(\alpha_1, \alpha_2) \implies$

      \[ q_1 \sim Beta(\alpha_1, \alpha_2)\quad\text{and}\quad q_2 = 1 - q_1 \]

  + more generally, the marginals of the Dirichlet distribution are also beta distribution.

    \[ q \sim Dirichlet(\alpha_1, \dots. \alpha_J) \;\implies\; q_i \sim Beta(\alpha_j,\; \sum_{i \neq j} \alpha_i) \]

  + the density of the Dirichlet distribution in the most convenient way

    \[ p(q\,|\,\alpha) = \frac{\Gamma(\alpha_1 + \cdots + \alpha_J)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_J)} \prod_{j=1}^J q_j^{\alpha_j - 1} \qquad (q_j \geq 0; \quad \sum_j q_j = 1) \]

    + performing standard (Lebesgue) integration of this density over the $J$-sim space $(q_q, \dots, q_J)$, the density integrates to 0, not 12 as a density should
    + cause: constraints that the $q$s must sum to 1 $\implies$ the Dirichlet distribution is effectively a $J-1$-dim distribution and not $J$-dim distribution
  + density function satisfying the constraint
    + let the $J$-sim Dirichlet distribution as a distribution on the $J-1$ numbers $(q_1, \dots, q_{J-1})$, satisfying $\sum_{j=1}^{J-1} q_j \leq 1$, and define $q_J := (1 - q_1 - q_2 - \cdots - q_{J-1})$
    + the density of the $J$-dim Dirichlet distribution

      \[ p(q_1, \dots, q_{J-1}\,|\,\alpha) = \frac{\Gamma(\alpha_1 + \cdots + \alpha_J)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_J)} \prod_{j=1}^{J-1} q_j^{\alpha_j - 1} (1 - q_1 - q_2 - \cdots - q_{j_1})^{\alpha_J} \\ \qquad\qquad\qquad\qquad (q_j \geq 0; \quad \sum_{j=1}^{J-1} q_j \leq 1) \]
  


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

    \[ \pi(\theta \,|\, X_1, \dots, X_n) \propto \mathcal{L}(\theta) \pi(\theta) \tag{2} \]

  + $\mathcal{L}(\theta)$: the likelihood function
  + finding an interval $C \to$

    \[ \int_C \pi(\theta \,|\, X_1, \dots, X_n) d\theta = 0.95 \]

  + the degree-of-belief probability statement about $\theta$ given the data

    \[ \mathbb{P}(\theta \in C \,|\, X_1, \dots, X_n) = 0.95 \]

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

      \[ \mathbb{P}(\theta \in C \,|\, X_1, \dots, X_N) = 1 - \alpha \]

+ Favors of Bayesian inference
  + subjective Bayesian: probability strictly as personal degrees of belief
  + objective Bayesian: finding prior distributions that formally express ignorance w/ the hope that the resulting posterior is , in some sense, objective
  + empirical Bayesian: estimating the prior distribution from the data
  + frequentist Bayesian: using Bayesian methods only when the resulting posterior has good frequency behavior


## 12.2 Basic Concepts

+ Notations and Assumptions
  + $X_1, \dots, X_n$: $n$ observations sampled from a probability density $p(x, \theta)$
  + $p(x\,|\,\theta)$:
    + $\theta$ as a random variable
    + representing the conditional probability density of $X$ conditioned on $\theta$
  + $p_\theta (x)$: $theta$ as a deterministic value


### 12.2.1 The Mechanics of Bayesian Inference

+ Bayesian procedure
  + choose the prior distribution
    + a probability density $\pi(\theta)$
    + expressing out beliefs about a parameter $\theta$ before observing any data
  + choose a statistical mode $p(x\,|\,\theta)$ to reflect our beliefs about $x$ given $\theta$
  + observe data $\mathcal(D) = \{X_1, \dots, X_n\}$, and then update our beliefs and calculate the posterior distribution $p(\theta\,|\,\mathcal{D})$

+ Bayesian approach
  + the posterior distribution

    \[ p(\theta \,|\, X_1, \dots, X_n) = \frac{p(X_1, \dots, X_n\,|\, \theta) \pi((\theta))}{p(X_1, \dots, X_n)}  = \frac{\mathcal{L}_n(\theta)\pi(\theta)}{c_n} \propto \mathcal{L}(\theta) \pi(\theta) \tag{3} \]

    + $\mathcal{L}(\theta) = \prod_{i=1}^n p(X_i \,|\, \theta)$: the likelihood function
    + the normalizing constant, a.k.a. the evidence

      \[ c_n = p(X_1, \dots, X_n) = \int p(X_1, \dots, X_n \,|\, \theta) \pi(\theta) d\theta = \int \mathcal{L}_n(\theta) \pi(\theta) d\theta \]

  + Bayesian point estimate
    + getting a Bayesian mean or mode by summing the center of the posterior $\to$ typically using the mean or mode of the posterior distribution
    + the posterior mean

      \[ \overline{\theta} = \int \theta p(\theta \,|\, \mathcal{D}) d\theta = \frac{\int \theta \mathcal{L}_n \pi(\theta) d\theta}{\int \mathcal{L} \pi(\theta) d\theta} \]

  + Bayesian interval estimate
    + $\exists\; \alpha \in (0, 1)$, find $a$ and $b \to$

      \[ \int_{-\infty}^a p(\theta \,|\, \mathcal{D}_n) d\theta = \int^{\infty}_b p(\theta \,|\, \mathcal{D}_n) d\theta = \alpha/2 \]

    + let $C = (a, b)$,

      \[ \mathbb{P}(\theta \in C \,|\, \mathcal{D}_n) = \int_b^a p(\theta \,|\, \mathcal{D}_n) d\theta = 1 - \alpha \]

    + $C$: a $1-\alpha$ Bayesian posterior interval or _credible interval_
    + _credible region_: $\theta$ w/ multi-dimensional

+ Prior uniform distribution
  + $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}, \;\; X_1, \dots, X_n \sim Bernoulli(\theta)$
  + prior distribution: uniform distribution as $\pi(\theta) = 1$
  + the posterior distribution

    \[\begin{align*} 
      p(\theta\,|\,\mathcal{D}_n) & \propto \pi(\theta) \mathcal{L}_n(\theta) = \theta^{S_n} (1-\theta)^{n - S_n} = \theta^{S_n+1-1} (1-\theta)^{n - S_n +1 -1} \\\\
       &= \frac{\Gamma(n+2)}{\Gamma(S_n + 1) \Gamma(n-S_n+1)} \theta^{(S_n+1)-1} (1-\theta)^{(n-S_n+1)-1} \qquad \bigg(Beta(S_n+1, n-S_n+1)\bigg) \\\\
       \theta\,|\,\mathcal{D}_n &\sim Beta(S_n+1, n-S_n +1)
    \end{align*}\]

    + $S_n = \sum_{i=1}^n X_i$: the number of success
  + the Bayesian posterior point estimator

    \[ \overline{\theta} = \frac{S_n + 1}{n+2} = \lambda_n \hat{\theta} + (1 - \lambda) \tilde{\theta} \]

    + $\hat{\theta} = S_n / n$: the maximum likelihood estimate
    + $\tilde{\theta} = 1/2$: the prior mean
    + $\lambda_n = n/(n+2) \approx 1$
  + the Bayesian posterior credible interval: 95% posterior interval = $\int_a^b p(\theta\,|\,\mathcal{D}_n) d\theta = .95$

+ Prior Beta distribution
  + the prior distribution: $\theta \sim Beta(\alpha, \beta)$
  + the posterior distribution: $\theta \,|\, \mathcal{D}_n \sim Beta(\alpha + S_n, \beta + n - S_n)$
  + the flat (uniform) prior: $\alpha = \beta = 1$
  + the posterior mean: prior mean = $\theta_0 = \alpha/(alpha+\beta)$

    \[ \overline{\theta} = \frac{\alpha + S_n}{\alpha + \beta + n} = \left(\frac{n}{\alpha+\beta+n}\right) \hat{\theta} + \left(\frac{\alpha+\beta}{\alpha+\beta+n} \theta_0 \right) \]
  + example
    + assumptions:
      + Bernoulli model: $n = 15, \theta = 0.4$
      + sample size: $s = 7$
    + maximum likelihood estimate: $\hat{\theta}(\theta) = 7/15 = 0.47$
    + left plot: prior w/ $Beta(4, 6) \to$ posterior mode = 0.43
    + right plot: prior w/ $Beta(4, 2) \to$ posterior mode = 0.67

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yx567vmm" ismap target="_blank">
        <img src="img/p04-01.png" style="margin: 0.1em;" alt="Illustration of Bayesian inference on Bernoulli data with two priors. The three curves are prior distribution (red-solid), likelihood function (blue-dashed), and the posterior distribution (black-dashed). The true parameter value theta = 0.4 is indicated by the vertical line." title="Illustration of Bayesian inference on Bernoulli data with two priors. The three curves are prior distribution (red-solid), likelihood function (blue-dashed), and the posterior distribution (black-dashed). The true parameter value theta = 0.4 is indicated by the vertical line." width=400>
      </a>
    </div>

+ Dirichlet prior distribution
  + the multinomial model w/ a Dirichlet prior
    + a generalization of the Bernoulli model and Beta prior
    + $\exists\; \mathbf{X} \sim Multinomial(n, \mathbf{\theta})$
    + $\mathbf{\theta} = (\theta_1, \dots, \theta_K)^T, \; (K > 1)$: a $K$-dim parameter
  + the Dirichlet distribution for $K$ outcomes
    + the exponential family distribution on the $K-1$ dimensional probability simplex
    + probability simplex defined as

      \[ \Delta_k = \{\mathbf{\theta} = (\theta)_1, \dots, \theta_K^T \in \mathbb{R}^K | \theta_i \geq 0\; \forall i, \sum_{i=1}^K \theta_i = 1 \} \]

    + the probability density of Dirichlet distribution

      \[ \pi_{\mathbf{\alpha}}(\mathbf{\theta}) = \frac{\Gamma(\sum_{j=1}^K \alpha_j)}{\prod_{j=1}^K \Gamma(\alpha_j)} \prod_{j=1}^K \theta_j^{\alpha_j -1} \]

      + $\mathbf{\alpha} = (\alpha_1, \dots, \alpha_K)^T \in \mathbb{R}_+^K$: a non-negative vector of scaling coefficients; the parameters of the model
  + the sample space of the multinomial w/ $K$ outcomes as the set of vertices of the $K$-dim hypercube $\mathbb{H}_K$, mad up of vectors w/ exactly only one 1 and remaining element 0

    \[ x = \underbrace{(0, 0, \dots, 00, 1, 0, \dots, 0)^T}_{\text{K places}} \]

  + $\exists\; \mathbf{X}_i = (X_{i1}, \dots, X_{iK})^T \in \mathbb{H}_K$,
  
    \[ \theta \sim Dirichlet(\mathbf{\alpha}) \text{ and } $\mathbf{X}_i \,|\, \theta \sim Multinomial(\mathbf{\theta}) \]

    $\implies$ the posterior satisfies

    \[ p(\mathbf{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) \propto \mathcal{L}(\theta)\pi(\theta) \propto \prod_{i=1}^n \prod_{j=1}^K \theta_j^{X_{ij}} \prod_{j=1}^K \theta_j^{\alpha_j - 1} = \prod_{j=1}^K \theta_j^{\sum_{i=1}^n X_{ij}+\alpha_j-1} \]

  + the posterior distribution w/ $\overline{\mathbf{X}} = \sum_{i=1}^n \mathbf{X}_i / n \in \Delta_K$

    \[ \mathbf{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n \sim Dirichlet(\alpha+ n \overline{\mathbf{X}})\]

  + the mean of a Dirichlet distribution $\pi_\alpha (\mathbf{\alpha})$

    \[ \mathbb{E}(\mathbf{\theta}) = \left( \frac{\alpha_1}{\sum_{i=1}^K \alpha_i}, \dots, \frac{\alpha_K}{\sum_{i=1}^K \alpha_i} \right)^T \]

  + the posterior mean of a multinomial w/ Dirichlet prior

    \[ \mathbb{E}(\theta \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) = \left(\frac{\alpha_1 + \sum_{i=1}^n X_{i1}}{\sum_{i=1}^K \alpha_j + n}, \dots, \frac{\alpha_K + \sum_{i=1}^n X_{iK}}{\sum_{i=1}^K \alpha_j + n} \right)^T \]

  + the posterior mean viewed as smoothing out the maximum likelihood estimated by allocating some additional probability mass to low frequency observation
  + the parameters $\alpha_1, \dots, \alpha_K$ act as "virtual counts" that don't actually appear in the observed data
  + prior conjugate w.r.t. the mode: the prior as Dirichlet distribution $\to$ the posterior as Dirichlet distribution
  + example
    + prior distribution $\sim Dirichlet(6, 6, 6)$
    + observed data size: $n = 20 \text{ and } 200$
    + parameter: $\mathbf{\theta} = (0.2, 0.3, 0.5)^T$
    + the contours of the prior, likelihood, and posteriors are plotted on a two-dimensional probability simplex (Starting from the bottom left vertex of each triangle, clock-wisely the three vertices correspond to $\theta_1, \theta_2, \theta_3$)
    + number of the observed data
      + small $\to$ the posterior affected by both the prior and the likelihood
      + large $\to$ the posterior mainly dominated by the likelihood

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yx567vmm" ismap target="_blank">
        <img src="img/p04-02a.png" style="margin: 0.1em;" alt="Illustration of Bayesian inference on multinomial data with the prior Dirichlet(6, 6, 6)" title="Illustration of Bayesian inference on multinomial data with the prior Dirichlet(6, 6, 6)" height=150>
        <img src="img/p04-02b.png" style="margin: 0.1em;" alt="Illustration of Bayesian inference on multinomial data with the prior Dirichlet(6, 6, 6)" title="Illustration of Bayesian inference on multinomial data with the prior Dirichlet(6, 6, 6)" height=150>
      </a>
    </div>

+ Conjugate prior
  + observed data: $\exists\; X \sim N(\theta, \sigma^2),\; \mathcal{D}_n = \{X_1, \dots, X_n\}$ w/ known $\sigma$
  + task: $\theta \in \mathbb{R}$
  + prior: $\theta \sim N(a, b^2)$
  + sample mean: $\overline{X} = \sum_{i=1}^n X_i/n$
  + the posterior for $\theta$

    \[\begin{align*}
      \theta \,|\, \mathcal{D}_n & \sim N(\overline{\theta}, \, \tau^2) \tag{4} \\\\
      \overline{\theta} = w\hat{\theta} + (1 - w), \quad \hat{\theta} = \overline{X}, &\quad w = \frac{\frac{1}{se^2}}{\frac{1}{se^2} + \frac{1}{b^2}}, \quad \frac{1}{\tau^2} = \frac{1}{se^2} + \frac{1}{b^2}
    \end{align*}\]

    + $se = \sigma/\sqrt{n}$: the standard error of the maximum likelihood estimate $\hat{\theta}$
    + $n \to \infty \implies w \to 1 \text{ and } \tau/se \to 1$
    + fixed $n, b \to \infty \implies$ flat prior
  + task: find posterior interval = find $C = (c, d) to \mathbb{P}(\theta \in C \,|\, \mathcal{D}_n) = 0.95$
  + $\exists\; c$ and $d \to \mathbb{P(\theta < c \,|\, \mathcal{D}_n) = 0.025$
  + find $ c \to$

    \[ \mathbb{P}(\theta < c \,|\, \mathcal{D}_n) = \mathbb{P} \left( \frac{\theta - \overline{\theta}}{\tau} < \frac{c - \overline{\theta}}{\tau}\right) = \mathbb{P}\left( Z < \frac{c - \overline{\theta}}{\tau} \right) = 0.025 \]

    where $Z \sim N(0, 1)$: a standard Gaussian random variable

    \[ \mathbb{P}(Z < -1.96) = 0.025) \to \frac{c - \overline{\theta}}{\tau} = -1.96  \implies c = \overline{\theta} - 1.96\tau \]

  + similarly, $d = \overline{\theta} = 1.96 \tau$
  + 95% Bayesian credible interval $\overline{\theta} \pm 1.96 \tau$
  + $n \gg 1 \to \overline{\theta} \approx \hat{\theta}, \; \tau \approx se \implies$ the 95% Bayesian credible interval approximated by $\hat{\theta} \pm 1.96 se$



### 12.2.2 Bayesian Prediction





### 12.2.3 Inference about Functions of Parameters





### 12.2.4 Multiparameter Problems




### 12.2.5 Flat Priors, Improper Priors, and "Noninformative" Priors




### 12.2.6 Conjugate Priors




### 12.2.7 Bayesian Hypothesis Testing




### 12.2.8 Model Comparison and Bayesian Information Criterion




### 12.2.9  Calculating the Posterior Distribution





