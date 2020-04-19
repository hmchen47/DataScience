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
    + let $q = (q_1, q_2)$, and $q \sim \text{Dirichlet}(\alpha_1, \alpha_2) \implies$

      \[ q_1 \sim \text{Beta}(\alpha_1, \alpha_2)\quad\text{and}\quad q_2 = 1 - q_1 \]

  + more generally, the marginals of the Dirichlet distribution are also beta distribution.

    \[ q \sim \text{Dirichlet}(\alpha_1, \dots. \alpha_J) \;\implies\; q_i \sim \text{Beta}(\alpha_j,\; \sum_{i \neq j} \alpha_i) \]

  + the density of the Dirichlet distribution in the most convenient way

    \[ p(q\,|\,\alpha) = \frac{\Gamma(\alpha_1 + \cdots + \alpha_J)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_J)} \prod_{j=1}^J q_j^{\alpha_j - 1} \qquad (q_j \geq 0; \quad \sum_j q_j = 1) \]

    + performing standard (Lebesgue) integration of this density over the $J$-dim space $(q_q, \dots, q_J)$, the density integrates to 0, not 12 as a density should
    + cause: constraints that the $q$s must sum to 1 $\implies$ the Dirichlet distribution is effectively a $J-1$-dim distribution and not $J$-dim distribution
  + density function satisfying the constraint
    + let the $J$-dim Dirichlet distribution as a distribution on the $J-1$ numbers $(q_1, \dots, q_{J-1})$, satisfying $\sum_{j=1}^{J-1} q_j \leq 1$, and define $q_J := (1 - q_1 - q_2 - \cdots - q_{J-1})$
    + the density of the $J$-dim Dirichlet distribution

      \[ p(q_1, \dots, q_{J-1}\,|\,\alpha) = \frac{\Gamma(\alpha_1 + \cdots + \alpha_J)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_J)} \prod_{j=1}^{J-1} q_j^{\alpha_j - 1} (1 - q_1 - q_2 - \cdots - q_{J-1})^{\alpha_J} \\ \hspace{15em} \left(q_j \geq 0; \quad \sum_{j=1}^{J-1} q_j \leq 1\right) \]

+ The Dirichlet distribution for $K$ outcomes
  + the exponential family distribution on the $K-1$ dimensional probability simplex
  + the parameters of the model: $\mathbf{\alpha} = (\alpha_1, \dots, \alpha_K)^T \in \mathbb{R}_+^K$, a non-negative vector of scaling coefficients
  + probability simplex defined as

    \[ \Delta_k = \left\{\mathbf{\theta} = (\theta_1, \dots, \theta_K)^T \in \mathbb{R}^K \,|\, \theta_i \geq 0\; \forall i, \sum_{i=1}^K \theta_i = 1 \right\} \]

  + the probability density of Dirichlet distribution

    \[ \pi_{\mathbf{\alpha}}(\mathbf{\theta}) = \frac{\Gamma(\sum_{j=1}^K \alpha_j)}{\prod_{j=1}^K \Gamma(\alpha_j)} \prod_{j=1}^K \theta_j^{\alpha_j -1} \]

  + the mean of a Dirichlet distribution $\pi_\alpha (\mathbf{\alpha})$

    \[ \mathbb{E}(\mathbf{\theta}) = \left( \frac{\alpha_1}{\sum_{i=1}^K \alpha_i}, \dots, \frac{\alpha_K}{\sum_{i=1}^K \alpha_i} \right)^T \]


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
  + sampling distribution: observing the data $X_1, \dots, X_n$, the posterior distribution for $\theta$ given the data using Bayes theorem

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
  + observe data $\mathcal{D}_n = \{X_1, \dots, X_n\} \to$ update our beliefs and calculate the posterior distribution $p(\theta\,|\,\mathcal{D}_n)$

+ General Bayesian inference
  + the posterior distribution

    \[ p(\theta \,|\, X_1, \dots, X_n) = \frac{p(X_1, \dots, X_n\,|\, \theta) \pi(\theta)}{p(X_1, \dots, X_n)}  = \frac{\mathcal{L}_n(\theta)\pi(\theta)}{c_n} \propto \mathcal{L}(\theta) \pi(\theta) \tag{3} \]

    + $\mathcal{L}(\theta) = p(X_1, \dots, X_n\,|\, \theta)$: the likelihood function
    + the normalizing constant, a.k.a. the evidence

      \[ c_n = p(X_1, \dots, X_n) = \int p\left(X_1, \dots, X_n \,|\, \theta\right) \pi(\theta) d\theta = \int \mathcal{L}_n(\theta) \pi(\theta) d\theta \]

  + Bayesian point estimate
    + getting a Bayesian mean or mode by summing the center of the posterior $\gets$ typically using the mean or mode of the posterior distribution
    + the posterior mean

      \[ \overline{\theta} = \int \theta\, p(\theta \,|\, \mathcal{D}_n)\, d\theta = \frac{\int \theta\, \mathcal{L}_n(\theta)\, \pi(\theta) d\theta}{\int \mathcal{L}_n(\theta) \pi(\theta) d\theta} \]

  + Bayesian interval estimate
    + $\exists\; \alpha \in (0, 1)$, find $a$ and $b \ni$

      \[ \int_{-\infty}^a p(\theta \,|\, \mathcal{D}_n) d\theta = \int^{\infty}_b p(\theta \,|\, \mathcal{D}_n) d\theta = \alpha/2 \]

    + let $C = (a, b)$,

      \[ \mathbb{P}(\theta \in C \,|\, \mathcal{D}_n) = \int_b^a p(\theta \,|\, \mathcal{D}_n) d\theta = 1 - \alpha \]

    + $C$: viz. a ($1-\alpha$) Bayesian posterior interval or _credible interval_
    + _credible region_: $\theta$ w/ multi-dimensional

+ Uniform-Bernoulli likelihood model
  + sampling distribution: $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}, \;\; X_1, \dots, X_n \sim Bernoulli(\theta)$
  + prior distribution: uniform distribution as $\pi(\theta) = 1$
  + $S_n = \sum_{i=1}^n X_i$: the number of success
  + the posterior distribution

    \[\begin{align*} 
      p(\theta\,|\,\mathcal{D}_n) & \propto \pi(\theta) \mathcal{L}_n(\theta) = \theta^{S_n} (1-\theta)^{n - S_n} = \theta^{S_n+1-1} (1-\theta)^{n - S_n +1 -1} \\\\
       &= \frac{\Gamma(n+2)}{\Gamma(S_n + 1) \Gamma(n-S_n+1)} \theta^{(S_n+1)-1} (1-\theta)^{(n-S_n+1)-1} \\\\
       \therefore\;\theta\,|\,\mathcal{D}_n &\sim \text{Beta}(S_n+1, n-S_n +1)
    \end{align*}\]

  + the Bayesian posterior point estimator

    \[ \overline{\theta} = \frac{S_n + 1}{n+2} = \lambda_n \hat{\theta} + (1 - \lambda_n) \tilde{\theta} \]

    + $\hat{\theta} = S_n / n$: the maximum likelihood estimate
    + $\tilde{\theta} = 1/2$: the prior mean
    + $\lambda_n = n/(n+2) \approx 1$
  + the Bayesian posterior credible interval: 95% posterior interval = $\int_a^b p(\theta\,|\,\mathcal{D}_n) d\theta = .95$

+ Beta-Bernoulli likelihood model
  + sampling distribution: $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}, \;\; X_1, \dots, X_n \sim \text{Bernoulli}(\theta)$ w/ $\hat{\theta} = S_n/n$
  + the prior distribution: $\theta \sim \text{Beta}(\alpha, \beta)$ w/ prior mean $\theta_0 = \alpha/(\alpha+\beta)$
  + the posterior distribution: $\theta \,|\, \mathcal{D}_n \sim \text{Beta}(\alpha + S_n, \beta + n - S_n)$
  + the flat (uniform) prior: $\alpha = \beta = 1$
  + the posterior mean:

    \[ \overline{\theta} = \frac{\alpha + S_n}{\alpha + \beta + n} = \left(\frac{n}{\alpha+\beta+n}\right) \hat{\theta} + \left(\frac{\alpha+\beta}{\alpha+\beta+n} \right) \theta_0 \]

  + example
    + assumptions:
      + Bernoulli model: $n = 15, \theta = 0.4$
      + number of success: $s = 7$
    + maximum likelihood estimate: $\hat{\theta}(\theta) = 7/15 = 0.47$
    + left plot: prior w/ $Beta(4, 6) \to$ posterior mode = 0.43
    + right plot: prior w/ $Beta(4, 2) \to$ posterior mode = 0.67

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yx567vmm" ismap target="_blank">
        <img src="img/p04-01.png" style="margin: 0.1em;" alt="Illustration of Bayesian inference on Bernoulli data with two priors. The three curves are prior distribution (red-solid), likelihood function (blue-dashed), and the posterior distribution (black-dashed). The true parameter value theta = 0.4 is indicated by the vertical line." title="Illustration of Bayesian inference on Bernoulli data with two priors. The three curves are prior distribution (red-solid), likelihood function (blue-dashed), and the posterior distribution (black-dashed). The true parameter value theta = 0.4 is indicated by the vertical line." width=400>
      </a>
    </div>

+ Dirichlet-Multinomial likelihood Model
  + sampling distribution: $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}, \;\; X_1, \dots, X_n \sim \text{Bernoulli}(\theta)$ w/ $\hat{\theta} = S_n/n$
  + prior distribution: Dirichlet prior
    + a multinomial distribution
    + a generalization of the Bernoulli model and Beta prior
    + $\exists\; \mathbf{X} \sim \text{Multinomial}(n, \mathbf{\theta})$
    + $\mathbf{\theta} = (\theta_1, \dots, \theta_K)^T, \; (K > 1)$: a $K$-dim parameter
  + the sample space of the multinomial w/ $K$ outcomes as the set of vertices of the $K$-dim hypercube $\mathbb{H}_K$, mad up of vectors w/ exactly only one 1 and the remaining elements 0

    \[ x = \underbrace{(0, 0, \dots, 0, 1, 0, \dots, 0)^T}_{K\text{ places}} \]

  + $\exists\; \mathbf{X}_i = (X_{i1}, \dots, X_{iK})^T \in \mathbb{H}_K$,
  
    \[ \underbrace{\theta \sim \text{Dirichlet}(\mathbf{\alpha})}_{\text{Prior}} \;\text{ and }\; \underbrace{\mathbf{X}_i \,|\, \theta \sim \text{Multinomial}(\mathbf{\theta})}_{\text{likeliehood}} \; \forall\; i=1, 2, \dots, n\]

    $\implies$ the posterior satisfies

    \[ p(\mathbf{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) \propto \mathcal{L}_n(\theta)\pi(\theta) \propto \prod_{i=1}^n \prod_{j=1}^K \theta_j^{X_{ij}} \prod_{j=1}^K \theta_j^{\alpha_j - 1} = \prod_{j=1}^K \theta_j^{\sum_{i=1}^n X_{ij}+\alpha_j-1} \]

  + the posterior distribution w/ $\overline{\mathbf{X}} = \sum_{i=1}^n \mathbf{X}_i / n \in \Delta_K$

    \[ \mathbf{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n \sim \text{Dirichlet}(\alpha+ n \overline{\mathbf{X}})\]

  + the posterior mean

    \[ \mathbb{E}(\theta \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) = \left(\frac{\alpha_1 + \sum_{i=1}^n X_{i1}}{\sum_{i=1}^K \alpha_i + n}, \dots, \frac{\alpha_K + \sum_{i=1}^n X_{iK}}{\sum_{i=1}^K \alpha_i + n} \right)^T \]

  + the posterior mean viewed as smoothing out the maximum likelihood estimated by allocating some additional probability mass to low frequency observation
  + the parameters $\alpha_1, \dots, \alpha_K$ act as "virtual counts" that don't actually appear in the observed data
  + prior conjugate w.r.t. the mode: the prior as Dirichlet distribution $\to$ the posterior as Dirichlet distribution
  + example
    + prior distribution $\sim \text{Dirichlet}(6, 6, 6)$
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
  + sampling distribution: $\exists\; X \sim N(\theta, \sigma^2),\; \mathcal{D}_n = \{X_1, \dots, X_n\}$ w/ known $\sigma$
  + point estimate: $\theta \in \mathbb{R}$
  + prior: $\theta \sim N(a, b^2)$
  + sample mean: $\overline{X} = \sum_{i=1}^n X_i/n$
  + the posterior for $\theta$

    \[\begin{align*}
      \theta \,|\, \mathcal{D}_n & \sim N(\overline{\theta}, \, \tau^2) \tag{4} \\\\
      \overline{\theta} = w\hat{\theta} + (1 - w)a, \quad \hat{\theta} = \overline{X}, &\quad w = \frac{\frac{1}{se^2}}{\frac{1}{se^2} + \frac{1}{b^2}}, \quad \frac{1}{\tau^2} = \frac{1}{se^2} + \frac{1}{b^2}
    \end{align*}\]

    + $se = \sigma/\sqrt{n}$: the standard error of the maximum likelihood estimate $\hat{\theta}$
    + $n \to \infty \implies w \to 1 \text{ and } \tau/se \to 1$
    + fixed $n, b \to \infty \implies$ flat prior
  + region estimate: find posterior interval = find $C = (c, d) \to \mathbb{P}(\theta \in C \,|\, \mathcal{D}_n) = 0.95$
  + $\exists\; c$ and $d \ni \mathbb{P}(\theta < c \,|\, \mathcal{D}_n) = 0.025$
  + find $ c \ni$

    \[ \mathbb{P}(\theta < c \,|\, \mathcal{D}_n) = \mathbb{P} \left( \frac{\theta - \overline{\theta}}{\tau} < \frac{c - \overline{\theta}}{\tau}\right) = \mathbb{P}\left( Z < \frac{c - \overline{\theta}}{\tau} \right) = 0.025 \]

    where $Z \sim N(0, 1)$: a standard Gaussian random variable

    \[ \mathbb{P}(Z < -1.96) = 0.025 \to \frac{c - \overline{\theta}}{\tau} = -1.96  \implies c = \overline{\theta} - 1.96\tau \]

  + similarly, $d = \overline{\theta} = 1.96 \tau$
  + 95% Bayesian credible interval $\overline{\theta} \pm 1.96 \tau$
  + $n \gg 1 \to \overline{\theta} \approx \hat{\theta}, \; \tau \approx se \implies$ the 95% Bayesian credible interval approximated by $\hat{\theta} \pm 1.96 se$



### 12.2.2 Bayesian Prediction

+ Bayesian predictive distribution
  + $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}$
  + task: to predict the distribution of a future data point $X$ conditioned on $\mathcal{D}_n$
  + the predictive distribution

    \[\begin{align*}
      p(x \,|\, \mathcal{D}_n) &= \int p(x, \theta \,|\, \mathcal{D}_n) \,d\theta  = \int p(x \,|\, \theta, \mathcal{D}_n)\, p(\theta \,|\, \mathcal{D}_n) \,d\theta  \\\\
       &= \int p(x \,|\, \theta) \, p(\theta \,|\, \mathcal{D}_N) \,d\theta \hspace{3em} (\text{cond. indep.})
    \end{align*}\]

  + the predictive distribution w/ conditionally independent viewed as a weighted average of the model $p(x \,|\, \theta)$
  + the weights determined by the posterior distribution of $\theta$


### 12.2.3 Inference about Functions of Parameters

+ Bayesian inference about a function
  + $\exists\; \tau = g(\theta)$ a function
  + the posterior CDF for $\tau$ w/ a given $A = \{\theta: g(\theta) \leq t\}$

    \[ H(t \,|\, \mathcal{D}_n) = \mathbb{P}(g(\theta) \leq t \,|\, \mathcal{D}_n) = \int_A p(\theta \,|\, \mathcal{D}_n) \,d\theta \]

  + the posterior density $p(\tau \,|\, \mathcal{D}_n) = H'(\tau \,|\, \mathcal{D}_n)$

+ Uniform-Bernoulli likelihood model w/ logarithm of odds ratio
  + sampling distribution: $X \sim \text{Bernoulli}(\theta)$
  + prior distribution: $\pi(\theta) = 1$
  + posterior distribution: $\theta \,|\, \mathcal{D}_n \sim \text{Beta}(S_n+1, \,n - S_n + 1)\;\;$ w/ $S_n = \sum_{i=1}^n X_i$
  + let $\psi = \log(\theta/(1 - \theta))$, the posterior CDF for $\psi$

    \[\begin{align*}
      H(t \,|\, \mathcal{D}_n) &= \mathbb{P}(\psi \leq t \,|\, \mathcal{D}_n) = \mathbb{P}\left( \log(\frac{\theta}{1 - \theta}) \leq t \,|\, \mathcal{D}_n \right)  \\\\
      &= \int_0^{e^t/(1+e^t)} p(\theta \,|\, \mathcal{D}_n) \,d\theta =\frac{\Gamma(n+2)}{\Gamma(S_n+1)\Gamma(n-S_n+1)} \int_0^{e^t/(1-e^t)} \theta^{S_n} (1-\theta)^{n - S_n} \,d\theta
    \end{align*}\]
  
  + the poster density w/ $\psi \in \mathbb{R}$

    \[ p(\psi \,|\, \mathcal{D}_n) = H'(\psi \,|\, \mathcal{D}_n) = \frac{\Gamma(n+2)}{\Gamma(S_n+1)\Gamma(n-S_n+1)} \left(\frac{w^\psi}{1+e^\psi}\right)^{S_n+1} \left(\frac{1}{1+e^\psi}\right)^{n-S_n+1} \]


### 12.2.4 Multiparameter Problems

+ Extracting inferences about one single parameters
  + sampling distribution: $\mathcal{D}_n = \{X_1, \dots, X_n \}$
  + prior distribution: $\pi(\mathbf{\theta}),\; \mathbf{\theta} = (\theta_1, \dots, \theta_d)^T$
  + the marginal posterior for $\theta_1$

    \[ p(\mathbf{\theta} \,|\, \mathcal{D}_n) = \int \cdots \int p(\theta_1, \dots, \theta_d \,|\, \mathcal{D}_n) \, d\theta_2 \cdots d\theta_d \]

  + probably not feasible to do the integral
  + Solution: simulation by drawing randomly from the posterior

    \[ \mathbf{\theta}^1, \dots, \mathbf{\theta}^B \sim p(\mathbf{\theta} \,|\, \mathcal{D}_n) \]

    + the superscript index: different draw
    + $\mathbf{\theta}^j = (\theta_1^j, \dots, \theta_d^j)^T$
  + the first component of each draw: $\theta_1^1, \dots, \theta_1^B$
  + a sample from $p(\theta_1 \,|\, \mathcal{D}_n) \to$ avoided doing any integrals
  + sampling $B$ data from a multivariate distribution $p(\mathbf{\theta} \,|\, \mathcal{D}_n) \to$ challenging especially w/ large dimensionality $d$

+ Comparing two binomials
  + control patients = $n_1$, treatment patients = $n_2$
  + sampling distribution
    + $X_1$: the number of survived patients in the control group
    + $X_2$: the number of survived patients in the treatment group
  + Binomial model

    \[ X_1 \sim \text{Binomial}(n_1, \theta_1) \quad\text{ and }\quad X_2 \sim \text{Binomial}(n_2, \theta_2) \]

  + task: estimate $\tau = g(\theta_1, \theta_2) = \theta_2 - \theta_1$
  + $\pi(\theta_1, \theta_2) = 1 \implies$ the posterior

    \[ p(\theta_1, \theta_2 \,|\, X_1, X_2) \propto \theta_1^{X_1} (1-\theta_1)^{n_1 - X_1} \theta_2^{X_2} (1 - \theta_2)^{n_2 - X_2} \]

  + $(\theta_1, \theta_2)$ as rectangle $\implies$ $\theta_1$ and $\theta_2$ independent under the posterior

    \[\begin{align*}
      p(\theta_1, \theta_2 \,|\, X_1, X_2) &= p(\theta_1 \,|\, X_1) p(\theta_2 \,|\, X_2) \hspace{5em} \\\\
      p(\theta_1 \,|\, X_1) \propto \theta_1^{X_1} (1 - \theta_1)^{n_1 - X_1} \quad&\&\quad p(\theta_2 \,|\, X_2) \propto \theta_2^{X_2} (1 - \theta_2)^{n_2 - X_2} \\\\
      \theta_1 \,|\, X_1 \sim \text{Beta}(X_1+1, n_1 - X_1 +1) \quad&\&\quad \theta_2 \,|\, X_2 \sim \text{Beta}(X_2 + 1, n_2 - X_2 + 1)
    \end{align*}\]

  + $\theta_1^1, \dots, \theta_1^B \sim \text{Beta}(X_1 + 1,\, n_1 - X_1 + 1)$ and $\theta_2^1, \dots, \theta_2^B \sim \text{Beta}(X_2 + 1,\, n_2 - X_2 + 1)$ $\implies \tau_b = \theta_2^b - \theta_1^b, \forall\; b=1,\dots,B \to p(\tau \,|\, X_1, X_2)$, a sample form
  


### 12.2.5 Flat Priors, Improper Priors, and "Noninformative" Priors

+ Subjective debate
  + subjectivism: the prior should reflect our subjective opinion about $\theta$ (before data collected)
  + impractical in complicated problems, in particular, many parameters
  + injecting subjective opinion into the analysis contrary to the goal of making scientific inference as objective as possible

+ Noninformative priors
  + candidate noninformative prior: a flat prior \(\pi(\theta) \propto \text{ constant}\)
  + example: Uniform-Bernoulli likelihood model: $\pi(\theta) =1 \to Beta(S_n+1, n-S_n+1)$

+ Improper priors
  + sampling distribution: $X \sim N(\theta, \sigma^2)$ w/ known $\sigma$
  + observed data: $\mathcal{D}_n = \{X_1, \dots, X_n\}$
  + prior distribution: flat prior $\pi(\theta) \propto c, \; c > 0$
    + $\int \pi(\theta) \;d\theta = \infty \to$ not a valid probability density
    + _improper prior_
  + still able to carrying out Bayes' theorem
  + the posterior density

    \[ p(\theta \,|\, \mathcal{D}_n) \propto \mathcal{L}_n(\theta) \pi(\theta) \propto \mathcal{L}_n(\theta) \]

  + $\theta \,|\, \mathcal{D}_n \sim N(\overline{X}, \,\sigma^2/n)$ w/ $\overline{X} = \sum_{i=1}^n X_i/n$
  + the resulting point and interval estimators agree exactly w/ their their frequentist counterparts
  + Summary: improper priors not a problem as long as the resulting posterior as a well-defined probability distribution

+ Flat priors not invariant
  + sampling distribution: $X \sim \text{Bernoulli}(\theta)$
  + flat prior $\pi(\theta) = 1 \implies$ lack of information about $\theta$ before the experiment
  + $\psi = \log(\theta/1-\theta)$: a transformation of $\theta$
  + the resulting distribution for $\psi$, not flat

    \[ p(\psi) = \frac{e^{\psi}}{(1+e^\psi)^2} \]
  
  + ignorant about $\theta$ and $\psi \implies$ a flat prior for $\psi$
  + contradiction
    + the notation of a flat prior not well define
    + a flat prior on a parameter $\nRightarrow$ a flat prior on a transformed version of this parameter
  + flat priors not transformed _invariant_

+ Jefferys' prior
  + transformation invariant priors
  + Harold Jefferys' rule
    + taking the prior distribution on parameter space proportional to the square root of the determinant of the Fisher infromation
    + the Fisher infromation

      \[ \pi(\theta) \propto \sqrt{|I(\theta)|}, \quad I(\theta) = -\mathbb{E} \left[ \left.\frac{\partial^2 \log p(X \,|\, \theta)}{\partial\theta\, \partial \theta^T} \,\right|\, \theta \right] \]

  + __Theorem:__ The Jefferys' prior is transformed invariant.
  + _Proof._
    + likelihood function: $p(x \,|\, \theta)$
    + $\psi$: transformation of $\theta$
    + shown that $\pi(\psi) \propto \sqrt{|I(\psi)|}$
    + using the change of variable theorem and the fact that the product of determinants is the determinant of matrix product. $\tag*{$\Box$}$

  + Bernoulli ($\theta$) model
    + the Fisher information

      \[ I(\theta) = \frac{1}{\theta(1 - \theta)} \]

    + Jefferys' rule using the prior

      \[ \pi(\theta) \propto \sqrt{|I(\theta)|} = \theta^{-1/2} (1-\theta)^{-1/2} \implies \pi(\theta) \sim \text{Beta}(1/2, 1/2)\]

    + $\pi(\theta)$ closed to a uniform density
  + Jeffers' prior: transformation invariant $\neq$  non-informative


### 12.2.6 Conjugate Priors

+ Definition of conjugate priors
  + a prior distribution closed under sampling distribution
  + $\mathcal{P}$: a family of prior distribution
  + __Definition__. $\forall\; \theta, \, \exists\; p(\cdot \,|\, \theta) \in \mathcal{F}$ over a sample space $\mathcal{X}$. The posterior

    \[ p(\theta \,|\, \mathbf{x}) = \frac{p(\mathbf{x} \,|\, \theta)\; \pi(\theta)}{\int p(\mathbf{x} \,|\, \theta)\; \pi(\theta) \;d\theta} \]

    satisfies $p(\cdot \,|\,\mathbf{x}) \in \mathcal{P} \implies$ the family $\mathcal{P}$ is _conjugate_ to the family of sampling distribution $\mathcal{F}$
  + the family $\mathcal{P}$ should be sufficiently restricted, and is typically taken to be a specific parametric family.

+ General exponential family models
  + $p(\cdot \,|\, \theta)$: a standard exponential family model
  + the density w.r.t. a positive measure $\mu$

    \[ p(\mathbf{x} \,|\, \mathbf{\theta}) = \exp\left(\mathbf{\theta}^T\,\mathbf{x} - A(\mathbf{\theta})\right) \tag{5} \]

    + $\mathbf{\theta} \in \mathbb{R}^d$: a $d$-dimensional parameter
    + $\Theta \subset \mathbb{R}^d$: an open parameter space $\ni$

      \[ \int \exp\left(\mathbf{\theta}^T\,\mathbf{x} - A(\mathbf{\theta})\right)\,d\mu(\mathbf{x}) < \infty \]

    + $A(\mathbf{\theta})$: the moment generation or log-normalizing constant

      \[A(\mathbf{\theta}) = \log\left(\int \exp(\mathbf{\theta} \mathbf{x} - A(\mathbf{\theta}))\, d\mu(\mathbf{x}) \right)\]

  + the density of a conjugate prior for the exponential family

    \[\pi_{\mathbf{x}_0,n_0}(\theta) = \frac{\exp(n_0 \mathbf{x}_0^T \mathbf{\theta} - n_0A(\mathbf{\theta}))}{\int \exp(n_0 \mathbf{x}_0^T \mathbf{\theta} - n_0A(\mathbf{\theta}))\,d\mathbf{\theta}} \]

    + $\mathbf{x}_0 \in \mathbb{R}^d$: a vector
    + $n_0 \in \mathbb{R}$: a scalar
  + the posterior

    \[\begin{align*}
      p(\mathbf{x} \,|\, \mathbf{\theta}) \pi_{\mathbf{x}_0,n_0}(\mathbf{\theta}) &= \exp(\mathbf{\theta}^T \mathbf{x} - A(\mathbf{\theta}))  \exp\left(n_0\mathbf{x}_0^T\mathbf{\theta} - n_0 \,A(\mathbf{\theta})\right) \\\\
      &= \exp\left((\mathbf{x} + \mathbf{x}_0)^T\mathbf{\theta} - (1+n_0)A(\mathbf{\theta})\right) \\\\
      &= \exp\left( (1+n_0) \left( \frac{\mathbf{x}}{1+n_0} + \frac{n_0\mathbf{x}_0}{1+n_0} \right)^T \mathbf{\theta} - (1+n_0)A(\mathbf{\theta}) \right) \\\\
      &\propto \pi_{\frac{\mathbf{x}}{1+n_0}+\frac{n_0\mathbf{x}_0}{1+n_0}, 1+n_0}(\mathbf{\theta})
    \end{align*}\]
  
  + the prior incorporating $n_0$ "virtual" observations of $\mathbf{x}_0 \in \mathbb{R}^d$
  + after making one "real" observation x: the parameters of the posterior as a mixture of the virtual and actual observation

    \[ n_0^\prime = 1 + n_0 \quad \text{ and } \quad \mathbf{x}_0^\prime = \frac{\mathbf{x}}{1 + n_0} + \frac{n_0 \mathbf{x}}{1 + n_0} \]

+ Generalized exponential family model
  + $n$ observations $\mathbf{X}_1, \dots, \mathbf{X}_n \implies$ the posterior

    \[\begin{align*}
      p(\mathbf{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) &= \pi_{\frac{\mathbf{n \mathbf{\hat{x}}}}{n+n_0} + \frac{n_0\mathbf{x}_0}{n+n_0}, n+n_0}(\mathbf{\theta}) \\\\
      &\propto \exp \left( (n + n_0) \left( \frac{n\overline{\mathbf{X}}}{n + n_0} + \frac{n_0\mathbf{x}_0}{n+n_0} \right)^T \mathbf{\theta} - (n + n_0)A(\mathbf{\theta}) \right) \\ 
      &\hspace{15em} \text{where }\left(\overline{\mathbf{X}} = \sum_{i=1}^n \mathbf{X}_i/n \right)
    \end{align*}\]

  + the parameters of the posterior

    \[ n_0^\prime = n + n_0 \quad \text{ and } \quad \mathbf{x}_0^\prime = \frac{n \overline{\mathbf{X}}}{n+n_0} + \frac{n_0\mathbf{x}_0}{n+n_0} \]

  + define $\pi_{\mathbf{x}_0, n_0}$ as

    \[\begin{align*}
      \pi_{\mathbf{x}_0, n_0} &= \exp\left(n_0 \mathbf{x}_0^T \mathbf{\theta}  -n_0 A(\mathbf{\theta})\right) \\\\
      \nabla \pi_{\mathbf{x}_0, n_0}(\mathbf{\theta}) &= n_0(\mathbf{x}_0 - \nabla A(\mathbf{\theta})) \pi_{\mathbf{x}_0, n_0}(\mathbf{\theta})
    \end{align*}\]

  + the expectation w.r.t. $\pi_{\mathbf{x}_0, n_0}$

    \[\begin{align*}
      \mathbb{E}[\nabla A(\mathbf{\theta})] = \int \nabla A(\mathbf{\theta}) \pi_{\mathbf{x}_0, n_0}(\mathbf{\theta})\,d\mathbf{\theta} &= \mathbf{x}_0 - \frac{1}{n_0} \int \nabla \pi_{\mathbf{x}_0, n_0} (\mathbf{\theta})\,d\mathbf{\theta} = \mathbf{x}_0 \\\\
      \text{with } \int\nabla \pi_{\mathbf{x}_0, n_0}(\mathbf{\theta})\,d\mathbf{\theta} &= \nabla \left( \int \pi_{\mathbf{x}_0, n_0}(\mathbf{\theta})\,d\mathbf{\theta} \right) = 0
    \end{align*}\]

  + more generally,

    \[ \mathbb{E}[\nabla A(\mathbf{\theta}) \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n] = \frac{n\overline{\mathbf{X}}}{n_0+n} + \frac{n_0 \mathbf{x}_0}{n_0+n} \]

  + under appropriate regularity conditions, the converse also holds, so that linearity of $\nabla A(\mathbf{\theta}) \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n$ is sufficient for conjugacy

+ __Theorem__
  + open space $\Theta \subset \mathbb{R}^d$
  + $\mathbf{X}$: a sample of size one from the exponential family $p(\cdot \,|\, \mathbf{\theta})$
  + the support of $\mu$ containing an open interval
  + $\pi(\mathbf{\theta})$: a prior density not concentrated at a single point
  + the posterior mean of $\nabla A(\mathbf{\theta})$ given a single observation $\mathbf{X}$ is linear

    \[ \mathbb{E}[\nabla A(\theta) \,|\, X] = a\mathbf{X} + \mathbf{b} \quad\iff\quad \pi(\mathbf{\theta}) \propto \exp\left( \frac{1}{a} \mathbf{b}^T \mathbf{\theta} - \frac{1 -a}{a} A(\mathbf{\theta}) \right)  \]

  + similar result holds w/ discrete measure $\mu$

+ Gamma-Poisson likelihood model
  + Poisson model w/ rate $\lambda \geq 0$ in the sample space $\mathcal{X} = \mathbb{Z}_+ \ni$

    \[ \mathbb{P}(X = x \,|\, \lambda) = \frac{\lambda^x}{x!} e^{-\lambda} \propto \exp(x\log\lambda - \lambda) \]

  + the natural parameter: $\theta = \log\lambda$
  + the conjugate prior

    \[ \pi_{x_0, n_0}(\lambda) \propto \exp(n_0x_0\log \lambda - n_0\lambda) \]
  
  + a better parameterization of the prior as the $\text{Gamma}(\alpha, \beta)$

    \[ \pi_{\alpha, \beta}(\lambda) \propto \lambda^{\alpha-1} (1-\lambda)^{-\beta\lambda} \]

  + sampling distribution: $\exists\; X_1, \dots, X_n$ observations from $\text{Poisson}(\lambda)$
  + the posterior

    \[ \lambda \,|\, X_1, \dots, X_n \sim \text{Gamma}(\alpha + n\overline{\mathbf{X}},\, \beta+n) \]

  + the prior acts as if $\beta$ virtual observations were made, with a total count of $\alpha -1$ among them

+ Gamma-Exponential likelihood model
  + exponential distribution w/ the sample space $\mathcal{X} \in \mathbb{R}_+ \ni$

    \[ p(x \,|\, \theta) = \theta e^{-x\theta} \]
  
  + exponential model widely used for survival times or waiting times btw events
  + the conjugate prior: Gamma distribution in the most convenient parameterization

    \[ \pi_{\alpha, \beta} \propto \theta^{\alpha - 1} e^{-\beta\theta} \]

  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $\text{Exponential}(\theta)$
  + the posterior

    \[ \theta \,|\, X_1, \dots, X_n \sim \text{Gamma}(\alpha + n,\, \beta + n\overline{X}) \]

  + the prior acts if $\alpha -1$ virtual example are used, w/ a total waiting time of $\beta$

+ Gamma-Geometric likelihood model
  + the geometric distribution
    + the discrete analogue of the exponential model
    + sample space $\mathcal{X} = \mathbb{Z}_{++}$, the strictly positive integers
    + the density
  
    \[ \mathbb{P}(X = x \,|\, \theta) = (1-\theta)^{x-1} \theta \]

  + the conjugate prior: $\text{Gamma}(\alpha, \beta)$
  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $\text{Geometric}(\theta)$
  + the posterior

    \[ \theta \,|\, X_1, \dots, X_n \sim \text{Gamma}(\alpha + n, \,\beta + n\overline{X}) \]

+ InvGamma-Gaussian likelihood model
  + sampling distribution: $N(\mu, \sigma^2)$
  + the likelihood function

    \[\begin{align*}
      p(X_1, \dots, X_n \,|\, \sigma^2) &\propto (\sigma^2)^{-n/2} \exp\left( -\frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu^2) \right) \\\\
        &= (\sigma^2)^{-n/2} \exp\left( -\frac{1}{2\sigma^2} n\, \overline{(X - \mu)^2} \right) \\
        & \hspace{10em} \text{with }\left(\overline{(X-\mu)^2} = \frac{1}{n} \sum_{i=1}^n (X_i - \mu)^2 \right)
    \end{align*}\]

  + the conjugate prior
    + inverse Gamma distribution: $1/\theta \sim \text{Gamma}(\alpha, \beta)$
    + the density

      \[ \pi_{\alpha, \beta}(\theta) \propto \theta^{-(\alpha+1)} e^{-\beta/\theta} \]

  + the posterior distribution of $\sigma^2$

    \[ \sigma^2 \,|\, X_1, \dots, X_n \propto \text{InvGamma}\left(\alpha + \frac{n}{2},\, \beta + \frac{n}{2}\, \overline{X - \mu)^2}\,\right) \]

+ ScaledInv-$\chi^2$-Gaussian likelihood model
  + the prior: scaled inverse $\chi^2$ distribution of $\sigma^2\nu_0Z\;$ w/ $Z \sim \chi_{\nu_0}^2$

    \[ \pi_{\nu_0, \sigma_0^2}(\theta) \propto \theta^{-(1+\nu_0/2)} \exp\left( -\frac{\nu_0 \sigma^2_0}{2\theta} \right) \]

  + the posterior

    \[ \sigma^2 \,|\, X_1, \dots, X_n \sim \text{ScaledInv-}\chi^2 \left( \nu_0 +n, \,\frac{\nu_0 \sigma_0^2}{\nu_0 + n} + \frac{n\, \overline{(X - \mu)^2}}{\nu_0 + n} \right) \]

+ InvWhishart-Gaussian likelihood Model
  + The Wishart distribution
    + a multidiemsional analogue of the Gamma distribution
    + a distribution over symmetric positive semi-definite $d \times d$ matrices $\mathbf{W}$
    + the density

      \[ \pi_{\nu_0, \mathbf{S}_0}(\mathbf{W}) \propto |\mathbf{W}|^{(\nu_0 + d + 1)/2} \exp\left( -\frac{1}{2} \text{tr}(\mathbf{S}_0^{-1} \mathbf{W}) \right) \]

      + $\nu_0$: the degrees of freedom
      + $\mathbf{S}_0$: the positive-definite matrix
  + $\mathbf{W}^{-1} \sim \text{Wishart}(\nu_0, \mathbf{S}_0) \implies \mathbf{W} \sim$ inverse Wishart distribution
  + the density of the inverse Wishart distribution

    \[ \pi_{\nu_0, \mathbf{S}_0}(\mathbf{W}) \propto |\mathbf{W}|^{-(\nu_0+d+1)/2} \exp \left( -\frac{1}{2} \text{tr}(\mathbf{S}_0 \mathbf{W}^{-1}) \right) \]

  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $N(\mathbf{0}, \mathbf{\Sigma}), \;\mathbf{\Sigma} \in \mathbb{R}^{n\times n}$ as covariance (positive semi-defined matrix)
  + the posterior: an inverse Wishart prior multiplies the likelihood

    \[\begin{align*}
      &p(\mathbf{X}_1, \dots, \mathbf{X}_n \,|\, \mathbf{\Sigma})\pi_{\nu_0, \mathbf{S}_0} \\\\
      & \hspace{3em}\propto |\mathbf{\Sigma}|^{-n/2} \exp \left( -\frac{n}{2} \text{tr}(\overline{\mathbf{S}}\mathbf{\Sigma}^{-1}) \right) |\mathbf{\Sigma}|^{-(\nu_0+d+1)/2} \exp \left( -\frac{1}{2}\text{tr}(\mathbf{S}_0 \mathbf{\Sigma}^{-1}) \right) \\\\
      &\hspace{1em}= |\mathbf{\Sigma}|^{-(n+\nu_0+d+1)/2} \exp \left( -\frac{1}{2} \text{tr}\left(\left(n \overline{\mathbf{S}} + \mathbf{S}_0\right) \mathbf{\Sigma}^{-1}\right) \right)
    \end{align*}\]

    + the empirical covariance: $\overline{\mathbf{S}} = \frac{1}{n} \sum_{i=1}^n \mathbf{X}_i\mathbf{X}_i^T$
  + the posterior

    \[ \mathbf{\Sigma} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n \sim \text{InvWishart}(\nu_0 + n,\, \mathbf{S}_0 + n\overline{\mathbf{S}}) \]

  + similarly, the conjugate prior for the inverse covariance $\mathbf{\Sigma}^{-1}$ (precision matrix) is a Wishart

+ Pareto-Uniform likelihood model
  + uniform distribution: $\text{Uniform}(0, \theta),\, \theta \geq 0$
  + Pareto distribution
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
  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $\text{Uniform}(0, \theta)$
  + the prior of $\theta$: $\text{Pareto}(k, \nu_0)$
  + let $X_{(n)} = \max_{1 \leq i \leq n} \{ X_i \}$
    + $\nu_0 > X_{(n)} \implies$

      \[ \mathcal{L}_n(\theta) \pi_{k, \nu_0}(\theta) = 0 \]

    + $\nu_o \leq X_{(n)} \implies$ the posterior ($\theta$ must be at least $X_{(n)}$)

      \[ \mathcal{L}_n(\theta) \pi_{k, \nu_0}(\theta) \propto \frac{1}{\theta^n} \frac{1}{\theta^{k+1}} \]
  + the posterior

    \[ \theta \,|\, X_1, \dots, X_n \sim \text{Pareto}\left(n + k, \max\{X_{(n)}, \,\nu_0\}\right) \]

  + $n \nearrow \;\to$ the decay of the posterior $\nearrow \implies$ a more peaked distribution around $X_{(n)}$
  + the parameter $K$ controls the sharpness of the decay for small $n$

+ Conjugate priors for discrete exponential family distributions

  <table style="font-family: arial,helvetica,sans-serif; width: 60vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.2em; margin: 0.2em;"><a href="http://www.stat.cmu.edu/~larry/=sml/">Conjugate priors for discrete exponential family distributions</a></caption>
    <thead>
    <tr>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Sample Space</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Sampling Dist.</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Conjugate Prior</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Posterior</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td style="text-align: center;">$X = \{0, 1\}$</td>
      <td style="text-align: center;">$\text{Bernoulli}(\theta)$</td>
      <td style="text-align: center;">$\text{Beta}(\alpha, \,\beta)$</td>
      <td style="text-align: center;">$\text{Beta}\left(\alpha + n\overline{X}, \,\beta + n\left(1-\overline{X}\right)\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$X = \mathbb{Z}_+$</td>
      <td style="text-align: center;">$\text{Poisson}(\lambda)$</td>
      <td style="text-align: center;">$\text{Gamma}(\alpha, \,\beta)$</td>
      <td style="text-align: center;">$\text{Gamma}\left(\alpha + n\overline{X}, \,\beta + n\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$X = \mathbb{Z}_{++}$</td>
      <td style="text-align: center;">$\text{Geometric}(\theta)$</td>
      <td style="text-align: center;">$\text{Gamma}\left(\alpha, \,\beta\right)$</td>
      <td style="text-align: center;">$\text{Gamma}\left(\alpha+n, \,\beta+n\overline{X}\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$X = \mathbb{H}_k$</td>
      <td style="text-align: center;">$\text{Multinomial}(\theta)$</td>
      <td style="text-align: center;">$\text{Dirichlet}\left(\alpha+n\overline{X}\right)$</td>
      <td style="text-align: center;">$\text{Dirichlet}\left(\alpha\right)$</td>
    </tr>
    </tbody>
  </table>

+ Conjugate priors for some continuous distributions

  <table style="font-family: arial,helvetica,sans-serif; width: 60vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.2em; margin: 0.2em;"><a href="http://www.stat.cmu.edu/~larry/=sml/">Conjugate priors for some continuous distributions</a></caption>
    <thead>
    <tr>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Sampling Dist.</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Conjugate Prior</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:25%;">Posterior</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td style="text-align: center;">$\text{Uniform}(\theta)$</td>
      <td style="text-align: center;">$\text{Pareto}\left(\nu_0, \,k\right)$</td>
      <td style="text-align: center;">$\text{Pareto}\left(\max\{\nu_0, \,X_{(n)}\}, \,n+k\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\text{Exponential}(\theta)$</td>
      <td style="text-align: center;">$\text{Gamma}\left(\alpha, \,\beta\right)$</td>
      <td style="text-align: center;">$\text{Gamma}\left(\alpha+n, \,\beta+n\overline{X}\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$N(\mu, \,\sigma^2)$, known $\sigma^2$</td>
      <td style="text-align: center;">$N(\mu_0, \,\sigma_0^2)$</td>
      <td style="text-align: center;">$N\left(\left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}\left(\frac{\mu_0}{\sigma_0^2} + \frac{n\overline{X}}{\sigma^2}\right), \,\left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$N(\mu, \,\sigma^2)$, known $\mu$</td>
      <td style="text-align: center;">$\text{InvGamma}(\alpha, \,\beta)$</td>
      <td style="text-align: center;">$\text{InvGamma}\left(\alpha+\frac{n}{2}, \,\beta + \frac{n}{2} \, \overline{(X - \mu)^2}\right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$N(\mu, \,\sigma^2)$, known $\mu$</td>
      <td style="text-align: center;">$\text{ScaledInv-}\chi^2(\nu_0, \,\sigma_0^2)$</td>
      <td style="text-align: center;">$\text{ScaledInv-}\chi^2\left(\nu_0+n, \,\beta + \frac{\nu_0+\sigma_0^2}{\nu_0 + n} + \frac{n\,\overline{(X-\mu)^{2}}}{\nu_0 + n} \right)$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$N(\mathbf{\mu}, \,\mathbf{\Sigma})$, known $\mathbf{\Sigma}$</td>
      <td style="text-align: center;">$N(\mathbf{\mu}_0, \,\mathbf{\Sigma}_0)$</td>
      <td style="text-align: center;">$N\left(\mathbf{K}\left(\Sigma_0^{-1} \mu_0 + n \Sigma^{-1} \overline{X}\right), \,\mathbf{K}\right), \\ \hspace{10em}\;\mathbf{K} = (\Sigma_0^{-1} + n\Sigma^{-1})^{-1}$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$N(\mathbf{\mu}, \,\mathbf{\Sigma})$, known $\mathbf{\mu}$</td>
      <td style="text-align: center;">$\text{InvWishart}(\nu_0, \,\mathbf{S}_0)$</td>
      <td style="text-align: center;">$\text{InvWishart}(\nu_0+n, \,\mathbf{S}_0+n \overline{\mathbf{S}}), \;\overline{\mathbf{S}}$ sample covariance</td>
    </tr>
    </tbody>
  </table>


### 12.2.7 Bayesian Hypothesis Testing




### 12.2.8 Model Comparison and Bayesian Information Criterion




### 12.2.9  Calculating the Posterior Distribution





