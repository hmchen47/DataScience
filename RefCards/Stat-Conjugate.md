# Conjugate Priors


## Overview of Conjugate Priors

+ [Prior conjugate family](../Notes/p03-BayesianBasics.md#11-the-bayes-rule)
  + conjugate family: prior densities and their leading posterior densities belonging to the same family
  + the choice of conjugate family not unique
  + exponential family of densities: sampling density w/ a sufficient statistic of constant dimension always finds a conjugate family of prior densities
  + _natural conjugate family_:
  + _subjective prior of informative prior_: the parameters of the prior density elicited using a previously collected data or expert knowledge
  + _noninformative prior_: no such prior information available or very little knowledge available about the parameter $\theta$

+ [Conjugate prior](../Notes/p04a-Bayesian.md#1221-the-mechanics-of-bayesian-inference)
  + prior: $\theta \sim N(a, b^2)$
  + the posterior for $\theta$

    \[ \theta \,|\, \mathcal{D}_n \sim N(\overline{\theta}, \, \tau^2) \tag{4} \]

  + region estimate: find posterior interval = find $C = (c, d) \to \mathbb{P}(\theta \in C \,|\, \mathcal{D}_n) = 0.95$
  + $\exists\; c$ and $d \text{ s.t. } \mathbb{P}(\theta < c \,|\, \mathcal{D}_n) = 0.025$
  + find $ c \text{ s.t. }$

    \[ \mathbb{P}(\theta < c \,|\, \mathcal{D}_n) = \mathbb{P} \left( \frac{\theta - \overline{\theta}}{\tau} < \frac{c - \overline{\theta}}{\tau}\right) = \mathbb{P}\left( Z < \frac{c - \overline{\theta}}{\tau} \right) = 0.025 \]

+ [Definition of conjugate priors](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + a prior distribution closed under sampling distribution
  + $\mathcal{P}$: a family of prior distribution
  + __Definition__. $\forall\; \theta, \, \exists\; p(\cdot \,|\, \theta) \in \mathcal{F}$ over a sample space $\mathcal{X}$. The posterior

    \[ p(\theta \,|\, \mathbf{x}) = \frac{p(\mathbf{x} \,|\, \theta)\; \pi(\theta)}{\int p(\mathbf{x} \,|\, \theta)\; \pi(\theta) \;d\theta} \]

    satisfies $p(\cdot \,|\,\mathbf{x}) \in \mathcal{P} \implies$ the family $\mathcal{P}$ is _conjugate_ to the family of sampling distribution $\mathcal{F}$
  + the family $\mathcal{P}$ should be sufficiently restricted, and is typically taken to be a specific parametric family.



## Conjugate Priors with Exponential Family

+ [General exponential family models](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + $p(\cdot \,|\, \theta)$: a standard exponential family model
  + the density w.r.t. a positive measure $\mu$

    \[ p(\mathbf{x} \,|\, \pmb{\theta}) = \exp\left(\pmb{\theta}^T\,\mathbf{x} - A(\pmb{\theta})\right) \tag{5} \]

    + $A(\pmb{\theta})$: the moment generation or log-normalizing constant

      \[A(\pmb{\theta}) = \log\left(\int \exp(\pmb{\theta} \mathbf{x} - A(\pmb{\theta}))\, d\mu(\mathbf{x}) \right)\]

  + the density of a conjugate prior for the exponential family

    \[\pi_{\mathbf{x}_0,n_0}(\theta) = \frac{\exp(n_0 \mathbf{x}_0^T \pmb{\theta} - n_0A(\pmb{\theta}))}{\int \exp(n_0 \mathbf{x}_0^T \pmb{\theta} - n_0A(\pmb{\theta}))\,d\pmb{\theta}} \]

  + the posterior

    \[\begin{align*}
      p(\mathbf{x} \,|\, \pmb{\theta}) \pi_{\mathbf{x}_0,n_0}(\pmb{\theta}) &= \exp(\pmb{\theta}^T \mathbf{x} - A(\pmb{\theta}))  \exp\left(n_0\mathbf{x}_0^T\pmb{\theta} - n_0 \,A(\pmb{\theta})\right) \\\\
      &\propto \pi_{\frac{\mathbf{x}}{1+n_0}+\frac{n_0\mathbf{x}_0}{1+n_0}, 1+n_0}(\pmb{\theta})
    \end{align*}\]
  
  + the prior incorporating $n_0$ "virtual" observations of $\mathbf{x}_0 \in \mathbb{R}^d$
  + after making one "real" observation x: the parameters of the posterior as a mixture of the virtual and actual observation

    \[ n_0^\prime = 1 + n_0 \quad \text{ and } \quad \mathbf{x}_0^\prime = \frac{\mathbf{x}}{1 + n_0} + \frac{n_0 \mathbf{x}}{1 + n_0} \]

+ [Generalized exponential family model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + $n$ observations $\mathbf{X}_1, \dots, \mathbf{X}_n \implies$ the posterior

    \[ p(\pmb{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) \propto \exp \left( (n + n_0) \left( \frac{n\overline{\mathbf{X}}}{n + n_0} + \frac{n_0\mathbf{x}_0}{n+n_0} \right)^T \pmb{\theta} - (n + n_0)A(\pmb{\theta}) \right) \]

  + the parameters of the posterior

    \[ n_0^\prime = n + n_0 \quad \text{ and } \quad \mathbf{x}_0^\prime = \frac{n \overline{\mathbf{X}}}{n+n_0} + \frac{n_0\mathbf{x}_0}{n+n_0} \]

  + the expectation w.r.t. $\pi_{\mathbf{x}_0, n_0}$

    \[ \mathbb{E}[\nabla A(\pmb{\theta})] = \int \nabla A(\pmb{\theta}) \pi_{\mathbf{x}_0, n_0}(\pmb{\theta})\,d\pmb{\theta} &= \mathbf{x}_0 - \frac{1}{n_0} \int \nabla \pi_{\mathbf{x}_0, n_0} (\pmb{\theta})\,d\pmb{\theta} = \mathbf{x}_0 \]

  + more generally,

    \[ \mathbb{E}[\nabla A(\pmb{\theta}) \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n] = \frac{n\overline{\mathbf{X}}}{n_0+n} + \frac{n_0 \mathbf{x}_0}{n_0+n} \]

  + under appropriate regularity conditions, the converse also holds, so that linearity of $\nabla A(\pmb{\theta}) \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n$ is sufficient for conjugacy

+ [__Theorem__ (exponential family)](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + open space $\Theta \subset \mathbb{R}^d$
  + $\mathbf{X}$: a sample of size one from the exponential family $p(\cdot \,|\, \pmb{\theta})$
  + the support of $\mu$ containing an open interval
  + $\pi(\pmb{\theta})$: a prior density not concentrated at a single point
  + the posterior mean of $\nabla A(\pmb{\theta})$ given a single observation $\mathbf{X}$ is linear

    \[ \mathbb{E}[\nabla A(\theta) \,|\, X] = a\mathbf{X} + \mathbf{b} \quad\iff\quad \pi(\pmb{\theta}) \propto \exp\left( \frac{1}{a} \mathbf{b}^T \pmb{\theta} - \frac{1 -a}{a} A(\pmb{\theta}) \right)  \]

  + similar result holds w/ discrete measure $\mu$

+ [Conjugate priors for discrete exponential family distributions](../Notes/p04a-Bayesian.md#1226-conjugate-priors)

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

+ [Conjugate priors for some continuous distributions](../Notes/p04a-Bayesian.md#1226-conjugate-priors)

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
      <td style="text-align: center;">$N(\pmb{\mu}, \,\pmb{\Sigma})$, known $\pmb{\Sigma}$</td>
      <td style="text-align: center;">$N(\pmb{\mu}_0, \,\pmb{\Sigma}_0)$</td>
      <td style="text-align: center;">$N\left(\mathbf{K}\left(\Sigma_0^{-1} \mu_0 + n \Sigma^{-1} \overline{X}\right), \,\mathbf{K}\right), \\ \hspace{10em}\;\mathbf{K} = (\Sigma_0^{-1} + n\Sigma^{-1})^{-1}$</td>
    </tr>
    <tr>
      <td style="text-align: center;">$N(\pmb{\mu}, \,\pmb{\Sigma})$, known $\pmb{\mu}$</td>
      <td style="text-align: center;">$\text{InvWishart}(\nu_0, \,\mathbf{S}_0)$</td>
      <td style="text-align: center;">$\text{InvWishart}(\nu_0+n, \,\mathbf{S}_0+n \overline{\mathbf{S}}), \;\overline{\mathbf{S}}$ sample covariance</td>
    </tr>
    </tbody>
  </table>



## Uniform-Bernoulli likelihood model

+ [Uniform-Bernoulli likelihood model](../Notes/p04a-Bayesian.md#1221-the-mechanics-of-bayesian-inference)
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

+ [Flat priors not invariant](../Notes/p04a-Bayesian.md#1225-flat-priors-improper-priors-and-noninformative-priors)
  + contradiction
    + the notation of a flat prior not well define
    + a flat prior on a parameter $\nRightarrow$ a flat prior on a transformed version of this parameter
  + flat priors not transformed _invariant_



## Beta-Bernoulli likelihood model

+ [Beta-Bernoulli likelihood model](../Notes/p04a-Bayesian.md#1221-the-mechanics-of-bayesian-inference)
  + sampling distribution: $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}, \;\; X_1, \dots, X_n \sim \text{Bernoulli}(\theta)$ w/ $\hat{\theta} = S_n/n$
  + the prior distribution: $\theta \sim \text{Beta}(\alpha, \beta)$ w/ prior mean $\theta_0 = \alpha/(\alpha+\beta)$
  + the posterior distribution: $\theta \,|\, \mathcal{D}_n \sim \text{Beta}(\alpha + S_n, \beta + n - S_n)$
  + the flat (uniform) prior: $\alpha = \beta = 1$
  + the posterior mean:

    \[ \overline{\theta} = \frac{\alpha + S_n}{\alpha + \beta + n} = \left(\frac{n}{\alpha+\beta+n}\right) \hat{\theta} + \left(\frac{\alpha+\beta}{\alpha+\beta+n} \right) \theta_0 \]



## Dirichlet-Multinomial likelihood Model

+ Dirichlet-Multinomial likelihood Model
  + sampling distribution: $\exists\; \mathcal{D}_n = \{X_1, \dots, X_n\}, \;\; X_1, \dots, X_n \sim \text{Bernoulli}(\theta)$ w/ $\hat{\theta} = S_n/n$
  + prior distribution: Dirichlet prior
  + the sample space of the multinomial w/ $K$ outcomes as the set of vertices of the $K$-dim hypercube $\mathbb{H}_K$, mad up of vectors w/ exactly only one 1 and the remaining elements 0

    \[ x = \underbrace{(0, 0, \dots, 0, 1, 0, \dots, 0)^T}_{K\text{ places}} \]

  + $\exists\; \mathbf{X}_i = (X_{i1}, \dots, X_{iK})^T \in \mathbb{H}_K$,
  
    \[ \underbrace{\theta \sim \text{Dirichlet}(\pmb{\alpha})}_{\text{Prior}} \;\text{ and }\; \underbrace{\mathbf{X}_i \,|\, \theta \sim \text{Multinomial}(\pmb{\theta})}_{\text{likeliehood}} \; \forall\; i=1, 2, \dots, n\]

    $\implies$ the posterior satisfies

    \[ p(\pmb{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) \propto \mathcal{L}_n(\theta)\pi(\theta) \propto \prod_{i=1}^n \prod_{j=1}^K \theta_j^{X_{ij}} \prod_{j=1}^K \theta_j^{\alpha_j - 1} = \prod_{j=1}^K \theta_j^{\sum_{i=1}^n X_{ij}+\alpha_j-1} \]

  + the posterior distribution w/ $\overline{\mathbf{X}} = \sum_{i=1}^n \mathbf{X}_i / n \in \Delta_K$

    \[ \pmb{\theta} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n \sim \text{Dirichlet}(\alpha+ n \overline{\mathbf{X}})\]

  + the posterior mean

    \[ \mathbb{E}(\theta \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n) = \left(\frac{\alpha_1 + \sum_{i=1}^n X_{i1}}{\sum_{i=1}^K \alpha_i + n}, \dots, \frac{\alpha_K + \sum_{i=1}^n X_{iK}}{\sum_{i=1}^K \alpha_i + n} \right)^T \]

  + prior conjugate w.r.t. the mode: the prior as Dirichlet distribution $\to$ the posterior as Dirichlet distribution


## Gamma-Poisson likelihood model

+ [Gamma-Poisson likelihood model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + Poisson model w/ rate $\lambda \geq 0$ in the sample space $\mathcal{X} = \mathbb{Z}_+ \text{ s.t. }$

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



## Gamma-Exponential likelihood model

+ [Gamma-Exponential likelihood model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + exponential distribution w/ the sample space $\mathcal{X} \in \mathbb{R}_+ \text{ s.t. }$

    \[ p(x \,|\, \theta) = \theta e^{-x\theta} \]
  
  + exponential model widely used for survival times or waiting times btw events
  + the conjugate prior: Gamma distribution in the most convenient parameterization

    \[ \pi_{\alpha, \beta} \propto \theta^{\alpha - 1} e^{-\beta\theta} \]

  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $\text{Exponential}(\theta)$
  + the posterior

    \[ \theta \,|\, X_1, \dots, X_n \sim \text{Gamma}(\alpha + n,\, \beta + n\overline{X}) \]

  + the prior acts if $\alpha -1$ virtual example are used, w/ a total waiting time of $\beta$


## Gamma-Geometric likelihood model

+ [Gamma-Geometric likelihood model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + the geometric distribution
    + the discrete analogue of the exponential model
    + sample space $\mathcal{X} = \mathbb{Z}_{++}$, the strictly positive integers
    + the density
  
    \[ \mathbb{P}(X = x \,|\, \theta) = (1-\theta)^{x-1} \theta \]

  + the conjugate prior: $\text{Gamma}(\alpha, \beta)$
  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $\text{Geometric}(\theta)$
  + the posterior

    \[ \theta \,|\, X_1, \dots, X_n \sim \text{Gamma}(\alpha + n, \,\beta + n\overline{X}) \]


## InvGamma-Gaussian likelihood model

+ [InvGamma-Gaussian likelihood model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
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


## ScaledInv-$\chi^2$-Gaussian likelihood model

+ [ScaledInv-$\chi^2$-Gaussian likelihood model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + the prior: scaled inverse $\chi^2$ distribution of $\sigma^2\nu_0Z\;$ w/ $Z \sim \chi_{\nu_0}^2$

    \[ \pi_{\nu_0, \sigma_0^2}(\theta) \propto \theta^{-(1+\nu_0/2)} \exp\left( -\frac{\nu_0 \sigma^2_0}{2\theta} \right) \]

  + the posterior

    \[ \sigma^2 \,|\, X_1, \dots, X_n \sim \text{ScaledInv-}\chi^2 \left( \nu_0 +n, \,\frac{\nu_0 \sigma_0^2}{\nu_0 + n} + \frac{n\, \overline{(X - \mu)^2}}{\nu_0 + n} \right) \]


## InvWhishart-Gaussian likelihood Model

+ [InvWhishart-Gaussian likelihood Model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + sampling distribution: $\exists\; X_1, \dots, X_n$ observed data from $N(\mathbf{0}, \pmb{\Sigma}), \;\pmb{\Sigma} \in \mathbb{R}^{n\times n}$ as covariance (positive semi-defined matrix)
  + the posterior: an inverse Wishart prior multiplies the likelihood

    \[\begin{align*}
      &p(\mathbf{X}_1, \dots, \mathbf{X}_n \,|\, \pmb{\Sigma})\pi_{\nu_0, \mathbf{S}_0} \\\\
      & \hspace{3em}\propto |\pmb{\Sigma}|^{-n/2} \exp \left( -\frac{n}{2} \text{tr}(\overline{\mathbf{S}}\pmb{\Sigma}^{-1}) \right) |\pmb{\Sigma}|^{-(\nu_0+d+1)/2} \exp \left( -\frac{1}{2}\text{tr}(\mathbf{S}_0 \pmb{\Sigma}^{-1}) \right) \\\\
      &\hspace{1em}= |\pmb{\Sigma}|^{-(n+\nu_0+d+1)/2} \exp \left( -\frac{1}{2} \text{tr}\left(\left(n \overline{\mathbf{S}} + \mathbf{S}_0\right) \pmb{\Sigma}^{-1}\right) \right)
    \end{align*}\]

    + the empirical covariance: $\overline{\mathbf{S}} = \frac{1}{n} \sum_{i=1}^n \mathbf{X}_i\mathbf{X}_i^T$
  + the posterior

    \[ \pmb{\Sigma} \,|\, \mathbf{X}_1, \dots, \mathbf{X}_n \sim \text{InvWishart}(\nu_0 + n,\, \mathbf{S}_0 + n\overline{\mathbf{S}}) \]

  + similarly, the conjugate prior for the inverse covariance $\pmb{\Sigma}^{-1}$ (precision matrix) is a Wishart


## Pareto-Uniform likelihood model

+ [Pareto-Uniform likelihood model](../Notes/p04a-Bayesian.md#1226-conjugate-priors)
  + uniform distribution: $\text{Uniform}(0, \theta),\, \theta \geq 0$
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



