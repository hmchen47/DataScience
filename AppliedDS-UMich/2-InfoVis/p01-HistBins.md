# Selecting the Number of Bins in a Histogram: A Decision Theoretic Approach

Author: K. He & G.Meeden
Publication: Journal of Statistical Planning and inference, 61(1), 49-59
Year: 1997

## Introduction

+ D. Scott, "On optimal and data-based histograms," Biometrika, 66:605{610, 1979.
    + A formula for the optimal histogram bin width which asymptotically minimizes the integrated mean squared error
    + Since the underlying density is usually unknown, it is not immediately clear how one should apply this in practice.
    + Using the Gaussian density as a reference standard, which leads to the data-based choice for the bin width of $a \times s \times n^{-1/3}$, where $a = 3.49$ and $s$ is an estimate of the standard deviation.
    + For real data sets histograms based on __5-20 bins__ usually suffice.

+ Mats Rudemo, "Empirical choice of histograms and kernel density estimators", Scandinavian J. of Stat., 9:65-78, 1982
    + a cross-validation technique for selecting the number of bins
    + large sampling variation


## Selecting the Number of Bins

+ Problem: decision on a histogram is an estimate of the unknown density $f$ where the bins are all of the same size, a typical decision $d$ consists of two components, the number of bins and the mass assigned to each bin.

+ Notation
    + ${\bf X} = (X_1, \ldots, X_n)$: a random sample from some unknown continuous distribution on the known interval $[a, b]$
    + $\Theta$: parameter space, the class of all probability density functions $f$ on $[a, b]$
    + $d = (k, {\bf m}_k)$: decision function
        + $k \in \mathbb{N}^+$ and $5 \leq k \leq 20$ in general
        + ${\bf m}_k = (m_{k,1}, \ldots, m_{k,k})$: measured vector, ${\bf m}_k > 0$ and $\sum_i m_{k, i} = 1$
    + $L(f, d, {\bf m})$: a loss function when $d = (k, {\bf m}_k)$ is an estimate of $f$
    + ${\bf p}_k (f) = {\bf p}_k = (p_{k,1}, \ldots, p_{k,k})$: predict vector
    + ${\bf V}_k({\bf x}) = {\bf v}_k$: the count vector for the number of observations that fall in each bin
    + ${\bf v}_k = (v_{k,1}, \ldots, {\bf v}_{k,k})$: $v_{k,i}$ = the number of observations that falls into bin $i$
    + ${bf X}$: unknown distribution
    + $(n, {\bf p}_k)$: multinomial distribution

+ Loss function $L(f, d, {\bf x})$
    + Assume that the loss function depends on $f$ only through the mass it assigns to the $k$ equal subintervals of $[a, b]$, i.e., on ${\bf p}_k (f) = {\bf p}_k = (p_{k,1}, \ldots, p_{k,k})$
    + Assume that the loss function $L(f, d, {\bf x})$ depends on the data ${\bf X} = {\bf x}$ and denote by (Eq.1)

        $$\begin{array}{rcl} 
            L(f, d, {\bf m}) & = & L({\bf p}_k, k, {\bf m}_k, {\bf x}) \\
                & = & 
                \left\{
                \begin{array}{ll} 
                    c(k, {\bf m}) \sum_1^k (p_{k,i} - m_{k,i})^2 & \text{if no  } m_{k,i}= 1 \\ 
                    1/k & \text{if  } \exists \text{  } m_{k,i} = 1
                \end{array}
                \right.
        \end{array}$$
    + Consider the second part $1/k$: With simple case of $4$ bins on the unit interval $[0, 1]$ and the unknown $f$ puts all its mass in the first  bin, i.e., $[0, .25]$. Consider two different decisions $d_1 = (2, (1, 0))$ and $d_2 = (4, (1, 0, 0, 0 ))$. Both decisions on the first part equals to $0$.  But $d_1$ is bettern than $d_2$.

+ Issue: how to trade off the increased 'accuracy' that comes from adding more bins with the increased `cost' of the additional bins.

+ Non-informative Bayesian Procedure
    + Donald B. Rubin, "The Bayesian bootstrap", Annals of Statistics, 9:130-134, 1981.
    + Glen Meeden, Malay Ghosh, and Stephen Vardeman, "Some admissible nonparametric and related finite population sampling estimators", Annals of Statistics, 13:811-817, 1985.
        + the admissibility of various non-parametric procedures by showing that they were stepwise Bayes procedures against the Bayesian bootstrap
        + Apply a similar approach to the problem of selecting the number of bins with a given $k$ <br/>
        $\Theta(k) = \{ {\bf p}_k = {\bf p}_k(f): f \in \Theta\}$
        + Given ${\bf X} = {\bf x}$, $f$ is true with unknown distribution ${\bf X} \rightarrow {\bf V}_k$ with multinomial $(n, {\bf P}_k)$. 
        + The posterior distribution over $\Theta(k)$, the Dirichlet distribution, with parameter vector ${\bf v}$ is the __Bayesian bootstrap__
    + NOT true as the posterior distribution on $\Theta(k)$ is the Bayesian bootstrap but __stepwise Bayes justification__

+ Sensible loss function with the Bayesian bootstrap
    + Suppose ${\bf X} = {\bf x}$ observed with fixed $k$, the number if bins and ${\bf v}_k$ computed
    + The choice of ${\bf m}_k$ to minimize the posterior expectation, $\sum^k_1 (p_{k,i} - m_{k,i})^2$ with the Bayesian bootstrap, is ${\bf v}_k / n$ where the sample size $n = \sum_{i=1}^k v_{k,i}$
    + Posterior risk:

        $$\sum_{i=1}^k v_{k,i} (n - v_{k,i}) / (n^2(n+1))$$

    + For a fixed ${\bf x}$, posterior risk $\uparrow$ as $k \uparrow$, $\therefore$ posterior risk never used to choose $k$.
    + $\because \max ((n+1)^{-1} \sum^k_{i=1} m_{k,i} (1 - m_{k,i}))$ by taking $m_{k,i} = 1/k \text{ } \forall i$, $\therefore \text{posterior risk } \leq (1 - 1/k)/(n+1)$.  The inequality ratio holds: 

        $$\frac{\sum_{i=1}^k v_{k,i} (n-v_i)/(n^2(n_1))}{(1 - 1/k) / (n+1)} \leq 1$$

    + The ratio, as a function of the number of bins $k$, vs. actual posterior risk under the Bayesian bootstrap to its max possible value: an attempt to calibrate the loss over the different problem
    + When sampling from a rough distribution the histogram has more bins than when sampling from a smooth distribution.

+ Entropy
    + A convenient measure of the smoothness or uncertainty of a probability distribution
    + Given ${\bf v}_k$, the estimate of the smoothness of the unknown $f$

        $$\mathcal{E}_{{\bf v}_k} = - \sum^k_{i=1} (v_{k.i}/n) \log(v_{k,i} / n)$$
    + The ratio (Eq.2)

        $$r({\bf X}, k) = \frac{\mathcal{E}_{{\bf v}_k}}{\log k}$$
    
        The estimated entropy of $f$ over $k$ bins divided by the entropy of the uniform distribution over $k$ bins, gives a standardized estimate of the measure of the smoothness of the distribution.
    + $r({\bf x}, k) \in [0, 1]$ and rougher data $\rightarrow$ $k \downarrow$
    + If denominator of Eq.2 increases to $1 + (1-r({\bf x}, k))$, then it is decreased as $r({\bf x}, k)$ decreases.
    + Because of the factor $1 - 1/k$, this decrease is proportionally greater for smaller values of $k$.
    + Histograms with fewer bins are penalized by a greater amount than histograms with more bins when the data are rough.





### [Bootstrapping vs Bayesian Bootstrapping conceptually?](https://stats.stackexchange.com/questions/181350/bootstrapping-vs-bayesian-bootstrapping-conceptually)

+ The (frequentist) bootstrap takes the data as a reasonable approximation to the unknown population distribution. Therefore, the sampling distribution of a statistic (a function of the data) can be approximated by repeatedly resampling the observations with replacement and computing the statistic for each each sample.

+ Let $y=(y_1, \ldots, y_n$) denote the original data. Let $y^b=(y^b_1, \ldots, y^b_n)$ denote a __bootstrap sample__. Such a sample will likely have some observations repeated one or more times and other observations will be absent. The mean of the bootstrap sample is given by $m_b = \frac{1}{n} \sum_{i=1}^n y^b_i$.

+ The distribution of $m_b$ over a number of bootstrap replications used to approximate the sampling distribution from the unknown population

+ Connection between the frequentist bootstrap and the Bayesian bootstrap
    + In each bootstrap sample $y_b$, each observation $y_i$ occurs anywhere from $0$ to $n$ times. 
    + $h^b_i$: the number of times $y_i$ occurs in $y^b$
    + $h^b = (h^b_1, \ldots, h^b_n)$
    + $h^b_i \in \{0,1, \ldots, n−1,n\}$ and $\sum^n_{i=1} h^b_i=n$. 
    + Given $h^b$, construct a collection of nonnegative weights that sum to one: $w^b = h^b / n$, where $w^b_i = h^b_i / n$. 
    + The mean of the bootstrap sample: $m_b = \sum^n_{i=1} w^b_i y_i$
    + The observations are chosen for a bootstrap sample determines the joint distribution for $w^b$. In particular, $h^b$ has a multinomial distribution and thus $(n w^b) \sim \text{Multinomial}(n, (1/n)^n_{i=1})$
    + Compute $m^b$ by drawing $w^b$ from its distribution and computing the dot product with $y$. From this new perspective, it appears that the observations are fixed while the weights are varying.

+ Bayesian inference
    + The calculation of the mean according to the Bayesian bootstrap differs only in the distribution of the weights.
    + The data $y$ are fixed and the weights $w$ are the unknown parameters. 
    + Some functional of the data that depends on the unknown parameters: $\mu = \sum_{i=1}^n w_i y_i$

+ Thumbnail sketch of the model behind the Bayesian bootstrap:
    + The sampling distribution for the observations is multinomial and the prior for the weights is a limiting Dirichlet distribution that puts all its weight on the vertices of the simplex.
    + Posterior distribution for the weights: $w \sim \text{Dirichlet}(1, \ldots, 1)$
    + The two distributions for the weights (frequentist and Bayesian) are quite similar: They have the same means and similar covariances. 
    + The Dirichlet distribution is 'smoother' than the multinomial distribution, so the Bayesian bootstrap may be call the smoothed bootstrap.
    + Interpret the frequentist bootstrap as an approximation to the Bayesian bootstrap.
    + Given the posterior distribution for the weights, we can approximate the posterior distribution of the functional $μ$ by repeated sampling $w$ from its Dirichlet distribution and computing the dot product with $y$.
    + Adopt the framework of estimating equations <br/>

        $$\sum^n_{i=1} w_i g(y_i, \theta) = {\bf 0}$$,
        + $g(y_i, \theta)$: a vector of _estimating functions_ that depends on the unknown parameter (vector) $\theta$
        + ${\bf 0}$: a vector of zeros
    + If this system of equations has a unique solution for $\theta$ given $y$ and $w$, then we can compute its posterior distribution by drawing $w$ from its posterior distribution and evaluating that solution. 
    + The framework of estimating equations is used with _empirical likelihood_ and with _generalized method of moments (GMM)_.

+ Example: the simplest case

    $$\sum^n_{i=1} w_i (y_i - \mu) = 0$$

    For the mean and the variance, $\theta = (\mu, v)$,

    $$g(y_i, \theta) = \left( 
        \begin{array}{c} 
            y_i - \mu \\ 
            (y_i - \mu)^2 - v
        \end{array} 
    \right)$$



## The stepwise Bayes justification




