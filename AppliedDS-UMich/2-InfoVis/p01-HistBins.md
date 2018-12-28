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

+ Proposal:
    + Consider the ratio (Eq.3)

        $$\frac{\sum_{i=1}^k v_{k,i} (n - v_{k,i})/(n^2(n+1))}{\{(1-1/k)/(n+1)\}^{(1+(1-r({\bf X}, k)))}}$$
    + Problem: select the histogram with $k_0$ bins where $k_0$ is the value of $k$ which minimizes the above ratio when $5 \leq k \leq 20$.
    + With sampling from smoother distribution the average histograms with fewer bins than when sampling from a rougher distribution.
    + Define (Eq.4): $c(k, {\bf x})^{-1} = \{(1 - 1/k)/(n+1)\}^{(1+(1-r({\bf X}, k)))}$
    + Given a justification of the loss function defined in Eq.1 with $c(k;,x)$ defined in Eq.4.
    + The general form of the loss function is quite reasonable and the particular choice of $c(k, x)$ was closely tied to our desire to exploit the Bayesian bootstrap to give a sensible solution for this problem.

+ Scott's rule
    + The average number of bins depends very little on the underlying distribution, especially for smaller sample sizes, and increases as the sample size increases.
    + Bayesian bootstrap: the average number of bins varies a good deal as the underlying population changes.
    + Smooth densities generate histograms with fewer bins than the rougher densities.
    + For smooth populations the average number of bins tends to decrease as the sample size increases, while for the rougher densities just the opposite is true.
    + Formula for the optimal histogram bin width: asymptotically minimizes the integrated mean squared error derived by Scott

        $$\{ 6 / \int_a^b f'(x)^2 dx\}^{-1/3} n^{-1/3}$$

        for a given density $f$, when the integral exists.

+ Rule of Thumb by Terrell & Scott
    + George R. Terrell and David W. Scott, "Oversmooth nonparametric density estimates", J. of the Amer. Stat'l. Assn., 80:209{214, 1985.
    + For a histogram of data from a density believed to be moderately smooth on the real line, use $(2n)^{1/3}$ equal-width bins or a convenient slightly larger number over the sample range.
    + E.g., $n=50 \rightarrow 4.6$ bins, $n=4000 \rightarrow 20$ bins
    + For most practical problems only considering histograms with 5 to 20 bins is not unreasonable.
    + Suppose $n$ is large enough so that $v_{k,i}/n$ is a very good estimate of $p_{k,i}$.
    + Notation:
        + $f$: the density to be estimated
        + $\mathcal{E}_f(k)$: as $\mathcal{E}_{{\bf v}_k}$ with ${\bf v}_k$ replaced with ${\bf p}_k(f)$
        + $r(f, k) = \mathcal{E}(k)/\log k$: the true entropy of $f$ over $k$ bins divided by the entropy of the Uniform distribution over $k$ bins.
    + Replacing ${\bf v}_k$ with ${\bf p}_k(f)$ on Eq.3: (Eq.5)

        $$(n+1)^{r(f,k)-1} (1 - 1/k)^{r(f,k)-2} \sum_{i=1}^k p_{k,i} (1-p_{k,i})$$
    + This equation measures how well a $k$ bin histogram approximates $f$ where for each bin we use the true bin probability under $f$.
    + Therefore, an optimal number bins for fitting the true $f$ is the value of $k$ which minimizes the above equation.
    + Uniform distribution $f$: Eq.5 is consistent in $k$
    + Other density: find the minimizing value of $k$ as $k$ ranges $[5, 20] \text{  }\forall n \in \mathbb{N}$
    + E.g., for $n=500$, densities $2, 3, 4,$, and $5 \rightarrow 20$ bins; densities $6, 7$, and $8 \rightarrow 5$ bins

+ [On Selecting The Number Of Bins For A Histogram](https://pdfs.semanticscholar.org/a207/5292fa63ecc88834b03ff24cfad348d8499a.pdf?_ga=2.108421506.1086990556.1545942104-1484135392.1545942104)
    <a href="https://pdfs.semanticscholar.org/a207/5292fa63ecc88834b03ff24cfad348d8499a.pdf?_ga=2.108421506.1086990556.1545942104-1484135392.1545942104"> <br/>
        <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/a2075292fa63ecc88834b03ff24cfad348d8499a/5-Table3-1.png" alt="Following abbreviations are used in the tables and figures displaying results: StM – Sturges Method; ScM – Scott Method; FDM – Freedman Diaconis Method; SM – Shimazaki et al. Method; KM – Knuth Method; LHM – method proposed in this paper; Define the error metrics ENN and EL for the nearest neighbor and linear interpolation reconstructions, respectively; the lowest roughness Rˆ is likely to be the most visually appealing" title="RESULTS FOR DF–1 & DF–2 USING VARIOUS METHODS" height="250">
    </a>
    <a href="https://pdfs.semanticscholar.org/a207/5292fa63ecc88834b03ff24cfad348d8499a.pdf?_ga=2.108421506.1086990556.1545942104-1484135392.1545942104"> 
        <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/a2075292fa63ecc88834b03ff24cfad348d8499a/5-Table4-1.png" alt="Following abbreviations are used in the tables and figures displaying results: StM – Sturges Method; ScM – Scott Method; FDM – Freedman Diaconis Method; SM – Shimazaki et al. Method; KM – Knuth Method; LHM – method proposed in this paper; Define the error metrics ENN and EL for the nearest neighbor and linear interpolation reconstructions, respectively; the lowest roughness Rˆ is likely to be the most visually appealing" title="RESULTS FOR DF–3 & DF–4 USING VARIOUS METHODS" height="250">
    </a>

+ The number of bins in a histogram should depend both on the sample size and the shape of the unknown density to be estimated.

+ Two more natural comparisons
    + Integrated squared error loss which is the criterion Scott used to derive his rule
    + The loss function given in Eq.3 which gives our procedure

+ Scott's rule gives fewer bins and will give better pictures for smooth densities but at the cost of increasing the probability of obscuring important structure for rougher densities.



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

+ Stepwise Bayes technique
    + Francis C. Hsuan. "A stepwise Bayes procedure". Annals of Statistics, 7:860-868, 1979.
    + Bruce McK. Johnson. "On admissible estimators for certain fixed sample binomial problems". Annals of Mathematical Statistics, 42:1579-1587,1971
    + The class of unique stepwise procedures is the minimal complete class when the parameter space is finite and the loss function is strictly convex.
    + Lawerence D. Brown. "A complete class theorem for statistical problems with finite sample space". Annals of Statistics, 9:1289{1300, 1981.
    + Brown gave a quite general complete class theorem for estimation problems having a finite sample space, again using the stepwise Bayes idea.
    + A finite sequence of disjoint subsets of the parameter space is selected, where the order of the specified subsets is important.
    + Procedure:
        1. define different prior distribution each of the subsets
        2. Bayes procedure found for each sample point that receives positive probability under the first prior
        3. Bayes procedure found for the second prior for each sample point which receives positive probability under the second prior and which was not taken care of under the first prior
        4. Bayes procedure is found for the sample points with positive probability under the third prior and which had not been considered in the first two stages.
        5. Continued over each subset of the sequence in the order given
        6. If the sequences of subsets and priors are such that a procedure is defined at every sample point of the sample space then the resulting procedure is admissible.
    + Assume that there is a random sample of size $n$ from an unknown population with some density function on the known interval $[a, b]$.
    + Given the sample to select a histogram with $k$ bins to represent the data. In each case the bins are always of the same size and $k$ must satisfy the conditions $k_1 \leq k \leq k_2$ where $k_1, k_2 \in \mathbb{N}^+$.

+ Notation:
    + $M$: the least common multiple of the intergers $k_1, k_1 + 1, \ldots, k_2$
    + $\Theta(M)$: parameter space
    + ${\bf u}$: a certain class of vectors ${bf u}$ of lengths $2, 3, \ldots, n+1$
    + ${\bf V}_M$: the count vector of the number of the observation which fall into each subinterval

+ Modeling 
    + Define a sequence of disjoint subsets of $\Theta(M)$
    + Given ${\bf u}$ to satisfy disjoint subsets with
        + $u_1 \in \{1, 2, \ldots, n\}$ and $\|{\bf u}\| = u_1 + 1$
        + arranged remaining entries $u_j \in \{1, 2, \ldots, M\}\backslash u_1, j=2,\ldots,M$, increasing monotonically
    + Associate with each such ${\bf u}$ the subset of $\Theta(M)$

        $$\Theta(M, u) = \{{\bf p}_M: p_{M,i} > 0 \text{ for } i=u_2, u_3, \ldots, u_{u_1+1} \text{ and } \sum_{i=2}^{u_1+1} p_{M, u_i} = 1\}$$
    + E.g. 1: ${\bf u} = (1, 3) \rightarrow \Theta(M, {\bf u})$ is the one vector with mass one on the 3rd subinterval of $[a, b]$
    + E.g. 2: ${\bf u} = (3, 1, 4, 7 \rightarrow \Theta(M, {\bf u})$ is the set of all those vectors which place all their mass on the 1st, 4th, and 7th subintervals and with positive mass on them
    + Generate ${\bf u}$: with a typical ${\bf u}$
        + select $u_1$ subintervals which are given by $u_2, u_3, \ldots, u_{u_1+1}$
        + ${\bf u}_1 \cap {\bf u}_2 = \emptyset \rightarrow \Theta(M, {\bf u}_1) \cap \Theta(M, {\bf u}_2) = \emptyset$
        + order ${\bf u}_i, i = 1, 2, \ldots, M$ by lexicographical ordering

+ Stepwise Bayes argument
    + Distributon of ${\bf V}_M$: $multinomial(n, {\bf p})$
    + Given ${bf u}$ with lexicographic order that all the sample points which get positive probability under all those vectors ${\bf u'}$
    + The points in the sample space of ${\bf V}_M$, assigned positive probability under ${\bf P}_M \in \Theta(M, {\bf u})$ if $v_i > 0, i = u_2, u_3, \ldots, u_{u_1 + 1}$
    + Only consider those sample points with $v_i > 0$, the prior choosen

        $$1/\prod_{i=2}^{u_1+1} p_{m,u_i}$$

    + The posterior for $v_i > 0$: $Dirichlet({\bf v})$, i.e., Bayesian bootstrap
    + Select $k$ to minimize the posterior expected loss = minimize Eq.3
    + The procedure given in the previous section is equivalent to stepwise Bayesian procedure




