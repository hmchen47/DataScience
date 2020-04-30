# Statistics: Basics

## Basic Concepts

+ [Terminology](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + standard deviation ($sd$): a measure of the amount of variation or dispersion of a set of values ($var = sd^2$) $\implies$ true population mean
  + standard error ($se$)
    + the standard deviation ($sd$) of its sampling distribution or an estimate of that standard deviation
    + not enough samples & different trials $\to$ different population means
  + sampling distribution: the probability distribution of a given random-sample-based statistic
  + frequency interpretation of probability: long-run properties of repeated random events
  + frequentist:
    + standard statistical methods
    + $p(x)$: the proportion of times $x$ will occur in an infinitely long series of repeated identical situations

+ [odds ($O$)](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the probability ($p$) that the event will occur divided by the probability ($1 - p$) that the event will not occur
  + used to describe the chance of an event occurring

  \[ O = \frac{p}{1 - p} \tag{Odds} \]

+ [logit: the natural logarithm of the odds](../Notes/p01-Bayesian.md#31-subjectivity-and-context)

  \[ \text{logit}(p) = \ln(\frac{p}{1 - p}) \tag{Odds.log} \]

+ [Bayes theorem](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + Formula

  \[ p(b|a) = \frac{p(a|b)}{p(a)} \times p(b) \tag{Bayes} \]

  + The odds form of Bayes theorem

    \[ \frac{p(b|a)}{p(\overline{b}|a)} = \frac{p(a|b)}{p(a| \overline{b})} \times \frac{p(b)}{p(\overline{b})} \tag{Bayes.odds} \]

+ [Basic concepts of mathematical foundation](../Notes/p04a-Bayesian.md#1201-math-fundamental)
  + __Definition__ (lower bound and infimum)
    + $a \in S$ as a _lower bound_  of $ S \subseteq P$, a partially ordered set $(P,\leq) \text{ s.t. } a \leq x, \;\forall\, x \in S$
    + a lower bound $a \in S$ as an _infimum_ (or _greatest lower bound_, or _meet_) of $S \text{ s.t. } \forall$ lower bounds $y \in S \subseteq P, y \leq a$ ($a$ is larger than or equal to any other lower bound)
  + __Definition__ (upper bound and supremum)
    + $b \in S$ as an _upper bound_  of $S \subseteq P$, a partially ordered set $(P,\leq), \,\text{ s.t. } b \geq x, \;\forall\, x \in S$
    + an upper bound $b$ of $S$ as a _supremum_ (or _least upper bound_, or _join_) of $S \,\text{ s.t. } \forall$ upper bounds $z \in S \subseteq P, z \geq b$ (b is less than or equal to any other upper bound)

  + __Definition__ (the arguments of the maxima)
    + $\exists$ an arbitrary set $X$, a totally ordered set $Y$ and a function $f: X \to Y$
    + the $\mathop{\arg\max}$ over some subset, $S$ of X define as

      \[ \mathop{\arg\max}_{x \in S \subseteq X} f(x) := \{ x \,|\, x \in S,  \forall\, y \in S: f(y) \leq f(x) \} \]

  + __Definition__ (argument of the minimum)

    \[\mathop{\arg\min}_{x \in S} f9x) := \{ x \,|\, \forall\, y \in S: f(y) \geq f(x) \}\]

+ [Basic concepts of Probability and Statistics](../Notes/p04a-Bayesian.md#1202-probability-and-statistics)
  + __Definition__ (moment)
    + a _moment_ is a specific quantitative measure of the shape of a function
    + the $n$-moment of a real-valued continuous fucntion $f(x)$ of a real variable about a value $c$ (usually $c=0$)

      \[ \mu_n = \int_{-\infty}^\infty (x - c)^n f(x) dx \]

    + $f$ as a probability density function $\implies$ the $n$-th moment of the probability distribution
    + $F$ as a cumulative probability distribution fucntion of any probability distribution, probably no density function $\implies$ the $n$-th moment of the probability distribution given by the Reimann-Stieltjes integral

      \[ \mu_n^\prime = E[X^n] = \int_{-\infty}^\infty x^n dF(x) \]

  + __Definition__ (converge in distribution) <br/>
    A sequence $X_1, X_2, \dots$ of real-valued random variables is said to _converge in distribution_, or _converge weakly_, or _converge in law_ to a random variable $X$, denoted as $\xrightarrow{D}$, if

    \[ \lim_{n \to \infty} F_n(x) = F(x) \]
  + __Definition__ (converge in probability) <br/>
    A sequence $\{X_n\}$ of random variables _converges in probability_, denoted as $\xrightarrow{P}$, towards the random variable $X$ if $\forall\, \varepsilon > 0$

    \[ \lim_{n \to \infty} \Pr(|X_n - X|) > \varepsilon) = 0 \]

    + $P_n$: the probability that $X_n$ is outside the ball of radius $\varepsilon$ centered at $X$
    + $X_n$ _converges in probability_ to $X$:  

      \[ \forall\, \varepsilon > 0, \delta > 0, \;\exists\, N \in \mathbb{N}, \,\text{ s.t. } P_n < \delta \;\forall n \geq N, \]

  + detailed balance
    + __Definition__ (Reversible Markov process or reversible Markov chain) <br/> A Markov process called _reverse Markov process_ or _reversible Markov chain_ if it satisfies the detailed balance equations
    + __Definition__ (detailed balance in discrete process) <br/> The transition probability matrix, $P$, for a Markov process posses a stationary distribution (i.e., equilibrium distribution) $\pi \text{ s.t. }$

      \[ \pi_i P_{ij} = \pi_j P_{ji} \]

  + __Theorem__ (univariate delta method) <br/>
    $\exists$ a sequence of random variables $X_n$ satisfying

      \[ \sqrt{n}[X_n - \theta] \xrightarrow{D} N(0, \sigma^2) \implies \sqrt{n} [g(X_n) - g(\theta)] \xrightarrow{D} N(0, \sigma^2 \cdot [g'(\theta)]^2) \]

  + the __mode__ of a set of data values is the value that happen most often.
  + Jensen's inequality
    + $\exists\, (\Omega, \mathcal{F}, P)$ as a probability space
    + $X$ as an integrable real-valued random variable
    + $\varphi$: a convex function

      \[ \varphi(E[X]) \leq E[\varphi(X)] \]

  + Hoeffding's inequality
    + providing an upper bound on the probability that the sum of bounded independent random variables deviates from its expected value by more than a certain amount
    + a generalization of the Chernoff bound, only applied to Bernoulli random variable



## Statistical Inference

+ [statistically significance](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + a determination by an analyst that the results in the data are not explainable by chance alone
  + the likelihood that a relationship btw two or more variables caused by something other than chance
  + used to provide evidence concerning the plausibility of the null hypothesis, which hypothesizes that there is nothing more than random chance at work in the data
  + a $p$-value of 5% or lower often considered to be statistically significant

+ [statistical hypothesis testing](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the method by which the analyst makes this determination
  + an act in statistics whereby an analyst tests an assumption regarding a population parameter
  + used to assess the plausibility of a hypothesis by using sample data

+ [$p$-value](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + the probability of observing results as extreme as those in the data, assuming the results are truly due to chance alone
  + the probability of obtaining results as extreme as the observed results of a statistical hypothesis test, assuming that the null hypothesis is correct
  + used as an alternative to rejection points to provide the smallest level of significance at which the null hypothesis would be rejected
  + smaller p-value $\implies$ stronger evidence in favor of the alternative hypothesis


## Analysis Methodologies

+ [meta-analysi](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + a statistical analysis that combines the results of multiple scientific studies
  + performed when multiple scientific studies address the same question w/ each study reporting measurements expected some degree of error
  + derived a pooled estimate closest to the unknown common true based on how this error is perceived

+ [prospective analysis](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + used as an analytical tool to assess and mitigate the occurrence of loss by analyzing a situation or process that carries with it some inherent risk
  + to identify the way in which a process might potentially fail, w/ the goal to eliminate or reduce the likelihood or outcome severity of such a failure
  + applied to process or equipment and systems
  + FEMA used proactively when designing a new system or process for a high-risk or complex process or during an inter-professional process w/ hands-off and interdependent steps
  + w/ its roots in the engineering industry



## Multiple Distributions

+ [Joint probability distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + joint probability: the probability of two events occurring simultaneously
  + a probability distribution giving the probability that each $X, Y, \dots$ falls in any particular range or discrete set of values specified for that variable
  + $f_{X, Y}(x, y)$: the joint probability density function of random variable $X$ and $Y$, the marginal probability density function of $X$ and $Y$

    \[ f_X(x) = \int f_{XY} (x, y) dy, \qquad f_Y(y) = \int f_{XY} (x, y) dx \]

+ [Marginal distribution](../Notes/p01-Bayesian.md#31-subjectivity-and-context)
  + marginal probability: the probability of an event irrespective of the outcome of another variable
  + the marginal distribution of a subset of a collection of random variables is the probability distribution of the variables contained in the subset
  + two random variables independent $\iff$ their joint distribution function equal to the product of their marginal distribution functions
  + marginal probability density function: two continuous random variables $X$ and $Y$ w/ $x \in [a, b]$ and $y \in [c, d]$

    \[ f_X(x) = \int_c^d f(x, y) dy, \qquad f_Y(y) = \int_a^b f(x, y) dx \]


## Hypothesis Test

+ [Neyman-Pearson lemma
  + performing a hypothesis test btw two simple hypotheses, $H_0: \theta = \theta_0$ and  $H_1: \theta = \theta_1$
  + using the likelihood ratio test  w/ threshold $\eta$
  + rejecting $H_0$ in favor of $H_1$ at a significance level of

    \[ \alpha = P(\Lambda(x) \leq \eta | H_0) \]

    + $\Lambda(x) = \frac{\mathcal{L}(\theta_0 | x)}{\mathcal{L}(\theta_1 | x)}$
    + $\mathcal{L}(\theta | x)$: the likelihood function
  + the Neyman-Pearson lemma: the likelihood ratio, $\Lambda(x)$, is the __most powerful test__ at significance level $\alpha$
  + Properties
    + the test is most powerful for $\theta_1 \in \Theta_1 \implies$ test as uniformly most powerful (UMP) for alternatives in the set $\Theta_1$
    + the likelihood ratio: used directly to construct tests
  + Example:
    + $X_1, X_2, \dots, X_n$: a random sample from $N(\mu, \sigma^2)$
    + test: $H_0: \sigma^2 = \sigma_0^2$ against $H_1: \sigma^2 = \sigma_1^2$
    + the likelihood for this set of normal distributed data

      \[ \mathcal{L}(\sigma^2 | x) \propto (\sigma^2)^{-n/2} \exp \left( - \frac{\sum_{i=1}^n (x_i - \mu)^2}{2\sigma^2} \right) \]

    + the likelihood ratio

      \[ \Lambda(x) = \frac{\mathcal{L}(\sigma_0^2 | x)}{\mathcal{L}(\sigma_1^2 | x)} = \left( \frac{\sigma_0}{\sigma_1} \right)^{-n/2} \exp \left( -\frac{1}{2}(\sigma_0^2 - \sigma_1^2) \right) \sum_{i=1}^n (x_i - \mu)^2 \]

    + the ratio only depends on the data through $\sum_{i=1}^n (x_i - \mu)^2$
    + by Heyman-Pearson lemma, the most powerful test for this data only depends on $\sum_{i=1}^n (x_i - \mu)^2$
    + $\sigma_1^2 > \sigma_0^2 \implies \Lambda(x)$ a decreasing function of $\sum_{i=1}^n (x_i - \mu)^2$
    + reject $H_0$ if $\sum_{i=1}^n (x_i - \mu)^2$ is sufficient large
    + the rejection threshold depending on the size of the test
    + $\therefore$ test statistic w/ a scaled $\chi^2$ distributed random variable $\implies$ obtaining an exact critical value $\eta$



