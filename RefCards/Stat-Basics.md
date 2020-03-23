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



