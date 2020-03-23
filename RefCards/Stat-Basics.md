# Statistics: Basics

## Basic Concepts

+ Terminology
  + standard deviation ($sd$): a measure of the amount of variation or dispersion of a set of values ($var = sd^2$) $\implies$ true population mean
  + standard error ($se$)
    + the standard deviation ($sd$) of its sampling distribution or an estimate of that standard deviation
    + not enough samples & different trials $\to$ different population means
  + sampling distribution: the probability distribution of a given random-sample-based statistic
  + frequency interpretation of probability: long-run properties of repeated random events
  + frequentist:
    + standard statistical methods
    + $p(x)$: the proportion of times $x$ will occur in an infinitely long series of repeated identical situations

+ odds ($O$):
  + the probability ($p$) that the event will occur divided by the probability ($1 - p$) that the event will not occur
  + used to describe the chance of an event occurring

  \[ O = \frac{p}{1 - p} \tag{Odds} \]

+ logit: the natural logarithm of the odds

  \[ \text{logit}(p) = \ln(\frac{p}{1 - p}) \tag{Odds.log} \]

+ Bayes theorem
  + Formula

  \[ p(b|a) = \frac{p(a|b)}{p(a)} \times p(b) \tag{Bayes} \]

  + The odds form of Bayes theorem

    \[ \frac{p(b|a)}{p(\overline{b}|a)} = \frac{p(a|b)}{p(a| \overline{b})} \times \frac{p(b)}{p(\overline{b})} \tag{Bayes.odds} \]


## Statistical Inference

+ [statistically significance](https://www.investopedia.com/terms/s/statistically_significant.asp)
  + a determination by an analyst that the results in the data are not explainable by chance alone
  + the likelihood that a relationship btw two or more variables caused by something other than chance
  + used to provide evidence concerning the plausibility of the null hypothesis, which hypothesizes that there is nothing more than random chance at work in the data
  + a $p$-value of 5% or lower often considered to be statistically significant

+ [statistical hypothesis testing](https://www.investopedia.com/terms/h/hypothesistesting.asp)
  + the method by which the analyst makes this determination
  + an act in statistics whereby an analyst tests an assumption regarding a population parameter
  + used to assess the plausibility of a hypothesis by using sample data

+ [$p$-value](https://www.investopedia.com/terms/p/p-value.asp)
  + the probability of observing results as extreme as those in the data, assuming the results are truly due to chance alone
  + the probability of obtaining results as extreme as the observed results of a statistical hypothesis test, assuming that the null hypothesis is correct
  + used as an alternative to rejection points to provide the smallest level of significance at which the null hypothesis would be rejected
  + smaller p-value $\implies$ stronger evidence in favor of the alternative hypothesis


## Analysis Methodologies

+ [meta-analysis](https://en.wikipedia.org/wiki/Meta-analysis)
  + a statistical analysis that combines the results of multiple scientific studies
  + performed when multiple scientific studies address the same question w/ each study reporting measurements expected some degree of error
  + derived a pooled estimate closest to the unknown common true based on how this error is perceived

+ [prospective analysis](https://www.longwoods.com/content/20972/healthcare-quarterly/looking-ahead-the-use-of-prospective-analysis-to-improve-the-quality-and-safety-of-care)
  + used as an analytical tool to assess and mitigate the occurrence of loss by analyzing a situation or process that carries with it some inherent risk
  + to identify the way in which a process might potentially fail, w/ the goal to eliminate or reduce the likelihood or outcome severity of such a failure
  + applied to process or equipment and systems
  + FEMA used proactively when designing a new system or process for a high-risk or complex process or during an inter-professional process w/ hands-off and interdependent steps
  + w/ its roots in the engineering industry



## Multiple Distributions

+ Joint probability distribution
  + joint probability: the probability of two events occurring simultaneously
  + a probability distribution giving the probability that each $X, Y, \dots$ falls in any particular range or discrete set of values specified for that variable
  + $f_{X, Y}(x, y)$: the joint probability density function of random variable $X$ and $Y$, the marginal probability density function of $X$ and $Y$

    \[ f_X(x) = \int f_{XY} (x, y) dy, \qquad f_Y(y) = \int f_{XY} (x, y) dx \]

+ Marginal distribution
  + marginal probability: the probability of an event irrespective of the outcome of another variable
  + the marginal distribution of a subset of a collection of random variables is the probability distribution of the variables contained in the subset
  + two random variables independent $\iff$ their joint distribution function equal to the product of their marginal distribution functions
  + marginal probability density function: two continuous random variables $X$ and $Y$ w/ $x \in [a, b]$ and $y \in [c, d]$

    \[ f_X(x) = \int_c^d f(x, y) dy, \qquad f_Y(y) = \int_a^b f(x, y) dx \]



