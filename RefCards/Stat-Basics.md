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


