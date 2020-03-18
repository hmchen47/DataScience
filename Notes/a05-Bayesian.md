# Bayesians and frequentists

[Original](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1114120/)

Author: J. Martin Bland and Douglas G. Altman


## Notes

+ Two competing philosophies of statistical analysis
  + the frequentist: almost all the statistical analyses
  + the Bayesian:
    + only snipe at the frequentists from the high ground of university departments of mathematical statistics
    + power of computers bringing to the fore

+ Bayesian methods
  + based on the idea of unknown quantities w/ probability distributions
  + unknown quantities including population means and proportions
  + prior knowledge / belief
    + the probability distribution proportion
    + the knowledge from data before knowing it
  
+ Example: prevalence of diabetes in a health district
  + task: to estimate the prevalence of diabetes in a health district
  + prior knowledge: the percentage of diabetics in the United Kingdom as a whole is about 2%
  + hypothesis: expect the prevalence in our health district to be fairly similar
  + confidence interval of prior distribution: information based on other datasets that such rates vary between 1% and 3%
  + construct a prior distribution which summarizes our beliefs about the prevalence in the absence of specific data
  + prior distribution: $\mu = 2$ and $\sigma = 0.5 \implies 2 \pm 2 \times 0.5$
  + the likelihood and posterior distribution
    + collecting some data by a sample survey of the district population
    + using the data to modify the prior probability distribution $\to$ posterior distribution
    + e.g., survey of 1000 subjects
      + 15 (1.5%) to be diabetic
      + posterior distribution: $\mu = 1.8\%, \sigma = 0.3\% \implies 95\%\,CI = (1.2\%, 2.4\%)$
    + the Bayesian method: an estimate nearer the prior mean and narrower interval
  + The frequentist analysis
    + ignore the prior information
    + final distribution: $\mu = 1.5\%, \sigma = 0.4\% \implies 95\%\, CI = (0.8\%, 2.5\%)$
    + similar to the results of the Bayesian method

+ Frequentist vs Bayesian methods
  + frequentist methods
    + the population value: fixed, unvarying (but unknown) quantity
    + w/o a probability distribution
    + calculating confidence intervals for the quantity or significance tests of hypotheses concerning it
  + Bayesian methods
    + not allowing to widen knowledge of the problem
    + not providing what researchers seem to want
    + able to provide
      + the probability of the 95% of the population value lies within the 95% CI
      + the probability of truth of the null hypothesis less than 5% $\to p(H_0 = \text{true}) < 5\%$ 

+ Issues of the Bayesian methods
  + how to decide on the prior distribution
    + a subjective synthesis of the available information
    + same data analyzed by different investigators leading to different conclusions
  + intractable computational problems
  + choice of Bayesian or frequentist: unknown which existed

+ Colutions for computational issue
  + computer intensive methods of analysis developed
  + new approaches to very difficult statistical problems, such as the location of geographical clusters of cases of a disease
  + a change in the statistical paradigm


