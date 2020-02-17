# An Overview of the Bayesian Approach

Book title: [Bayesian Approaches to Clinical Trials and Health-Care Evaluation](http://93.174.95.29/main/791000/1c3cccffb374be94e8940aa087c433c0/%28Statistic%20in%20practice%29%20David%20J.%20Spiegelhalter%2C%20Keith%20R.%20Abrams%2C%20Jonathan%20P.%20Myles%20-%20Bayesian%20Approaches%20to%20Clinical%20Trials%20and%20Health-Care%20Evaluation-Wiley%20%282004%29.pdf)

Author: David J. Spiegelhalter, Keith R. Abrams, Jonathan P. Myles

Series: Statistic in practice

Publisher: Wiley

Year: 2004

[Chapter 3](http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf)


## 3.1 Subjectivity and Context

+ Terminology
  + __the prior distribution__: a reasonable opinion concerning the plausibility of different values of the treatment effect _excluding_ the evidence from the trail
  + __likelihood__: the support  for different values of the treatment effect based _solely_ on data from the trial
  + __the posterior distribution__: a final opinion about the treatment effect
  + frequency interpretation of probability: long-run properties of repeated random events
  + frequentist: standard statistical methods

+ The Bayesian approach
  + resting on an essentially _subjective_ interpretation of probability
  + allowed to express generic uncertainty or _degree of belief_ about any unknown but potentially observable quantity
  + rules of probability
    + Lindley, D. V. (2000) [The philosophy of statistics (with discussion)](https://www.phil.vt.edu/dmayo/personal_website/Lindley_Philosophy_of_Statistics.pdf). The Statistician, 49, 293–337.
    + not assumed as self-evident
    + able to derived from 'deeper' axioms of reasonable behavior of an individual
  + probabilities _for_ events rather than probabilities _of_ events
  + the probability is a reflection of personal uncertainty ranther than necessarily being based on future unknown events illustrated by a gambling game

+ Bayesian statistics
  + Berger, J. and Berry, D. A. (1988) [Statistical analysis and the illusion of objectivity](http://ifmlab.for.unb.ca/people/kershaw/Courses/Research_Methods/Readings/BergerJO1988a.pdf). American Scientist, 76, 159–65.
  + treating subjectivity with respect by placing it in the open and under the control of the consumer of data


## 3.2 Bayes theorem for two hypotheses

+ Prior to posterior analysis
  + hypotheses $H_0$ and $H_1$: mutually exhaustive and exclusive
  + the prior probability for each of two hypotheses: $p(H_0)$ and $p(H_1)$
  + $y$: the result of a test
  + $p(y | H_0)$ and $p(y | H_1)$
    + the probability of observing $y$ under each of the two hypotheses
    + the _likelihoods_
  + posterior probabilities:

    \[ p(H_0 | y) = \frac{p(y | H_0)}{p(y)} \times p(H_0)  \tag{1} \]

    + the overall probability of $y$ occuring: $p(y) = p(y | H_0) p(H_0) + p(y | H_1) p(H_1)$
  + Bayes theorem

    \[ \frac{p(H_0 | y)}{p(H_1 | y)} = \frac{p(y | H_0)}{p(y | H_1)} \times \frac{p(H_0)}{p(H_1)} \tag{2} \]

  + the prior odds: $p(H_0)/p(H_1)$
  + the posterior odds: $p(H_0 | y) / p(H_1 | y)$
  + the ratio of the likelihood:

    \[\begin{align*}
      \text{posterior odds} &= \text{likelihood ratio} \times \text{prior odds} \\
      \log(\text{posterior odds}) &= \log(\text{likelihood ratio}) + \log(\text{prior odds})
    \end{align*}\]

  + the weight of evidence: $\log(\text{posterior odds})$

+ Example 1: Diagnosis: Bays theorem in diagnostic testing
  + Assumption:
    + a new home HIV test
    + %95\%$ sensitivity
    + $98\%$ specificity
    + used in a population w/ an HIV prevalence of $1/1000$
  + Expected status of 100,000 tested individuals in a population w/ an HIV prevalence of $1/1000$

    |     | HIV- | HIV+ |   |
    |-----|-----:|-----:|--:|
    | Test - | 87,902 | 5 | 97,907 |
    | Test + | 1, 998 | 95 | 2,093 |
    |Total   | 99,900 | 100 | 100,000 |

  + $H_0$: the hypothesis that individual is truly HIV positive
  + $H_1$: the hypothesis truly HIV negative
  + $y$: the observation tested positive
  + the prior probability of the disease prevalence: $p(H_0) = 0.001$
  + the posterior probability $p(H_0 | y)$: the chance that someone who tests positive is truly HIV positive
  + Analysis
    + $95\%$ sensitivity: $p(y | H_0) = 0.95$
    + $98\%$ specificity: $p(y | H_1) = 0.02$
    + the prior odds: $p(H_0)/p(H_1) = 1/999$
    + the likelihood ratio: $p(y | H_0) / p(y | H_1) = 0.95/0.02 = 95/2$
    + the posterior odds: $p(H_0 | y) / p(H_1 | y) = (95/2) \times (1/999) = 95/1998$
    + the posterior probability: $p(H_0 | y) = 95/(95 + 1998) = 0.045$
  + Bayes theorem (Eq. (1))

    \[\begin{align*}
      p(y ) & = p(y | H_0) p(H_0) | p(y | H_1) p(H_1) = 0.95 \cdot 0.001 + 0.02 \cdot 0.999 = 0.02093 \\
      p(H_o | y) &= 0.95 \cdot 0.001 / 0.02093 = 0.045
    \end{align*}\]

  + Ans: over $95\%$ pf those testing positive will not have HIV
  + Bayes theorem for two hypotheses $H_0$ and $H_1$
    + by specifying the prior probability or odds, and likelihood ratio $p(y|H_0)/p(y|H_1)$, the posterior probability or odds can be read off the graphs
    + using the logarithmic scaling, under which Bayes theorem gives a linear relationship (fig.(b))

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf" ismap target="_blank">
        <img src="img/p01-01a.png" style="margin: 0.1em;" alt="Bayes theorem for two hypotheses H0 and H1 in probability p(H0) form" title="Bayes theorem for two hypotheses H0 and H1 in probability p(H0) form" height=200>
        <img src="img/p01-01b.png" style="margin: 0.1em;" alt="Bayes theorem for two hypotheses H0 and H1 in odd p(H0)/p(H1) form" title="Bayes theorem for two hypotheses H0 and H1 in odd p(H0)/p(H1) form" height=200>
      </a>
    </div>



## 3.3 Comparing simple hypotheses: likelihood ratios and Bayes factors 54







## 3.4 Exchangeability and parametric modelling* 56







## 3.5 Bayes theorem for general quantities 57







## 3.6 Bayesian analysis with binary data 57







### 3.6.1 Binary data with a discrete prior distribution 58







### 3.6.2 Conjugate analysis for binary data 59







## 3.7 Bayesian analysis with normal distributions 62







## 3.8 Point estimation, interval estimation and interval hypotheses 64







## 3.9 The prior distribution 73







## 3.10 How to use Bayes theorem to interpret trial results 74







## 3.11 The ‘credibility’ of significant trial results* 75







## 3.12 Sequential use of Bayes theorem* 79







## 3.13 Predictions 80







### 3.13.1 Predictions in the Bayesian framework 80







### 3.13.2 Predictions for binary data* 81







### 3.13.3 Predictions for normal data 83







## 3.14 Decision-making 85







## 3.15 Design 90







## 3.16 Use of historical data 90







## 3.17 Multiplicity, exchangeability and hierarchical models 91







## 3.18 Dealing with nuisance parameters* 100







### 3.18.1 Alternative methods for eliminating nuisance parameters* 100







### 3.18.2 Profile likelihood in a hierarchical model* 102







## 3.19 Computational issues 102







### 3.19.1 Monte Carlo methods 103







### 3.19.2 Markov chain Monte Carlo methods 105







### 3.19.3 WinBUGS 107







## 3.20 Schools of Bayesians 112







## 3.21 A Bayesian checklist 113







## 3.22 Further reading 115







## 3.23 Key points 116







## Exercises






