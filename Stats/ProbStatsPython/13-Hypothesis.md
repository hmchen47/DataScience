# Topic13: Hypothesis Testing


## 13.1 Hypothesis Test - Introduction

### Lecture Notes

+ Hypotheses
  + assumptions (statements) about <font style="color: magenta; font-wight: bold;">parameters</font> of
    + distribution
    + population
  + examples
    + a coin is biased
    + average student GPA < 3.0
    + amazon average delivery time > 2 days
    + people tweet more on weekend
    + men play more video games than women on average

+ Hypotheses types
  + simple
    + parameters taking a single specific value
    + e.g. $\mu = 2.5, \sigma = 2.4$
  + composite
    + parameter taking one of several values
    + e.g., $\mu \in \{4.5, 6.3\}, \mu > \sigma, \sigma \in \{4.5, 6.3\}$
  + one-sided: $\mu \le 2.3, \mu > 4.5$
  + two-sided: $\mu \le 2.3 \text{ or } > 4.5, \mu < 2.3 \text{ or } > 2.3, \mu \ne 2.3$

+ Null and Alternative Hypotheses
  + null hypothesis
    + more often
    + assumption believed to be true
    + Status quo
    + notation: $H_0$
  + alternative hypothesis
    + complementary view
    + research
    + notation: $H_1$ or $H_A$
  + $H_A$ often complement or "one-side complement" of $H_0$

+ Simple $H_0$
  + coin:
    + null hypothesis: unbiased $\to p_h = 0.5$
    + alternative hypothesis
      + 2-sided: biased $\to p_h \ne 0.5$
      + 1-sided: heads more likely $\to p_h > 0.5$
  + gender equality average GPS
    + null hypothesis: same average GPA (not exactly simple: {(x, x)})
    + alternative hypothesis
      + 2-sided: different average GPS
      + 1-sided: men's average GPS higher

+ Example: one-sided $H_0$
  + smartphones iOS x Android
    + null hypothesis: $\ge 50\%$ using iOS
    + alternative hypothesis: < 60% of phones use iOS
  + checkout self x cashier
    + null hypothesis: self checkout faster (not exactly one sided: {(x, y): x < y})
    + alternative hypothesis: self checkout slower

+ How to test
  + design experiment
  + gather data
  + data consistent w/ null hypothesis?
    + no: reject null in favor of alternative
    + yes: do not reject null
  + equivalently, strong evidence for alternative hypothesis?
    + yes: reject null in favor of alternative
    + no: do not reject null
  + conservative
    + reject bull (status quo)
    + only if stronger evidence against it
    + two analogies

+ Test vs. trial
  + hypothesis test: strong evidence for alternative hypothesis?
    + yes: reject null in favor of alternative
    + no: do not reject null
  + legal trial: strong incriminating evidence? presumed innocence
    + guilty: innocent = null
    + not guilty: reject only by strong evidence

+ Test vs. Myth
  + hypothesis test: strong evidence for alternative hypothesis?
    + yes: reject null in favor of alternative
    + no: do not reject null
  + myth: strong evidence for myth?
    + yes: accept
    + no: keep default belief

+ Testing hypotheses
  + test: design experiment
  + test statistic
    + define numerical outcome, $T$
    + related to hypothesis
  + $P_{H_0}(T=t)$: determine distribution of $T$ under $H_0$
  + observe data: calculate value $t$ of the test statistic $T$
  + $P_{H_0}(t)$ value:
    + large $t$ toward $H_0$
      + $H_0$ consistent w/ data
      + do not reject $H_0$
      + accept $H_0$
    + small $t$ toward $H_A$
      + $H_0$ inconsistent w/ data
      + reject $H_0$ in favor of $H_A$
      + intuitive $\to$ formal

+ Example: biased coin, 1-sided $H_A$
  + simple null:
    + unbiased
    + $H_0: p_h = 0.5$
  + 1-sided alternative:
    + biased towards heads
    + $H_A: p_h > 0.5$
  + test statistic
    + $X$ as number of heads
    + test 20 times
  + intuitive: $X \ge 16$ (left diagram)
    + unlikely under $H_0$
    + more likely under $H_A$
    + reject null in favor of $H_A$

+ Example: biased coin, 2-sided $H_A$
  + simple null:
    + unbiased
    + $H_0: p_h = 0.5$
  + 2-sided alternative hypothesis
    + biased
    + $H_A: p_h \ne 0.5$
  + test statistics
    + $X$: number of heads
    + 20 times
  + intuitive (middle diagram)
    + $5 \le X \le 15$
      + do not reject $H_0$
      + accept $H_0$
    + otherwise: reject $H_0$ in favor of $H_A$

<div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
  <a href="https://tinyurl.com/y4k2be2r" ismap target="_blank">
    <img src="img/t13-01.png" style="margin: 0.1em;" alt="Hypothesis testing: 1-sided alternative hypothesis" title="Hypothesis testing: 1-sided alternative hypothesis" height=200>
    <img src="img/t13-02.png" style="margin: 0.1em;" alt="Hypothesis testing: 2-sided alternative hypothesis" title="Hypothesis testing: 2-sided alternative hypothesis" height=200>
  </a>
  <a href="https://tinyurl.com/y2jyy58c" ismap target="_blank">
    <img src="https://tinyurl.com/yxvv7trj" style="margin: 0.1em;" alt="Summary of the Different Tests. Note that in a one-tailed test, when H1 involves values that are greater than μ_X, we have a right-tail test. Similarly, when H1 involves values that are less than μ_X, we have a left-tail test. For example, an alternative hypothesis of the type H1 : μ_X > 100 is a right-tail test while an alternative hypothesis of the type H1 : μ_X < 100 is a left-tail test." title="Summary of the Different Tests" height=200>
  </a>
</div>

+ [Original Slides](https://tinyurl.com/y4k2be2r)


### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 13.2 Hypothesis Testing - p-Values

### Lecture Notes






+ [Original Slides]()


### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 13.3 Lady Tasting Tea

### Lecture Notes






+ [Original Slides]()


### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 13.4 Hypothesis Testing - Z and T Tests

### Lecture Notes






+ [Original Slides]()


### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 13







## Programming Assignment 13







