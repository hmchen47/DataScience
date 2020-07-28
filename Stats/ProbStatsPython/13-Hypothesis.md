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

+ Example: simple $H_0$
  + coin:
    + null hypothesis: unbiased $\to p_h = 0.5$
    + alternative hypothesis
      + 2-sided: biased $\to p_h \ne 0.5$
      + 1-sided: heads more likely $\to p_h > 0.5$
  + gender equality average GPA
    + null hypothesis: same average GPA (not exactly simple: {(x, x)})
    + alternative hypothesis
      + 2-sided: different average GPA
      + 1-sided: men's average GPA higher

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
    + reject null (status quo)
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

+ [Test statistic](https://tinyurl.com/vkvx9b8)
  + a random variable that is calculated from sample data and used in a hypothesis test
  + determining whether to reject the null hypothesis
  + comparing data with what is expected under the null hypothesis
  + used to calculate the p-value
  + measuring the degree of agreement between a sample of data and the null hypothesis
  + containing information about the data relevant for deciding whether to reject the null hypothesis
  + <font style="color: cyan;">null distribution</font>: the sampling distribution of the test statistic under the null hypothesis
  + When the data show strong evidence against the assumptions in the null hypothesis, the magnitude of the test statistic becomes too large or too small depending on the alternative hypothesis.
  + the test's p-value small enough to reject the null hypothesis
  + Z-statistic
    + the test statistic for a Z-test
    + the standard normal distribution under the null hypothesis
  + Different hypothesis tests using different test statistics based on the probability model assumed in the null hypothesis (hypothesis test $\to$ test statistic)
    + z-test $\to$ z-statistic
    + t-tests $\to$ t-statistic
    + ANOVA $\to$ F-statistic
    + $\chi^2$-tests $\to \chi^2$ statistic

+ Critical value
  + a point on the test distribution compared to the test statistic to determine whether to reject the null hypothesis
  + the absolute value of test statistic is greater than the critical value $\implies$ statistical significance and reject the null hypothesis
  + corresponding to $\alpha$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/tq5prhd" ismap target="_blank">
      <img src="https://tinyurl.com/y4qu4g4s" style="margin: 0.1em;" alt="Figure A shows that results of a one-tailed Z-test are significant if the test statistic is equal to or greater than 1.64, the critical value in this case. The shaded area is 5% (α) of the area under the curve." title="Figure A shows that results of a one-tailed Z-test are significant if the test statistic is equal to or greater than 1.64, the critical value in this case. The shaded area is 5% (α) of the area under the curve." height=150>
      <img src="https://tinyurl.com/y2gyrtne" style="margin: 0.1em;" alt="Figure B shows that results of a two-tailed Z-test are significant if the absolute value of the test statistic is equal to or greater than 1.96, the critical value in this case. The two shaded areas sum to 5% (α) of the area under the curve." title="Figure B shows that results of a two-tailed Z-test are significant if the absolute value of the test statistic is equal to or greater than 1.96, the critical value in this case. The two shaded areas sum to 5% (α) of the area under the curve." height=150>
    </a>
  </div>


+ [Original Slides](https://tinyurl.com/y4k2be2r)


### Problem Sets

0. If we fail to reject the null hypothesis, does it mean that the null hypothesis is correct?<br/>
  a. Yes, it must be correct.<br/>
  b. No, we just don't have enough evidence to reject it.<br/>

  Ans: b


1. The distribution of the test statistic T depends on<br/>
  a. Null hypothesis $H_0$,<br/>
  b. Alternative hypothesis $H_A$,<br/>
  c. Observed data t,<br/>
  d. None of above.<br/>

  Ans: a


2. The null hypothesis says that  Z  follows normal distribution $N(0,\sigma^2)$. If the null hypothesis is correct, which of the following is the most unlikely event?<br/>
  a. $Z \in [−\sigma, \sigma]$<br/>
  b. $Z \ni [−2\sigma,2\sigma]$<br/>
  c. $Z \ge \sigma$<br/>

  Ans: <font style="color: cyan;">b</font><br/>
  Explanation: By the 68-95-99.7 Rule:
    + $\Pr(Z \in [−\sigma,\sigma]) \approx 68\%$
    + $\Pr(Z \ni [−2\sigma,2\sigma]) \approx 100−95=5\%$
    + $\Pr(Z \ge \sigma) \approx (100−68)/2=16\%$,

  hence the second is smallest.


### Lecture Video

<a href="https://tinyurl.com/y3f3np38" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 13.2 Hypothesis Testing - p-Values

### Lecture Notes

+ Example: biased coin
  + $\underbrace{H_0: p_h = 0.5}_{\text{Unbiased}} \quad \underbrace{H_A: p_h > 0.5}_{\substack{\text{Heads more}\\\text{likely}}}$
  + data: 20 trials
  + test statistic: $X$ as the number of heads
  + Type-I error
    + $H_0$: coin unbiased
    + declare $H_A$: heads more likely
  + under $H_0$: $X \sim B_{0.5, 20} \to P_{H_0}$ (see diagram)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-10.png" style="margin: 0.1em;" alt="pmf of Bernoulli distribution w/ n=20, p = 0.5" title="pmf of Bernoulli distribution w/ n=20, p = 0.5" width=250>
    </a>
  </div>

+ Nomencalture
  + complicated work for terminology, e.g., appropos = appropriate
  + $H_A$ is true, allowed statements
    + reject $H_0$ in favor of $H_A$
    + reject $H_0$
    + <font style="color: cyan;">accept $H_A$</font>
  + $H_0$ is true, allowed statements
    + do not reject $H_0$
    + data not significant
    + $H_0$ plausible
    + <font style="text-decoration: line-through;">accept $H_0$</font>
    + (reluctantly) <font style="color: cyan;">retain $H_0$</font> 
  + why verbal gymnastics?
    + test asymmetric
  + example
    + 20 trials w/ $X= 12 \to$ accept $H_0$?
    + what if $p_H$ = 0.6 or 0.55?
    + better explain 12 than $p_h = 0.5$
    + not knowing that $H_0$ is true
    + just not enough data to reject it

+ Significant level
  + reject null (status quo) hypothesis $H_0$ only if strong evidence for alternative $H_A$
  + precise probabilistic formulation
  + $\alpha$: significant level, typically 5%, 1%
  + if $H_0$ is true, accept $H_A$ w/ probability $\le \alpha$
  
    \[ P_{H_0}(\underbrace{\text{accept} H_A}_{\text{Type-I Error}}) \le \alpha \]

  + two methods
    + critical values
    + p-values

+ Critical value
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h > 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + $H_0 \to X \approx 10$
  + accept $H_A$ when $X = 16, 17, 18, 19, 20$
  + critical value: $x_\alpha$ as a threshold
    + $X \ge x_\alpha \to$ accept $H_A$
    + $X < x_\alpha \to$ retain $H_A$
  + $x_\alpha \gets$ significance level $\alpha$
  + $\alpha$: upper bound on $P_{H_0}(\text{accept } H_A)$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-03.png" style="margin: 0.1em;" alt="Bernoulli distribution w/ p=0.5, n=20 as x_α = 16 for significance level α = 5%" title="Bernoulli distribution w/ p=0.5, n=20 as x_α = 16 for significance level α = 5%" width=200>
    </a>
  </div>

+ Finding $x_\alpha$
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h > 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + critical value $x_\alpha$
    + $X < x_\alpha \to$ retain $H_0$
    + $X \ge x_\alpha \to$ accept $H_A$
  + significance level $\alpha$
    + typical $\alpha = 5\%, 1\%$
    + $P_{H_0} (X \ge x\alpha) = P_{H_0}(\text{falsely accept } H_A) \le \alpha$
  + requirement: $P_{H_0}( X \ge x_\alpha) \le \alpha$
    + $x_\alpha$ large: almost never declare $H_A$
    + smallest $x$ s.t. $P_{H_0}(X \ge x) \le \alpha$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-11.png" style="margin: 0.1em;" alt="Bernoulli distribution w/ p=0.5, n=20 as x_α = 16 for significance level α = 5% and the area of probability" title="Bernoulli distribution w/ p=0.5, n=20 as x_α = 16 for significance level α = 5% and the area of probability" width=250>
    </a>
  </div>

+ Example: $X_{5\%}$ and $X_{1\%}$
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h > 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + significance level: $\alpha = 5\%$
  + critical value: $x_{5\%}$
  + finding smallest $x$ s.t. $P_{H_0}(X \ge x) \le 5\%$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-04.png" style="margin: 0.1em;" alt="Example of critical value calculation" title="Example of critical value calculation" width=650>
    </a>
  </div>

+ Room for improvement: critical value
  + significance level: $\alpha = 5\%$
  + critical value: $x_\alpha = 15$
  + testing
    + $X < x_\alpha$: retain $H_0$
    + $X \ge x_\alpha$: accept $H_A$
  + practically observe one value $x$ of $X$
  + $X = 13 \to$ retain $H_0$ or accept $H_A$?
    + get around finding smallest $x$ for accepting $H_A$? $\to$ critical value
    + find a rule just for $X$ itself? $\to$ p-value

+ p (probability) value
  + critical value recall
    + $x_{5\%} = 15$
    + finding smallest $x$ s.t. $P_{H_0}(X \ge x) \le 5\%$
    + $X \ge x_{x_{5\%}} \to$ accept $H_A$: $x \ge x_{5\%} \iff P_{H_0}(X \ge x) \le P_{H_0}(X \ge x_{5\%}) \le 5\%$
    + $X < x_{x_{5\%}} \to$ retain $H_0$: $x < x_{5\%} \iff P_{H_0}(X \ge x) > 5\%$
  + p values of $x$
    + accept $H_A$: $P_{H_0}(X \ge x) \le 5\% \to x \ge x_{5\%}$
    + retain $H_0$: $P_{H_0}(X > x) > 5\% \to x < x_{5\%}$
  + same $H_0$ and $H_A$ regions as in critical values
  + intuitively $P$ under $H_0$ small $\to H_A$ more likely

+ Example: p-value w/ 1-sided hypothesis testing
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h > 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + significance level: $\alpha = 5\%$
  + critical value testing
    + $x_{5\%} = 15$
    + $X \ge 15 \to H_A$
    + $X < 15 \to H_0$
  + p-value testing
    + $\le 5\%: x \ge x_\{5\%} \to $ accept $H_A$
    + $> 5\% : x < x_{5\%} to$ retain $H_0$
  + p-value of $x$ = Probability $X \ge x$
    + $\le \alpha$: accept $H_A$
    + $> \alpha$: retain $H_0$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-05.png" style="margin: 0.1em;" alt="Example of p-value calculation" title="Example of p-value calculation" width=350>
    </a>
  </div>

+ Example: critical value & p-value w/ 1-sided hypothesis testing in opposite alternative
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h {\color{Magenta}<}\, 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + significance level: $\alpha = 5\% \text{ or } 1\% \to P_{H_0}(\text{falsely accept } H_A) \le \alpha$
  + critical value $x_\alpha$ (refer to diagram in example for p-value)
    + <font style="color: magenta;">largest</font> $x$ s.t. $P_{H_0}(X {\color{magenta}\le}\, x) \le \alpha \xrightarrow{\text{snsures}} P_{H_0}(\text{falsely accept } H_A) \le \alpha$
    + largest $x \to$ left tail
  + $X \le x_\alpha$: accept $H_0$
  + $x > x_\alpha$: retain $H_0$
  + $\alpha = 5\% \xrightarrow{\text{by symmetry}} x_{5\%} = 5$

+ Example: p-value w/ 1-sided hypothesis testing in opposite alternative
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h {\color{Magenta}<}\, 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + significance level: $\alpha = 5\%$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-06.png" style="margin: 0.1em;" alt="Application of p-value" title="Application of p-value" width=650>
    </a>
  </div>

+ Example: two-sided alternative hypothesis
  + $\underbrace{H_0: p_h = 0.5}_{\text{unbiased}} \quad \underbrace{H_A: p_h {\color{Magenta}\ne}\, 0.5}_{\text{2-sided}}$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + $H_0$ mean: $\mu_x = 10$
  + probability under the area
    + $H_0$: $X$ close to 10
    + $H_A$: $X$ far from 10
  + $P(|X - 10| > x)$: two far ends
    + small: retain $H_0$
    + large: accept $H_A$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-07.png" style="margin: 0.1em;" alt="Example of 2-sided alternative hypothesis" title="Example of 2-sided alternative hypothesis" width=250>
    </a>
  </div>

+ Critical value for 2-sided $H_A$
  + hypotheses: $H_0 p_h = 0.5 \quad H_A: p_h {\color{Magenta}\ne}\, 0.5$
  + sampling data: 20 trials
  + test statistic: $X$ as number of heads
  + significance level $\alpha$
    + upper bound on Type-I error
    + $H_o$ mean: $\mu_x = 10$
  + critical value $x_\alpha$: $x$ close to 10 s.t. $P_{H-0}(|X - 10| \ge |x - 10|) \le \alpha$
  + testing
    + $|X - 10| \ge |x - 10|$: accept $H_A$
    + $|X - 10| < |x - 10|$: retain $H_0$
  
    \[ P_{H_0} (\text{type-I error}) = P_{H_0}(|X - 10| \ge |x_{\alpha} -10|) \le \alpha \]

  + $x_\alpha$ closest to 10 minimizes type-II error

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-08.png" style="margin: 0.1em;" alt="Example of critical value" title="Example of critical value" width=250>
    </a>
  </div>

+ p values for 2-sided $H_A$
  + p value of $x$
    + $P_{H_0}(X \text{ is at least as far from 10 as } x)$
    + $P_{H_0}(|X - 10| \ge |x - 10|)$
  + low p-value
    + $x$ far from mean
    + low $H_0$ prob of outcome $x$ or further towards $H_A$
    + $x$ less likely to be generated under $H_0$
  + higher p-value
    + $x$ far from mean
    + high $H_0$ prob of outcome $x$ or further towards $H_A$
    + $x$ more likely to be generated under $H_0$
  + p-values $\to$ hypothesis

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y4s4tdt5" ismap target="_blank">
      <img src="img/t13-09.png" style="margin: 0.1em;" alt="Critical value and hypothesis" title=4Critical value and hypothesis" width=450>
    </a>
  </div>

+ General p-value
  + p value of statistic $t$ of $T$: $P_{H_0}(T \text{ is $t$ or further towards } H_A)$
  + significance level: $\alpha$
  + p-value
    + $\le \alpha$: accep $H_A$
    + $> \alpha$: retain $H_0$


+ [Original Slides](https://tinyurl.com/y4s4tdt5)


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







