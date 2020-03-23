# Statistics: Help Cards

## Statistics Notation

<table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
  <thead>
  <tr style="font-size: 2em$">
    <th style="text-align: center; background-color: #3d64ff; color: #ffff00; width:15%;">Name</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffff00; width:10%;">Population Notation</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffff00; width:10%;">Sample Notation</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffff00; width:10%;">Python DataFrame</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffff00; width:10%;">Notation used in R</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <th colspan="5" style="text-align: center; background-color: #3d64cc; color: #ffffff; ">Summary Measures</th>
  </tr>
  <tr>
    <th style="text-align: left;">Mean</th>
    <td style="text-align: center;">$\mu$ (read as "mu")</td>
    <td style="text-align: center;">$\overline{x}$ (x-bar)</td>
    <td style="text-align: center;">df.mean()</td>
    <td style="text-align: center;">Mean</td>
  </tr>
  <tr>
    <th style="text-align: left;">Proportion</th>
    <td style="text-align: center;">$p$</td>
    <td style="text-align: center;">$\hat{p}$ (p-hat)</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <th style="text-align: left;">Standard deviation</th>
    <td style="text-align: center;">$\sigma$ (sigma)</td>
    <td style="text-align: center;">$s$</td>
    <td style="text-align: center;">df.std()</td>
    <td style="text-align: center;">Varies, often "sd"</td>
  </tr>
  <tr>
    <th style="text-align: left;">Variance</th>
    <td style="text-align: center;">$\sigma^2$</td>
    <td style="text-align: center;">$s^2$</td>
    <td style="text-align: center;">df.var()</td>
    <td style="text-align: center;">Variance</td>
  </tr>
  <tr>
    <th style="text-align: left;">Sample size</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$n$</td>
    <td style="text-align: center;">ser.size, df.index, df.shape</td>
    <td style="text-align: center;">$n$ (sometimes $N$</td>
  </tr>
  <tr>
    <th colspan="5" style="text-align: center; background-color: #3d64cc; color: #ffffff;">Confidence intervals</th>
  </tr>
  <tr>
    <th rowspan="2" style="text-align: left;">Multipliers</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$z^{\ast}$ (z-star)</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$t^{\ast}$ (t-star)</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <th style="text-align: left;">Margin of error</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">m, m.e.</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <th colspan="5" style="text-align: center; background-color: #3d64cc; color: #ffffff;">Hypothesis Testing</th>
  </tr>
  <tr>
    <th rowspan="4" style="text-align: left;">Test statistic<br/><span style="font-size: 0.8em; font-wight: 100;">Note: $t$, $F$, and $\chi^2$ statistics have degrees of freedom ($df$) associated with them</span></th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$z$</td>
    <td style="text-align: center;"><a href="https://stackoverflow.com/questions/45949160/finding-z-scores-of-data-in-a-test-dataframe-in-pandas">Example1</a>,<a href="https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce">Example2</a></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$t$</td>
    <td style="text-align: center;"><a href="https://stackoverflow.com/questions/13404468/t-test-in-pandas">Example1</a>,<a href="https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce">Example2</a></td>
    <td style="text-align: center;">$t$</td>
  </tr>
  <tr>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$F$</td>
    <td style="text-align: center;"><a href="https://towardsdatascience.com/fisher-test-for-regression-analysis-1e1687867259">Example1</a></td>
    <td style="text-align: center;">$F$</td>
  </tr>
  <tr>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$\chi^2$ (chi-square)</td>
    <td style="text-align: center;"><a href="https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.chisquare.html">API</a>,<a href="https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce">Example</a></td>
    <td style="text-align: center;">Chi-square</td>
  </tr>
  <tr>
    <th style="text-align: left;">Significance level</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$\alpha$ (alpha)</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <th style="text-align: left;">$p$-value</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$p$-value</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">Pr(*) (* depends on what test used)</td>
  </tr>
  <tr>
    <th colspan="5" style="text-align: center; background-color: #3d64cc; color: #ffffff;">Analysis of Variance (ANOVA)</th>
  </tr>
  <tr>
    <th style="text-align: left;">Sum of squares for groups</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">SSG</td>
    <td style="text-align: center;"><a href="https://pythonfordatascience.org/anova-python/">Example</a></td>
    <td style="text-align: center;">Row: grouping<br/>Col: Sum Sq</td>
  </tr>
  <tr>
    <th style="text-align: left;">Sum of squares for error</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">SSE</td>
    <td style="text-align: center;"><a href="https://pythonfordatascience.org/anova-python/">Example</a></td>
    <td style="text-align: center;">Row: Residuals<br/>Col: Sum Sq</td>
  </tr>
  <tr>
    <th style="text-align: left;">Mean square for groups</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">MSG</td>
    <td style="text-align: center;"><a href="https://pythonfordatascience.org/anova-python/">Example</a></td>
    <td style="text-align: center;">Row: grouping<br/>Col: Mean Sq</td>
  </tr>
  <tr>
    <th style="text-align: left;">Mean square error</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">MSE</td>
    <td style="text-align: center;"><a href="https://pythonfordatascience.org/anova-python/">Example</a></td>
    <td style="text-align: center;">Row: Residuals<br/>Col: Mean Sq</td>
  </tr>
  <tr>
    <th colspan="5" style="text-align: center; background-color: #3d64cc; color: #ffffff;">Regression</th>
  </tr>
  <tr>
    <th style="text-align: left;">Response (dependent) variable</th>
    <td style="text-align: center;">$y$</td>
    <td style="text-align: center;">$y$</td>
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/#linear-regression">Example1</a>, <a href="https://datatofish.com/multiple-linear-regression-python/">Example2</a></td>
    <td style="text-align: center;">($y$-variable)</td>
  </tr>
  <tr>
    <th style="text-align: left;">Predicted (estimated) response</th>
    <td style="text-align: center;">$E(y)$ (expected value of $y$)</td>
    <td style="text-align: center;">$\hat{y}$ (y-hat)</td>
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/#linear-regression">Example1</a>, <a href="https://datatofish.com/multiple-linear-regression-python/">Example2</a></td>
    <td style="text-align: center;"></td>
  </tr>
  <tr>
    <th style="text-align: left;">Explanatory (independent) variable</th>
    <td style="text-align: center;">$x$</td>
    <td style="text-align: center;">$x$</td>
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/#linear-regression">Example1</a>, <a href="https://datatofish.com/multiple-linear-regression-python/">Example2</a></td>
    <td style="text-align: center;">($x$-variable)</td>
  </tr>
  <tr>
    <th style="text-align: left;">y-intercept</th>
    <td style="text-align: center;">$\beta_0$(beta-naught)</td>
    <td style="text-align: center;">$b_0$</td>
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/#linear-regression">Example1</a>, <a href="https://datatofish.com/multiple-linear-regression-python/">Example2</a></td>
    <td style="text-align: center;">B (row: intercept)</td>
  </tr>
  <tr>
    <th style="text-align: left;">Slope</th>
    <td style="text-align: center;">$\beta_1$ (beta-one)</td>
    <td style="text-align: center;">$b_1$</td>
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/#linear-regression">Example1</a>, <a href="https://datatofish.com/multiple-linear-regression-python/">Example2</a></td>
    <td style="text-align: center;">B (row: $x$)</td>
  </tr>
  <tr>
    <th style="text-align: left;">Coefficient of correlation</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$r$</td>
    <td style="text-align: center;"><a href="https://realpython.com/numpy-scipy-pandas-correlation-python/">Example1</a>, <a href="https://medium.com/analytics-vidhya/linear-regression-using-python-ce21aa90ade6">Example2</a></td>
    <td style="text-align: center;">Values in Correlation Matrix</td>
  </tr>
  <tr>
    <th style="text-align: left;">Coefficient of determination</th>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">$r^2$</td>
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/">Example1</a>, <a href="https://medium.com/analytics-vidhya/linear-regression-using-python-ce21aa90ade6">Example2</a></td>
    <td style="text-align: center;">Multiple-R Squared</td>
  </tr>
  <tr>
    <th style="text-align: left;">Error terms vs Residuals</th>
    <td style="text-align: center;">$\varepsilon$ (error terms)</td>
    <td style="text-align: center;">$e$ (residuals)</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">Unstandardized residuals</td>
  </tr>
  </tbody>
  <tfoot style="border-top: 1px double;">
  <tr>
    <td colspan="5"><div style="font-weight: bold;">Python:</div> <div style="font-family: courier">import pandas as pd;<br/>ser = pd.Series([1,2,3]); ser.describe(); <br/>df = pd.read_csv("filename.csv"); df.info(); df.describe();</div></td>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>
  </tfoot>
</table>


## Basic Formula

### Summary Measures

+ Sample mean

  \[ \overline{x} = \frac{x_1 + x_2 + \cdots + x_n}{n} = \frac{\sum x_1}{n} \]

+ Sample standard deviation
  
  \[ s = \sqrt{\frac{\sum(x_i - \overline{x})^2}{n-1}} = \sqrt{\frac{\sum x_i^2 - n\overline{x}^2}{n-1}} \]


### Probability Rules

+ Complement rule

  \[ P(A^C) = 1 - P(A) \]

+ Addition rule
  + general
  
    \[ P(A \cup B) = P(A) + P(B) - P(A \cap B) \]

  + for independent event

    \[ P(A \cup B) = P(A) + P(B) - P(A) P(B)\]

  + for mutually exclusive events

    \[ P(A \cup B) = P(A) + P(B) \]

+ Multiplication rule
  + general

    \[ P(A \cap B) = P(A) P(B | A) \]

  + for independent events

    \[ P(A \cap B) = P(A) P(B) \]

  + for mutuallu exclusive events

    \[ P(A \cap B) = 0 \]

+ Conditional probability
  + general

    \[ P(A | B) = \frac{P(A \cap B)}{{P(B)}} \]

  + for independent events

    \[ P(A | B) = P(A) \]

  + for mutual exclusive events

    \[ P(A | B) = 0 \]


### Discrete Random Variables

+ Mean

  \[ E(X) = \mu = \sum x_ip_i = x_1p_1 + x_2p_2 + \cdots + x_kp_k \]

+ Standard deviation

  \[ s.d.(X) = \sigma = \sqrt{\sum (x_i - \mu)^2 p_i} = \sqrt{\sum (x_i^2 p_i) - \mu^2} \]


### Binomial Random Variables

+ Distribution

  \[ P(X = k) = \begin{pmatrix}n\\k\end{pmatrix} p^k (1-p)^{n-k}, \qquad\qquad \text{where } \begin{pmatrix} n  \\ k \end{pmatrix} = \frac{n!}{k!(n-k)!} \]

+ Mean

  \[ E(X) = \mu_X = np \]

+ Standard deviation

  \[ s.d.(X) = \sigma_X = \sqrt{np(1-p)} \]


### Normal Random Variables

+ $z$-score

  \[ z\text{-score} = \frac{\text{observation - mean}}{\text{standard deviation}} = \frac{x = \mu}{\sigma} \]

+ Percentile:

  \[ x = \mu + z\sigma \]

+ Normalization

  \[X \sim N(\mu, \sigma) \implies Z = \frac{X - \mu}{\sigma} \sim N(0, 1)\]


### Normal Approximation to the Binomial Distribution

  \[X \sim B(n, p),\; np \geq 10, n(1-p) \geq 10 \implies X \approx B\left(np, \sqrt{np(1-p)}\right)\]


### Sample Proportion

+ Proportion

    \[ \hat{p} = \frac{x}{n} \]

+ Mean
  \[ E(\hat{p}) = \mu_{\hat{p}} = p \]

+ Standard deviation

  \[ s.d.(\hat{p}) = \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}} \]

+ Sampling distribution of $\hat{p}$: 

  \[ \exists \text{ sample size } n \ni np \geq 10, \;n(1-p) \geq 10 \implies \hat{p} \approx N\left( p, \sqrt{\frac{p(1-p)}{n}} \right)\]

### Sample means

+ Mean

  \[ E(X) = \mu_{\overline{X}} = \mu \]

+ Standard deviation

  \[ s.d.(\overline{X}) = \sigma_{\overline{X}}  = \frac{\sigma}{\sqrt{n}} \]

+ Sampling distribution of $\overline{X}$ 
  + if $X$ has the $N(\mu, \sigma)$ distribution, the $\overline{X}$ is

    \[ N(\mu_{\overline{X}}, \sigma_{\overline{X}}) \iff N\left(\mu, \frac{\sigma}{\sqrt{n}}\right) \]
  
  + __Central Limit Theorem__: 
  
    \[\exists \;X \text{ with } \mu, \; \sigma \text{ and } n \gg 1 \implies \overline{X} \sim N\left( \mu, \frac{\sigma}{\sqrt{n}} \right)\]



## Naming Scenarios for Inference

+ How many populations are there?
  + one
  + two
  + more than two

+ How many variables are there?
  + one
  + two

+ What type of variable(s)?
  + categorical
  + quantitative


<table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Type of Inference</a></caption>
  <thead>
  <tr>
    <th colspan="2"style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;"></th>
    <th colspan="3" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Number of Populations</th>
  </tr>
  <tr>
    <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Number of Variables & Type</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">One</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Two</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">More than Two</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <th rowspan="2" style="text-align: left;">One</th>
    <th>Categorical</th>
    <td style="vertical-align: top;"><span style="color: lightgreen">1-sample inference for popul. proportion ($p$)</span><br/>[approx. cond: 1 r.s., $\min(np, (n(1-p)) \geq 10$)]<br/><br/><br/><span style="color: lightgreen">$\chi^2$: Goodness of Fit</span><br/>[cond: 1 r.s., 1 response w/ $k$ outcomes, <span style="font-weight: bold; color: cyan;">Condition</span>]</td>
    <td style="vertical-align: top;"><span style="color: lightgreen">2-indep. samples inference for the difference btw 2 popul. proportions ($p_1 - p_2$)</span><br/>[approx. cond.: 2 r.s., $\min(n_1p_1, n_1(1-p_1),$ $n_2p_2, n_2(1-p_2)) \geq 10$]<br/><br/><br/><span style="color: lightgreen">$\chi^2$: Homogeneity</span><br/>[cond: 2 indep. r.s., 1 response w/ $r$ outcomes, <span style="font-weight: bold; color: cyan;">Condition</span>]</td>
    <td style="vertical-align: top;"><span style="color: lightgreen">$\chi^2$: homogeneity</span><br/>[cond.: $c$ indep. r.s., 1 response w/ $r$ outcomes,  <span style="font-weight: bold; color: cyan;">Condition</span>]</td>
  </tr>
  <tr>
    <th>Quantitative</th>
    <td style="vertical-align: top;"><span style="color: lightgreen;">1-same inference for popul. mean ($\mu$)</span><br/>[approx. cond: 1 r.s. & $n \geq 25$, no outlier]<br/><br/><br/><span style="color: lightgreen;">Paired samples inference for a popul. mean difference ($\mu_0$)</span><br/>[approx. cond.: 1 rs & $n \geq 25$, no outlier]</td>
    <td style="vertical-align: top;"><span style="color: lightgreen;">2 indep. samples inference for the difference btw 2 popul. means ($\mu_1 - \mu_2$)</span><br/>[approx. cond.: 2 indep. r.s., normal popul., same $\sigma^2$ (pooled only)] </td>
    <td style="vertical-align: top;"><span style="color: lightgreen;">ANOVA <br/>($\mu_i$ - one $\mu_i$ for each popul.)</span><br/>[cond.: 1 indep. r.s., normal popul., same $\sigma^2$]</span></td>
  </tr>
  <tr>
    <th rowspan="2" style="text-align: left;">Two</th>
    <th>Categorical</th>
    <td style="vertical-align: top;"><span style="color: lightgreen;">$\chi^2$: indep.</span><br/>[cond.: 1 r.s., $c$ & $r$ outcomes, <span style="font-weight: bold; color: cyan;">Condition</span>]</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Quantitative</th>
    <td><span style="color: lightgreen;">Regression ($\beta_1$)</span><br/>[cond.: linear relationship, normal error, constant variance, no pattern</td>
    <td></td>
    <td></td>
  </tr>
  <tfoot style="border-top: 1px double;">
  <tr>
    <th colspan="2" style="text-align: left;">Note</th>
    <td colspan="3">the corresponding parameter is in parentheses, where appropriate.</td>
  </tr>
  <tr>
    <th colspan="2" style="text-align: left;">Distributions</th>
    <td colspan="3">$N(\mu, \sigma), t(df), F(df_1, df_2), \chi^2(df)$ where $df_1 = gps - 1$, $df_2 = n - gps$</td>
  </tr>
  <tr>
    <th colspan="2" style="text-align: left;"><span style="font-weight: bold; color: cyan;">Condition</span></th>
    <td colspan="3">80% expected count > 5; none of them < 1</td>
  </tr>
  <tr>
    <th colspan="2" style="text-align: left;">Abbreviations</th>
    <td colspan="3">Popul. = population; r.s. = random sample; indep. = independent; gps = groups</td>
  </tr>
  </tbody>
</table>


## Population Proportion


<table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
  <thead>
  <tr>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Population Proportion</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Two Population Proportions</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td><span style="font-weight: bold;">Parameter</span><span style="padding-left: 2em;">$p$</span></td>
    <td><span style="font-weight: bold;">Parameter</span><span style="padding-left: 2em;">$p_1 - p_2$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Statistic</span><span style="padding-left: 3em;">$\hat{p}$</span></td>
    <td><span style="font-weight: bold;">Statistic</span><span style="padding-left: 3em;">$\hat{p}_1 - \hat{p}_2$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Standard error</span><br/><br/><span style="padding-left: 5em;">$s.e.(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$</span></td>
    <td><span style="font-weight: bold;">Standard error</span><br/><br/><span style="padding-left: 5em;">$s.e.(\hat{p_1} - \hat{p}_2) = \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+ \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Confidence Interval</span><br/><br/><span style="padding-left: 5em;">$\hat{p} \pm z^\ast s.e.(\hat{p})$</span><br/><br/><span style="font-weight: bold;">Conservative Confidence Interval</span><br/><br/><span style="padding-left: 5em;">$\hat{p} \pm \frac{z^\ast}{2\sqrt{n}}$</span></td>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Confidence Interval</span><br/><br/><span style="padding-left: 5em;">$(\hat{p}_1 - \hat{p}_2) \pm z^\ast s.e.(\hat{p}_1 - \hat{p}_2)$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Large-Sample $z$-Test</span><br/><br/><span style="padding-left: 5em;">$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$</span></td>
    <td rowspan="2" style="vertical-align: top;"><span style="font-weight: bold;">Large-Sample $z$-Test</span><br/><br/><span style="padding-left: 5em; padding-top: 1em;">$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left( \frac{1}{n_1} + \frac{1}{n_2} \right)}}$</span><div style="padding-left: 10em; padding-top: 1.5em;"> where $\hat{p} = \frac{n_1\hat{p}_1 + n_2\hat{p}_2}{n_1+n_2}$</div></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Sample Size</span><br/><span style="padding-left: 5em;">$n = \left(\frac{z^\ast}{2m}\right)^2$</span></td>
  </tr>
  </tbody>
</table>


## Population Mean


<table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
  <thead>
  <tr>
    <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Population Mean</th>
    <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Two Population Means</th>
  </tr>
  <tr>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">General</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Pooled</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td><span style="font-weight: bold;">Parameter</span><span style="padding-left: 2em;">$\mu$</span></td>
    <td><span style="font-weight: bold;">Parameter</span><span style="padding-left: 2em;">$\mu_1 - \mu_2$</span></td>
    <td><span style="font-weight: bold;">Parameter</span><span style="padding-left: 2em;">$\mu_1 - \mu_2$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Statistic</span><span style="padding-left: 3em;">$\overline{x}$</span></td>
    <td><span style="font-weight: bold;">Statistic</span><span style="padding-left: 3em;">$\overline{x}_1 - \overline{x}_2$</span></td>
    <td><span style="font-weight: bold;">Statistic</span><span style="padding-left: 3em;">$\overline{x}_1 - \overline{x}_2$</span></td>
  </tr>
  <tr>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Standard Error</span><br/><br/><span style="padding-left: 1em;">$s.e.(\overline{x}) = \frac{s}{\sqrt{n}}$</span></td>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Standard Error</span><br/><br/><span style="padding-left: 1em;">$s.e.(\overline{x}_1 - \overline{x}_2) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$</span></td>
    <td><span style="font-weight: bold;">Standard Error</span><br/><br/><span style="padding-left: 1em;">$\text{pooled } s.e.(\overline{x}_1 - \overline{x}_2) = s_p \sqrt{\frac{1}{n_1}+\frac{1}{n_2}}$</span><br/><br/><span style="padding-left: 2em;">where $s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2 -1)s_2^2}{n_1+n_2-2}}$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">Confidence Interval</span><br/><br/><span style="padding-left: 1em;">$\overline{x} \pm t^\ast s.e.(\overline{x})$</span><br/><br/><span style="padding-left: 2em;">$df = n -1$</span><br/><br/><br/><span style="font-weight: bold;">Paired Confidence Interval</span><br/><br/><span style="padding-left: 1em;">$\overline{d} \pm t^\ast s.e.(\overline{d})$</span><br/><br/><span style="padding-left: 2em;">$df = n -1$</span></td>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Confidence Interval</span><br/><br/><span style="padding-left: 0.5em;">$(\overline{x}_1 - \overline{x}_2) \pm t^\ast \left(s.e.(\overline{x}_1 - \overline{x}_2)\right)$</span><br/><br/><span style="padding-left: 2em;">$df = \min(n_1 -1, n_2 -1)$</span></td>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Confidence Interval</span><br/><br/><span style="padding-left: 0.5em;">$(\overline{x}_1 - \overline{x}_2) \pm t^\ast (\text{pooled }s.e.(\overline{x}_1 - \overline{x}_2))$</span><br/><br/><span style="padding-left: 2em;">$df = n_1 + n_2 -2$</span></td>
  </tr>
  <tr>
    <td><span style="font-weight: bold;">One-Sample $t$-Test</span><br/><br/><span style="padding-left: 1em;">$t = \frac{\overline{x}-\mu_0}{s.e.(\overline{x})} = \frac{\overline{x} - \mu_0}{s/\sqrt{n}}$</span><br/><br/><span style="padding-left: 2em;">$df = n -1$</span><br/><br/><br/><span style="font-weight: bold;">Paired $t$-Test</span><br/><br/><span style="padding-left: 1em;">$t = \frac{\overline{d}-0}{s.e.(\overline{d})} = \frac{\overline{d}}{s_d/\sqrt{n}}$</span><br/><br/><span style="padding-left: 2em;">$df = n -1$</span></td>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Two-Sample $t$-Test</span><br/><br/><span style="padding-left: 1em;">$t = \frac{\overline{x}_1 - \overline{x}_2 - 0}{s.e.(\overline{x}_1 - \overline{x}_2)} = \frac{\overline{x}_1 - \overline{x}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$</span><br/><br/><span style="padding-left: 2em;">$df = \min(n_1 - 1, n_2 - 1)$</span></td>
    <td style="vertical-align: top;"><span style="font-weight: bold;">Pooled Two-Sample $t$-Test</span><br/><br/><span style="padding-left: 1em;">$t = \frac{\overline{x}_1 - \overline{x}_2-0}{\text{pooled }s.e.(\overline{x}_1 - \overline{x}_2)} = \frac{\overline{x}_1 - \overline{x}_2}{s_p \sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}$</span><br/><br/><span style="padding-left: 2em;">$df = n_1 + n_2 - 2$</span></td>
  </tr>
  </tbody>
</table>



## One-Way ANOVA

<table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
  <caption style="font-size: 1.5em; margin: 0.2em;">One-Way ANOVA</caption>
  <tbody>
  <tr>
    <td style="width: 10%;">SS Group = SSG = <br/><div style="padding-top: 1.0em; padding-left: 1.0em;">$\displaystyle\sum_{\text{groups}} n_i (\overline{x}_i - \overline{x})^2$</div></td>
    <td style="width: 15%;">MS Groups = MSG = <br/><div style="padding-top: 1.0em; padding-left: 1.0em;">$\frac{\text{SSG}}{k-1}$</div></td>
    <td style="width: 20%;" rowspan="3">
      <table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
        <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">ANOVA Table</a></caption>
        <thead>
        <tr>
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Source</th>
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:25%;">SS</th>
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">DF</th>
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:25%;">MS</th>
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">F</th>
        </tr>
        </thead>
        <tbody>
        <tr>
          <th style="text-align: left;">Groups</th>
          <td>SS Groups</td>
          <td>$k-1$</td>
          <td>MS Groups</td>
          <td>F</td>
        </tr>
        <tr>
          <th style="text-align: left;">Error</th>
          <td>SS Error</td>
          <td>$N-k$</td>
          <td>MS Error</td>
          <td></td>
        </tr>
        <tr>
          <th style="text-align: left;">Total</th>
          <td>SSTO</td>
          <td>$N-1$</td>
          <td></td>
          <td></td>
        </tr>
        </tbody>
      </table>
    </td>
  </tr>
  <tr>
    <td>SS Error = SSE = <br/><div style="padding-top: 1.0em; padding-left: 1.0em;">$\displaystyle \sum_{\text{groups}} (n_i - 1) s_i^2$</div></td>
    <td>MS Error = MSE = <br/><div style="padding-top: 1.0em; padding-left: 1.0em;">$s_p^2 = \displaystyle \frac{\text{SSE}}{N-k}$</div></td>
  </tr>
  <tr>
    <td>SS Total = SSTO = <br/><div style="padding-top: 1.0em; padding-left: 1.0em;">$\displaystyle \sum_{\text{values}} \left(x_{ij} - \overline{x}\right)^2$</div></td>
    <td>F = $\frac{\text{MS Groups}}{\text{MS Error}}$</div></td>
  </tr>
  <tr>
    <td colspan="2"><span style="font-weight: bold; padding-right: 1em;">Confidence Interval</span>$\overline{x}_i \pm t^\ast \frac{s_p}{\sqrt{n_i}} \quad df=N-k$</td>
    <td>Under $H_0$, $F \sim F(k-1, N-k)$</td>
  </tr>
  </tbody>
</table>


## Regression

### Linear Regression Model

+ Population Version
  + Mean

    \[ \mu_Y (x) = E(Y) = \beta_0 + \beta_1 x \]

  + Individual

    \[ y_i = \beta_0 + \beta_1 x_i + \varepsilon_i \qquad\text{where } \varepsilon_i \sim N(0, \sigma) \]

+ Sample version
  + Mean

    \[ \hat{y} = b_0 + b_1 x \]

  + Individual

    \[ y_i = b_0 + b_1 x_i + e_i \]


### Parameter Estimators

  \[\begin{align*}
    b_1 &= \frac{S_{XY}}{S_{XX}} = \frac{\sum (x - \overline{x})(y - \overline{y})}{\sum (x - \overline{x})^2} = \frac{\sum (x-\overline{x})y}{\sum (x - \overline{x})^2} \\\\
    b_0 &= \overline{y} - b_1 \overline{x}
  \end{align*}\]


### Residuals

  \[ e = y - \hat{y} = \text{observed } y - \text{predicted } y \]


### Correlation and its square

  \[\begin{align*}
    r &= \frac{S_{XY}}{\sqrt{S_{XX}S_{YY}}} \\\\
    r^2 &= \frac{SSTO - SSE}{SSTO} = \frac{SSREG}{SSTO} \qquad\qquad \text{where } SSTO = S_{YY} = \sum (y - \hat{y})^2
  \end{align*}\]


### Estimate of $\sigma$

  \[ s = \sqrt{MSE} = \sqrt{\frac{SSE}{n-2}} \qquad\qquad \text{where } SSE = \sum (y - \hat{y})^2 = \sum e^2 \]


### Sample Slope

+ Standard Error of Sample Slop

  \[ s.e.(b_1) = \frac{s}{\sqrt{S_{XX}}} = \frac{s}{\sqrt{\sum (x - \overline{x})^2}} \]


+ Confidence Interval of $\beta_1$

  \[ b_1 \pm t^\ast s.e.(b_1) \qquad df = n-2 \]


+ $t$-Test for $\beta_1$

  \[\begin{align*}
    \text{To test } & H_0: \beta_1 = 0 \\\\
    t &= \frac{b_1 -0}{s.e.(b_1)} \qquad df = n-2 \\\\
    \text{or } F &= \frac{MSREG}{MSE} \qquad df = 1, n-2
  \end{align*}\]


### Confidence Interval for the Mean Response

  \[\begin{align*}
    \hat{y} \pm t^\ast & s.e.(fit)   \qquad df = n-2 \\\\
    & \text{where } s.e.(fit) = s \sqrt{\frac{1}{n}+\frac{(x - \overline{x})^2}{S_{XX}}}
  \end{align*}\]

### Prediction Interval for the Mean Response

  \[\begin{align*}
    \hat{y} \pm t^\ast & s.e.(pred) \qquad df = n-2 \\\\
    & \text{where } s.e.(pred) = \sqrt{s^2 + \left(s.e.(fit)\right)^2}
  \end{align*}\]


### Sample Intercept

+ Standard Error of the Sample Intercept

  \[ s.e.(b_0) = s \sqrt{\frac{1}{n} + \frac{\overline{x}^2}{S_{XX}}} \]


+ Confidence Interval for $\beta_0$

  \[ b_0 \pm t^\ast s.e.(b_0) \qquad df = n-2\]


+ $t$-Test for $\beta_0$

  \[\begin{align*}
    \text{To test } & H_0: \beta_0 = 0 \\\\
    t &= \frac{b_0 - 0}{s.e.(b_0)}  \qquad df = n-2
  \end{align*}\]



## Chi-Square Tests

<table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
  <caption style="font-size: 1.5em; margin: 0.2em;">$\chi^2$ Tests</caption>
  <thead>
  <tr>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Test of Independence & Test of Homogeneity</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Test for Goodness of Fit</th>
  </tr>
  </thead>
  <tbody>
  <tr style="vertical-align: top;">
    <td><div style="font-weight: bold;">Expected Count</div><div style="padding-left: 2em; padding-top: 1em;">$E = \text{expected} = \frac{\text{row total } \times \text{ column total}}{\text{total } n}$</div></td>
    <td><div style="font-weight: bold;">Expected Count</div><div style="padding-left: 2em; padding-top: 1em;">$E_i = \text{expected} = n p_{i0}$</div></td>
  </tr>
  <tr style="vertical-align: top;">
    <td><div style="font-weight: bold;">Test Statistic</div><div style="padding-left: 2em; padding-top: 1em;">$X^2 = \sum \frac{(O - E)^2}{E} = \sum \frac{(\text{observed } - \text{ expected})^2}{\text{expected}}$</div><div style="padding-left: 15em; padding-top: 1em;"> $df = (r-1)(c-1)$</div></td>
    <td><div style="font-weight: bold;">Test Statistic</div><div style="padding-left: 2em; padding-top: 1em;">$X^2 = \sum \frac{(O - E)^2}{E} = \sum \frac{(\text{observed } - \text{ expected})^2}{\text{expected}}$</div><div style="padding-left: 15em; padding-top: 1em;"> $df = k-1$</div></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center;">$Y \sim \chi^2(df) \implies E(Y) = df,\; Var(Y) = 2\cdot df$</td>
  </tbody>
</table>



