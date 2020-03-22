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
    <td style="text-align: center;"><a href="https://realpython.com/linear-regression-in-python/#linear-regression">Example</a>, <a href="https://datatofish.com/multiple-linear-regression-python/">Example2</a></td>
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
    <td colspan="5"><span style="font-weight: bold;">Python</span>: <span style="font-family: courier">import pandas as pd; ser = pd.sSeries([1,2,3]); ser.describe(); df = pd.read_csv("filename.csv"); df.info(); df.describe();</td>
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
  
  \[ s = \sqrt{\frac{\sum(x_i - \overline{x})^2}}{n-1} = \sqrt{\sum x_i^2 - n\overline{x}^2}{n-1} \]


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

+ Normalization: <br/>if $X$ has the $N(\mu, \sigma)$ distribution, then the variable $Z = \frac{X - \mu}{\sigma}$ has the $N(0, 1)$ distribution


### Normal Approximation to the Binomial Distribution

  + If $X$ has the $B(n, p)$ distribution and the sample size $n$ is large enough (namely $np \geq 10$ and $n(1-p) \geq 10$), the $X$ is approximately $B\left(np, \sqrt{np(1-p)}\right)$


### Sample Proportion

+ Proportion

    \[ \hat{p} = \frac{x}{n} \]

+ Mean
  \[ E(\hat{p}) = \mu_{\hat{p}} = p \]

+ Standard deviation

  \[ s.d.(\hat{p}) = \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}} \]

+ Sampling distribution of $\hat{p}$: <br/>if the sample size $n$ is large enough (namely, $np \geq 10$, and $n(1-p) \geq 10$), then $\hat{p}$ is _approximately_ $N\left( p, \sqrt{\frac{p(1-p)}{n}} \right)$

### Sample means

+ Mean

  \[ E(X) = \mu_{\overline{X}} = \mu \]

+ Standard deviation

  \[ s.d.(\overline{X}) = \sigma_{\overline{X}}  = \frac{\sigma}{\sqrt{n}} \]

+ Sampling distribution of $\overline{X}$ 
  + if $X$ has the $N(\mu, \sigma)$ distribution, the $\overline{X}$ is

    \[ N(\mu_{\overline{X}}, \sigma_{\overline{X}}) \iff N\left(\mu, \frac{\sigma}{\sqrt{n}}\right) \]
  
  + __Central Limit Theorem__: if $X$ follows _any_ distribution w/ mean $\mu$ and standard deviation $\sigma$ and $n$ is large, the $\overline{X}$ is approximately $N\left( \mu, \frac{\sigma}{\sqrt{n}} \right)$



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
    <td><span style="color: lightgreen">1-sample inference for popul. proportion ($p$)</span><br/>[approx. cond: 1 r.s., $min(np, (n(1-p)) \geq 10$)]<br/><br/><br/><span style="color: lightgreen">$\chi^2$: Goodness of Fit</span><br/>[cond: 1 r.s., 1 response w/ $k$ outcomes, <span style="text-weight: bold; color: cyan;">Condition</span>]</td>
    <td><span style="color: lightgreen">2-indep. samples inference for the difference btw 2 popul. proportions ($p_1 - p_2$)</span><br/>[approx. cond.: 2 r.s., $min(n_1p_1, n_1(1-p_1),$ $n_2p_2, n_2(1-p_2) \geq 10$]<br/><br/><br/><span style="color: lightgreen">$\chi^2$: Homogeneity</span><br/>[cond: 2 indep. r.s., 1 response w/ $r$ outcomes, <span style="text-weight: bold; color: cyan;">Condition</span>]</td>
    <td>$\chi^2$: homogeneity<br/>[cond.: $c$ indep. r.s., 1 response w/ $r$ outcomes,  <span style="text-weight: bold; color: cyan;">Condition</span>]</td>
  </tr>
  <tr>
    <th>Quantitative</th>
    <td><span style="color: lightgreen;">1-same inference for popul. mean ($\mu$)</span><br/>[approx. cond: 1 r.s. & $n \geq 25$, no outlier]<br/><br/><br/><span style="color: lightgreen;">Paired samples inference for a popul. mean difference ($\mu_0$)</span><br/>[approx. cond.: 1 rs & $n \geq 25$, no outlier]</td>
    <td><span style="color: lightgreen;">2 indep. samples inference for the difference btw 2 popul. means ($\mu_1 - \mu_2$)</span><br/>[approx. cond.: 2 indep. r.s., normal popul., same $\sigma^2$ (pooled only)] </td>
    <td><span style="color: lightgreen;">ANOVA <br/>($\mu_i$ - one $\mu_i$ for each popul.)</span><br/>[cond.: 1 indep. r.s., normal popul., same $\sigma^2$]</span></td>
  </tr>
  <tr>
    <th rowspan="2" style="text-align: left;">Two</th>
    <th>Categorical</th>
    <td><span style="color: lightgreen;">$\chi^2$: indep.</span><br/>[cond.: 1 r.s., $c$ & $r$ outcomes, <span style="text-weight: bold; color: cyan;">Condition</span>]</td>
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
    <td colspan="3">$N(\mu, sigma), t(df), F(df_1, df_2), \chi^2(df)$ where $df_1 = gps - 1$, $df_2 = n - gps$</td>
  </tr>
  <tr>
    <th colspan="2" style="text-align: left;"><span style="text-weight: bold; color: cyan;">Condition</span></th>
    <td colspan="3">80% expected count > 5; none of them < 1</td>
  </tr>
  <tr>
    <th colspan="2" style="text-align: left;">Abbreviations</th>
    <td colspan="3">Popul. = population; r.s. = random sample' indep. = independent, gps = groups/td>
  </tr>
  </tbody>
</table>


## Population Proportion




## Population Mean





## One-Way ANOVA





## Regression




## Chi-Square Tests





