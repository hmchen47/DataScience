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




## Naming Scenarios




## Population Proportion




## Population Mean





## One-Way ANOVA





## Regression




## Chi-Square Tests





