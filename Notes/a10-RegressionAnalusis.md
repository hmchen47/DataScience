# Going Deeper into Regression Analysis with Assumptions, Plots & Solutions

Authot: ANALYTICS VIDHYA

Date: JULY 14, 2016


[Original](https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/)


## Introduction

+ Regression analysis
  + the first step in prediction modeling
  + with R, regression analysis returns 4 plots using `plot(model_name)`
    + providing significant information or rather an interesting story about the data
    + easily failing to decipher the information or not care about what these plots say

+ Regression
  + a parametric approach
  + parametric: making assumptions about data for the purpose of analysis
  + successful regression analysis: validate assumptions
  

## Assumptions in Regression

+ The important assumptions in regression analysis:
  + __linear and additive relationship__:
    + between dependent (response) variable and independent (predictor) variable(s).
    + linear relationship: a change in response $Y$ due to one unit change in $X^1$ is constant, regardless of the value of $X^1$
    + additive relationship: the effect of $X^¹$ on $Y$ is independent of other variables
  + __no autocorrelation__: no correlation between the residual (error) terms
  + __no multicollinearity__: independent variables not be correlated
  + __heteroscedasticity__: error terms not constant variance, homoscedasticity - variables w/ same variance
  + __normally distributed__: the distribution of error terms


## What if these assumptions get violated?

+ Linear and Additive
  + fit a linear model to a non-linear, non-additive data set $\to$ fail to capture the trend mathematically
  + resulting in an inefficient model
  + erroneous predictions on an unseen data set
  + __Check:__
    + residual vs fitted value plots
    + including polynomial terms ($X, X^2, X^3, \dots$) to capture the non-linear effect

+ Autocorrelation
  + presence of correlation in error terms drastically reduces model’s accuracy
  + time series models: the next instance depending on previous instance
  + correlated error terms: underestimate the true standard error
    + narrower confidence interval: a 95% confidence interval lesser probability than 0.95 to contain the actual value of coefficients
    + narrower prediction intervals, e.g.,
      + the least square coefficient of $X^1$: 15.02
      + standard error: 0.28 (w/o autocorrelation) & 1.20 (w/ autocorrelation)
      + prediction intervals: (13.82, 16.22) $\to$ (12.94, 17.10)
  + lower standard errors $\to$ lower p-values $\to$ incorrect conclusion for a parameter as statistically significant
  + __Check__:
    + Durbin-Watson (DW) statistic: $DW \in (0, 4)$
      + $DW = 2 \implies$ no autocorrelation
      + $DW \in (0, 2) \implies$ positive autocorrelation
      + $DW \in (2, 4) \implies$ negative autocorrelation
    + residual vs time plot: observing the seasonal or correlated pattern in residual values

+ Multicollinearity
  + the independent variables moderately or highly correlated
  + model with correlated variables
    + difficult to figure out the true relationship of a predictors with response variable
    + hard to find out which variable actually contributing to predict the response variable
  + correlated predictors
    + larger standard errors $\to$ wider confidence interval $\to$ less precise estimates of slope parameters
    + estimated regression coefficient: depending on other predictors in the model
    + incorrect conclusion: a variable strongly / weakly affecting target variable
    + changing the estimated regression coefficients as a correlated variable drops off
  + __Check__:
    + scatter plot to visualize correlation effect among variables
    + VIF factor
      + $VIF \le 4$: no multicollinearity
      + $VIF \ge 10$: serious multicollinearity
    + correlation table

+ Heteroscedasticity
  + non-constant variance in the error terms
  + arising in presence of outliers or extreme leverage values
  + disproportionately influences the model’s performance
  + confidence interval for out of sample prediction tends to be unrealistically wide or narrow
  + __Check__:
    + residual vs fitted values plot: exhibit a funnel shape pattern
    + Breusch-Pagan / Cook–Weisberg test or White general test

+ Normal distribution of error terms
  + non-normally distributed:
    + CI too wide or narrow
    + a few unusual data points $\to$ investigate closely to make a better model
  + unstable CI: difficulty in estimating coefficients based on minimization of least squares
  + __Check__:
    + Q-Q plot
    + statistical tests of normality, including Kolmogorov-Smirnov test, Shapiro-Wilk test


## Interpretation of Regression Plots

+ Residual vs Fitted Values

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/z5bu6a3" ismap target="_blank">
      <img style="margin: 0.1em;" height=200
        src  ="https://tinyurl.com/jjzp7x3"
        alt  ="interpretation of residual vs fitted regression plot"
        title="interpretation of residual vs fitted regression plot"
      >
    </a>
    <a href="url" ismap target="_blank">
      <img style="margin: 0.1em;" height=150
        src  ="https://tinyurl.com/zy443vu"
        alt  ="residual vs fitted value heteroskedasticity plot interpretation"
        title="residual vs fitted value heteroskedasticity plot interpretation"
      >
    </a>
  </div>

  + scatter plot: (right & middle diagrams)
    + the distribution of residuals (errors) vs fitted values (predicted values)
    + various useful insights including outliers
    + outliers: labeled by observation number to make them easy to detect
  + key points
    + existence of any patterns:
      + signs of non-linearity in the data
      + model not capturing non-linearity
    + funnel shaped: sign of non constant variance, ir.e. heteroscedasticity
  + solution
    + executing a non-linear transformation, such as $\log(x)$, $\sqrt{x}$, or $x^2$
    + overcome heteroscedasticity:
      + transforming variable such as $\log(Y)$ or $\sqrt{Y}$
      + weighted least square method

+ Normal Q-Q Plot

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
      onclick="window.open('https://tinyurl.com/z5bu6a3')"
      src    ="https://tinyurl.com/z7wfo5m"
      alt    ="normal q-q plot regression interpretation"
      title  ="normal q-q plot regression interpretation"
    />
  </figure>

  + a scatter plot to validate the assumption of normal distribution in a data set
  + normal distribution: points shown fairly straight line
  + non-normality: deviation in the straight line
  + quantile:
    + points in data below a certain proportion of data falls
    + often referred to as percentiles
    + e.g., value of 50th percentile = 120 $\implies$ half og the data lies below 120
  + solution: non-linear transformation of variables (response or predictors)

+ Scale Location Plot

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
      onclick="window.open('https://tinyurl.com/z5bu6a3')"
      src    ="https://tinyurl.com/zhdnerb"
      alt    ="3. Scale Location Plot scale location regression plot"
      title  ="3. Scale Location Plot scale location regression plot"
    />
  </figure>

  + used to detect homoscedasticity
  + how the residual spreading along the range of predictors
  + similar to residual vs fitted value plot except using standardized residual values
  + normally distributed: no discernible pattern in the plot
  + solution: (same as Residual vs Fitted Values for heteroscedasticity)
    + transforming variable such as $\log(Y)$ or $\sqrt{Y}$
    + weighted least square method

+ Residuals vs Leverage Plot

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
      onclick="window.open('https://tinyurl.com/z5bu6a3')"
      src    ="https://tinyurl.com/jyzjfro"
      alt    ="residual vs leverage regression plot interpretation"
      title  ="residual vs leverage regression plot interpretation"
    />
  </figure>

  + known as Cook’s Distance plot
  + Cook’s distance: identifying the points more influence than other points
  + influential points: a sizable impact of the regression line
  + adding or removing such points from the model able to completely change the model statistics
  + influential points =? outliers: investigating the data required
  + solution: influential point = outlier
    + removing those data if not many
    + treating as missing values or scale down






