# Regression Analysis - Statistics


## Overview

+ [Important assumptions](../Notes/a10-RegressionAnalusis.md#assumptions-in-regression)
  + __linear and adaptive__ relationship btw dependent (response) variable and independent (predictor) variables
  + __no autocorrelation__: no correlation btw residual (error) terms
  + __not multicolinearity__: independent variables not correlated
  + __homoscedasticity__: error terms w/ constant variance
  + __normally distributed__ error terms


## Assumption Violation and Solutions

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
    + narrower prediction intervals
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


## Regression Plots

+ [Residual vs Fitted Values](../Notes/a10-RegressionAnalusis.md#interpretation-of-regression-plots)
  + scatter plot:
    + the distribution of residuals (errors) vs fitted values (predicted values)
    + various useful insights including outliers
    + outliers: labeled by observation number to make them easy to detect
  + key points
    + existence of any patterns:
      + signs of non-linearity in the data
      + model not capturing non-linearity
    + funnel shaped: sign of non constant variance, i.e. heteroscedasticity
  + solution
    + executing a non-linear transformation, such as $\log(x)$, $\sqrt{x}$, or $x^2$
    + overcome heteroscedasticity:
      + transforming variable such as $\log(Y)$ or $\sqrt{Y}$
      + weighted least square method

+ [Normal Q-Q Plot](../Notes/a10-RegressionAnalusis.md#interpretation-of-regression-plots)
  + a scatter plot to validate the assumption of normal distribution in a data set
  + normal distribution: points shown fairly straight line
  + non-normality: deviation in the straight line
  + quantile:
    + points in data below a certain proportion of data falls
    + often referred to as percentiles
    + e.g., value of 50th percentile = 120 $\implies$ half og the data lies below 120
  + solution: non-linear transformation of variables (response or predictors)

+ [Scale Location Plot](../Notes/a10-RegressionAnalusis.md#interpretation-of-regression-plots)
  + used to detect homoscedasticity
  + how the residual spreading along the range of predictors
  + similar to residual vs fitted value plot except using standardized residual values
  + normally distributed: no discernible pattern in the plot
  + solution: (same as Residual vs Fitted Values for heteroscedasticity)
    + transforming variable such as $\log(Y)$ or $\sqrt{Y}$
    + weighted least square method

+ [Residuals vs Leverage Plot](../Notes/a10-RegressionAnalusis.md#interpretation-of-regression-plots)
  + known as Cook’s Distance plot
  + Cook’s distance: identifying the points more influence than other points
  + influential points: a sizable impact of the regression line
  + adding or removing such points from the model able to completely change the model statistics
  + influential points =? outliers: investigating the data required
  + solution: influential point = outlier
    + removing those data if not many
    + treating as missing values or scale down



