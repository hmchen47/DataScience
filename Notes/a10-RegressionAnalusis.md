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





