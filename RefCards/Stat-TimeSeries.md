# Time Series Analysis


## Overview

+ [Univariate analysis](../Notes/a13-TimeSeries.md#time-series-analysis-a-primer): popular univariate methods
  + exponential smoothing
  + ARIMA (Autoregressive Integrated Moving Average)
  
+ [Causal Modeling](../Notes/a13-TimeSeries.md#time-series-analysis-a-primer)
  + assumption risk: future ~ past
  + causal (predictor) variables
    + improving accuracy
    + understanding which most influence
  + ROMI (Return on Marketing Investment): market response/marketing mix modeling
  + specialized regression procedures
    + transfer function models
    + ARMAX
    + dynamic regression

+ [Multiple time series](../Notes/a13-TimeSeries.md#time-series-analysis-a-primer) - frequently used approaches
  + Vector Autoregresion (VAR)
  + Vector error correction model (VECM)
  + State space framework

+ [Other models](../Notes/a13-TimeSeries.md#time-series-analysis-a-primer)
  + panel model
    + category - level analysis
    + handy when data are infrequent
  + longitudinal analysis
    + analysis of data w/ many time periods (e.g., > 25)
    + confusingly used term $\to$
      + panel modeling w/ a small number of periods
      + repeated measures, growth curve analysis or multilevel analysis
  + popular: structural equation modeling (SEM)
  + structural analysis
    + analyzing the expected length of time until one/more events happen
    + duration analysis in economics
    + event history analysis sociology
  + clusters of volatility
    + structural changes within the series and model parameters vary across time
    + models
      + breakpoint tests and models, e.g., state space, swithcing regression
      + GARCH (Generalized Autoregressive Conditional Heteroscedasticity) & ARCH
      + MGARCH (Multivariate GRACH): two or more series to analyze jointly
  + Non-parametric econometrics
    + time series and longitudinal data
    + feasible and useful
    + hard to interpret
  + machine learning
    + ANN, DL, ...
    + hard to interpret



