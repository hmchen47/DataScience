# Time Series Articles


## Time Series Analysis: A Primer

Author: Kevin Gray

Date: 2016-09-08

[Original](https://tinyurl.com/y89zhl4b)

+ Categories of data
  + cross-sectional data
    + observing many subjects
    + single slice of time
  + time series data
    + specialized branch of statistics
    + many periods

+ Cross-sectional analysis vs. time-series analysis
  + different objectives
  + statistical methods learned for cross-sectional data $\to$ missleading when applied to time-series data

+ Issues applying cross-sectional techniques on time-series data
  + standard errors far off: p-values too small & variables appearing more significant
  + seriously biased regression coefficients
  + not considering serial correlation in time

+ Univariate analysis: popular univariate methods
  + exponential smoothing
  + ARIMA (Autoregressive Integrated Moving Average)
  
+ Causal Modeling
  + assumption risk: future ~ past
  + causal (predictor) variables
    + improving accuracy
    + understanding which most influence
  + ROMI (Return on Marketing Investment): market response/marketing mix modeling
  + specialized regression procedures
    + transfer function models
    + ARMAX
    + dynamic regression

+ Multiple time series - frequently used approaches
  + Vector Autoregresion (VAR)
  + Vector error correction model (VECM)
  + State space framework

+ Other models
  + panel model
    + category - level analysis
    + handy when data are infrequent
  + longitudinal analysis
    + analysis of data w/ many time periods (e.g., > 25)
    + confusingly used term $\to$
      + panel modeling w/ a small number of periods
      + repeated measures, growth curve analysis or multilevel analysis
    + popular: structural equation modeling (SEM)
  + survival analysis
    + analyzing the expected length of time until one/more events happen
    + duration analysis in economics
    + event history analysis sociology
  + clusters of volatility
    + structural changes within the series and model parameters vary across time
    + models
      + breakpoint tests and models, e.g., state space, swithcing regression
      + GARCH (Generalized Autoregressive Conditional Heteroskedasticity) & ARCH
      + MGARCH (Multivariate GRACH): two or more series to analyze jointly
  + Non-parametric econometrics
    + time series and longitudinal data
    + feasible and useful
    + hard to interpret
  + machine learning
    + ANN, DL, ...
    + hard to interpret


