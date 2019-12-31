# 6 Powerful Feature Engineering Techniques For Time Series Data (using Python)

Author: Aishwarya Singh

Date: Dec. 9, 2019

[Origin](https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/)

## Overview

+ Feature engineering: a skill every data scientist should know how to perform, especially in the case of time series
+ 6 powerful feature engineering techniques for time series in this article
+ feature engineering technique w/ Python


## Introduction

+ a lot of nuance to time series data to consider when working with datasets that are time-sensitive

+ feature engineering for time series: the potential to transform time series model from just a good one to a powerful forecasting model


## Quick Introduction to Time Series

+ In a time series, the data is captured at equal intervals and each successive data point in the series depends on its past values.

+ Examples:
  + predicting stock price at a certain company
  + predicting the traffic on a website

+ time series data may also have certain trends or seasonality


## Setting up the Problem Statement for Time Series Data

+ Problem
  + historical data for ‘JetRail’, a form of public rail transport, that uses advanced technology to run rails at a high speed
  + to forecast the traffic on JetRail for the next 7 months based on past data
  + detailed problem statement and the dataset: [download](https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/?utm_source=blog&utm_medium=6-powerful-feature-engineering-techniques-time-series)

+ Loading data

  ```python
  import pandas as pd
  data = pd.read_csv('Train_SU63ISt.csv')
  data.dtypes
  ```

+ Convert categorical variable into a DateTime variable

  ```python
  import pandas as pd
  data = pd.read_csv('Train_SU63ISt.csv')
  data['Datetime'] = pd.to_datetime(data['Datetime'],format='%d-%m-%Y %H:%M')
  data.dtypes
  ```


## Date-Related Features

+ the sales pattern for weekdays and weekends based on historical data

+ forecast the count of people who will take the JetRail on an hourly basis for the next 7 months

+ important factor: the day of the week (weekday or weekend) or month

  ```python
  import pandas as pd
  data = pd.read_csv('Train_SU63ISt.csv')
  data['Datetime'] = pd.to_datetime(data['Datetime'],format='%d-%m-%Y %H:%M')

  data['year']=data['Datetime'].dt.year 
  data['month']=data['Datetime'].dt.month 
  data['day']=data['Datetime'].dt.day

  data['dayofweek_num']=data['Datetime'].dt.dayofweek  
  data['dayofweek_name']=data['Datetime'].dt.weekday_name

  data.head()
  ```

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/url" ismap target="_blank">
      <img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/11/fets1.png" style="margin: 0.1em;" alt="Extracting these features" title="capExtracting these featurestion" width=450>
    </a>
  </div>


## Time-Related Features



## Lag Features



## Rolling Window



## Expanding Window



## Domain-Specific




