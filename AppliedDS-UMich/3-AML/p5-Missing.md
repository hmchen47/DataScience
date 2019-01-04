# Handling Missing Values when Applying Classification Models

Author: Maytal Saar-Tsechansky and Foster Provost 
Publication: Journal of Machine Learning Research 8 (2007) 1625-1657

[Original document](http://jmlr.csail.mit.edu/papers/volume8/saar-tsechansky07a/saar-tsechansky07a.pdf)

## Abstract

+ Methods compared: predictive value imputation, the distribution based imputation used by C4.5, and reduced method

+ Applying calssification trees to instances with missing values and generalizing to bagged trees and to logistic regression

+ The former two preferable under difference conditions.

+ Reduced-models approach consistently outperform the other two, sometimes by a large margin.

+ Hybrid approaches scale gracefully to the amount of investment in computation/storage and outperform imputation even for small investment.

## Introduction




## Treatments for Missing Values at Prediction Time




## Experimental Comparison of Prediction-time Treatments for Missing Values


### Experimental Setup




### Comparison of PVI, DBI and Reduced Modeling




### Feature Imputability and Modeling Error




### Evaluation using Ensembles of Trees




### Evaluation using Logistic Regression




### Evaluation with “Naturally Occurring” Missing Values




### Evaluation with Multiple Missing Values




## Hybrid Models for Efficient Prediction with Missing Values




### Likelihood-based Hybrid Solutions




### Reduced-Feature Ensembles




### Larger Ensembles




### ReFEs with Increasing Numbers of Missing Values




## Related Work




## Limitations




## Conclusions




