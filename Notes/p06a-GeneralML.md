# A Few Useful Things to Know about Machine Learning

Author: Pedro Domingos

[Original](https://tinyurl.com/y4fz3jja)

[Article in Communications of the ACM](https://tinyurl.com/kcd5ztd)


## 1. Introduction

+ Machine learning
  + systems automatically learning programs from data
  + a.k.a. data mining or predictive analytics
  + used in 
    + Web search
    + spam filters
    + recommender systems
    + ad placement
    + credit scoring
    + fraud detection
    + stock trading
    + drug design
    + and many other applications

+ Classifier
  + a system
  + input (typically): a vector of discrete and/or continuous feature values
  + output: a single discrete value, the class
  + example: spam filter to classify email messages
    + input: a Boolean vector $\vec{x} = (x_1, \dots, x_j, \dots, x_n)$ where $x_j = 1$ if $i$th word in the dictionary appears in in the mail and $x_j = 0$ otherwise
    + _learner_ w/ a _training set_ of _examples_ $(\bf{x}_i, y_i)$ as input
    + $\bf{x_i} = (x_{i, 1}, \dots, x_{i, d})$: an observation input
    + $y_i$: the corresponding output and the output of a classifier
    + $\bf{x_t}$:  the test input of the learner


## 2. Learning = Representation + Evaluation + Optimization




## 3. It's Generalization that Counts



## 4. Data Alone is not Enough




## 5. Overfitting Has Many Faces




## 6. Intuition Falls in High Dimensions





## 7. Theoretical Guarantees are not What They Seem




## 8. Feature Engineering is the Key




## 9. More Date Beats a Clever Algorithms




## 10. Learn Many Models Not Just One




## 11. Simplicity Does Not Imply Accuracy




## 12. Presentable Does Not Imply Learnable




## 13. Correlation Does Not Imply Causation



## 14. Conclusion 



## 15 References





