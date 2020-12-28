# Machine Learning Articles

## Making Sense of Machine Learner

Author: Kevin Gray

Date: 2016-02-01

[Original](https://tinyurl.com/y8oyhf9e)

+ Machine learner: computer algorithm designed for
  + pattern recognition
  + curve fitting
  + classification
  + clustering

+ Common ML application
  + predict cx behavior
  + estimate cx spending
  + identify cx segmentation
  + find key driver
  + identify payoff activity
  + recommendation system
  + social media analytic

+ Type of machine learner
  + supervised methods
    + using a dependent variable
    + 'label' used for dependent variable
    + categories
      + classification problem in statistics
      + quantity: regression problem
  + unsupervised methods
  + time-series methods
    + data collected at many points of time
    + cross-sectional research for marketing
    + utilizing discriminant analysis, regression and factor analysis commonly
  + pattern mining: used for rationalize self placement and for recommend system
  + special methods
    + text analysis
    + social network analysis
    + web analysis
    + mining stream data
    + anomaly detection

+ Popular machine learner
  + Artificial Neural Network (ANN)
    + inspired by notions of how the human brain functions
    + used for classification, regression, clustering, text mining, and assortment of real-time analytics
    + cons: high time complexity, tendency of overfit, and hard to interpret
  + Support Vector Machine (SVM) (left diagram)
    + originally binary classification problems
    + extended to multi-group classification and quantitative dependent variables
    + basic idea: constructing a hyperplane or set of hyperplanes used for classification, regression, or other tasks
  + Random Forest (right diagram)
    + employ a committee fool's strategy
    + fast and parallel computing
    + predicting either group memberships or quantities
    + randomly select cases and variables
    + mini-models: predict poorly but better than chance
  + Adaboost / boosting
    + common fool's strategies
    + using all cases and weighted up or down depending on how difficult they are to predict accurately
    + sensitive to noisy data $\to$ perform poorly by chasing outliers
    + stochastic gradient boosting gaining popular

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y8oyhf9e" ismap target="_blank">
      <img style="margin: 0.1em;" height=180
        src  ="https://cnx.org/resources/5846bc7558e0fb464f99ef468248337ae91d214b/SVM%20classifier.gif"
        alt  ="SVM Classifier"
        title="SVM Classifier"
      >
    </a>
    <a href="https://imgur.com/BmEWJhA" ismap target="_blank">
      <img style="margin: 0.1em;" height=180
        src  ="https://i.imgur.com/BmEWJhA.png"
        alt  ="committee of fool's strategy"
        title="committee of fool's strategy"
      >
    </a>
  </div>



