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


## Decision Tree Classifiers: A Concise Technical Overview

Author: Matthew Mayo

[Original](https://tinyurl.com/y8kljks4)

+ Decision tree
  + a type of flow chart to assist in the decision making process
  + internal node: tests on particular attributes
  + branches: a single test outcome
  + leaf nodes: class labels

+ Decision tree classifiers
  + most important classifiers
    + [Iteractive Dichotimiser 3 (ID3)](https://tinyurl.com/jpxx8c3) - Ross Quinlan's precursor to the C4.5
    + [C4.5](https://tinyurl.com/7dr6b28) - one of the [most popular classifiers](https://tinyurl.com/y7slvaad) of all time, also from Quinlan
    + CART - independently invented around the same time as C4.5, also still very popular
  + all adopt a top-down, recursive, divide-and-conquer approach to decision tree induction
  + C4.5: a benchmark against which the performance of newer classification algorithms are often measured
  + example: 'Buys Computer?' Decision Tree ([Han, Kamber & Pei](https://tinyurl.com/ycseb7b5)).

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
      onclick="window.open('https://tinyurl.com/y8kljks4')"
      src    ="https://tinyurl.com/yc6dhp78"
      alt    ="'Buys Computer?' Decision Tree"
      title  ="'Buys Computer?' Decision Tree"
    />
  </figure>

+ Decision tree induction
  + the process of constructing a decision tree from a set of training data and computations of attribute selection measures
  + main tasks of decision tree classification algorithms
    + tree induction
    + tree pruning
  + taking a set of pre-classified instances as input, deciding which attributes are best to split on, splitting the dataset, and recursing on the resulting split datasets until all training instances are categorized
  + goal
    + split on the attributes which create the purest child node possible
    + minimize the number
  + algorithm: Decision Tree Induction Algorithm

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/y8kljks4')"
        src    ="https://tinyurl.com/yc9tjzv2"
        alt    ="Decision Tree Induction Algorithm"
        title  ="Decision Tree Induction Algorithm"
      />
    </figure>

+ Attribute Selection Measures
  + 3 prominent decision tree classifiers.
    + [Information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) - used in the ID3 algorithm
    + [Gain ratio](https://en.wikipedia.org/wiki/Information_gain_ratio) - used in the C4.5 algorithm
    + [Gini index](https://en.wikipedia.org/wiki/Gini_coefficient) - used in the CART algorithm
  + information gain
    + based on information theory from Claude Shannon
    + how much would need to be known about a previously-unseen instance in order for it to be properly classified
    + measured by comparing entropy, or the amount of information needed to classify a single instance of a current dataset partition, to the amount of information to classify a single instance if the current dataset partition were to be further partitioned on a given attribute
    + the required number of information gain comparisons enough to tell us how much is actually gained
      + given attribute
      + the expected reduction in the info requirements caused by knowing the value of attribute
    + max info gain $\to$ attribute upon which to split this partition

+ Decision Tree Raising
  + completed decision tree model
    + overlay complex
    + containing unnecessary structure
    + difficult to interpret
  + tree pruning: the process of removing the unnecessary structure from a decision tree in order to make it more efficient
  + overfitting
    + models build from algorithms too specifically-tailored to the particular training dataset that was used to generrate them
    + perform poorly on another set of unseeing test data
    + applied to nearly all ML classification algorithm
  + C4.5 pruning methods
    + post-pruning, with subtree raising
    + subtree raising entails raising entire subtrees to replace nodes closer to the root $\to$ reclassifying leaves of subtrees closer to the root which may have been replaced during this process
  + Decision Tree Raising Algorithm

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick="window.open('https://tinyurl.com/y8kljks4')"
        src    ="https://www.kdnuggets.com/wp-content/uploads/dt-raising-algo.jpg"
        alt    ="Decision Tree Raising Algorithm"
        title  ="Decision Tree Raising Algorithm"
      />
    </figure>

  + procedure of decision tree
    1. build tree via induction
    2. solidified tree w/ pruning
    3. classifying data

+ Decision tree
  + classification strategy as oppose to some single well-known classification algorithm
  + any algorithms seeks to classify data, and take a top-down, recursive. divide-conquer approach to crafting a tree-based graph for subsequent instance classification, regardless of any other particulars (including attribution split selection methods and optional tree-pruning approach) would be consider a decision tree



