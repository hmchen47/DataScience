# Rules of Machine Learning: Best Practices for ML Engineering

Author: Martin Zinkevich @ Google

## Terminology

+ __Instance__: The thing about which you want to make a prediction. For example, the instance might be a web page that you want to classify as either "about cats" or "not about cats".
+ __Label__: An answer for a prediction task either the answer produced by a machine learning system, or the right answer supplied in training data. For example, the label for a web page might be "about cats".
+ __Feature__: A property of an instance used in a prediction task. For example, a web page might have a feature "contains the word 'cat'".
+ __Feature Column__: A set of related features, such as the set of all possible countries 1 i n which users might live. An example may have one or more features present in a feature column. A feature column is referred to as a “namespace” in the VW system (at Yahoo/Microsoft), or a field .
+ __Example__: An instance (with its features) and a label.
+ __Model__: A statistical representation of a prediction task. You train a model on examples then use the model to make predictions.
+ __Metric__: A number that you care about. May or may not be directly optimized.
+ __Objective__: A metric that your algorithm is trying to optimize.
+ __Pipeline__: The infrastructure surrounding a machine learning algorithm. Includes gathering the data from the front end, putting it into training data files, training one or more models, and exporting the models to production.

## Overview

+ Do machine learning like the great engineer you are, not like the great machine learning expert you aren’t.

+ The basic approach
    1. make sure your pipeline is solid end to end
    2. start with a reasonable objective
    3. add commonsense features in a simple way
    4. make sure that your pipeline stays solid.

## Before Machine Learning

### Rule #1: Don’t be afraid to launch a product without machine learning.

+ If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there.

+ If machine learning is not absolutely required for your product, don't use it until you have data.


### Rule #2: Make metrics design and implementation a priority.

+ Track as much as possible in your current system.
    1. It is easier to gain permission from the system’s users earlier on.
    2. If you think that something might be a concern in the future, it is better to get historical data now.
    3. If you design your system with metric instrumentation in mind, things will go better for you in the future.
    4. You will notice what things change and what stays the same.

+ Note that an experiment framework, where you can group users into buckets and aggregate statistics by experiment, is important.

+ By being more liberal about gathering metrics, you can gain a broader picture of your system.


### Rule #3: Choose machine learning over a complex heuristic.

+ A simple heuristic can get your product out the door. 

+ A complex heuristic is unmaintainable.

+ The machine-learned model is easier to update and maintain.


## ML Phase I: Your First Pipeline

### Rule #4: Keep the first model simple and get the infrastructure right.

+ The first model provides the biggest boost to your product, so it doesn't need to be fancy.

+ Before anyone can use your fancy new machine learning system, you have to determine:
    1. How to get examples to your learning algorithm.
    2. A first cut as to what “good” and “bad” mean to your system.
    3. How to integrate your model into your application.

+ Choosing simple features makes it easier to ensure that:
    1. The features reach your learning algorithm correctly.
    2. The model learns reasonable weights.
    3. The features reach your model in the server correctly.

+ Simple model provides you with baseline metrics and a baseline behavior that you can use to test more complex models.


### Rule #5: Test the infrastructure independently from the machine learning.

+ Make sure that the infrastructure is testable, and that the learning parts of the system are encapsulated so that you can test everything around it.

+ Test getting data into the algorithm.
    + Check that feature columns that should be populated are populated.
    + Where privacy permits, manually inspect the input to your training algorithm.
    + If possible, check statistics in your pipeline in comparison to elsewhere, such as RASTA.

+ Test getting models out of the training algorithm. Make sure that the model in your training environment gives the same score as the model in your serving environment.

+ Machine learning has an element of unpredictability, so make sure that you have tests for the code for creating examples in training and serving, and that you can load and use a fixed model during serving.


### Rule #6: Be careful about dropped data when copying pipelines.

+ Often we create a pipeline by copying an existing pipeline (i.e. cargo cult programming), and the old pipeline drops data that we need for the new pipeline.

+ Log data that was seen by the user is useless if we want to model why a particular post was not seen by the user, because all the negative examples have been dropped.


### Rule #7: Turn heuristics into features, or handle them externally. 

+ These same heuristics for an existing system can give you a lift when tweaked with machine learning.

+ Mined for whatever information they have
    + The transition to a machine learned system will be smoother
    + Usually those rules contain a lot of the intuition about the system you don’t want to throw away

+ Four ways to use an existing heuristic:
    1. Preprocess using the heuristic.
    2. Create a feature: Directly creating a feature from the heuristic but start by using the raw value produced by the heuristic
    3. Mine the raw inputs of the heuristic: feeding these inputs of the heuristic  into the learning separately
    4. Modify the label: the heuristic captures information not currently contained in the label

+ Do be mindful of the added complexity when using heuristics in an ML system.


## Monitoring


### Rule #8: Know the freshness requirements of your system.


### Rule #9: Detect problems before exporting models.


### Rule #10: Watch for silent failures.


### Rule #11: Give feature sets owners and documentation.


## Your First Objective


### Rule #12: Don’t overthink which objective you choose to directly optimize.


### Rule #13: Choose a simple, observable and attributable metric for your first objective.


### Rule #14: Starting with an interpretable model makes debugging easier.


### Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer.


## ML Phase II: Feature Engineering


### Rule #16: Plan to launch and iterate.


### Rule #17: Start with directly observed and reported features as opposed to learned features.


### Rule #18: Explore with features of content that generalize across contexts.


### Rule #19: Use very specific features when you can.


### Rule #20: Combine and modify existing features to create new features in humanunderstandable ways.


### Rule #21: The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.


### Rule #22: Clean up features you are no longer using.


## Human Analysis of the System


### Rule #23: You are not a typical end user.


### Rule #24: Measure the delta between models.


### Rule #25: When choosing models, utilitarian performance trumps predictive power.


### Rule #26: Look for patterns in the measured errors, and create new features.


### Rule #27: Try to quantify observed undesirable behavior.


### Rule #28: Be aware that identical shortterm behavior does not imply identical longterm behavior.


## TrainingServing Skew


### Rule #29: The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time.


### Rule #30: Importance weight sampled data, don’t arbitrarily drop it!


### Rule #31: Beware that if you join data from a table at training and serving time, the data in the table may change.


### Rule #32: Reuse code between your training pipeline and your serving pipeline whenever possible.


### Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after.


### Rule #34: In binary classification for filtering (such as spam detection or determining interesting emails), make small shortterm sacrifices in performance for very clean data.


### Rule #35: Beware of the inherent skew in ranking problems.


### Rule #36: Avoid feedback loops with positional features.


### Rule #37: Measure Training/Serving Skew.


## ML Phase III: Slowed Growth, Optimization Refinement, and Complex Models


### Rule #38: Don’t waste time on new features if unaligned objectives have become the issue.


### Rule #39: Launch decisions will depend upon more than one metric.


### Rule #40: Keep ensembles simple.


### Rule #41: When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.


### Rule #42: Don’t expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.


### Rule #43: Your friends tend to be the same across different products. Your interests tend not to be.

