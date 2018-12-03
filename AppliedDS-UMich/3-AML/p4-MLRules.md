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

### <a name="rule-01"></a> Rule #01: Don’t be afraid to launch a product without machine learning.

+ If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there.

+ If machine learning is not absolutely required for your product, don't use it until you have data.


### <a name="rule-02"></a> Rule #02: Make metrics design and implementation a priority.

+ Track as much as possible in your current system.
    1. It is easier to gain permission from the system’s users earlier on.
    2. If you think that something might be a concern in the future, it is better to get historical data now.
    3. If you design your system with metric instrumentation in mind, things will go better for you in the future.
    4. You will notice what things change and what stays the same.

+ Note that an experiment framework, where you can group users into buckets and aggregate statistics by experiment, is important.

+ By being more liberal about gathering metrics, you can gain a broader picture of your system.


### <a name="rule-03"></a> Rule #03: Choose machine learning over a complex heuristic.

+ A simple heuristic can get your product out the door. 

+ A complex heuristic is unmaintainable.

+ The machine-learned model is easier to update and maintain.


## ML Phase I: Your First Pipeline

### <a name="rule-04"></a> Rule #04: Keep the first model simple and get the infrastructure right.

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


### <a name="rule-05"></a>  Rule #5: Test the infrastructure independently from the machine learning.

+ Make sure that the infrastructure is testable, and that the learning parts of the system are encapsulated so that you can test everything around it.

+ Test getting data into the algorithm.
    + Check that feature columns that should be populated are populated.
    + Where privacy permits, manually inspect the input to your training algorithm.
    + If possible, check statistics in your pipeline in comparison to elsewhere, such as RASTA.

+ Test getting models out of the training algorithm. Make sure that the model in your training environment gives the same score as the model in your serving environment.

+ Machine learning has an element of unpredictability, so make sure that you have tests for the code for creating examples in training and serving, and that you can load and use a fixed model during serving.


### <a name="rule-06"></a>  Rule #6: Be careful about dropped data when copying pipelines.

+ Often we create a pipeline by copying an existing pipeline (i.e. cargo cult programming), and the old pipeline drops data that we need for the new pipeline.

+ Log data that was seen by the user is useless if we want to model why a particular post was not seen by the user, because all the negative examples have been dropped.


### <a name="rule-07"></a>  Rule #7: Turn heuristics into features, or handle them externally. 

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

### <a name="rule-08"></a>  Rule #8: Know the freshness requirements of your system.

+ How much does performance degrade if you have a model that is a day old? A week old? A quarter old?

+ Information to understand the priorities of your monitoring.

+ Most ad serving systems have new advertisements to handle every day, and must update daily.

+ Freshness can change over time, especially when feature columns are added or removed from your model.

### <a name="rule-09"></a>  Rule #9: Detect problems before exporting models.

+ Many machine learning systems have a stage where you export the model to serving.

+ Do sanity checks right before you export the model.

+ Specifically, make sure that the model's performance is reasonable on held out data.

+ Many teams continuously deploying models check the area under the ROC curve (or AUC) before exporting. 

+ Issues about models that haven’t been exported require an email alert, but issues on a user-facing model may require a page.


### <a name="rule-10"></a>  Rule #10: Watch for silent failures.

+ A problem occurs more for machine learning systems than for other kinds of systems.

+ For example, a particular table that is being joined is no longer being updated.


### <a name="rule-11"></a>  Rule #11: Give feature sets owners and documentation.

+ If the system is large, and there are many feature columns, know who created or is maintaining each feature column.

+ Having a more detailed description of what the feature is, where it came from, and how it is expected to help.


## Your First Objective

+ Objective: a number that your algorithm is “trying” to optimize.

+ Metric: any number that your system reports

### <a name="rule-12"></a>  Rule #12: Don’t overthink which objective you choose to directly optimize.

+ Tons of metrics care about and should measure them all (Rule #02)

+ In early stage of the machine learning process, all going up, even those that you do not directly optimize.

+ Keep it simple and don’t think too hard about balancing different metrics when you can still easily increase all the metrics.

+ Do not confuse your objective with the ultimate health of the system (Rule #39)

+ If you find yourself increasing the directly optimized metric, but deciding not to launch, some objective revision may be required.


### <a name="rule-13"></a>  Rule #13: Choose a simple, observable and attributable metric for your first objective.

+ The ML objective should be something that is easy to measure and is a proxy for the “true” objective.

+ Train on the simple ML objective, and consider having a "policy layer" on top that allows you to add additional logic (hopefully very simple logic) to do the final ranking.

+ User behavior directly observed and attributable to an action of the system:
    1. Was this ranked link clicked?
    2. Was this ranked object downloaded?
    3. Was this ranked object forwarded/replied to/emailed?
    4. Was this ranked object rated?
    5. Was this shown object marked as spam/pornography/offensive?

+ Avoid modeling indirect effects at first:
    1. Did the user visit the next day?
    2. How long did the user visit the site?
    3. What were the daily active users?

+ Indirect effects make great metrics, and can be used during A/B testing and during launch decisions.

+ Don’t try to get the machine learning to figure out:
    1. Is the user happy using the product?
    2. Is the user satisfied with the experience?
    3. Is the product improving the user’s overall wellbeing?
    4. How will this affect the company’s overall health?

+ Use proxies: if the user is happy, they will stay on the site longer.


### <a name="rule-14"></a>  Rule #14: Starting with an interpretable model makes debugging easier.

+ Linear regression, logistic regression, and Poisson regression are directly motivated by a probabilistic model.

+ Each prediction is interpretable as a probability or an expected value.

+ Easier to debug than models that use objectives (zero-one loss, various hinge losses, et cetera) that try to directly optimize classification accuracy or ranking performance.

+ In linear, logistic, or Poisson regression, there are subsets of the data where the average predicted expectation equals the average label (1moment calibrated, or just calibrated).

+ With simple models, it is easier to deal with feedback loops (Rule #36)

+ Remember when it comes time to choose which model to use, the decision matters more than the likelihood of the data given the model (Rule #27)


### <a name="rule-15"></a>  Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer.

+ Quality ranking is a fine art, but spam filtering is a war.

+ Quality ranking should focus on ranking content that is posted in good faith.

+ "Racy" content should be handled separately from Quality Ranking.

+ Spam filtering: the features that you need to generate will be constantly changing.

+ Integrate the output of the two systems at some stage.


## ML Phase II: Feature Engineering

+ After you have a working end to end system with unit and system tests instrumented, Phase II begins .

+ Pulling in as many features as possible and combining them in intuitive ways


### <a name="rule-16"></a>  Rule #16: Plan to launch and iterate.

+ Three basic reasons to launch new models:
    1. you are coming up with new features,
    2. you are tuning regularization and combining old features in new ways, and/or
    3. you are tuning the objective.

+ Looking over the data feeding into the example can help find new signals as well as old, broken ones.

+ Think about how easy it is to create a fresh copy of the pipeline and verify its correctness.

+ Think about whether it is possible to have two or three copies running in parallel.

+ Finally, don’t worry about whether feature 16 of 35 makes it into this version of the pipeline.


### <a name="rule-17"></a>  Rule #17: Start with directly observed and reported features as opposed to learned features.

+ A learned feature is a feature generated either by an external system (such as an unsupervised clustering system) or by the learner itself (e.g. via a factored model or deep learning).

+ They should not be in the first model.

+ The external system's objective may be only weakly correlated with your current objective.

+ The primary issue with factored models and deep models is that they are non-convex.

+ By creating a model without deep features, you can get an excellent baseline performance. After this baseline is achieved, you can try more esoteric approaches.


### <a name="rule-18"></a>  Rule #18: Explore with features of content that generalize across contexts.

+ Often a machine learning system is a small part of a much bigger picture.

+ Note that this is not about personalization: figure out if someone likes the content in this context first, then figure out who likes it more or less.


### <a name="rule-19"></a>  Rule #19: Use very specific features when you can.

+ Don’t be afraid of groups of features where each feature applies to a very small fraction of your data, but overall coverage is above 90%.

+ Use regularization to eliminate the features that apply to too few examples.


### <a name="rule-20"></a>  Rule #20: Combine and modify existing features to create new features in humanunderstandable ways.

+ Machine learning systems such as TensorFlow allow you to preprocess your data through transformations.

+ The two most standard approaches are "discretizations" and "crosses".

+ Discretization consists of taking a continuous feature and creating many discrete features from it.

+ Crosses
    + Combine two or more feature columns.
    + A feature column, in TensorFlow's terminology, is a set of homogenous features.
    + A new feature column with features
    + Massive amounts of data to learn models with crosses of three, four, or more base feature columns
    + Produce very large feature columns may overfit
    + __Dot product__: the simplest form simply counts the number of common words between the query and the document
    + __Intersection__: thus, we will have a feature which is present if and only if the word “pony” is in the document and the query, and another feature which is present if and only if the word “the” is in the document and the query.


### <a name="rule-21"></a>  Rule #21: The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.

+ Fascinating statistical learning theory results concerning the appropriate level of complexity for a model, but this rule is basically all you need to know.

+ Scale your learning to the size of your data:
    + 1000 labeled examples: search ranking system: use a dot product between document and query features, TFIDF, and a half-dozen other highly human-engineered features.
    + Million examples: Intersect the document and query feature columns, using regularization and possibly feature selection.
    + billions or hundreds of billions of examples: cross the feature columns with document and query tokens, using feature selection and regularization.


### <a name="rule-22"></a>  Rule #22: Clean up features you are no longer using.

+ Unused features create technical debt.

+ A feature not used -> combine other features -> drop out

+ Keep coverage in mind when considering what features to add or keep.


## Human Analysis of the System

### <a name="rule-23"></a>  Rule #23: You are not a typical end user.

+ While there are a lot of benefits to fishfooding (using a prototype within your team) and dogfooding (using a prototype within your company), employees should look at whether the performance is correct.

+ Anything that looks reasonably near production should be tested further, either by paying laypeople to answer questions on a crowdsourcing platform, or through a live experiment on real users.

+ Two reasons
    + too close to the code
    + time is too valuable

+ Using user experience methodologies
    + Create user personas (one description is in Bill Buxton’s Designing User Experiences ) early in a process
    + Do usability testing (one description is in Steve Krug’s Don’t Make Me Think ) later


### <a name="rule-24"></a>  Rule #24: Measure the delta between models.

+ Calculate just how different the new results are from production

+ Difference
    + Samll: without running an experiment that there will be little change
    + Large: make sure that the change is good

+ Make sure that the system is stable. Make sure that a model when compared with itself has a low (ideally zero) symmetric difference.


### <a name="rule-25"></a>  Rule #25: When choosing models, utilitarian performance trumps predictive power.

+ The key question is what you do with that prediction
    + If you are using click-through-rate to rank documents, then the quality of the final ranking matters more than the prediction itself.
    + If you predict the probability that a document is spam and then have a cutoff on what is blocked, then the precision of what is allowed through matters more.

+ If there is some change that improves log loss but degrades the performance of the system, look for another feature.


### <a name="rule-26"></a>  Rule #26: Look for patterns in the measured errors, and create new features.

+ Model got “wrong” after training
    + classification: a false positive or a false negative
    + ranking: a pair where a positive was ranked lower than a negative

+ The machine learning system knows it got wrong and would like to fix if given the opportunity.
    + If a feature allows the model to fix the error, the model will try to use it.
    + If create a feature based upon examples the system doesn’t see as mistakes, the feature will be ignored.

+ Once you have examples that the model got wrong, look for trends that are outside your current feature set.

+ Don’t be too specific about the features you add. If you are going to add post length, don’t try to guess what long means, just add a dozen features and the let model figure out what to do with them


### <a name="rule-27"></a>  Rule #27: Try to quantify observed undesirable behavior.

+ Some members of your team will start to be frustrated with properties of the system they don’t like which aren’t captured by the existing loss function. At this point, they should do whatever it takes to turn their gripes into solid numbers.

+ If your issues are measurable, then you can start using them as features, objectives, or metrics.

+ General rule: measure first, optimize second


### <a name="rule-28"></a>  Rule #28: Be aware that identical shortterm behavior does not imply identical longterm behavior.

+ Imagine that you have a new system that looks at every doc_id and exact_query, and then calculates the probability of click for every doc for every query. You find that its behavior is nearly identical to your current system in both side by sides and A/B testing, so given its simplicity, you launch it. However, you notice that no new apps are being shown.

+ Since your system only shows a doc based on its own history with that query, there is no way to learn that a new doc should be shown.

+ The only way to understand how such a system would work longterm is to have it train only on data acquired when the model was live.


## Training-Serving Skew

+ Training-serving skew is a difference between performance during training and performance during serving.

+ Skew caused by
    + a discrepancy between how you handle data in the training and serving pipelines, or
    + a change in the data between when you train and when you serve, or
    + a feedback loop between your model and your algorithm.

+ The best solution is to explicitly monitor it so that system and data changes don’t introduce skew unnoticed.


### <a name="rule-29"></a>  Rule #29: The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time.

+ Verify the consistency between serving and training (Rule #37)

+ Logging features at serving time with significant quality improvements and a reduction in code complexity


### <a name="rule-30"></a>  Rule #30: Importance weight sampled data, don’t arbitrarily drop it!

+ Dropping data in training has caused issues in the past for several teams (Rule #06)

+ Importance weighting means that if you decide that you are going to sample example X with a $30\%$ probability, then give it a weight of $10/3$. 

+ With importance weighting, all of the calibration properties discussed in Rule #14 still hold.


### <a name="rule-31"></a>  Rule #31: Beware that if you join data from a table at training and serving time, the data in the table may change.

+ Between training and serving time, features in the table may be changed.

+ The easiest way to avoid this sort of problem is to log features at serving time (Rule #32)

+ Note that this still doesn’t completely resolve the issue.


### <a name="rule-32"></a>  Rule #32: Reuse code between your training pipeline and your serving pipeline whenever possible.

+ In online processing, handle each request as it arrives (e.g. you must do a separate lookup for each query), whereas in batch processing, combine tasks (e.g. making a join).

+ At serving time, doing online processing, whereas training with batch processing task.

+ Example:
    + Create an object that is particular to your system where the result of any queries or joins can be stored in a very human readable way, and errors can be tested easily.
    + Once information gathered all the information, during serving or training, run a common method to bridge between the human-readable object that is specific to your system, and whatever format the machine learning system expects.

+ Eliminates a source of training-serving skew

+ Try not to use two different programming languages between training and serving


### <a name="rule-33"></a>  Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after.

+ Measure performance of a model on the data gathered after the data you trained the model on, as this better reflects what your system will do in production.

+ Area under the curve, which represents the likelihood of giving the positive example a score higher than a negative example, should be reasonably close.


### <a name="rule-34"></a>  Rule #34: In binary classification for filtering (such as spam detection or determining interesting emails), make small shortterm sacrifices in performance for very clean data.

+ In a filtering task, examples which are marked as negative are not shown to the user.

+ Sampling bias: Suppose you have a filter that blocks 75% of the negative examples at serving. You might be tempted to draw additional training data from the instances shown to users.

+ Note that if your filter is blocking $95\%$ of the negative examples or more, this becomes less viable.


### <a name="rule-35"></a>  Rule #35: Beware of the inherent skew in ranking problems.

+ When you switch your ranking algorithm radically enough that different results show up, you have effectively changed the data that your algorithm is going to see in the future.

+ Approaches to design model around it
    + Have higher regularization on features that cover more queries as opposed to those features that are on for only one query. -> prevent very popular results from leaking into irrelevant queries.
    + Only allow features to have positive weights. Thus, any good feature will be better than a feature that is “unknown”.
    + Don’t have document-only features.


### <a name="rule-36"></a>  Rule #36: Avoid feedback loops with positional features.

+ The position of content dramatically affects how likely the user is to interact with it.

+ One way to deal with this is to add positional features, i.e. features about the position of the content in the page.

+ Note that it is important to keep any positional features somewhat separate from the rest of the model because of this asymmetry between training and testing.

+ Having the model be the sum of a function of the positional features and a function of the rest of the features is ideal.


### <a name="rule-37"></a>  Rule #37: Measure Training/Serving Skew.

+ Things causing skew
    1. The difference between the performance on the training data and the holdout data.
    2. The difference between the performance on the holdout data and the “next-day” data.  Should tune your regularization to maximize the next-day performance.
    3. The difference between the performance on the “next-day” data and the live data.


## ML Phase III: Slowed Growth, Optimization Refinement, and Complex Models


### <a name="rule-38"></a>  Rule #38: Don’t waste time on new features if unaligned objectives have become the issue.


### <a name="rule-39"></a>  Rule #39: Launch decisions will depend upon more than one metric.


### <a name="rule-40"></a>  Rule #40: Keep ensembles simple.


### <a name="rule-41"></a>  Rule #41: When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.


### <a name="rule-42"></a>  Rule #42: Don’t expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.


### <a name="rule-43"></a>  Rule #43: Your friends tend to be the same across different products. Your interests tend not to be.





