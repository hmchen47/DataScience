# Machine Learning System Design

## Building a Spam Classifier

### Prioritizing What to Work On

#### Lecture Notes

+ Building a spam classifier
  + Classes: Spam (1), Non-spam (0)
  + Supervised learning
  + $x\;$ = features of emails
  + $y\;$ = spam(1) or not spam (0)
  + Features: choose 100 word indicative of spam/not spam, e.g. deal, buy, discount, andrew, now, ...

    $$x_j = \begin{cases} 1 & \text{if word } j \text{ appears in email} \\ 0 & \text{otherwose} \end{cases}$$

    $$X = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \\ \vdots \\ 1 \\ \vdots \end{bmatrix} \quad \begin{matrix} \text{andrew} \\ \text{buy} \\ \text{deal} \\ \text{discount} \\ \vdots \\ \text{now} \\ \vdots \end{matrix} \quad\implies X \;\in\; \mathbb{R}^{100}$$
  
  + Note: In practice, take most frequently occurring $n$ words (10,000 to 50,000) in training set, rather than manually pick 100 words.
  + How to spend your time to make it have low error?
    + Collect lots of data, e.g., "honeypot" project
    + Develop sophisticated features based on email routing information (from email header)
    + Develop sophisticated feature for message body
      + Should "discount" and "discounts" be treated as the same word?
      + How about "deal" and "Dealer"?
      + Features about punctuation?
    + Develop sophisticated algorithm to detect misspellings (e.g. m0rtgage, med1cine, w4tches)

+ IVQ: Which of the following statements do you agree with? Check all that apply.

  1. For some learning applications, it is possible to imagine coming up with many different features (e.g. email body features, email routing features, etc.). But it can be hard to guess in advance which features will be the most helpful.
  2. For spam classification, algorithms to detect and correct deliberate misspellings will make a significant improvement in accuracy.
  3. Because spam classification uses very high dimensional feature vectors (e.g. n = 50,000 if the features capture the presence or absence of 50,000 different words), a significant effort to collect a massive training set will always be a good idea.
  4. There are often many possible ideas for how to develop a high accuracy learning system; “gut feeling” is not a recommended way to choose among the alternatives.

  Ans: 14


----------------------------------------

System Design Example:

Given a data set of emails, we could construct a vector for each email. Each entry in this vector represents a word. The vector normally contains 10,000 to 50,000 entries gathered by finding the most frequently used words in our data set. If a word is to be found in the email, we would assign its respective entry a 1, else if it is not found, that entry would be a 0. Once we have all our x vectors ready, we train our algorithm and finally, we could use it to classify if an email is a spam or not.

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.coursera.org/learn/machine-learning/supplement/0uu7a/prioritizing-what-to-work-on">
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ys5NKOLJEeaPWBJZo44gSg_aba93cf4ce4507175d7e47ab5f9b7ce4_Screenshot-2017-01-24-22.29.45.png?expiry=1555286400000&hmac=zU20vvlv-s5rkAliYcExaNnWD3EZTUaOXhaG_oJTV18" style="margin: 0.1em;" alt="ML System Design Example with a spam classifier" title="Build a spam classifier" width="350">
  </a></div>
</div>

So how could you spend your time to improve the accuracy of this classifier?

+ Collect lots of data (for example "honeypot" project but doesn't always work)
+ Develop sophisticated features (for example: using email header data in spam emails)
+ Develop algorithms to process your input in different ways (recognizing misspellings in spam).

It is difficult to tell which of the options will be most helpful.


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/11.1-MachineLearningSystemDesign-PrioritizingWhatToWorkOn.bfb78210b22b11e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1555286400&Signature=S~sggyayuandyK4fPtfle7sR9o-53TNRbA2ozmg-YC6FadWe62CvSBHIkdSWRws1CY26u26yilFLduVu162hEOA-NKBDeGXYpKj2GCLm3C~8Pv9sqlUbr~ATVO2ge0g6QG3yKZSPjgOks249JGsDABE7OzhytEspePTfQ9iPiLU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/12kP7pG7Q1CpD-6Ru1NQFQ?expiry=1555286400000&hmac=rFs2miXwp8J1GTWjJezlrZYzNbGsXQlgZBRsdnBIJ8w&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Error Analysis

#### Lecture Notes



----------------------------------------



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Handling Skewed Data

### Error Metrics for Skewed Classes

#### Lecture Notes



----------------------------------------



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


### Trading Off Precision and Recall

#### Lecture Notes



----------------------------------------



#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Using Large Data Sets

### Lecture Notes



----------------------------------------



### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>
<br/>


## Review

### Lecture Slides

#### Prioritizing What to Work On

Different ways we can approach a machine learning problem:

+ Collect lots of data (for example "honeypot" project but doesn't always work)
+ Develop sophisticated features (for example: using email header data in spam emails)
+ Develop algorithms to process your input in different ways (recognizing misspellings in spam).

It is difficult to tell which of the options will be helpful.


#### Error Analysis

The recommended approach to solving machine learning problems is:

+ Start with a simple algorithm, implement it quickly, and test it early.
+ Plot learning curves to decide if more data, more features, etc. will help
+ Error analysis: manually examine the errors on examples in the cross validation set and try to spot a trend.

It's important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance.

You may need to process your input before it is useful. For example, if your input is a set of words, you may want to treat the same word with different forms (fail/failing/failed) as one word, so must use "stemming software" to recognize them all as one.


#### Error Metrics for Skewed Classes

It is sometimes difficult to tell whether a reduction in error is actually an improvement of the algorithm.

+ For example: In predicting a cancer diagnoses where 0.5% of the examples have cancer, we find our learning algorithm has a 1% error. However, if we were to simply classify every single example as a 0, then our error would reduce to 0.5% even though we did not improve the algorithm.

This usually happens with __skewed classes__; that is, when our class is very rare in the entire data set.

Or to say it another way, when we have lot more examples from one class than from the other class.

For this we can use __Precision/Recall__.

+ Predicted: 1, Actual: 1 --- True positive
+ Predicted: 0, Actual: 0 --- True negative
+ Predicted: 0, Actual, 1 --- False negative
+ Predicted: 1, Actual: 0 --- False positive

__Precision__: of all patients we predicted where $y=1$, what fraction actually has cancer?

$$\dfrac{\text{True Positives}}{\text{Total number of predicted positives}} = \dfrac{\text{True Positives}}{\text{True Positives} + \text{False positives}}$$

__Recall__: Of all the patients that actually have cancer, what fraction did we correctly detect as having cancer?

$$\dfrac{\text{True Positives}}{\text{Total number of actual positives}}= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False negatives}}$$
​	 
These two metrics give us a better sense of how our classifier is doing. We want both precision and recall to be high.

In the example at the beginning of the section, if we classify all patients as 0, then our __recall__ will be $\dfrac{0}{0 + f} = 0$, so despite having a lower error percentage, we can quickly see it has worse recall.

$$\text{Accuracy} = \frac{\text{true positive } + \text{ true negative }}{\text{ total population }}$$

Note 1: if an algorithm predicts only negatives like it does in one of exercises, the precision is not defined, it is impossible to divide by 0. F1 score will not be defined too.


#### Trading Off Precision and Recall

We might want a __confident__ prediction of two classes using logistic regression. One way is to increase our threshold:

+ Predict 1 if: $h_\theta(x) \geq 0.7$
+ Predict 0 if: $h_\theta(x) < 0.7$

This way, we only predict cancer if the patient has a $70\%$ chance.

Doing this, we will have __higher precision__ but __lower recall__ (refer to the definitions in the previous section).

In the opposite example, we can lower our threshold:

+ Predict 1 if: $h_\theta(x) \geq 0.3$
+ Predict 0 if: $h_\theta(x) < 0.3$

That way, we get a very __safe__ prediction. This will cause __higher recall__ but __lower precision__.

The greater the threshold, the greater the precision and the lower the recall.

The lower the threshold, the greater the recall and the lower the precision.

In order to turn these two metrics into one single number, we can take the __F value__.

One way is to take the __average__:

$$\dfrac{P+R}{2}$$

This does not work well. If we predict all $y=0$ then that will bring the average up despite having 0 recall. If we predict all examples as $y=1$, then the very high recall will bring up the average despite having 0 precision.

A better way is to compute the __F Score__ (or F1 score):

$$\text{F Score} = 2\dfrac{PR}{P + R}$$

In order for the F Score to be large, both precision and recall must be large.

We want to train precision and recall on the __cross validation set__ so as not to bias our test set.


#### Data for Machine Learning

How much data should we train on?

In certain cases, an "inferior algorithm," if given enough data, can outperform a superior algorithm with less data.

We must choose our features to have __enough__ information. A useful test is: Given input $x$, would a human expert be able to confidently predict $y$?

__Rationale for large data__: if we have a __low bias__ algorithm (many features or hidden units making a very complex function), then the larger the training set we use, the less we will have overfitting (and the more accurate the algorithm will be on the test set).


#### Quiz instructions

When the quiz instructions tell you to enter a value to "two decimal digits", what it really means is "two significant digits". So, just for example, the value 0.0123 should be entered as "0.012", not "0.01".

References:
+ [Machine Learning - Coursera/Stanford](https://class.coursera.org/ml/lecture/index)
+ [Bias vs. Variance](http://www.cedar.buffalo.edu/~srihari/CSE555/Chap9.Part2.pdf)
+ [Managing Bias - Variance Tradeoff in Machine Learning](http://blog.stephenpurpura.com/post/13052575854/managing-bias-variance-tradeoff-in-machine-learning)
+ [Bias-Variance Tradeoff in ML](https://cedar.buffalo.edu/~srihari/CSE574/Chap3/3.3-Bias-Variance.pdf)


### Quiz: Machine Learning System Design





