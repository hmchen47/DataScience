# Logistic Regression

Organization: ML Glossary

[Original](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)


## Introduction

+ Logistic regression
  + a classification algorithm used to assign observations to a discrete set of classes
  + transforming output using the sigmoid function to return a probability value
  + mapping probability value to two pr more discrete classes
  + linear regression: outputs w/ continuous number values

+ Linear regression vs logistic regression
  + linear regression
    + predictions w/ continuous value (numbers in a range)
    + example: predicting the student's test score on a scale of 0 - 100
  + logistic regression
    + predictions w/ discrete value (only specific values or categories allowed)
    + viewing probability score underlying the model's classifications
    + example: predicting the student passed or failed

+ Types of logistic regression
  + binary: pass/fail
  + multi: cats, dogs, sheep
  + ordinal: low, medium, high

## Binary logistic Regression

+ Sigmoid activation
  + mapping any real value into another value $\in [0, 1]$
  + ML: mapping predictions to probabilities
  + formula

    \[ S(z) = \frac{1}{1 + e^{-z}} \]

    + $S(z)$: output btw 0 and 1 (probability estimate)
    + $z$: input to the function
    + $e$: base of natural log

  + function graph

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 10vw;"
        onclick= "window.open('https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html')"
        src    = "https://ml-cheatsheet.readthedocs.io/en/latest/_images/sigmoid.png"
        alt    = "Sigmoid function"
        title  = "Sigmoid function"
      />
    </figure>

+ Decision boundary
  + mapping a probability to a discrete class, e.g., true/false, cat/dog
  + selecting a threshold value or tipping point

    \[ class = \begin{cases} 1 & p \ge 0.5 \\ 0 & p < 0.5 \end{cases}\]

  + example: predicted value = 0.7
    + threshold = 0.5 $\to$ positive
    + threshold = 0.2 $\to$ negative
  + multiple classes: selecting the class w/ the highest predicted probability

  <figure style="margin: 0.5em; text-align: center;">
    <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
      onclick= "window.open('https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html')"
      src    = "https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png"
      alt    = "Decision boundary and sigmoid function"
      title  = "Decision boundary and sigmoid function"
    />
  </figure>

+ Example
  + problem and data
    + goal: to predict whether a student will pass or fail
    + data:
      + features: hours slept and hours studied
      + results: student exam results w/ two classes, passed (1) and failed (0)

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick= "window.open('https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html')"
        src    = "https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_exam_scores_scatter.png"
        alt    = "Plot of hours slept, study and exam results"
        title  = "Plot of hours slept, study and exam results"
      />
    </figure>

  + making predictions
    + $P(class = 1)$:
      + class 1
      + prediction function returning the probability of observation
      + probability being positive, True, or "Yes"
    + the probability closer to 1 $\implies$ more confident that the observation in class 1
    + multiple linear regression equation

      \[ z = W_0 + W_1 \cdot H_{studied} + W_2 \cdot H_{slept} \]

    + transforming the output using sigmod function to return a probability

      \[ P(class = 1) = \frac{1}{1 + e^{-z}} \]

    + result: $P(class = 1) = .4 \to$ only 40\% change of passing
    + decision boundary : 0.5
    + #\therefore\;$ this observation "Fail"
    + python snippet

      ```python
      def predict(features, weights):
      '''
      Returns 1D array of probabilities
      that the class label == 1
      '''
      z = np.dot(features, weights)
      return sigmoid(z)
      ```



## Multiclass logistic Regression






