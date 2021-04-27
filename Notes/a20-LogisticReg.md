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

+ Cross-Entropy cost function
  + a.k.a. Log Loss
  + unable to use the same cost function. Mean Squared Error or L2 loss, as linear regression
  + the non-linear prediction function of sigmoid
    + squaring prediction as in MSE in a non-convex function $\to$ many local minimums
    + gradient descent probably unable to find the optimal global minimum
  + divided into two separate cost functions for different classes

    \[\begin{align*} &J(\theta) = \frac1m \sum_{i=1}^m \text{Cost}\left(h_\theta(x^{(i)}), y^{(i)}\right) \\
      & \text{Cost}\big(h_\theta(x), y\big) = \begin{cases} -\log\big(h_\theta(x)\big) & \text{if } y=1 \\ -\log\big(1 - h_\theta(x)\big) & \text{if } y = 0 \end{cases}\\\\
      \implies\hspace{0.5em} &J(\theta) = -\frac1m \sum_{i=1}^m \left[ y^{(i)} \log\big(h_\theta(x^{(i)})\big) + \big(1 - y^{(i)}\big) \log\big(1 - h_\theta(x^{(i)})\big)\right]
    \end{align*}\]

  + benefits:
    + smooth monotonic functions
    + easy to calculate the gradient and minimize cost

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick= "window.open('https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html')"
        src    = "https://ml-cheatsheet.readthedocs.io/en/latest/_images/y1andy2_logistic_function.png"
        alt    = "Cross Entropy cost function for different classes"
        title  = "Cross Entropy cost function for different classes"
      />
    </figure>

  + limitation: penalizing confident and wrong predictions than reward confident and correct predictions
  + vectorized cost function

    \[\begin{align*}
      h &= g(X \theta) \\\\
      J(\theta) &= \frac1m \cdot \Big( -y^T \log(h) - (1- y)^T \log(1 - h)\Big)
    \end{align*}\]

  + Python snippet

    ```python
    def cost_function(features, labels, weights):
      '''
      Using Mean Absolute Error

      Features:(100,3)
      Labels: (100,1)
      Weights:(3,1)
      Returns 1D matrix of predictions
      Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
      '''
      observations = len(labels)

      predictions = predict(features, weights)

      #Take the error when label=1
      class1_cost = -labels*np.log(predictions)

      #Take the error when label=0
      class2_cost = (1-labels)*np.log(1-predictions)

      #Take the sum of both costs
      cost = class1_cost - class2_cost

      #Take the average cost
      cost = cost.sum() / observations

      return cost
    ```

+ Gradient descent
  + minimizing the cost function
  + derivative of the sigmoid function:
    + $\frac{d}{d x}\sigma(x) = \sigma(x) \big(1 - \sigma(x)\big)$
    + ref: [detailed derivative](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x)
  + cost function derivative

    \[ C^\prime = x\big(s(z) - y \big) \]

    + $C^\prime$: the derivative of cost w.r.t. weights
    + $y$: the actual class label (0 or 1)
    + $s(z)$: model's prediction
    + $x$: the feature or feature vector
  + pseudocode

    ```shell
    Repeat {
      1. Calculate gradient average
      2. Multiply by learning rate
      3. Subtract from weights
    }
    ```

  + python snippet

    ```python
    def update_weights(features, labels, weights, lr):
      '''
      Vectorized Gradient Descent

      Features:(200, 3)
      Labels: (200, 1)
      Weights:(3, 1)
      '''
      N = len(features)

      #1 - Get Predictions
      predictions = predict(features, weights)

      #2 Transpose features from (200, 3) to (3, 200)
      # So we can multiply w the (200,1)  cost matrix.
      # Returns a (3,1) matrix holding 3 partial derivatives --
      # one for each feature -- representing the aggregate
      # slope of the cost function across all observations
      gradient = np.dot(features.T,  predictions - labels)

      #3 Take the average cost derivative for each feature
      gradient /= N

      #4 - Multiply the gradient by our learning rate
      gradient *= lr

      #5 - Subtract from our weights to minimize cost
      weights -= gradient

      return weights
    ```

+ Mapping probabilities to classes
  + the final step: assign class labels (0 or 1) to the predicted probabilities
  + procedure
    + decision boundary
    + convert probabilities to classes

  ```python
  # decision boundary
  def decision_boundary(prob):
    return 1 if prob >= .5 else 0

  # convert probabilities to classes
  def classify(predictions):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize(decision_boundary)
    return decision_boundary(predictions).flatten()
  ```

+ Training

  ```python
  def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 1000 == 0:
            print "iter: "+str(i) + " cost: "+str(cost)

    return weights, cost_history
  ```

+ Model evaluation
  + cost history plot
  + accuracy: measuring how correct predictions were

    ```python
    def accuracy(predicted_labels, actual_labels):
      diff = predicted_labels - actual_labels
      return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
    ```

  + decision boundary
    + technique to plot the decision boundary on the top of the predictions
    + used to observe how the predicted labels compare to the actual labels

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

  + model evaluation
    + mapping probabilities to classes
      + decision boundary
      + convert probabilities to classes
      + example output
        + Probabilities = [ 0.967, 0.448, 0.015, 0.780, 0.978, 0.004]
        + Classifications = [1, 0, 0, 1, 1, 0]
    + cost and weights
      + iter: 0 cost: 0.635
      + iter: 1000 cost: 0.302
      + iter: 2000 cost: 0.264  
      + Final cost: 0.2487. Final weights: [-8.197, .921, .738]

      <figure style="margin: 0.5em; text-align: center;">
        <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
          onclick= "window.open('https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html')"
          src    = "https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_loss_history.png"
          alt    = "Plot of cost value w/ epochs"
          title  = "Plot of cost value w/ epochs"
        />
      </figure>

    + accuracy: simply comparing predicted labels to true labels and divide by the total
    + decision boundary

      <figure style="margin: 0.5em; text-align: center;">
        <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
          onclick= "window.open('https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html')"
          src    = "https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_loss_history.png"
          alt    = "Visualization of decision boundary"
          title  = "Visualization of decision boundary"
        />
      </figure>

      ```python
      def plot_decision_boundary(trues, falses):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        no_of_preds = len(trues) + len(falses)

        ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
        ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

        plt.legend(loc='upper right');
        ax.set_title("Decision Boundary")
        ax.set_xlabel('N/2')
        ax.set_ylabel('Predicted Probability')
        plt.axhline(.5, color='black')
        plt.show()
      ```


## Multiclass logistic Regression

+ Procedure
  + divide the problem into n+1 binary classification problems
  + for each class ...
  + predict the probability the observation are in that single class
  + prediction = &lt;math&gt;max(probability of the classes)

+ Softmax activation
  + a.k.a. softargmax or normalized exponential function
  + taking input as a vector of $K$ real numbers and normalizing it into a probability distribution
  + distribution consisting of $k$ probabilities proportional to the exponentials of the input numbers
  + applying softmax to values of vector components
    + prior: probably negative, greater than 1 and not summing to 1
    + post: components $\in [0, 1]$ and sum to 1
  + formula for standard (unit) softmax functions

    \[ \sigma(z_i) = \frac{e^{z_(i)}}{\sum_{j=1}^K s^{z_{(j)}}}, \hspace{1em}\text{for } i = 1, \dots, K, z = z_1, \dots, z_K \]

  + applying the standard exponential function to each element $z_i$ of the input vector $z$ and normalize these values by dividing by the sum of these exponentials
  + normalization ensuring $\sum \sigma(z) = 1$

+ Scikit-Learn example

  ```python
  import sklearn
  from sklearn.linear_model import LogisticRegression
  from sklearn.cross_validation import train_test_split

  # Normalize grades to values between 0 and 1 for more efficient computation
  normalized_range = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

  # Extract Features + Labels
  labels.shape =  (100,) #scikit expects this
  features = normalized_range.fit_transform(features)

  # Create Test/Train
  features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.4)

  # Scikit Logistic Regression
  scikit_log_reg = LogisticRegression()
  scikit_log_reg.fit(features_train,labels_train)

  #Score is Mean Accuracy
  scikit_score = clf.score(features_test,labels_test)
  print 'Scikit score: ', scikit_score

  #Our Mean Accuracy
  observations, features, labels, weights = run()
  probabilities = predict(features, weights).flatten()
  classifications = classifier(probabilities)
  our_acc = accuracy(classifications,labels.flatten())
  print 'Our score: ',our_acc
  ```





