# Module 3: [Evaluation](./03-Evaluation.md)

## Module 3 Notebook

+ [Web Note Launch Page](https://www.coursera.org/learn/python-machine-learning/notebook/g7cJG/module-3-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/user/elkljxyoytcwjbmkgctrtg/notebooks/Module%203.ipynb)
+ [Local Notebook](./notebooks/Module03.ipynb)
+ [Local Python Code](./notebooks/Module03.py)

## Model Evaluation & Selection

### Note

+ Learning objectives
    + Understand why accuracy only gives a partial picture of a classifier's performance.
    + Understand the motivation and definition of important evaluation metrics in machine learning.
    + Learn how to use a variety of evaluation metrics to evaluate supervised machine learning models.
    + Learn about choosing the right metric for selecting between models or for doing parameter tuning.

+ Represent / Train / Evaluate / Refine Cycle
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
        <br/><img src="images/fig3-01.png" alt="So let's return for a moment to this workflow diagram that we introduced earlier in the course. You see that evaluation is a key part of this development cycle in applied machine learning. Once a model is trained, the evaluation step provides critical feedback on the trained model's performance characteristics. Particularly those that might be important for your application. The results of the evaluation step, for example, might help you understand which data instances are being classified or predicted incorrectly. Which might in turn suggest better features or different kernel function or other refinements to your learning model in the feature and model refinement phase. As we discussed earlier, the objective function that's optimized during the training phase may be a different, what's called a surrogate metric. That's easier to use in practice for optimization purposes than what's used for the evaluation metric. For example, a commercial search engine might use a ranking algorithm that is trained to recommend relevant web pages that best match a query. In other words, trying to predict a relevant label for a page. And that might be the objective in the training phase. But there are many evaluation methods in the evaluation phase that could be applied to measure aspects of that search engine's performance using that ranking algorithm, that are important to the search company's business, for example. Such as how many unique users the system sees per day. Or how long the typical user search session is and so on. " title= "caption" height="250">
    </a>
    + surrogate metric: the objective function that's optimized during the training phase may be a different

+ Evaluation
    + Different applications have very different goals
    + Accuracy is widely used, but many others are possible, e.g.:
        + User satisfaction (Web search)
        + Amount of revenue (e-commerce)
        + Increase in patient survival rates (medical)
    + It's very important to choose evaluation methods that match the goal of your application.
    + Compute your selected evaluation metric for multiple different models.
    + Then select the model with 'best' value of evaluation metric.

+ Accuracy with __imbalanced classes__
    + Suppose you have two classes:
        + Relevant (__R__): the _positive_ class
        + Not_Relevant(__N__): the _negative_ class
    + Out of 1000 randomly selected items, on average
        + One item is relevant and has an R label
        + The rest of the items (999 of them) are not relevant and labelled N.
    + Recall that:

        $$\text{Accuracy} = \frac{\text{\# correct predictions}}{\text{\# total instances}}$$
    + You build a classifier to predict relevant items, and see that its accuracy on a test set is $99.9\%$.
    + Wow! Amazingly good, right?
    + For comparison, suppose we had a "dummy" classifier that didn't look at the features at all, and always just blindly predicted the most frequent class (i.e. the negative N class).
    + Assuming a test set of $1000$ instances, what would this dummy classifier's accuracy be?
    + Answer: $\text{Accuracy}_{\text{DUMMY}} = 999 / 1000 = 99.9\%$

+ Dummy classifiers completely ignore the input data!
    + Dummy classifiers serve as a sanity check on your classifier's performance.
    + They provide a __null metric__ (e.g. null accuracy) baseline.
    + Dummy classifiers should not be used for real problems.
    + Some commonly-used settings for the `strategy` parameter for `DummyClassifier` in scikit-learn:
        + __most_frequent__: predicts the most frequent label in the training set.
        + __stratified__: random predictions based on training set class distribution.
        + __uniform__: generates predictions uniformly at random.
        + __constant__: always predicts a constant label provided by the user.
            + A major motivation of this method is F1-scoring, when the positive class is in the minority.
    + Use metrics other than accuracy
    + AUC: under the curve

+ What if my classifier accuracy is close to the null accuracy baseline? <br/>
    This could be a sign of:
    + Ineffective, erroneous or missing features
    + Poor choice of kernel or hyperparameter
    + Large class imbalance 

+ Dummy regressors <br/>
    + `DummyRegression` for Regression as counterpart of `DummyClassifier` for Classifier
    + `strategy` parameter options:
        + _mean_: predicts the mean of the training targets.
        + _median_: predicts the median of the training targets.
        + _quantile_: predicts a user-provided quantile of the training targets.
        + _constant_: predicts a constant user-provided value.

+ Binary prediction outcomes
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
        <br/><img src="images/fig3-02.png" alt="Now let's look more carefully at the different types of outcomes we might see using a binary classifier. This will give us some insight into why using just accuracy doesn't give a complete picture of the classifier's performance. And will motivate our definition and exploration of additional evaluation metrics. With a positive and negative class, there are four possible outcomes that we can break into two cases corresponding to the first and second row of this matrix. If the true label for an instance is negative, the classifier can predict either negative, which is correct, and call the true negative. Or it can erroneously predict positive, which is an error and called a false positive. If the true label for an instance is positive, the classifier can predict either negative, which is an error and called a false negative. Or it can predict positive, which is correct and that's called a true positive. So maybe a quick way to remember this is that the first word in these matrix cells is false, if it's a a classifier error, or true if it's a classifier success. The second word is negative if the true label is negative and positive if the true label is positive. Another name for a false positive that you might know from statistics is a type one error. And another name for a false negative is a type two error. We're going to use these two-letter combinations, TN, FN, FP, and TP, as variable names, when defining some new evaluation metrics shortly. We'll also use capital N here to denote the total number of instances, of the sum of all the values in the matrix, the number of data points we're looking at. This matrix of all combinations of predicted label and true label is called a confusion matrix. " title= "confusion matrix" height="150">
    </a>
    <a href="https://www.researchgate.net/publication/230614354_How_to_evaluate_performance_of_prediction_methods_Measures_and_their_interpretation_in_variation_effect_analysis/figures?lo=1">
        <img src="https://www.researchgate.net/profile/Mauno_Vihinen/publication/230614354/figure/fig4/AS:216471646019585@1428622270943/Contingency-matrix-and-measures-calculated-based-on-it-2x2-contigency-table-for_W640.jpg" alt="Contingency matrix and measures calculated based on it 2x2 contigency table for displaying the outcome of predictions. Based on the table it is possible to calculate row and column wise parameters, PPV and NVP, and sensitivity and specificity, respectively. These parameters are useful, but are not based on all the information in the table. Accuracy is a measure that is calculated based on all the four figures in the table." title= "Contingency matrix and measures" height="250">
    </a>
    + confusion matrix: This matrix of all combinations of predicted label and true label
    + N: total number of instances

+ Confusion matrix for binary prediction task
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
        <br/><img src="images/fig3-03.png" alt="We can take any classifier prediction on a data instance and associate it with one of these matrix cells, depending on the true label of the instance and the classifier's predicted label. This also applies to multi-class classification, in addition to the special case of binary classification I've shown here. In the multi-class case with k classes, we simply have a k by k matrix instead of a two by two matrix. Scikit-learn makes it easy to compute a confusion matrix for your classifier.  " title= "caption" height="150">
    </a>
    + Every test instance is in exactly one box (integer counts).
    + Breaks down classifier results by error type.
    + Thus, provides more information than simple accuracy.
    + Helps you choose an evaluation metric that matches project goals.
    + Not a single number like accuracy.. but there are many possible metrics that can be derived from the confusion matrix.
    + Multi-class classifier with k classes: $k x k$ matrix


+ Demo
    ```python
    %matplotlib notebook
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits

    dataset = load_digits()
    X, y = dataset.data, dataset.target

    for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
        print(class_name,class_count)
    # 0 178
    # 1 182
    # 2 177
    # 3 183
    # 4 181
    # 5 182
    # 6 181
    # 7 179
    # 8 174
    # 9 180

    # Creating a dataset with imbalanced binary classes:  
    # Negative class (0) is 'not digit 1' 
    # Positive class (1) is 'digit 1'
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0

    print('Original labels:\t', y[1:30])
    print('New binary labels:\t', y_binary_imbalanced[1:30])
    # Original labels:  [1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]
    # New binary labels:  [1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]

    np.bincount(y_binary_imbalanced)    # Negative class (0) is the most frequent class
    # array([1615,  182], dtype=int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    # 0.90888888888888886

    # Accuracy of Support Vector Machine classifier
    from sklearn.svm import SVC

    svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
    svm.score(X_test, y_test)

    # ### Dummy Classifiers
    # DummyClassifier is a classifier that makes predictions using simple rules, which 
    # can be useful as a baseline for comparison against actual classifiers, especially 
    # with imbalanced classes.
    from sklearn.dummy import DummyClassifier

    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)
    # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        ...    ...     ...     ...     ...     ...     ...     ...     ...
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    dummy_majority.score(X_test, y_test)

    svm = SVC(kernel='linear', C=1).fit(X_train, y_train)   # 0.9044444444444445
    svm.score(X_test, y_test)                               # 0.97777777777777775

    # ### Confusion matrices
    # #### Binary (two-class) confusion matrix
    from sklearn.metrics import confusion_matrix

    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_majority_predicted = dummy_majority.predict(X_test)
    confusion = confusion_matrix(y_test, y_majority_predicted)
    print('Most frequent class (dummy classifier)\n', confusion)
    # Most frequent class (dummy classifier)
    #  [[407   0]
    #  [ 43   0]]

    # produces random predictions w/ same class proportion as training set
    dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)
    y_classprop_predicted = dummy_classprop.predict(X_test)
    confusion = confusion_matrix(y_test, y_classprop_predicted)
    print('Random class-proportional prediction (dummy classifier)\n', confusion)
    # Random class-proportional prediction (dummy classifier)
    #  [[372  35]
    #  [ 39   4]]

    svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    confusion = confusion_matrix(y_test, svm_predicted)
    print('Support vector machine classifier (linear kernel, C=1)\n', confusion)
    # Support vector machine classifier (linear kernel, C=1)
    #  [[402   5]
    #  [  5  38]]

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression().fit(X_train, y_train)
    lr_predicted = lr.predict(X_test)
    confusion = confusion_matrix(y_test, lr_predicted)
    print('Logistic regression classifier (default settings)\n', confusion)
    # Logistic regression classifier (default settings)
    #  [[401   6]
    #  [  6  37]]

    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
    tree_predicted = dt.predict(X_test)
    confusion = confusion_matrix(y_test, tree_predicted)
    print('Decision tree classifier (max_depth = 2)\n', confusion)
    # Decision tree classifier (max_depth = 2)
    #  [[400   7]
    #  [ 17  26]]
    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Confusion Matrices & Basic Evaluation Metrics

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Classifier Decision Functions

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Precision-recall and ROC curves

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Multi-Class Evaluation

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Regression Evaluation

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Practical Guide to Controlled Experiments on the Web (optional)

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Model Selection: Optimizing Classifiers for Different Evaluation Metrics

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Quiz: Module 3 Quiz

### Note


+ Demo
    ```python

    ```

    <a href="url">
        <br/><img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

