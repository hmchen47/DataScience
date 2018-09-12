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

<a href="https://d3c33hcgiwev3.cloudfront.net/n4ge2T6FEee3MRIl4lCYSA.processed/full/360p/index.mp4?Expires=1536883200&Signature=Nb8qRgdnN4JrgKmkaiI7JXbQdRBVWFNvWfW4peY5JQTJg9wL4OeM0ny5Vj9q8~BuvfXsdWtAlGjQUUtQH2hRv6~byZbIbUbZR-~8yuPM14ecmjoQL2oGeaFjnX3B7u-mPaQqgiZf-ZxJ0kptP5Pbo51Lhk0J2BeVK0SmMIIKeSo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Model Evaluation & Selection" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Confusion Matrices & Basic Evaluation Metrics

### Note

+ Confusion Matrix for Binary Prediction Task
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
        <br/><img src="images/fig3-04.png" alt="So, let's go back to the matrix of possible binary classification outcomes. This time filled out with the actual counts from the notebooks decision tree output. Remember our original motivation for creating this matrix was to go beyond a single number accuracy, to get more insight into the different types of prediction successes and failures of a given classifier. Now we have these four numbers that we can examine and compare manually. " title= "Confusion table" height="150">
    </a>

    + Always look at the confusion matrix for your classifier

+ Visualization of Different Error Types
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection"> 
        <br/><img src="images/fig3-05.png" alt="Let's look at this classification result visually to help us connect these four numbers to a classifier's performance. What I've done here is plot the data instances by using two specific feature values out of the total 64 feature values that make up each instance in the digits dataset. The black points here are the instances with true class positive namely the digit one and the white points have true class negative, that is, there are all the other digits except for one. The black line shows a hypothetical linear classifier's decision boundary for which any instance to the left of the decision boundary is predicted to be in the positive class and everything to the right of the decision boundary is predicted to be in the negative class. The true positive points are those black points in the positive prediction region and false positives are those white points in the positive prediction region. Likewise, true negatives are the white points in the negative prediction region and false negatives are black points in the negative prediction region. " title= "caption" height="200">
    </a>
    + Black points: True class positive, namely digit one
    + White pointss: True class negative
    + Black line: a hypothetical linear classifier's decision boundary, positive class on left and negative class on right
    + True Positive (TP): black points in the positive prediction region
    + True Negative (TF): white points in the positive prediction region
    + False Positive (FP): white points in the positive prediction region
    + False Negative (FN): black points in the positive prediction region

+ Confusion Matrix for Binary Prediction Task
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection"> <br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <img src="images/fig3-04.png" alt="We've already seen one metric that can be derived from the confusion matrix counts namely accuracy. The successful predictions of the classifier, the ones where the predicted class matches the true class are along the diagonal of the confusion matrix. So, if we add up all the accounts along the diagonal, that will give us the total number of correct predictions across all classes, and dividing this sum by the total number of instances gives us accuracy. But, let's look at some other evaluation metrics we can compute from these four numbers. Well, a very simple related number that's sometimes used is classification error, which is the sum of the counts off the diagonal namely all of the errors divided by total instance count, and numerically, this is equivalent to just one minus the accuracy. Now, for a more interesting example, let's suppose, going back to our medical tumor detecting classifier that we wanted an evaluation metric that would give higher scores to classifiers that not only achieved the high number of true positives but also avoided false negatives. That is, that rarely failed to detect a true cancerous tumor. Recall, also known as the true positive rate, sensitivity or probability of detection is such an evaluation metric and it's obtained by dividing the number of true positives by the sum of true positives and false negatives. You can see from this formula that there are two ways to get a larger recall number. First, by either increasing the number of true positives or by reducing the number of false negatives. Since this will make the denominator smaller. In this example there are 26 true positives and 17 false negatives which gives a recall of 0.6. Now suppose that we have a machine learning task, where it's really important to avoid false positives. In other words, we're fine with cases where not all true positive instances are detected but when the classifier does predict the positive class, we want to be very confident that it's correct. A lot of customer facing prediction problems are like this, for example, predicting when to show a user A query suggestion in a web search interface might be one such scenario. Users will often remember the failures of a machine learning prediction even when the majority of predictions are successes. So, precision is an evaluation metric that reflects the situation and is obtained by dividing the number of true positives by the sum of true positives and false positives. So to increase precision, we must either increase the number of true positives the classifier predicts or reduce the number of errors where the classifier incorrectly predicts that a negative instance is in the positive class. Here, the classifier has made seven false positive errors and so the precision is 0.79. Another related evaluation metric that will be useful is called the false positive rate, also known as specificity. This gives the fraction of all negative instances that the classifier incorrectly identifies as positive. Here, we have seven false positives, which out of a total of 407 negative instances, gives a false positive rate of 0.02. " title= "Confusion Matrix for Binary Prediction Task" height="150">
    </a>
    + __Accuracy__: for what fraction of all instances is the classifier's prediction correct (for either positive or negative class)?

        $$\text{Accuracy} = \frac{NP + T{}{TN + TP + FN + FP}} = \frac{400 + 26}{400 + 26 + 17 + 7} = 0.95$$
    + __Classification error (1 – Accuracy)__: for what fraction of all instances is the classifier's prediction incorrect?

        $$ ClassificationError = \frac{FP + FN}{TN + TP + FN + FP} = \frac{7 + 17}{400 + 26 + 17 + 7} = 0.060$$
    + __Recall__, or __True Positive Rate (TPR)__: what fraction of all positive instances does the classifier correctlyidentify as positive?

        $$Recall = \frac{TP}{TP + FN} = \frac{26}{26 + 17} = 0.60$$
        + Recall is also known as
            + True Positive Rate (TPR)
            + Sensitivity
            + Probability of detection
    + __Precision__: what fraction of positive predictions are correct?

        $$ Precision = \frac{TP}{TP + FP} = \frac{26}{26 + 7} = 0.79 $$
    + __False positive rate (FPR)__: what fraction of all negative instances does the classifier incorrectly identify as positive?

        $$ FPR = \frac{FP}{TN + FP} = \frac{7}{400 + 7} = 0.02$$
        + a.k.a. __Specificity__

+ A Graphical Illustration of Precision & Recall
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
        <br/><img src="images/fig3-06.png" alt="Going back to our classifier visualization, let's look at how precision and recall can be interpreted. The numbers that are in the confusion matrix here are derived from this classification scenario. " title= "Graphical Illustration of Precision & Recall" height="200">
    </a>

    + The Precision-Recall Tradeoff
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
            <br/><img src="images/fig3-07.png" alt="We can see that a precision of 0.68 means that about 68 percent of the points in the positive prediction region to the left of the decision boundary or 13 out of the 19 instances are correctly labeled as positive. A recall of 0.87 means, that of all true positive instances, so all black points in the figure, the positive prediction region has 'found about 87 percent of them' or 13 out of 15. " title= "Precision-Recall Tradeoff" height="200">
        </a>

    + High Precision, Lower Recall
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
            <br/><img src="images/fig3-08.png" alt="If we wanted a classifier that was oriented towards higher levels of precision like in the search engine query suggestion task, we might want a decision boundary instead that look like this. Now, all the points in the positive prediction region seven out of seven are true positives, giving us a perfect precision of 1.0. Now, this comes at a cost because out of the 15 total positive instances eight of them are now false negatives, in other words, they're incorrectly predicted as being negative. And so, recall drops to 7 divided by 15 or 0.47. " title= "High Precision, Lower Recall" height="200">
        </a>

    + Low Precision, High Recall
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection">
            <br/><img src="images/fig3-09.png" alt="On the other hand, if our classification task is like the tumor detection example, we want to minimize false negatives and obtain high recall. In which case, we would want the classifier's decision boundary to look more like this. Now, all 15 positive instances have been correctly predicted as being in the positive class, which means these tumors have all been detected. However, this also comes with a cost since the number of false positives, things that the detector triggers as possible tumors for example that are actually not, has gone up. So, recall is a perfect 1.0 score but the precision has dropped to 15 out of 42 or 0.36. " title= "Low Precision, High Recall" height="200">
        </a>

+ There is often a tradeoff between precision and recall
    + Recall-oriented machine learning tasks:
        + Search and information extraction in legal discovery
        + Tumor detection
        + Often paired with a human expert to filter out false positives
    + Precision-oriented machine learning tasks:
        + Search engine ranking, query suggestion
        + Document classification
        + Many customer-facing tasks (users remember failures!)
    + F1-score: combining precision & recall into a single number

        $$ F_1 = 2 \cdot \frac{Precision \cdot Recall}{precision + Recall} = \frac{2 \cdot TP}{2 \cdot TP + FN + FP}$$
    + F-score: generalizes F1-score for combining precision & recall into a single number

        $$F_{\beta} = (1 + \beta^2) \cdot \frac{Precision \cdot Recall}{(\beta^2 \cdot Precision) + Recall} = \frac{(1 + \beta^2) \cdot TP}{(1 + \beta^2) \cdot TP + \beta \cdot FN + FP}$$
        + $\beta$ allows adjustment of the metric to control the emphasis on recall vs precision:
            + Precision-oriented users: $\beta = 0.5$ (false positives hurt performance more than false negatives)
            + Recall-oriented users: $\beta = 2$ (false negatives hurt performance more than false positives)

+ Demo
    ```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Accuracy = TP + TN / (TP + TN + FP + FN)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
    # F1 = 2 * Precision * Recall / (Precision + Recall) 
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))
    # Accuracy: 0.95
    # Precision: 0.79
    # Recall: 0.60
    # F1: 0.68

    # Combined report with all above metrics
    from sklearn.metrics import classification_report

    print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
    #                 precision  recall  f1-score   support
    #       not 1     0.96       0.98    0.97       407
    #           1     0.79       0.60    0.68        43
    # avg / total     0.94       0.95    0.94       450

    print('Random class-proportional (dummy)\n', 
        classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))
    print('SVM\n', 
        classification_report(y_test, svm_predicted, target_names = ['not 1', '1']))
    print('Logistic regression\n', 
        classification_report(y_test, lr_predicted, target_names = ['not 1', '1']))
    print('Decision tree\n', 
        classification_report(y_test, tree_predicted, target_names = ['not 1', '1']))
    # Random class-proportional (dummy)
    #               precision    recall  f1-score   support
    # 
    #       not 1       0.91      0.91      0.91       407
    #           1       0.10      0.09      0.10        43
    # 
    # avg / total       0.83      0.84      0.83       450
    # 
    # SVM
    #               precision    recall  f1-score   support
    # 
    #       not 1       0.99      0.99      0.99       407
    #           1       0.88      0.88      0.88        43
    # 
    # avg / total       0.98      0.98      0.98       450
    # 
    # Logistic regression
    #               precision    recall  f1-score   support
    # 
    #       not 1       0.99      0.99      0.99       407
    #           1       0.86      0.86      0.86        43
    # 
    # avg / total       0.97      0.97      0.97       450
    # 
    # Decision tree
    #               precision    recall  f1-score   support
    # 
    #       not 1       0.96      0.98      0.97       407
    #           1       0.79      0.60      0.68        43
    # 
    # avg / total       0.94      0.95      0.94       450
    ```

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/4gasyz6FEee2TA5yccyTSg.processed/full/360p/index.mp4?Expires=1536883200&Signature=JVw~A~1O9I1P2g8TM3V8w9flpCq037msbx0ihmQ24tyZoOv11XNoZbOTdW7i1TWp062vIUh8Coo3Nc~2mtLgmhD820CdHVqkLYkjH1zn0hgGaE09SMkQhYiJmUnANXQmQFp52qjadBJ5zbKonXrk~AErCYcX02y2Z01L2nr96gE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Confusion Matrices & Basic Evaluation Metrics" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Classifier Decision Functions

### Note

+ Decision Functions (decision_function)
    + Each classifier score value per test point indicates how confidently the classifier predicts the positive class (large-magnitude positive values) or the negative class (large-magnitude negative values).
    + Choosing a fixed decision threshold gives a classification rule.
    + By sweeping the decision threshold through the entire range of possible score values, we get a series of classification outcomes that form a curve.

+ Predicted Probability of Class Membership (predict_proba)
    + Typical rule: choose most likely class
        + e.gclass 1 if threshold > 0.50.
    + Adjusting threshold affects predictions of classifier.
    + Higher threshold results in a more conservative classifier
        + e.g. only predict Class 1 if estimated probability of class 1 is above 70%
        + This increases precision. Doesn't predict class 1 as often, but when it does, it gets high proportion of class 1 instances correct.
    + Not all models provide realistic probability estimates

+ Varying the Decision Threshold

    | True Label | Classifier score | | True Label | Classifier score |
    |------------|------------------|-|------------|------------------|
    | 0 | -27.6457 | | 0 | -25.8486 |
    | 0 | -25.1011 | | 0 | -24.1511 |
    | 0 | -23.1765 | | 0 | -22.575 |
    | 0 | -21.8271 | | 0 | -21.7226 |
    | 0 | -19.7361 | | 0 | -19.5768 |
    | 0 | -19.3071 | | 0 | -18.9077 |
    | 0 | -13.5411 | | 0 | -12.8594 |
    | 1 | -3.9128 |  | 0 | -1.9798 |
    | 1 | 1.824 |  | 0 | 4.74931 |
    | 1 | 15.234624 |  | 1 | 21.20597 |

    | Classifier score threshold | Precision | Recall |
    |----------------------------|-----------|--------|
    | -20 | 4/12=0.34 | 4/4=1.00 |
    | -10 | 4/6=0.67 | 4/4=1.00 |
    | 0 | 3/4=0.75 | 3/4=0.75 |
    | 10 | 2/2=1.0 | 2/4=0.50 |
    | 20 | 1/1=1.0 | 1/4 = 0.25 |

    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0YPe1/classifier-decision-functions">
        <img src="images/fig3-10.png" alt="Now, we can use these decision scores or prediction probabilities for getting more complete evaluation picture of a classifiers performance. For a particular application, we might pick a specific decision threshold depending on whether we want the classifier to be more or less conservative about making false-positive or false-negative errors. It might not be entirely clear when developing a new model, what the right decision threshold would be, and how that choice will affect evaluation metrics like precision and recall. So instead, what we'll do is, look at how classifier performs for all possible decision thresholds. This example shows how that works. On the left here is a list of test instances with their true label and classifier score. If we set a decision threshold, then all the instances above that line, for example if we set the decision threshold to be -20 here. Then, all the instances above the line are below the threshold of -20. So -20 or less and all the instances in this direction are above the threshold of -20. And so the ones below the threshold will be predicted to be in the- class. And the ones above the threshold will be predicted to be in the + class. So, if we pick the specific threshold, so in this case, -20. And we partition the test points in this way. We can compute partition and recall for the points that are predicted to be in the positive class. So in this case, we have 12 instances here, 12 total instances. They're being predicted as positive and only four of them, this one, this one, this one, and this one are actually positive and so the precision here is 4 divided by 12 or approximately 0.34. The recall on the other hand, there are four positive labeled instances in the whole set of test examples here and we've found all of them with this particular threshold setting. So the recall here is 4 out of 4, we found all four positive labeled examples. And so, for this particular threshold of -20, we can obtain precision on re cost score for that threshold. Let's pick a different threshold let's look at what happened when the threshold is -10? Right here, so again anything below this line is treated and has a higher value than -10 here, so those would be treated as + predictions. Things above the line have a score below -10, so these would be predicted to be And again, we can compute a precision and recall for this decision threshold setting, and we can see here that there are a total of six instances in the + prediction class. Of which four are actually of the positive class, and so the precision here is 4 over 6 or about 0.67. And again, the recall here is going to be 4 out of 4, and it's going to be 1.0. Again, so that corresponds to this point in the table over here. And then as were computing these different precision and recalls for different Thresholds. We can also plot them on this precision recall chart. So the first pair of precision recall numbers that I got, 0.34 and 1.0, we can plot on this point in precision recall space. The second example, so this was for the threshold of -20. When the threshold was -10, we got precision of .67 and a recall of 1 corresponding to this point that we can plot. And so you can see that if we do this for a number of other thresholds, for example the threshold of 0, we'll get a precision of 0.75. And a recall of 0.75 that corresponds to this point. And in that choice of decision threshold. And we can keep doing that for different thresholds. And we actually are plotting a series of points through the space which we can be connected at as a curve. And so in this way, we can get a more complete picture by varying the threshold of how the precision and recall of the result and classifier output changes as a function of the decision threshold. And this resulting chart here is called a precision recall curve." title= "Decision Threshold" height="200">
    </a>


+ Demo
    ```python
    # ### Decision functions
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
    y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))
    # [(0, -23.176547400757663), (0, -13.541223742469006), (0, -21.722500473360618),
    #  (0, -18.907578437722535), (0, -19.736034587372778), (0, -9.7493459511792651),
    #  (1, 5.2349002124953099),  (0, -19.30716117885968),  (0, -25.101015601857377),
    #  (0, -21.827250934235906), (0, -24.150855663826746), (0, -19.576844844946265),
    #  (0, -22.574902551102674), (0, -10.823739601445064), (0, -11.912425566043064),
    #  (0, -10.97956652705531),  (1, 11.205846086251944),  (0, -27.645770221552823),
    #  (0, -12.859636015637092), (0, -25.848590145556187)]

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
    y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))
    # [(0, 8.6010872706740499e-11), (0, 1.3155903495453823e-06), (0, 3.6816111034009875e-10),
    #  (0, 6.1452989618944584e-09), (0, 2.6837934145133791e-09), (0, 5.8329401240781557e-05),
    #  (1, 0.99470087426871634),    (0, 4.1210362715903745e-09)  (0, 1.2553575357627774e-11),
    #  (0, 3.3154719959007555e-10), (0, 3.2465093048358345e-11), (0, 3.1469099051059103e-09),
    #  (0, 1.5698002448420801e-10), (0, 1.9920533537070619e-05), (0, 6.706507243234968e-06),
    #  (0, 1.7046194538057202e-05), (1, 0.99998640569605668),    (0, 9.8535912965519826e-13),
    #  (0, 2.6009374594983658e-06), (0, 5.9442892596185542e-12)]

    # ### Precision-recall curves
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0YPe1/classifier-decision-functions">
        <img src="images/plt3-01.png" alt="text" title= "Precision-recall curves" height="300">
    </a>

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/zxFPUz6FEeeR4AqenwJvyA.processed/full/360p/index.mp4?Expires=1536883200&Signature=KjE0qGwonCOca4-bh2ZZFfNBML6WyABR4RGyijJbKA~S7HS1YwuL9R4LEvzg0Ii2Mv78SBveugPxUgLO8qLD01Lu49MgwxuhaiRYi9doGLWccg2v7gmn3a8acWR5Zks7GcO-1i-ZpPZXNvNArbYOjnSqVq2jERGPTzp47Zzjyng_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Classifier Decision Functions" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Precision-recall and ROC curves

### Note


+ Classifier Decision Functions
    + X-axis: Precision
    + Y-axis: Recall
    + Top right corner:
        + The “ideal” point
        + Precision = 1.0
        + Recall = 1.0
    + “Steepness” of P-R curves is important:
        + Maximize precision
        + while maximizing recall
    
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves">
        <img src="images/plt3-08.png" alt="Precision-Recall Curves are very widely used evaluation method from machine learning. As we just saw in example, the x axis shows precision and the y axis shows recall. Now an ideal classifier would be able to achieve perfect precision of 1.0 and perfect recall of 1.0. So the optimal point would be up here in the top right. And in general, with precision recall curves, the closer in some sense, the curve is to the top right corner, the more preferable it is, the more beneficial the tradeoff it gives between precision and recall. And we saw some examples already of how there is a tradeoff between those two quantities, between precision and recall, with many classifiers. This example here is an actual precision recall curve that we generated using the following notebook code. The red circle indicates the precision and recall that's achieved when the decision threshold is zero. So I created this curve using exactly the same method as we saw in the previous example, by looking at the decision function output from a support vector classifier. Applying very end decision boundary, looking at how the precision of recall change as the decision boundary changed. Fortunately, learn has a function that's built in that does all of that, that could compute the precision of recall curve. And that's what we've been using in the notebook here. So you can see that in this particular application there is a general downward trend. So as the precision of the classifier goes up, the recall tends to go down. In this particular case you'll see also that It's not exactly a smooth curve. There are some jaggy errors and, in fact, the jumps tend to get a little bigger as we approach maximum precision. This is a consequence of how the formulas for precision and recall are computed. They use discrete counts that include the number of true positives. And so as the decision threshold increases, there are fewer and fewer points that remain as positive predictions. So the fractions that are computed for these smaller numbers can change pretty dramatically with small changes in the decision threshold. And that's why these sort of trailing edges of the Precision-recall curve can appear a bit jagged when you plot them. " title= "Classifier Decision Functions" height="250">
    </a>

+ ROC Curves
    + ROC: Receiver Operating Characteristics
    + X-axis: False Positive Rate
    + Y-axis: True Positive Rate
    + Top left corner:
        + The “ideal” point
        + False positive rate of zero
        + True positive rate of one
    + “Steepness” of ROC curves is important:
        + Maximize the true positive rate
        + while minimizing the false positive rate
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves"><br/>
        <img src="images/fig3-11.png" alt="ROC curves or receiver operating characteristic curves are a very widely used visualization method that illustrate the performance of a binary classifier. ROC curves on the X-axis show a classifier's False Positive Rate so that would go from 0 to 1.0, and on the Y-axis they show a classifier's True Positive Rate so that will also go from 0 to 1.0. The ideal point in ROC space is one where the classifier achieves zero, a false positive rate of zero, and a true positive rate of one. So that would be the upper left corner. So curves in ROC space represent different tradeoffs as the decision boundary, the decision threshold is varied for the classifier. So just as in the precision recall case, as we vary decision threshold, we'll get different numbers of false positives and true positives that we can plot on a chart. The dotted line here that I'm showing is the classifier curve that secretly results from a classifier that randomly guesses the label for a binary class. It's basically like flipping a coin. If you have two classes with equal numbers of positive and negative incidences, then flipping a coin will get you randomly equal numbers of false positives and true positives for a large virus data sets. So the dotted line here is used as a base line. So bad classifier will have performance that is random or maybe even worse than random or be slightly better than random. Reasonably good classifier will give an ROC curve that is consistently better than random across all decision threshold choices. And then an excellent classifier would be one like I've shown here, which is way up into the left. This particular example is an example of a logistic regression classifier using the notebook example you've seen. So, the shape of the curve can be important as well, the steepness of the curve, we want classifiers that maximize the true positive rate while minimizing the false positive rate. " title= "ROC Curves" height="200">
    </a>

+ ROC curve examples
    + Random guessing
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves"><br/>
            <img src="images/fig3-12.png" alt="text" title= "ROC curve examples: random guessing" height="200">
        </a>
    + Perfect classifier
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves"><br/>
            <img src="images/fig3-13.png" alt="text" title= "ROC curve examples: perfect classifier" height="200">
        </a>
    + Bad, okay,
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves"><br/>
            <img src="images/fig3-12.png" alt="text" title= "ROC curve examples: bad, okay" height="200">
        </a>

+ Summarizing an ROC curve in one number: Area Under the Curve (AUC)
    + AUC = 0 (worst) AUC = 1 (best)
    + AUC can be interpreted as:
        1. The total area under the ROC curve.
        2. The probability that the classifier will assign a higher score to a randomly chosen positive example than to a randomly chosen   + negative example.
    + Advantages:
        + Gives a single number for easy comparison.
        + Does not require specifying a decision threshold.
    + Drawbacks:
        + As with other single-number metrics, AUC loses information, e.g. about tradeoffs and the shape of the ROC curve.
        + This may be a factor to consider when e.g. wanting to compare the performance of classifiers with overlapping ROC curves.
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves"><br/>
        <img src="images/fig3-15.png" alt="We can qualify the goodness of a classifier in some sense by looking at how much area there is underneath the curve. So the area underneath the random classifier is going to be 0.5 but then the area, as you can see, the size of the bumpiness of the classifier as it approaches the top left corner. Well, the area underneath the curve will get larger and larger. It will approach 1.  We use something called area under the curve, AUC. That's the single number that measures this total area underneath the ROC curve as a way to summarize a classifier's performance. So, an AUC of zero represents a very bad classifier, and an AUC of one will represent an optimal classifier." title= "Area Under the Curve (AUC)" height="200">
    </a>


+ Demo 1
    ```python
    from sklearn.metrics import roc_curve, auc

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

    y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    ```
    <a href="url"> <br/>
        <img src="images/plt3-02.png" alt="ROC curves, Area-Under-Curve (AUC)" title= "ROC curves, Area-Under-Curve (AUC)" height="250">
    </a>

+ Demo 2
    ```python
    from matplotlib import cm

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    for g in [0.01, 0.1, 0.20, 1]:
        svm = SVC(gamma=g).fit(X_train, y_train)
        y_score_svm = svm.decision_function(X_test)
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        accuracy_svm = svm.score(X_test, y_test)
        print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, roc_auc_svm))
        plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
                label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
    plt.legend(loc="lower right", fontsize=11)
    plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
    plt.axes().set_aspect('equal')

    plt.show()
    # gamma = 0.01  accuracy = 0.91   AUC = 1.00
    # gamma = 0.10  accuracy = 0.90   AUC = 0.98
    # gamma = 0.20  accuracy = 0.90   AUC = 0.66
    # gamma = 1.00  accuracy = 0.90   AUC = 0.50
    ```
    <a href="url"> <br/>
        <img src="images/plt3-03.png" alt="ROC curves, Area-Under-Curve (AUC)" title= "ROC curves, Area-Under-Curve (AUC) with differeent gamma parameters" height="250">
    </a>

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/7QzJvD6FEee2TA5yccyTSg.processed/full/360p/index.mp4?Expires=1536883200&Signature=Z7qcBxZq7~rGuyqNG14q2NjuUodasmZjY8vnrHJFlisdQ0vfvAt07NkmrP-F2UM3LsvPYgW~L-9Oasbz9604aV2MF~wqka2F7dmbjQSyfuwWSrElXeNkx41nFo2ObEmynogoyk~LuFwriGcvfWJNj37uGR5Rqk4Cy3OE8Rrlnkg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Precision-recall and ROC curves" target="_blank">
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

