# Module 4: Supervised Machine Learning - Part 2

## Module 4 Notebook

+ [Launch Notebook Web Page](https://www.coursera.org/learn/python-machine-learning/notebook/TR0yt/module-4-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/user/elkljxyoytcwjbmkgctrtg/notebooks/Module%204.ipynb)
+ [Local Notebook](notebooks/Module04.ipynb)
+ [Python Code](notebooks/Module04.py)

+ Demo: Preamble and Datasets
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification, make_blobs
    from matplotlib.colors import ListedColormap
    from sklearn.datasets import load_breast_cancer
    from adspy_shared_utilities import load_crime_dataset

    cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

    # fruits dataset
    fruits = pd.read_table('fruit_data_with_colors.txt')

    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']

    # synthetic dataset for simple regression
    from sklearn.datasets import make_regression
    plt.figure()
    plt.title('Sample regression problem with one input variable')
    X_R1, y_R1 = make_regression(n_samples = 100, n_features=1, n_informative=1, 
        bias = 150.0, noise = 30, random_state=0)
    plt.scatter(X_R1, y_R1, marker= 'o', s=50)
    plt.show()      # Fig.1

    # synthetic dataset for more complex regression
    from sklearn.datasets import make_friedman1
    plt.figure()
    plt.title('Complex regression problem with one input variable')
    X_F1, y_F1 = make_friedman1(n_samples = 100, n_features = 7, random_state=0)

    plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
    plt.show()      # Fig.2

    # synthetic dataset for classification (binary)
    plt.figure()
    plt.title('Sample binary classification problem with two informative features')
    X_C2, y_C2 = make_classification(n_samples = 100, n_features=2, n_redundant=0, 
        n_informative=2, n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
    plt.scatter(X_C2[:, 0], X_C2[:, 1], marker= 'o', c=y_C2, s=50, cmap=cmap_bold)
    plt.show()      # Fig.3

    # more difficult synthetic dataset for classification (binary)
    # with classes that are not linearly separable
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, 
        cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2
    plt.figure()
    plt.title('Sample binary classification problem with non-linearly separable classes')
    plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
    plt.show()      # Fig.4

    # Breast cancer dataset for classification
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

    # Communities and Crime dataset
    (X_crime, y_crime) = load_crime_dataset()
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/plt4-01.png" alt="text" title= "Fig.1" height="200">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/plt4-02.png" alt="text" title= "Fig.2" height="200">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/plt4-03.png" alt="text" title= "Fig.3" height="200">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/plt4-04.png" alt="text" title= "Fig.4" height="200">
    </a>

## Naive Bayes Classifiers

+ Naïve Bayes Classifiers: a simple, probabilistic classifier family
    + These classifiers are called 'Naïve' because they assume that features are conditionally independent, given the class.
    + In other words: they assume that, for all instances of a given class, the features have little/no correlation with each other.
    + Highly efficient learning and prediction.
    + But generalization performance may worse than more sophisticated learning methods.
    + Can be competitive for some tasks.

+ Naïve Bayes classifier types
    + `Bernoulli`: binary features (e.g. word presence/absence)
    + `Multinomial`: discrete features (e.g. word counts)
    + `Gaussian`: continuous/real-valued features - Statistics computed for each class: For each feature: mean, standard deviation
    + See the Applied Text Mining course for more details on the `Bernoulli` and `Multinomial` Naïve Bayes models

+ Gaussian Naïve Bayes classifier
    <a href="https://www.researchgate.net/publication/255695722_Smoothness_without_Smoothing_Why_Gaussian_Naive_Bayes_Is_Not_Naive_for_Multi-Subject_Searchlight_Studies/figures?lo=1"> <br/>
        <img src="https://www.researchgate.net/profile/Yune_Lee/publication/255695722/figure/fig8/AS:341300424527882@1458383770548/Illustration-of-how-a-Gaussian-Naive-Bayes-GNB-classifier-works-For-each-data-point.png" alt="Figure 1. Illustration of how a Gaussian Naive Bayes (GNB) classifier works. For each data point, the z-score distance between that point and each class-mean is calculated, namely the distance from the class mean divided by the standard deviation of that class. Note that this schematic just shows one dimension, whereas a crucial distinction between GNBs and other classifiers arises only when there is more than one input dimension: the GNB does not model the covariance between dimensions, but other types of classifier do." title= "Gaussian Naive Bayes (GNB) classifier" height="200">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/fig4-01.png" alt="More specifically, the Gaussian Naive Bayes Classifier assumes that the data for each class was generated by a simple class specific Gaussian distribution. Predicting the class of a new data point corresponds mathematically to estimating the probability that each classes Gaussian distribution was most likely to have generated the data point. Classifier then picks the class that has the highest probability. During training, the Gaussian Naive Bayes Classifier estimates for each feature the mean and standard deviation of the feature value for each class. For prediction, the classifier compares the features of the example data point to be predicted with the feature statistics for each class and selects the class that best matches the data point. More specifically, the Gaussian Naive Bayes Classifier assumes that the data for each class was generated by a simple class specific Gaussian distribution. Predicting the class of a new data point corresponds mathematically to estimating the probability that each classes Gaussian distribution was most likely to have generated the data point. Classifier then picks the class that has the highest probability. Without going into the mathematics involved, it can be shown that the decision boundary between classes in the two class Gaussian Naive Bayes Classifier. In general is a parabolic curve between the classes. And in the special case where the variance of these feature is the same for both classes. The decision boundary will be linear. Here's what that looks like, typically, on a simple binary classification data set. The gray ellipses given idea of the shape of the Gaussian distribution for each class, as if we were looking down from above. You can see the centers of the Gaussian's correspond to the mean value of each feature for each class. More specifically, the gray ellipses show the contour line of the Gaussian distribution for each class, that corresponds to about two standard deviations from the mean. The line between the yellow and gray background areas represents the decision boundary. And we can see that this is indeed parabolic. " title= "Gaussian Naive Bayes Classifier" height="200">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/fig4-02.png" alt="And in the special case where the variance of these feature is the same for both classes. The decision boundary will be linear. Here's what that looks like, typically, on a simple binary classification data set. The gray ellipses given idea of the shape of the Gaussian distribution for each class, as if we were looking down from above. You can see the centers of the Gaussian's correspond to the mean value of each feature for each class. More specifically, the gray ellipses show the contour line of the Gaussian distribution for each class, that corresponds to about two standard deviations from the mean. The line between the yellow and gray background areas represents the decision boundary. And we can see that this is indeed parabolic. " title= "Example of Gaussian Naive Bayes Classifier with decision boundary" height="200">
    </a>
    + `partial_fit` method: train model incrementally in case working with a huge data set that doesn't fit into memory
    + Used for high-dimensional data, e.g., hundreds, thousands or maybe even more features

+ The Bernoulli and Nultinomial flavors of Naive Bayes:
    + Used for text classification with very large number of distinct words (features)
    + Used for the sparse future vectors because any given document uses only a small fraction of the overall vocabulary. 

+ Naïve Bayes classifiers: Pros and Cons
    + Pros:
        + Easy to understand
        + Simple, efficient parameter estimation
        + Works well with high-dimensional data
        + Often useful as a baseline comparison against more sophisticated methods
    + Cons:
        + Assumption that features are conditionally independent given the class is not realistic.
        + As a result, other classifier types often have better generalization performance.
        + Their confidence estimates for predictions are not very accurate.


+ Demo
    ```python
    from sklearn.naive_bayes import GaussianNB
    from adspy_shared_utilities import plot_class_regions_for_classifier

    X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

    nbclf = GaussianNB().fit(X_train, y_train)
    plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test, 
        'Gaussian Naive Bayes classifier: Dataset 1')

    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

    nbclf = GaussianNB().fit(X_train, y_train)
    plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
        'Gaussian Naive Bayes classifier: Dataset 2')

    # ### Application to a real-world dataset
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

    nbclf = GaussianNB().fit(X_train, y_train)
    print('Breast cancer dataset')
    print('Accuracy of GaussianNB classifier on training set: {:.2f}'
        .format(nbclf.score(X_train, y_train)))
    print('Accuracy of GaussianNB classifier on test set: {:.2f}'
        .format(nbclf.score(X_test, y_test)))
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/0XFms/naive-bayes-classifiers">
        <img src="images/plt4-05.png" alt="To use the Gaussian Naive Bayes classifier in Python, we just instantiate an instance of the Gaussian NB class and call the fit method on the training data just as we would with any other classifier. It's worth noting that the Naive Bayes models are among a few classifiers in scikit learn that support a method called partial fit, which can be used instead of fit to train the classifier incrementally in case you're working with a huge data set that doesn't fit into memory. More details on that are available in the scikit learn documentation for Naive Bayes. For the Gaussian NB class there are no special parameters to control the models complexity. Looking at one example in the notebook from our synthetic two class dataset, we can see that, in fact, the Gaussian Naive Bayes classifier achieves quite good performance on this simple classification example. When the classes are no longer as easily separable as with this second, more difficult binary example here. Like linear models, Naive Bayes does not perform as well. " title= "Gaussian Naive Bayes classifier with synthetic datasets" height="250">
    </a>

### Lecture Video

<a href="https://d18ky98rnyall9.cloudfront.net/Ld6WAECbEee4_A7ezGAgwg.processed/full/360p/index.mp4?Expires=1537488000&Signature=VMGXs1je8kFOomj5Yk1bNqi2lKFRDS~LcCE4AhXWb2ubC2zpzTtRucIuiNYRsPllZjmnFBkKKcm2aouGDa1b9bneFMFwWbVZodxBbAd5R7d4F3br-By1V-gj44iSRZItOI1LtDmVfoDhxu7JxuP5lX9spS7pqNIu28IraN6tEcg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Naive Bayes Classifiers" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Random Forests

+ Ensemble:
    + take multiple individual learning models and combine them to produce an aggregate model 
    + more powerful than any of its individual learning models alone
    + different learning models tend to make different kinds of mistakes on the data set
    + each individual model might overfit to a different part of the data
    + combining different individual models into an ensemble → average out their individual mistakes to reduce the risk of overfitting while maintaining strong prediction performance

+ Random Forests
    + An ensemble of trees, not just one tree.
    + Widely used, very good results on many problems.
    + `sklearn.ensemble` module:
        + Classification: `RandomForestClassifier`
        + Regression: `RandomForestRegressor`
    + One decision tree → Prone to overfitting.
    + Many decision trees → More stable, better generalization
    + Ensemble of trees should be diverse: introduce random variation into tree-building.

+ Random Forest Process
    +Original Dataset
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
            <img src="images/fig4-03.png" alt="This random variation during tree building happens in two ways. First, the data used to build each tree is selected randomly and second, the features chosen in each split tests are also randomly selected. To create a random forest model you first decide on how many trees to build. This is set using the n_estimated parameter for both RandomForestClassifier and RandomForestRegressor. Each tree were built from a different random sample of the data called the bootstrap sample. " title= "Random Forest Process" height="250">
        </a>

    + Bootstrap Samples
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
            <img src="images/fig4-04.png" alt="Bootstrap samples are commonly used in statistics and machine learning. If your training set has N instances or samples in total, a bootstrap sample of size N is created by just repeatedly picking one of the N dataset rows at random with replacement, that is, allowing for the possibility of picking the same row again at each selection. You repeat this random selection process N times. The resulting bootstrap sample has N rows just like the original training set but with possibly some rows from the original dataset missing and others occurring multiple times just due to the nature of the random selection with replacement." title= "Bootstrap Samples" height="250">
        </a>

    + Randomized Feature Splits
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
            <img src="images/fig4-05.png" alt="When building a decision tree for a random forest, the process is almost the same as for a standard decision tree but with one important difference. When picking the best split for a node, instead of finding the best split across all possible features, a random subset of features is chosen and the best split is found within that smaller subset of features. The number of features in the subset that are randomly considered at each stage is controlled by the max_features parameter. This randomness in selecting the bootstrap sample to train an individual tree in a forest ensemble, combined with the fact that splitting a node in the tree is restricted to random subsets of the features of the split, virtually guarantees that all of the decision trees and the random forest will be different. " title= "Randomized Feature Splits" height="250">
        </a>

+ Random Forest `max_features` Parameter
    + Learning is quite sensitive to `max_features`.
    + Setting `max_features= 1` leads to forests with diverse, more complex trees.
    + Setting `max_features= <close to number of features>` will lead to similar forests with simpler trees.

+ Prediction Using Random Forests
    1. Make a prediction for every tree in the forest.
    2. Combine individual predictions
        + Regression: mean of individual tree predictions.
        + Classification:
            + Each tree gives probability for each class.
            + Probabilities averaged across trees.
            + Predict the class with highest probability.
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
        <img src="images/fig4-06.png" alt="Once a random forest model is trained, it predicts the target value for new instances by first making a prediction for every tree in the random forest. For regression tasks the overall prediction is then typically the mean of the individual tree predictions. For classification the overall prediction is based on a weighted vote. Each tree gives a probability for each possible target class label then the probabilities for each class are averaged across all the trees and the class with the highest probability is the final predicted class. " title= "Prediction Using Random Forests" height="200">
    </a>

+ Random Forest: Fruit Dataset
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
        <img src="images/fig4-07.png" alt="Here's an example of learning a random forest of the example fruit dataset using two features, height and width. Here we're showing the training data plotted in terms of two feature values with height on the x axis and width on the y axis. As usual, there are four categories of fruit to be predicted. Because the number of features is restricted to just two in this very simple example, the randomness in creating the tree ensemble is coming mostly from the bootstrap sampling of the training data. You can see that the decision boundaries overall have the box like shape that we associate with decision trees but with some additional detail variation to accommodate specific local changes in the training data. Overall, you can get an impression of the increased complexity of this random forest model in capturing both the global and local patterns in the training data compared to the single decision tree model we saw earlier. Notice that we did not have to perform scaling or other pre-processing as we did with a number of other supervised learning methods. This is one advantage of using random forests. Also note that we passed in a fixed value for the random state parameter in order to make the results reproducible. " title= "Random Forest: Fruit Dataset" height="200">
    </a>

+ Random Forest: Pros and Cons
    + Pros:
        + Widely used, excellent prediction performance on many problems.
        + Doesn't require careful normalization of features or extensive parameter tuning.
        + Like decision trees, handles a mixture of feature types.
        + Easily parallelized across multiple CPUs.
    + Cons:
        + The resulting models are often difficult for humans to interpret.
        + Like decision trees, random forests may not be a good choice for very high-dimensional tasks (e.g. text classifiers) compared to fast, accurate linear models.

+ Random Forests: `RandomForestClassifier` Key Parameters
    + `n_estimators`: number of trees to use in ensemble (default: 10).
        + Should be larger for larger datasets to reduce overfitting(but uses more computation).
    + `max_features`: has a strong effect on performance. Influences the diversity of trees in the forest.
        + Default works well in practice, but adjusting may lead to some further gains.
    + `max_depth`: controls the depth of each tree (default: None. Splits until all leaves are pure).
    + `n_jobs`: How many cores to use in parallel during training.
    + Choose a fixed setting for the `random_state` parameter if you need reproducible results.



+ Demo
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

    clf = RandomForestClassifier().fit(X_train, y_train)
    title = 'Random Forest Classifier, complex binary dataset, default settings'
    plot_class_regions_for_classifier_subplot(
        clf, X_train, y_train, X_test, y_test, title, subaxes)

    plt.show()      # Fig.7

    # ### Random forest: Fruit dataset
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

    X_train, X_test, y_train, y_test = train_test_split(
        X_fruits.as_matrix(), y_fruits.as_matrix(), random_state = 0)
    fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

    title = 'Random Forest, fruits dataset, default settings'
    pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

    for pair, axis in zip(pair_list, subaxes):
        X = X_train[:, pair]
        y = y_train
        
        clf = RandomForestClassifier().fit(X, y)
        plot_class_regions_for_classifier_subplot(
            clf, X, y, None, None, title, axis, target_names_fruits)
        
        axis.set_xlabel(feature_names_fruits[pair[0]])
        axis.set_ylabel(feature_names_fruits[pair[1]])
        
    plt.tight_layout()
    plt.show()          # Fig.8

    clf = RandomForestClassifier(
        n_estimators = 10, random_state=0).fit(X_train, y_train)

    print('Random Forest, Fruit dataset, default settings')
    print('Accuracy of RF classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of RF classifier on test set: {:.2f}'
        .format(clf.score(X_test, y_test)))
    # Random Forest, Fruit dataset, default settings
    # Accuracy of RF classifier on training set: 1.00
    # Accuracy of RF classifier on test set: 0.80

    # #### Random Forests on a real-world dataset
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

    clf = RandomForestClassifier(max_features = 8, random_state = 0)
    clf.fit(X_train, y_train)

    print('Breast cancer dataset')
    print('Accuracy of RF classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of RF classifier on test set: {:.2f}'
        .format(clf.score(X_test, y_test)))
    # Breast cancer dataset
    # Accuracy of RF classifier on training set: 1.00
    # Accuracy of RF classifier on test set: 0.99
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
        <img src="images/plt4-07.png" alt="This code also plots the decision boundaries for the other five possible feature pairs. Again, to use the RandomForestClassifier we import the random forest classifier class from the sklearn ensemble library. After doing the usual train test split and setting up the pipe plot figure for plotting, we iterate through pairs of feature columns in the dataset. For each pair of features we call the fit method on that subset of the training data X using the labels y. We then use the utility function plot class regions for classifier that's available in the shared module for this course to visualize the training data and the random forest decision boundaries. " title= "Random Forest Classifier, complex binary dataset, default settings" height="250">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lF9QN/random-forests"> <br/>
        <img src="images/plt4-08.png" alt="Let's apply random forest to a larger dataset with more features. For comparison with other supervised learning methods, we use the breast cancer dataset again. We create a new random forest classifier and since there are about 30 features, we'll set max_features to eight to give a diverse set of trees that also fit the data reasonably well. We can see that random forest with no feature scaling or extensive parameter tuning achieve very good test set performance on this dataset, in fact, it's as good or better than all the other supervised methods we've seen so far including current life support vector machines and neural networks that require more careful tuning. " title= "Random forest: Fruit dataset" height="600">
    </a>



### Lecture Video

<a href="https://d18ky98rnyall9.cloudfront.net/_4VeolzrEeeQywpoSy5QrA.processed/full/360p/index.mp4?Expires=1537488000&Signature=BuFNm32Z2UaDHHcbL32WyXCgU6iJzlGtdRtTazEkBobN0lPZr4fVTDjPHsgZcArbW99evZH6cwuzcBm-oRPC4gw0iMzP1m3hVZNr4EUg1MB46gKNlwNcnJ1F4yKEMy-tOd4wcHhBSqHYx9S7CGcss8yb3CUjCLi8RFx-K0AkHbk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="andom Forests" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Gradient Boosted Decision Trees

+ Gradient Boosted Decision Trees (GBDT)
    + Training builds a series of small decision trees.
    + Each tree attempts to correct errors from the previous stage.
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/emwn3/gradient-boosted-decision-trees"> <br/>
            <img src="images/fig4-08.png" alt="Unlike the random forest method that builds and combines a forest of randomly different trees in parallel, the key idea of gradient boosted decision trees is that they build a series of trees. Where each tree is trained, so that it attempts to correct the mistakes of the previous tree in the series. Typically, gradient boosted tree ensembles use lots of shallow trees known in machine learning as weak learners. Built in a nonrandom way, to create a model that makes fewer and fewer mistakes as more trees are added. Once the model is built, making predictions with a gradient boosted tree models is fast and doesn't use a lot of memory. Like random forests, the number of estimators in the gradient boosted tree ensemble is an important parameter in controlling model complexity. " title= "Gradient Boosted Decision Trees (GBDT) - data flow" height="200">
        </a>
    + The learning rate controls how hard each new tree tries to correct remaining mistakes from previous round.
        + High learning rate: more complex trees
        + Low learning rate: simpler trees
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/emwn3/gradient-boosted-decision-trees"> <br/>
        <img src="images/fig4-09.png" alt="A new parameter that does not occur with random forest is something called the learning rate. The learning rate controls how the gradient boost the tree algorithms, builds a series of collective trees. When the learning rate is high, each successive tree put strong emphases on correcting the mistakes of its predecessor. And thus may result in a more complex individual tree, and those overall are more complex model. With smaller settings of the learning rate, there's less emphasis on thoroughly correcting the errors of the previous step, which tends to lead to simpler trees at each step. " title= "Gradient Boosted Decision Trees (GBDT) - example" height="200">
    </a>

+ GBDT: Pros and Cons
    + Pros:
        + Often best off-the-shelf accuracy on many problems.
        + Using model for prediction requires only modest memory and is fast.
        + Doesn't require careful normalization of features to perform well.
        + Like decision trees, handles a mixture of feature types.
    + Cons:
        + Like random forests, the models are often difficult for humans to interpret.
        + Requires careful tuning of the learning rate and other parameters.
        + Training can require significant computation.
        + Like decision trees, not recommended for text classification and other problems with very high dimensional sparse features, for accuracy and computational cost reasons.

+ GBDT: `GradientBoostingClassifier` Key Parameters
    + `n_estimators`: sets # of small decision trees to use (weak learners) in the ensemble.
    + `learning_rate`: controls emphasis on fixing errors from previous iteration.
    + The above two are typically tuned together.
    + `n_estimatorsis` adjusted first, to best exploit memory and CPUs during training, then other parameters.
    + `max_depthis` typically set to a small value (e.g. 3-5) for most applications.



+ Demo
    ```python
    # ### Gradient-boosted decision trees
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

    clf = GradientBoostingClassifier().fit(X_train, y_train)
    title = 'GBDT, complex binary dataset, default settings'
    plot_class_regions_for_classifier_subplot(
        clf, X_train, y_train, X_test, y_test, title, subaxes)

    plt.show()      # Fig.9

    # #### Gradient boosted decision trees on the fruit dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_fruits.as_matrix(), y_fruits.as_matrix(), random_state = 0)
    fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

    pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

    for pair, axis in zip(pair_list, subaxes):
        X = X_train[:, pair]
        y = y_train
        
        clf = GradientBoostingClassifier().fit(X, y)
        plot_class_regions_for_classifier_subplot(
            clf, X, y, None, None, title, axis, target_names_fruits)
        
        axis.set_xlabel(feature_names_fruits[pair[0]])
        axis.set_ylabel(feature_names_fruits[pair[1]])
        
    plt.tight_layout()
    plt.show()      # Fig.10

    clf = GradientBoostingClassifier().fit(X_train, y_train)

    print('GBDT, Fruit dataset, default settings')
    print('Accuracy of GBDT classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}'
        .format(clf.score(X_test, y_test)))
    # GBDT, Fruit dataset, default settings
    # Accuracy of GBDT classifier on training set: 1.00
    # Accuracy of GBDT classifier on test set: 0.80

    # #### Gradient-boosted decision trees on a real-world dataset
    from sklearn.ensemble import GradientBoostingClassifier

    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

    clf = GradientBoostingClassifier(random_state = 0)
    clf.fit(X_train, y_train)

    print('Breast cancer dataset (learning_rate=0.1, max_depth=3)')
    print('Accuracy of GBDT classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}\n'
        .format(clf.score(X_test, y_test)))
    # Breast cancer dataset (learning_rate=0.1, max_depth=3)
    # Accuracy of GBDT classifier on training set: 1.00
    # Accuracy of GBDT classifier on test set: 0.96

    clf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)
    clf.fit(X_train, y_train)

    print('Breast cancer dataset (learning_rate=0.01, max_depth=2)')
    print('Accuracy of GBDT classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}'
        .format(clf.score(X_test, y_test)))
    # Breast cancer dataset (learning_rate=0.01, max_depth=2)
    # Accuracy of GBDT classifier on training set: 0.97
    # Accuracy of GBDT classifier on test set: 0.97
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/emwn3/gradient-boosted-decision-trees"> <br/>
        <img src="images/plt4-09.png" alt="Here's an example showing how to use gradient boosted trees in scikit-learn on our sample fruit classification test, plotting the decision regions that result. The code is more or less the same as what we used for random forests. But from the sklearn.ensemble module, we import the GradientBoostingClassifier class. We then create the GradientBoostingClassifier object, and fit it to the training data in the usual way. By default, the learning rate parameter is set to 0.1, the n_estimators parameter giving the number of trees to use is set to 100, and the max depth is set to 3. As with random forests, you can see the decision boundaries have that box-like shape that's characteristic of decision trees or ensembles of trees. " title= "Gradient-boosted decision trees" height="250">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/emwn3/gradient-boosted-decision-trees"> <br/>
        <img src="images/plt4-10.png" alt="Not covered i  Lecture" title= "Gradient boosted decision trees on the fruit dataset" height="600">
    </a>



### Lecture Video

<a href="https://d18ky98rnyall9.cloudfront.net/2YlAZ1zrEeeliw7ADgKLdA.processed/full/360p/index.mp4?Expires=1537488000&Signature=CyD0KOnbXSk7Zf-q1XDP249mSCKXST4r4y52GuxsoftRnYrcKPDBEx25fTkuSNcacZlMkBhwAwThrV6lYwjGfrdvYg7gcPs~Q6o9dOMcbm3vsMI5ey7Su0vzUjvxwQ1w6psBs85OHoXoDluoHEbArrHKGIYwzyU0o1ABk8ssqnc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Gradient Boosted Decision Trees" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Neural Networks

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Neural Networks Made Easy (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Play with Neural Networks: TensorFlow Playground (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Deep Learning (Optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Deep Learning in a Nutshell: Core Concepts (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Assisting Pathologists in Detecting Cancer with Deep Learning (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Data Leakage

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## The Treachery of Leakage (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Leakage in Data Mining: Formulation, Detection, and Avoidance (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Data Leakage Example: The ICML 2013 Whale Challenge (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Rules of Machine Learning: Best Practices for ML Engineering (optional)

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Quiz: Module 4 Quiz






