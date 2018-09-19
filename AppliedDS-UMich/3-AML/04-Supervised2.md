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

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Random Forests

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Gradient Boosted Decision Trees

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
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






