# Module 2: [Supervised Machine Learning](./02-Supervised1.md)

## Module 2 Notebook

+ [Launching Web Page](https://www.coursera.org/learn/python-machine-learning/notebook/7u2va/module-2-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=n1itCiowSXKYrQoqMPly9A&next=%2Fnotebooks%2FModule%25202.ipynb)
+ [Local Notebook](notebooks/Module02.ipynb)
+ [Local Python Code](notebooks/Module02.py)



## Introduction to Supervised Machine Learning

+ Learning objectives
    + Understand how a number of different supervised learning algorithms learn by estimating their parameters from data to make new predictions.
    + Understand the strengths and weaknesses of particular supervised learning methods.
    + Learn how to apply specific supervised machine learning algorithms in Python with scikit-learn.
    + Learn about general principles of supervised machine learning, like overfitting and how to avoid it.

+ Review of important terms
    + Feature representation, e.g. `mass`, `width`, `height`, `color_score`
    + Data instances/samples/examples (X); i.e., rows, e.g. row(0) = `| 1 | 0 | 1 | apple | granny_smith | 192 | 8.4 | 7.3 | 0.55 |`, row(2) = `| 3 | 2 | 1 | apple | granny_smith | 176 | 7.4 | 7.2 | 0.60 |`; in Python `X` represent the set of features, e.g., row(0) = `| 192 | 8.4 | 7.3 | 0.55 |`
    + Target value (y), e.g., label = `fruit_label`, `fruit_name` & `fruit_subtype`: only for labeling purpose as more readable for humans
    + Training and test sets: using `train_test_split` function from `sklearn.model_selection` module, default as $75\%:25\%$, e.g., `X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)`
    + Model/Estimator
        + Model fitting produces a 'trained model'
        + Training is the process of estimating model parameters
        + Demo
            ```python
            # estimator/model - Classifier selection
            knn = KNeighborsClassifier(n_neighbors = 5)
            # model fit - training to get model parameters with given training data set
            knn.fit(X_train_scaled, y_train)
            # Apply model to predict a given instance
            knn.predict(example_fruit_scaled)
            ```
    + Evaluation method
        ```python
        # accuracy of the model
        knn.score(X_test_scaled, y_test)
        ```
    + Example - Table terminologies

        |   | fruit_label | fruit_name | fruit_subtype | mass | width | height | color_score |
        |---|-------------|------------|---------------|------|-------|--------|-------------|
        | 0 | 0 | 1 | apple | granny_smith | 192 | 8.4 | 7.3 | 0.55 |
        | 1 | 1 | 1 | apple | granny_smith | 180 | 8.0 | 6.8 | 0.59 |
        | 2 | 2 | 1 | apple | granny_smith | 176 | 7.4 | 7.2 | 0.60 |
        | 3 | 3 | 2 | mandarin | mandarin | 86 | 6.2 | 4.7 | 0.80 |
        | 4 | 4 | 2 | mandarin | mandarin | 84 | 6.0 | 4.6 | 0.79 |

+ Classification and Regression
    + Both classification and regression take a set of training instances and learn a mapping to a __target value__.
    + For classification, the target value is a _discrete_ class value
        + Binary: target value = $0$ (negative class) or $1$ (positive class), e.g., detecting a fraudulent credit card transaction
        + Multi-class: target value is one of a set of discrete values, e.g., labelling the type of fruit from physical attributes
        + Multi-label: there are multiple target values (labels), e.g., labelling the topics discussed ion a Web page
    + For regression, that target value is _continuous_ (floating point/real-value), e.g., predicting the selling price of house from its attributes
    + Looking at the target value's type will guide you on what supervised learning method to use
    + Many supervised learning methods have 'flavors' for both classification and regression

+ Supervised learning methods: Overview
    + To start with, we'll look at two simple but powerful prediction algorithms
        + K-nearest neighbors (review from week 1, plus regression)
        + Linear model fit using least-squares
    + These represent two complementary approaches to supervised learning
        + K-nearest neighbors makes few assumptions about the structure of the data and gives potentially accurate but sometimes unstable predictions (sensitive to small changes in the training data)
        + Linear models make strong assumptions about structure of the data and give stable but potentially inaccurate predictions
    

+ Supervised learning methods: Overview
    + To start with, we'll look at two simple but powerful prediction algorithms:
        + K-nearest neighbors (review from week 1, plus regression)
        + Linear model fit using least-squares
    + These represent two complementary approaches to supervised learning:
        + K-nearest neighbors makes few assumptions about the structure of the data and gives potentially accurate but sometimes unstable predictions (sensitive to small changes in the training data).
        + Linear models make strong assumptions about the structure of the data and give stable but potentially inaccurate predictions.
    + We'll cover a number of widely-used supervised learning methods for classification and regression.
    + For each supervised learning method we'll explore:
        + How the method works conceptually at a high level.
        + What kind of feature preprocessing is typically needed.
        + Key parameters that control model complexity, to avoid under-and over-fitting.
        + Positives and negatives of the learning method.
    + Other models: decision trees, kernelized supported vector machines (SVM) and neural networks

+ The relationship between model complexity and training/test performance
    <a href="https://datascience.stackexchange.com/questions/33720/i-am-trying-to-make-a-classifier-using-machine-learning-to-detect-malwares-am-i">
        <br/><img src="https://i.stack.imgur.com/4nVgI.png" alt="The first thing to do if you want to validate your results is to cut your set into a training set and a validation set. This way you train the K-NN method on your training set, and you use the trained classifier on the validation set. Then you can monitor the validation error, and the training error." title= "K-NN best case scenario" width="250">
    </a>
        <a href="https://www.coursera.org/learn/python-machine-learning/lecture/EiQjD/linear-regression-least-squares">
        <img src="images/fig2-06.png" alt="Underfitting vs. overfitting" title= "relationship between model complexity and training/test performance" width="285">
    </a>

+ Models and Variables
    + Model: a specific mathematical or computational description that express the relationship between a set of input variables and one or more outcome variables that are being studied or predicted
    + Statistics: input variables = independent variables; output variable = dependent variables
    + Machine learning: input variables = features; output variables = target values / target labels
    + Unsupervised learning models used to understand and explore the structure within a given dataset
    + Supervised learning used to develop predict models that can accurately predict the outcomes, target values/target labels

+ Demo: Preamble and Review
    ```python
    %matplotlib notebook
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    np.set_printoptions(precision=2)


    fruits = pd.read_table('fruit_data_with_colors.txt')

    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']

    X_train, X_test, y_train, y_test = \
        train_test_split(X_fruits, y_fruits, random_state=0)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train_scaled, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(X_train_scaled, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(X_test_scaled, y_test)))

    example_fruit = [[5.5, 2.2, 10, 0.70]]
    example_fruit_scaled = scaler.transform(example_fruit)
    print('Predicted fruit type for ', example_fruit, ' is ', 
            target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])
    # Accuracy of K-NN classifier on training set: 0.95
    # Accuracy of K-NN classifier on test set: 1.00
    # Predicted fruit type for  [[5.5, 2.2, 10, 0.7]]  is  mandarin
    ```

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/tPIu3lzrEeeQywpoSy5QrA.processed/full/360p/index.mp4?Expires=1536278400&Signature=QJpYlD0vOtufdV2wDh49dA7eMIu7XUHPJLOoxIwvPDpcsGrjhSZvac1dgTn0dD1UpdLkCkcYtUqBvOKklUEfMDAkMnp8Sz4vKiLHVSnAcKQ96B0xhfpMG3KORoWOo7i3~XcRC5oDpYNN-P-B35xYGJsPDyAEkpEi2oFbEuCCnOw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Introduction to Supervised Machine Learning" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Overfitting and Underfitting

+ Generalization, Overfitting, and Underfitting
    + __Generalization ability__ refers to an algorithm's ability to give accurate predictions for new, previously unseen data.
    + Assumptions:
        + Future unseen data (test set) will have the _same properties_c as the current training sets.
        + Thus, models that are accurate on the training set are expected to be accurate on the test set.
        + But that may not happen if the trained model is tuned too specifically to the training set.
    + Models that are too complex for the amount of training data available are said to __overfit__ and are not likely to generalize well to new examples.
    + Models that are too simple, that don't even do well on the training data, are said to __underfit__ and also not likely to generalize well.
    + Not enough training data to constraint the mode to respect these global trends -> Training set accuracy is a hopelessly optimistic indicator for likely test set accuracy if the mode is overfitting
    + Understanding, detecting, and avoiding overfitting is perhaps the most important aspect of applying supervised machine learning algorithms

+ Overfitting vs. Underfitting in Regression
    <a href="http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html">
        <br/><img src="http://scikit-learn.org/stable/_images/sphx_glr_plot_underfitting_overfitting_001.png" alt="This example demonstrates the problems of underfitting and overfitting and how we can use linear regression with polynomial features to approximate nonlinear functions. The plot shows the function that we want to approximate, which is a part of the cosine function. In addition, the samples from the real function and the approximations of different models are displayed. The models have polynomial features of different degrees. We can see that a linear function (polynomial with degree 1) is not sufficient to fit the training samples. This is called underfitting. A polynomial of degree 4 approximates the true function almost perfectly. However, for higher degrees the model will overfit the training data, i.e. it learns the noise of the training data. We evaluate quantitatively overfitting / underfitting by using cross-validation. We calculate the mean squared error (MSE) on the validation set, the higher, the less likely the model generalizes correctly from the training data." title= "Underfitting vs. Overfitting" width="600">
    </a>

+ Underfitting and Overfitting in Classification
    <a href="https://www.safaribooksonline.com/library/view/deep-learning/9781491924570/ch01.html">
        <br/><img src="https://www.safaribooksonline.com/library/view/deep-learning/9781491924570/assets/dpln_0107.png" alt="A straight line cutting across a curving scatterplot would be a good example of underfitting.If the line fits the data too well, we have the opposite problem, called “overfitting.” Solving underfitting is the priority, but much effort in machine learning is spent attempting not to overfit the line to the data. When we say a model overfits a dataset, we mean that it may have a low error rate for the training data, but it does not generalize well to the overall population of data in which we’re interested.  Another way of explaining overfitting is by thinking about probable distributions of data. The training set of data that we’re trying to draw a line through is just a sample of a larger unknown set, and the line we draw will need to fit the larger set equally well if it is to have any predictive power. We must assume, therefore, that our sample is loosely representative of a larger set." title= "Underfitting and overfitting in machine learning" width="450">
    </a>

+ Overfitting with k-NN classifiers
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/fVStr/overfitting-and-underfitting">
        <br/><img src="images/fig1-19.png" alt="text" title= "caption" width="600">
    </a>

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/Ygb0LT7yEee4_A7ezGAgwg.processed/full/360p/index.mp4?Expires=1536278400&Signature=agNy6H49Wn~dvZEbWoshbc1ODtxp160Y6PadOwWaOHgBJh6yBCSZyVuGCHco4TZ0HBsmDF6KMEX45tR8baj8o~pbe8JoLkQangbEwbpbFAWP4uDVtK0BqOYHS1Yj6EZJYtBM~ULx3I~x9Ns~yP8bmBPqz3odlskrWf6bCdnayis_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Overfitting and Underfitting" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Supervised Learning: Datasets

+ Simple Regression Dataset
    + Demo - Dataset
        ```python
        from sklearn.datasets import make_classification, make_blobs
        from matplotlib.colors import ListedColormap
        from sklearn.datasets import load_breast_cancer
        from adspy_shared_utilities import load_crime_dataset

        cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

        # synthetic dataset for simple regression
        from sklearn.datasets import make_regression
        plt.figure()
        plt.title('Sample regression problem with one input variable')
        X_R1, y_R1 = make_regression(
            n_samples = 100, n_features=1, n_informative=1, 
            bias = 150.0, noise = 30, random_state=0)
        plt.scatter(X_R1, y_R1, marker= 'o', s=50)
        plt.show()

        # synthetic dataset for more complex regression
        from sklearn.datasets import make_friedman1
        plt.figure()
        plt.title('Complex regression problem with one input variable')
        X_F1, y_F1 = make_friedman1(
            n_samples = 100, n_features = 7, random_state=0)

        plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
        plt.show()
        ```
    <img src="images/plt2-01.png" alt="For example, high dimensional data sets in some sense have most of their data in corners with lots of empty space and that's kind of difficult to visualize. We'll go through some examples later in the course. But the low dimensional examples are still useful so that we can understand things like how a model's complexity changes with changes in some key parameters So for basic regression we'll start with the simple problem that has one informative input variable. One noisy linear output and 100 data set samples. Here's a plot of a data set using scatter plot with each point represented by one dot. The x-axis shows the future value, and the y-axis shows the regression target. To create this we use the make regression function in SK learned data sets. Here is the code in the notebook." title="synthetic dataset for simple regression" width="300">&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="images/plt2-02.png" alt="To illustrate binary classification we will include a simple two class dataset with two informative features. Here's a scatterplot showing each data instance as a dot with the first feature value corresponding to the x-axis. And the second feature value corresponding to the y-axis. The color of a point shows which class that data instance is labeled. " title= "synthetic dataset for more complex regression" width="300">


+ Simple Binary Classification Dataset
    + Demo
        ```python
        # synthetic dataset for classification (binary) 
        plt.figure()
        plt.title('Sample binary classification problem with two informative features')
        X_C2, y_C2 = make_classification(
            n_samples = 100, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
        plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
                marker= 'o', s=50, cmap=cmap_bold)
        plt.show()
        ```
    <img src="images/plt2-03.png" alt="To illustrate binary classification we will include a simple two class dataset with two informative features. Here's a scatterplot showing each data instance as a dot with the first feature value corresponding to the x-axis. And the second feature value corresponding to the y-axis. The color of a point shows which class that data instance is labeled. I'm calling this dataset simple because it has only two features, both of which are informative. " title= "synthetic dataset for classification (binary)" width="350">

+ Complex Binary Classification Dataset
    + Demo
        ```python
        # more difficult synthetic dataset for classification (binary) 
        # with classes that are not linearly separable
        X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8,
                            cluster_std = 1.3, random_state = 4)
        y_D2 = y_D2 % 2
        plt.figure()
        plt.title('Sample binary classification problem with non-linearly separable classes')
        plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,
                marker= 'o', s=50, cmap=cmap_bold)
        plt.show()

        # Breast cancer dataset for classification
        cancer = load_breast_cancer()
        (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

        # Communities and Crime dataset
        (X_crime, y_crime) = load_crime_dataset()
        ```
    <img src="images/plt2-04.png" alt="We'll also look at a more complex binary classification problem that uses two features. But where the two classes are not really linearly separable, instead forming into various clusters in different parts of the feature space. This dataset was created in two steps. First using the make_blobs function in SK learn datasets to randomly generate 100 samples in 8 different clusters. And then by changing the cluster label assigned by make_blobs, which is a number from 1 to 8, to a binary number by converting it using a modulo 2 function. Assigning the even index points to class 0 and odd index points to class 1." title= "more difficult synthetic dataset for classification (binary) with classes that are not linearly separable" width="350">
 
+ Fruit Multi-class Classification Dataset  
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/71PMP/supervised-learning-datasets">
        <img src="images/fig1-15.png" alt="To illustrate multi-class classification, we'll use our familiar fruits dataset, which, as you may remember has four features and four possible target labels. Here on the left, I'm showing the array of scatter plots that we saw in week one that shows the relationship between all possible pairs of features and the class labels, with the distribution of values for each feature along the diagonal. " title= "Fruit Multi-class scatterplots" width="450">
    </a>
    + Features: width, height, mass, color_score
    + Classes: 0: apple; 1: mandarin orange; 2: orange; 3: lemon


+ Supervised Learning: Datasets
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/71PMP/supervised-learning-datasets">
        <br/><img src="images/fig2-02.png" alt="To illustrate a real-world regression problem, we'll use a dataset derived from the communities and crime dataset in the UCI repository. Our dataset uses a subset of the original features and target values. Which were originally created from combining several U.S. government data sources, like the U.S. census. Each data instance corresponds to a particular geographic area, typically a town or a region of a city. Our version of this dataset has 88 features that encode various demographic and social economic properties of each location. With 1994 location data instances. The target value that we'll try to predict is the per capita violent crime rate. To use this data set, we use the load_crime_dataset function that's included with the share utilities module for this course" title= "Table of communities and crime dataset" width="450">
    </a>
    + Input features: socio-economic data by location from U.S. Census
    + Target variable: Per capita violent crimes
    + Derived from the original UCI dataset at: https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
    + Import Python code
        ```python
        from adspy_shared_utilities import load_crime_dataset
        crime = load_crime_dataset()
        ```

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/9W-NIzzCEeeW0g5QrK3QnA.processed/full/360p/index.mp4?Expires=1536278400&Signature=hiR0kqhBB1RBP4F4Sxe0W9i2CMCQrKugrV0J15f1npMuvmTGJVy96VvZtirnb-iUzkMWtptbFLwg-R73k6Gizk8tb7G0bgFIkht1U5IGPiHEJvmVbFYtg1HK2apYvQqWd1xoYvl2zKvo2tAIiUKeqRyvhKE09jxODtXz3ofpfWk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Supervised Learning: Datasets" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## K-Nearest Neighbors: Classification and Regression

+ The k-Nearest Neighbor (k-NN) Classifier Algorithm <br/>
    Given a training set `X_train` with labels `y_train`, and given a new instance `x_test` to be classified:
    1. Find the most similar instances (let's call them `X_NN`) to `x_test` that are in `X_train`.
    2. Get the labels `y_NN` for the instances in `X_NN`
    3. Predict the label for `x_test` by combining the labels `y_NN` e.g. simple majority vote

+ Nearest Neighbors Classification (k=1 & 11)
    + K = 1: variant decision boundary; high model complexity; overfitting
    + K = 11: smoother decision boundary; lower model complexity; underfitting (?)
    + Python code
        ```python
        from adspy_shared_utilities import plot_two_class_knn

        X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

        plot_two_class_knn(X_train, y_train, 1, 'uniform', X_test, y_test)
        plot_two_class_knn(X_train, y_train, 3, 'uniform', X_test, y_test)
        plot_two_class_knn(X_train, y_train, 11, 'uniform', X_test, y_test)
        ```
    <img src="images/plt2-05.png" alt="Here's how a k-Nearest Neighbor Classifier using only one nearest neighbor, that is with k equal to 1, makes these predictions for the simple binary synthetic dataset. So as you might recall from week one where we applied a nearest neighbors classifier to our multi-class fruit dataset. Here we're applying the nearest neighbors classifier to our simple binary classification problem. Where the points in class zero are labeled with yellow dots and the points in class one are labeled with black dots. And just as we did for the week one problem with fruit classification, here we're also showing how the entire feature space is broken up into different decision regions according to the predictions that the k-Nearest Neighbor Classifier would make at each point in the decision space. So for example, a point out here in the yellow region represents a point that the classifier would classify as class zero. And a point, let's say, over here, the classifier would classify as class one. So because this is a one nearest neighbors classifier, to make a classification prediction for any given query point, the Classifier simply looks back into its trading set. So these points here represent all the points on the training set. So for any given point, let's say here, The Classifier would simply find the training point that's closest, namely this one, and assign the predict a class to simply the class of the nearest point in the training set. Likewise, if we have a point over here. The nearest point in the training says actually this point right here that has a class zero label and so that point would get assigned a class zero. And in fact, this whole region right here represents all the points that are closer to the class zero training point than any of the other class one training points. So this whole region here represents a one nearest neighbors prediction of class zero. So the k-Nearest Neighbor's Classifier with k = 1, you can see that the decision boundaries that derived from that prediction are quite jagged and have high variance. This is an example of a model, classification model, it has high model complexity. " title= "Classification with KNN (K=1)" width="200">
    <img src="images/plt2-06.png" alt="If you run this code and compare the resulting training and test scores for k equals 1, 3, and 11, which are shown in the title of each plot, you can see the effect of model complexity on a models ability to generalize." title= "Classification with KNN (K=3)" width="200">
    <img src="images/plt2-07.png" alt="And here is what happens when we increase k from 1 to 11. Now the classifier must combine the votes of the 11 nearest points, not just 1. So single training data points no longer have as dramatic an influence on the prediction. The result is a much smoother decision boundary, which represents a model with lower model complexity where the decision boundary has much less variance. Actually if we increased k even higher to be the total number of points in the training set, the result would be a single decision region where all predictions would be the most frequent class in the training data. As we saw for the fruit data set, k-Nearest Neighbor Classifiers can be applied to any number of classes, not just 2. The code for this example in the notebook uses a special function, in the shared utilities library for this course, called plot_two_class_knn. If you run this code and compare the resulting training and test scores for k equals 1, 3, and 11, which are shown in the title of each plot, you can see the effect of model complexity on a models ability to generalize. In the k = 1 case, the training score is a perfect 1.0. But the test score is only 0.80. As k increases to 3, the training score drops to 0.88 but the test score rises slightly 2.88, indicating the model is generalizing better to new data. When k = 11, the training score drops a bit further to 0.81, but the test score even better at 0.92, indicating that this simple model is much more effective at ignoring minor variations in training data. And instead capturing the more important global trend in where the classes tend to be located with the best overall generalization performance as a result. " title= "Classification with KNN (K=11)" width="200">

+ k-Nearest Neighbors Regression
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/I1cfu/k-nearest-neighbors-classification-and-regression">
        <br/><img src="images/fig2-05.png" alt="The nearest neighbors approach isn't useful just for classification. You can use it for regression too. So here are three plots that show the same simple regression problem with one input feature and the corresponding target values in the training data. The left most plot here, this one, shows just the original training data points. And the middle and right plots show the predictions made by k and n regression algorithm, when k = 1 and k = 3. So in these plots, you can see the training points are actually in green. These green circles are the training points and the blue triangles are the output of the k-nearest neighbor regression for any given input value of x. So for example the knn regression prediction for this point here is this y value here. " title= "1-NN Classification" width="550">
    </a>
    + Diagrams: original, k = 1, k = 3
    + Green dot = training point; blue triangle = test point
    + For k=1, $\text{x_test} = -0.6$ and the nearest point is $(-0.5, 105)$, therefore, $\text{predict} = 105$
    + For k=3, $\text{x_test} = -1.25$ and 3 nearest points $(-1.6, 55)$, $(-1.4, 90)$, and $(-0.9, 90)$, $\text{predict} = (55+90+90)/3$
    + Demo
        ```python
        from sklearn.neighbors import KNeighborsRegressor

        X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

        knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

        print(knnreg.predict(X_test))
        print('R-squared test score: {:.3f}'.format(knnreg.score(X_test, y_test)))
        # [ 231.71  148.36  150.59  150.59   72.15  166.51  141.91  235.57  208.26
        #   102.1   191.32  134.5   228.32  148.36  159.17  113.47  144.04  199.23
        #   143.19  166.51  231.71  208.26  128.02  123.14  141.91]
        # R-squared test score: 0.425
        ```

+ The $R^2$ ("r-squared") Regression Score
    + Measures how well a prediction model for regression fits the given data.
    + The score is between $0$ and $1$:
        + A value of $0$ corresponds to a constant model that predicts the mean value of all training target values.
        + A value of $1$ corresponds to perfect prediction
    + Also known as "__coefficient of determination__"
    + Demo: Regression model complexity as a function of K
        ```python
        fig, subaxes = plt.subplots(1, 2, figsize=(8,4))
        X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

        for thisaxis, K in zip(subaxes, [1, 3]):
            knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
            y_predict_output = knnreg.predict(X_predict_input)
            thisaxis.set_xlim([-2.5, 0.75])
            thisaxis.plot(X_predict_input, y_predict_output, '^', markersize = 10,
                    label='Predicted', alpha=0.8)
            thisaxis.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
            thisaxis.set_xlabel('Input feature')
            thisaxis.set_ylabel('Target value')
            thisaxis.set_title('KNN regression (K={})'.format(K))
            thisaxis.legend()
        plt.tight_layout()
        ```
    <img src="images/plt2-08.png" alt="Just as we did for classification, let's look at the connection between model complexity and generalization ability as measured by the r-squared training and test values on the simple regression dataset. " title= "Regression model complexity as a function of K" width="450">

+ KNeighborsClassifier and KNeighborsRegressor: important parameters
    + Model complexity
        + `n_neighbors`: number of nearest neighbors ($k$) to consider: Default = 5
    + Model fitting
        + `metric`: distance function between data points: Default: Minkowski distance with power parameter p = 2 (Euclidean)

    + Demo: Regression model complexity as a function of K¶
        ```python
        # plot k-NN regression on sample dataset for different values of K
        fig, subaxes = plt.subplots(1, 5, figsize=(20, 5))
        X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)
        for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]):
            knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
            y_predict_output = knnreg.predict(X_predict_input)
            train_score = knnreg.score(X_train, y_train)
            test_score = knnreg.score(X_test, y_test)
            thisaxis.plot(X_predict_input, y_predict_output)
            thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
            thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
            thisaxis.set_xlabel('Input feature')
            thisaxis.set_ylabel('Target value')
            thisaxis.set_title('KNN Regression (K={})\nTrain $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$'
                .format(K, train_score, test_score))
            thisaxis.legend()
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        ```
        <img src="images/plt2-09.png" alt="Just as we did for classification, let's look at the connection between model complexity and generalization ability as measured by the r-squared training and test values on the simple regression dataset. The series of plots on the notebook shows how the KNN regression algorithm fits the data for k = 1, 3, 7, 15, and in an extreme case of k = 55. It represents almost half the training points. We can see the same pattern in model complexity for k and N regression that we saw for k and N classification. Namely, that small values of k give models with higher complexity. And large values of k result in simpler models with lower complexity. Starting on the left when k = 1, the regression model fits the training data perfectly with a r-squared score of 1.0. But it's very bad at predicting the target values for new data samples, as reflected in the r-squared test score of only 0.155. As the value of k increases, which we can see acts to smooth out these local variations to capture more of the global trend. Again the training set score drops, but the model gets better at generalizing to new data and the test score goes up as K increases. Finally in this series, the model with k = 15 has the best test set performance, with an r-squared score of 0.485. Increasing k much further however to k = 55, results in both the training and test set scores dropping back down to lower levels, as the model now starts to under-fit. In other words, it's too simple to do well, even on the training data. The pro's of the nearest neighbor approach are that it's simple and easy to understand why a particular prediction is made. A k-nearest neighbor approach can be a reasonable baseline against what you can compare more sophisticated methods. When the training data has many instances, or each instance has lots of features, this can really slow down the performance of a k-nearest neighbors model. So in general, if your data set has hundreds or thousands of features, you should consider alternatives to k-nearest neighbors models, especially if your data is sparse. Meaning that each instance has lots of features, but most of them are zero. 
" title= "plot k-NN regression on sample dataset for different values of K" width="800">


### Lecture Video 

<a href="https://d3c33hcgiwev3.cloudfront.net/qdyBZzy_Eee_fhKOKKDDtA.processed/full/360p/index.mp4?Expires=1536278400&Signature=MnTBLit6pHuRL1oyKq6YuXJHdS3iZdP6W52ounY5OQSLlNiXdEMzoHn~6P7u-uTPi~oQosRPFPnH8MkfeSV~gr91Xtbnt3Xl8Dy-OuynwnqX197xvbtVp7kebdjHiohYSMmwBKw-sFUfKb6q-TskZyX-p3yiYdeYYL66QKojfbw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="K-Nearest Neighbors: Classification and Regression" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Linear Regression: Least-Squares

+ Linear Models
    + A linear model is a sum of weighted variablesthat predicts a target output value given an input data instance. 
    + Example: predicting housing prices
        + House features: taxes per year ($X_{TAX}$), age in years ($X_{AGE}$)

            $$\hat{Y_{PRICE}} = 21200 + 109 \cdot X_{TAX} - 2000 \cdot X_{AGE}$$ 
        + A house with feature values $(X_{TAX}, X_{AGE})$ of $(10000, 75)$ would have a predicted selling price of:

            $$\hat{Y_{PRICE}} = 21200 + 109 \cdot 1000 - 2000 \cdot 75 = 1,152,000$$

+ Linear Regression is an Example of a Linear Model
    + Input instance - feature vector: $ {\bf x} = (x_0, x_1, \cdots, x_n)$
    + Predict output: $\hat{y} = \hat(w_0) x_0 + \hat{w_1} x_1 + \cdots + \hat{w_n} x_n + b$
    + Parameters to estimate: train parameters or coefficients
        + $\hat{\bf w} = (\hat{w_0}, \hat{w_1}, \cdots , \hat{w_n})$: feature weights/model coefficients
        + $\hat{\bf b}$: constant bias term / intercept
    + Example - house price: $\hat{w_0} = 109$, $x_0$ = tax paid, $\hat{w_1} = -20$, $x_1$ = house age, $\hat{b} = 212,000$

+ A Linear Regression Model with one Variable (Feature)
    + Input instance: ${\bf x} = (x_0)$
    + Predicted output: $\hat{y} = \hat{w_0} x_0 + \hat{b}$
    + Parameters to estimate: $\hat{w_0}$ (slope, $\hat{b}$ (y-intercept)
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/EiQjD/linear-regression-least-squares">
        <br/><img src="images/fig2-07.png" alt="Here's an example of a linear regression model with just one input variable or feature x0 on a simple artificial example dataset. The blue cloud of points represents a training set of x0, y pairs. In this case, the formula for predicting the output y hat is just w0 hat times x0 + b hat, which you might recognize as the familiar slope intercept formula for a straight line, where w0 hat is the slope, and b hat is the y intercept. The grand red lines represent different possible linear regression models that could attempt to explain the relationship between x0 and y." title= "Multiple regression lines with a given dataset" width="300">
    </a>

+ Least-Squares Linear Regression("Ordinary least-squares")
    + Finds $w$ and $b$ that minimizes the mean squared error of the linear model: the sum of squared differences between predicted target and actual target values.
    + No parameters to control model complexity.
    <a href="https://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression/">
        <br/><img src="https://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression_files/simple_regression.png" alt="In Ordinary Least Squares (OLS) Linear Regression, our goal is to find the line (or hyperplane) that minimizes the vertical offsets. Or in other words, we define the best-fitting line as the line that minimizes the sum of squared errors (SSE) or mean squared error (MSE) between our target variable (y) and our predicted output over all samples i in our dataset of size n." title= "Least-square linear regression" width="350">
    </a>

+ How are Linear Regression Parameters w, bEstimated?
    + Parameters are estimated from training data.
    + There are many different ways to estimate wand b:
        + Different methods correspond to different "fit" criteria and goals and ways to control model complexity.
    + The learning algorithm finds the parameters that optimize an __objective function__, typically to minimize some kind of __loss function__ of the predicted target values vs.actual target values.


+ Least-Squares Linear Regression("Ordinary least-squares")
    + Finds $w$ and $b$ that minimizes the __sum of squared differences(RSS)__ over the training data between predicted target and actual target values.
    + a.k.a. mean squared error of the linear model
    + No parameters to control model complexity.

    $$RSS({\bf w}, b) = \sum^N_{i=1} (y_i - (w_i \cdot x_i + b))^2$$

    + Training set target value: ${\bf y_i}$
    + Predicted target value using model: $({\bf w \cdot x_i} + b)$

+ Least-Squares Linear Regression in Scikit-Learn
    + `linreg.coef_`: ${\bf w_0}$
    + `linreg.intercept_`: $b$
    + Underscore denotes a quantity derived from training data, as opposed to a user setting
    + Demo: Linear regression
        ```python
        from sklearn.linear_model import LinearRegression

        X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)
        linreg = LinearRegression().fit(X_train, y_train)

        print('linear model coeff (w): {}'.format(linreg.coef_))
        print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
        print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
        print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))
        # linear model coeff (w): [ 45.71]
        # linear model intercept (b): 148.446
        # R-squared score (training): 0.679
        # R-squared score (test): 0.492
        ```
    + Demo: Linear regression: example plot
        ```python
        plt.figure(figsize=(5,4))
        plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
        plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
        plt.title('Least-squares linear regression')
        plt.xlabel('Feature value (x)')
        plt.ylabel('Target value (y)')
        plt.show()
        ```
    <img src="images/plt2-10.png" alt=" Here is the same code in the notebook. With additional code to score the quality of the regression model, in the same way that we did for K nearest neighbors regression using the R-squared metric. And here is the notebook code we use to plot the least-squares linear solution for this dataset. " title= "apLinear reqgression" width="250">

+ K-NN Regression vs Least-Squares Linear Regression
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/EiQjD/linear-regression-least-squares">
        <br/><img src="images/fig2-08.png" alt="Here we can see how these two regression methods represent two complementary types of supervised learning. The K nearest neighbor regresser doesn't make a lot of assumptions about the structure of the data, and gives potentially accurate but sometimes unstable predictions that are sensitive to small changes in the training data. So it has a correspondingly higher training set, R-squared score, compared to least-squares linear regression. K-NN achieves an R-squared score of 0.72 and least-squares achieves an R-squared of 0.679 on the training set.On the other hand, linear models make strong assumptions about the structure of the data, in other words, that the target value can be predicted using a weighted sum of the input variables. And linear models give stable but potentially inaccurate predictions. However, in this case, it turns out that the linear model strong assumption that there's a linear relationship between the input and output variables happens to be a good fit for this dataset. " title= "K-NN Regression vs Least-Squares Linear Regression" width="450">
    </a>

+ Demo: Linear model
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)

    print('Crime dataset')
    print('linear model intercept: {}'.format(linreg.intercept_))
    print('linear model coeff:\n{}'.format(linreg.coef_))
    print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))
    # Crime dataset
    # linear model intercept: -1728.1306725806212
    # linear model coeff:
    # [  1.62e-03  -9.43e+01   1.36e+01  -3.13e+01  -8.15e-02  -1.69e+01
    #   -2.43e-03   1.53e+00  -1.39e-02  -7.72e+00   2.28e+01  -5.66e+00
    #    ...        ...        ...          ...         ...      ...
    #    5.97e-01   1.98e+00  -1.36e-01  -1.85e+00]
    # R-squared score (training): 0.673
    # R-squared score (test): 0.496
    ```


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/mmfAvlzrEeejtgqYK5OBTg.processed/full/360p/index.mp4?Expires=1536278400&Signature=c5t~aFcm-nbBmSXNiFUsgAp~0t12pE3u-SXSLqQ2-I5lJxWg7xu3g1N6HK18H6RTdYhCKm2WzBbKR70jmLyWki4w87Yn8oxGPot-BXE7e5WSFJLk1~4CnyFJcy5NIuSkyaDoizDCKhxjis-69LMwUOwfIjGbP2NSCAPrxAXgZVQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Linear Regression: Least-Squares" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## Linear Regression: Ridge, Lasso, and Polynomial Regression

+ Ridge Regression
    + Ridge regression learns $w$, $b$ using the same least-squares criterion but adds a penalty for large variations in $w$ parameters

        $$ RSS_{RIDGE}({\bf w}, b) = \sum_{i=1}^N ({\bf y_i} - (w_i \cdot x_i + b))^2 + \alpha \sum_{j=1}^p w_j^2$$
        
        where $\alpha \sum_{j=1}^p w_j^2$ is the penalty
    + Once the parameters are learned, the __ridge regression prediction formula__ is the __same__ as ordinary least-squares.
    + The addition of a parameter penalty is called __regularization__. Regularization prevents overfitting by restricting the model, typically to reduce its complexity.
    + Ridge regression uses __L2 regularization__: minimize sum of squares of $w$ entries
    + The influence of the regularization term is controlled by the $\alpha$ parameter.
    + Higher alpha means more regularization and simpler models.
    + Demo: Ridge regression
        ```python
        from sklearn.linear_model import Ridge
        X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

        linridge = Ridge(alpha=20.0).fit(X_train, y_train)

        print('Crime dataset')
        print('ridge regression linear model intercept: {}'.format(linridge.intercept_))
        print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
        print('R-squared score (training): {:.3f}'.format(linridge.score(X_train, y_train)))
        print('R-squared score (test): {:.3f}'.format(linridge.score(X_test, y_test)))
        print('Number of non-zero features: {}'.format(np.sum(linridge.coef_ != 0)))
        # Crime dataset
        # ridge regression linear model intercept: -3352.4230358466525
        # ridge regression linear model coeff:
        # [  1.95e-03   2.19e+01   9.56e+00  -3.59e+01   6.36e+00  -1.97e+01
        #   -2.81e-03   1.66e+00  -6.61e-03  -6.95e+00   1.72e+01  -5.63e+00
        #     ...         ...         ...     ...         ...         ...
        #    3.31e-01   3.36e+00   1.61e-01  -2.68e+00]
        # R-squared score (training): 0.671
        # R-squared score (test): 0.494
        # Number of non-zero features: 88
        ```

+ The Need for Feature Normalization
    + Important for some machine learning methods that all features are on the same scale (e.g. faster convergence in learning, more uniform or 'fair' influence for all weights)
        + e.g. regularized regression, k-NN, support vector machines, neural networks, …
    + Can also depend on the data. More on feature engineering later in the course. For now, we do MinMax scaling of the features:
        + For each feature $x_i$: compute the min value $x_i^{MIN}$ and the max value $x_i^{MAX}$ achieved across all instances in the training set.
        + For each feature: transform a given feature $x_i$ value to a scaled version $x_i^{\prime}$ using the formula

            $$ x_i^{\prime} = (x_i - x_i^{MIN}) / (x_i^{MAX} - x_i^{MIN}) $$
    + Demo: 
        ```python
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        from sklearn.linear_model import Ridge
        X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state = 0)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

        print('Crime dataset')
        print('ridge regression linear model intercept: {}'.format(linridge.intercept_))
        print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
        print('R-squared score (training): {:.3f}'.format(linridge.score(X_train_scaled, y_train)))
        print('R-squared score (test): {:.3f}'.format(linridge.score(X_test_scaled, y_test)))
        print('Number of non-zero features: {}'.format(np.sum(linridge.coef_ != 0)))
        # Crime dataset
        # ridge regression linear model intercept: 933.3906385044163
        # ridge regression linear model coeff:
        # [  88.69   16.49  -50.3   -82.91  -65.9    -2.28   87.74  150.95   18.88
        #   -31.06  -43.14 -189.44   -4.53  107.98  -76.53    2.86   34.95   90.14
        #    ...      ...     ...     ...     ...     ...     ...     ...     ...
        #   205.2    75.97   61.38  -79.83   67.27   95.67  -11.88]
        # R-squared score (training): 0.615
        # R-squared score (test): 0.599
        # Number of non-zero features: 88
        ```

+ Feature Normalization with MinMaxScaler
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/M7yUQ/linear-regression-ridge-lasso-and-polynomial-regression">
        <br/><img src="images/fig2-09.png" alt="transformation for each feature as shown here. Here's an example of how it works with two features. Suppose we have one feature 'height' whose values fall in a fairly narrow range between 1.5 and 2.5 units. But a second feature, 'width' has a much wider range between five and 10 units. After applying minmax scaling, values for both features are transformed because they are on the same scale, with the minimum value getting mapped to zero, and the maximum value being transformed to one. And everything else getting transformed to a value between those two extremes." title= "Unnormalized and Normalized data points" width="450">
    </a>

+ Demo: Using a scaler object - fit and transform methods
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled= scaler.transform(X_train)
    X_test_scaled= scaler.transform(X_test)
    clf= Ridge().fit(X_train_scaled, y_train)
    r2_score = clf.score(X_test_scaled, y_test)

    # Tip: It can be more efficient to do fitting and transforming together 
    # on the training set using the fit_transform method.
    scaler = MinMaxScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    ```

+ Feature Normalization: The test set must use identical scaling to the training set
    + Fit the scaler using the training set, then apply the same scaler to transform the test set.
    + Do not scale the training and test sets using different scalers: this could lead to random skew in the data.
    + Do not fit the scaler using any part of the test data: referencing the test data can lead to a form of _data leakage_. More on this issue later in the course.


+ Demo: Ridge regression with regularization parameter - alpha
    ```python
        print('Ridge regression: effect of alpha regularization parameter\n')
        for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
            linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
            r2_train = linridge.score(X_train_scaled, y_train)
            r2_test = linridge.score(X_test_scaled, y_test)
            num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
            print('Alpha = {:.2f}\n  num abs(coeff) > 1.0: {}, \
                r-squared training: {:.2f}, r-squared test: {:.2f}'
                .format(this_alpha, num_coeff_bigger, r2_train, r2_test))

        # Ridge regression: effect of alpha regularization parameter
        # 
        # Alpha = 0.00
        #   num abs(coeff) > 1.0: 88, r-squared training: 0.67, r-squared test: 0.50
        # Alpha = 1.00
        #   num abs(coeff) > 1.0: 87, r-squared training: 0.66, r-squared test: 0.56
        # Alpha = 10.00
        #   num abs(coeff) > 1.0: 87, r-squared training: 0.63, r-squared test: 0.59
        # Alpha = 20.00
        #   num abs(coeff) > 1.0: 88, r-squared training: 0.61, r-squared test: 0.60
        # Alpha = 50.00
        #   num abs(coeff) > 1.0: 86, r-squared training: 0.58, r-squared test: 0.58
        # Alpha = 100.00
        #   num abs(coeff) > 1.0: 87, r-squared training: 0.55, r-squared test: 0.55
        # Alpha = 1000.00
        #   num abs(coeff) > 1.0: 84, r-squared training: 0.31, r-squared test: 0.30
    ```

+ Lasso regression is another form of regularized linear regression that uses an L1 regularization penalty for training (instead of ridge's L2 penalty)
    + __L1 penalty__: Minimize the sum of the __absolute values__ of the coefficients

        $$ RSS_{LASSO} ({\bf w}, b) = \sum_{i=1}^N (y_i - (w_i \cdot x_i + b))^2 + \alpha \sum_{j=1}^p |w_j|$$
    + This has the effect of setting parameter weights in $w$ to __zero__ for the least influential variables. This is called a __sparse solution__: a kind of feature selection
    + The parameter $\alpha$ controls amount of L1 regularization (default = 1.0).
    + The prediction formula is the same as ordinary least-squares.
    + When to use ridge vs lasso regression:
        + Many small/medium sized effects: use _ridge_.
        + Only a few variables with medium/large effect: use _lasso_.
    + Demo: Lasson Regression
        ```python
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

        print('Crime dataset')
        print('lasso regression linear model intercept: {}'.format(linlasso.intercept_))
        print('lasso regression linear model coeff:\n{}'.format(linlasso.coef_))
        print('Non-zero features: {}'.format(np.sum(linlasso.coef_ != 0)))
        print('R-squared score (training): {:.3f}'.format(linlasso.score(X_train_scaled, y_train)))
        print('R-squared score (test): {:.3f}\n'.format(linlasso.score(X_test_scaled, y_test)))
        print('Features with non-zero weight (sorted by absolute magnitude):')

        for e in sorted (list(zip(list(X_crime), linlasso.coef_)),
                        key = lambda e: -abs(e[1])):
            if e[1] != 0:
                print('\t{}, {:.3f}'.format(e[0], e[1]))
        # Crime dataset
        # lasso regression linear model intercept: 1186.6120619985786
        # lasso regression linear model coeff:
        # [    0.       0.      -0.    -168.18    -0.      -0.       0.     119.69
        #      0.      -0.       0.    -169.68    -0.       0.      -0.       0.
        #      ...      ...     ...     ...       ...       ...     ...     ...
        #   -104.57   264.93     0.      23.45   -49.39     0.       5.2      0.  ]
        # Non-zero features: 20
        # R-squared score (training): 0.631
        # R-squared score (test): 0.624
        # 
        # Features with non-zero weight (sorted by absolute magnitude):
        #  | PctKidsBornNeverMar, 1488.365
        #  | PctKids2Par, -1188.740
        #   ...
        #  | PctLargHouseFam, 20.144
        #  | PctSameCity85, 5.198
        ```

+ Demo: Lasso regression with regularization parameter - alpha
    ```python
    print('Lasso regression: effect of alpha regularization\n\
    parameter on number of features kept in final model\n')
    
    for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
        linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
        r2_train = linlasso.score(X_train_scaled, y_train)
        r2_test = linlasso.score(X_test_scaled, y_test)
        
        print('Alpha = {:.2f}\n  Features kept: {}, r-squared training: {:.2f}, \
    r-squared test: {:.2f}'
             .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
    # Lasso regression: effect of alpha regularization
    # parameter on number of features kept in final model
    # 
    # Alpha = 0.50
    #   Features kept: 35, r-squared training: 0.65, r-squared test: 0.58
    # Alpha = 1.00
    #   Features kept: 25, r-squared training: 0.64, r-squared test: 0.60
    # Alpha = 2.00
    #   Features kept: 20, r-squared training: 0.63, r-squared test: 0.62
    # Alpha = 3.00
    #   Features kept: 17, r-squared training: 0.62, r-squared test: 0.63
    # Alpha = 5.00
    #   Features kept: 12, r-squared training: 0.60, r-squared test: 0.61
    # Alpha = 10.00
    #   Features kept: 6, r-squared training: 0.57, r-squared test: 0.58
    # Alpha = 20.00
    #   Features kept: 2, r-squared training: 0.51, r-squared test: 0.50
    # Alpha = 50.00
    #   Features kept: 1, r-squared training: 0.31, r-squared test: 0.30
    ```

+ Lasso Regression on the Communities and Crime Dataset
    + For alpha = $2.0$, $20$ out of $88$ features have non-zero weight.
    + Top features (sorted by abs. magnitude):
        
        > PctKidsBornNeverMar, 1488.365 # percentage of kids born to people who never married  
        > <span style="color:red"> PctKids2Par, -1188.740 </span> # percentage of kids in family housing with two parents  
        > HousVacant, 459.538 # number of vacant households  
        > PctPersDenseHous, 339.045 # percent of persons in dense housing (more than 1 person/room)  
        > NumInShelters, 264.932 # number of people in homeless shelters  

+ Polynomial Features with Linear

    $$ {\bf x} = (x_0, x_1) \Longrightarrow {\bf x^{\prime}} = (x_0, x_1, x_0^2, x_0 x_1, x_1^2)$$

    $$ \hat{y} = \hat{w_0}x_0 + \hat{w_1}x_1 + \hat{w_{00}}x_0^2 + \hat{w_{01}} x_0 x_1 + \hat{w_{11}}x_1^2 + b$$
    + Generate new features consisting of all polynomial combinations of the original two features $(x_0, x_1)$.
    + The _degree_ of the polynomial specifies how many variables participate at a time in each new feature (above example: degree 2)
    + This is still a weighted linear combination of features, so it's __still a linear model__, and can use same least-squares estimation method for $w$  and $b$.

+ Least-Squares Polynomial Regression
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/M7yUQ/linear-regression-ridge-lasso-and-polynomial-regression">
        <br/><img src="images/fig2-10.png" alt="So you can think of this intuitively as allowing polynomials to be fit to the training data instead of simply a straight line, but using the same least-squares criterion that minimizes mean squared error. We'll see later that this approach of adding new features like polynomial features is also very effective with classification. And we'll look at this kind of transformation again in kernelized support vector machines. When we add these new polynomial features, we're essentially adding to the model's ability to capture interactions between the different variables by adding them as features to the linear model. For example, it may be that housing prices vary as a quadratic function of both the lat size that a house sits on, and the amount of taxes paid on the property as a theoretical example. A simple linear model could not capture this nonlinear relationship, but by adding nonlinear features like polynomials to the linear regression model, we can capture this nonlinearity. Or generally, we can use other types of nonlinear feature transformations beyond just polynomials. This is beyond the scope of this course but technically these are called nonlinear basis functions for regression, and are widely used. Of course, one side effect of adding lots of new features especially when we're taking every possible combination of K variables, is that these more complex models have the potential for overfitting. So in practice, polynomial regression is often done with a regularized learning method like ridge regression." title= "complex regression problem with one input variable" width="600">
    </a>

+ Polynomial Features with Linear Regression
    + Why would we want to transform our data this way?
        + To capture interactions between the original features by adding them as features to the linear model.
        + To make a classification problem easier (we'll see this later).
        + E.g., housing price as a quadratic function of house size and tax paid
    + More generally, we can apply other non-linear transformations to create new features
        + (Technically, these are called _non-linear basis functions_)
    + Beware of polynomial feature expansion with high as this can lead to complex models that overfit
        + Thus, polynomial feature expansion is often combined with a regularized learning method like ridge regression.

+ Demo: Polynomial regression
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1,
                                                    random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)

    print('linear model coeff (w): {}'.format(linreg.coef_))
    print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
    print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))

    print('\nNow we transform the original input data to add\n\
    polynomial features up to degree 2 (quadratic)\n')
    poly = PolynomialFeatures(degree=2)
    X_F1_poly = poly.fit_transform(X_F1)

    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)

    print('(poly deg 2) linear model coeff (w):\n{}'.format(linreg.coef_))
    print('(poly deg 2) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
    print('(poly deg 2) R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
    print('(poly deg 2) R-squared score (test): {:.3f}\n'.format(linreg.score(X_test, y_test)))

    print('\nAddition of many polynomial features often leads to\n\
    overfitting, so we often use polynomial features in combination\n\
    with regression that has a regularization penalty, like ridge\n\
    regression.\n')

    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                    random_state = 0)
    linreg = Ridge().fit(X_train, y_train)

    print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
        .format(linreg.coef_))
    print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
        .format(linreg.intercept_))
    print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
        .format(linreg.score(X_train, y_train)))
    print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
        .format(linreg.score(X_test, y_test)))
    # linear model coeff (w): [  4.42   6.     0.53  10.24   6.55  -2.02  -0.32]
    # linear model intercept (b): 1.543
    # R-squared score (training): 0.722
    # R-squared score (test): 0.722
    # 
    # Now we transform the original input data to add
    # polynomial features up to degree 2 (quadratic)
    # 
    # (poly deg 2) linear model coeff (w):
    # [  3.41e-12   1.66e+01   2.67e+01  -2.21e+01   1.24e+01   6.93e+00
    # 1.05e+00   3.71e+00  -1.34e+01  -5.73e+00   1.62e+00   3.66e+00
    # 5.05e+00  -1.46e+00   1.95e+00  -1.51e+01   4.87e+00  -2.97e+00
    # -7.78e+00   5.15e+00  -4.65e+00   1.84e+01  -2.22e+00   2.17e+00
    # -1.28e+00   1.88e+00   1.53e-01   5.62e-01  -8.92e-01  -2.18e+00
    # 1.38e+00  -4.90e+00  -2.24e+00   1.38e+00  -5.52e-01  -1.09e+00]
    # (poly deg 2) linear model intercept (b): -3.206
    # (poly deg 2) R-squared score (training): 0.969
    # (poly deg 2) R-squared score (test): 0.805
    # 
    # Addition of many polynomial features often leads to
    # overfitting, so we often use polynomial features in combination
    # with regression that has a regularization penalty, like ridge
    # regression.
    # 
    # (poly deg 2 + ridge) linear model coeff (w):
    # [ 0.    2.23  4.73 -3.15  3.86  1.61 -0.77 -0.15 -1.75  1.6   1.37  2.52
    # 2.72  0.49 -1.94 -1.63  1.51  0.89  0.26  2.05 -1.93  3.62 -0.72  0.63
    # -3.16  1.29  3.55  1.73  0.94 -0.51  1.7  -1.98  1.81 -0.22  2.88 -0.89]
    # (poly deg 2 + ridge) linear model intercept (b): 5.418
    # (poly deg 2 + ridge) R-squared score (training): 0.826
    # (poly deg 2 + ridge) R-squared score (test): 0.825
    ```


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/zySS11zrEeeP6hLXsz0H0g.processed/full/360p/index.mp4?Expires=1536364800&Signature=itQQbJ3rem54AhQiqzXxstWOb1YeMTRmDGNQEs-QGR~jGfV2qhQNCyMAprFfG-~0qXs5xonMYBaqvqj8ZiqG5Oi5~gH5Y8AcImrh5e~Aijk1~QCsE40vIJJUG2u8S5atWtd6qAQ0mdDl7mSIrf3eagahYyHX5L0B1UZJuHeD0aw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Linear Regression: Ridge, Lasso, and Polynomial Regression" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Logistic Regression

+ Linear regression 
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/bEtYh/logistic-regression">
        <br/><img src="images/fig2-11.png" alt="Linear regression predicts a real valued output y based on a weighted sum of input variables or features xi, plus a constant b term. This diagram shows that formula in graphical form. The square boxes on the left represent the input features, xi. And the values above the arrows represent the weights that each xi is multiplied by. The output variable y in the box on the right is the sum of all the weighted inputs that are connected into it. Note that we're adding b as a constant term by treating it as the product of a special constant feature with value 1 multiplied by a weight of value b. This formula is summarized in equation form below the diagram. The job of linear regression is to estimate values for the model coefficients, wi hat and b hat. They give a model that best fit the training data with minimal squared error. " title= "Linear Regression Block Diagram" width="200">
    </a>

+ Linear models for classification: Logistic Regression
    + a kind of generalized linear model
    + take a set of variables, the features, and estimate a target value
    + binary variable instead of a continuous value, generalized to multi-class categorical variable
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/bEtYh/logistic-regression">
        <br/><img src="images/fig2-12.png" alt="Logistic regression is similar to linear regression, but with one critical addition. Here, we show the same type of diagram that we showed for linear regression with the input variables, xi in the left boxes and the model coefficients wi and b above the arrows. The logistic regression model still computes a weighted sum of the input features xi and the intercept term b, but it runs this result through a special non-linear function f, the logistic function represented by this new box in the middle of the diagram to produce the output y. The logistic function itself is shown in more detail on the plot on the right. It's an S shaped function that gets closer and closer to 1 as the input value increases above 0 and closer and closer to 0 as the input value decreases far below 0. The effect of applying the logistic function is to compress the output of the linear function so that it's limited to a range between 0 and 1. Below the diagram, you can see the formula for the predicted output y hat which first computes the same linear combination of the inputs xi, model coefficient weights wi hat and intercept b hat, but runs it through the additional step of applying the logistic function to produce y hat. If we pick different values for b hat and the w hat coefficients, we'll get different variants of this s shaped logistic function, which again is always between 0 and 1. Because the job of basic logistic regression is to predict a binary output value, you can see how this might used for binary classification. We could identify data instances with the target value of 0 as belonging to the negative class and data instances with a target value of 1 belonging to the positive class. Then the value of y hat, that's the output from the logistic regression formula, can be interpreted as the probability that the input data instance belongs to the positive class, given its input features. " title= "Logistic Regression: functional plot and block diagram" width="450">
    </a>

    + The logistic function transforms real-valued input to an output number $y$ between $0$ and $1$, interpreted as the __probability__ the input object belongs to the positive class, given its input features $x_0, x_1, \cdots,x_n)$

+ Linear models for classification: Logistic Regression
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/bEtYh/logistic-regression">
        <br/><img src="images/fig2-13.png" alt="Suppose we want to whether or not a student will pass a final exam based on a single input variable that's the number of hours they spend studying for the exam. Students who end up failing the exam are assigned to the negative class, which corresponds to a target value of 0. And students who pass the exam are assigned to the positive class and associated with a target value of 1. This plot shows an example training set. The x-axis corresponds to the number of hours studied and the y-axis corresponds to the probability of passing the exam. The red points to the left, with a target value of 0 represent points in the training set, which are examples of students who failed the exam, along with the number of hours they spent studying. Likewise, the blue points with target value 1 on the right represent points in the training set, corresponding to students who passed the exam. With their x values representing the number of hours those students spent studying. " title= "Logistic regression for classification" width="350">
    </a>
    + Training set to represent the hours of study and passing/failing of the exam: red dot = negative class = failing = 0; blue dot = positive class = passing = 1

+ Logistic Regression for binary classification
    <a href="https://helloacm.com/a-short-introduction-logistic-regression-algorithm/">
        <br/><img src="https://helloacm.com/wp-content/uploads/2016/03/logistic-regression-example.jpg" alt="The logistic function looks like a big S and will transform any value into the range 0 to 1. This is useful because we can apply a rule to the output of the logistic function to snap values to 0 and 1 (e.g. IF less than 0.5 then output 1) and predict a class value.  Because of the way that the model is learned, the predictions made by logistic regression can also be used as the probability of a given data instance belonging to class 0 or class 1. This can be useful on problems where you need to give more rationale for a prediction." title= "Logistic regression: binary classification" width="200">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/bEtYh/logistic-regression">
        <img src="images/fig2-14.PNG" alt="We can apply logistic regression to learn a binary classifier using this training set, using the same idea we saw in the previous exam example. To do this, we'll add a third dimension shown here as the vertical y-axis. Corresponding to the probability of belonging to the positive class. We'll say that the red points are associated with the negative class and have a target value of 0, and the blue points are associated with the positive class and have a target value of 1. Then just as we did in the exam studying example, we can estimate the w hat and b hat parameters of the logistic function that best fits this training data. The only difference is that the logistic function is now a function of two input features and not just one. So it forms something like a three dimensional S shaped sheet in this space. " title= "Logistic regression with 3D diagram" width="240">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/bEtYh/logistic-regression">
        <img src="images/fig2-15.png" alt="Once this logistic function has been estimated from the training data, we can use it to predict the class membership for any point, given its Feature 1 and Feature 2 values, same way we did for the exam example. Any data instances whose logistic probability estimate y hat is greater than or equal to 0.5 are predicted to be in the positive blue class, otherwise, in the other red class. Now if we imagine that there's a plane representing y equals 0.5, as shown here, that intersects this logistic function. It turns out that all the points that have a value of y = 0.5, when you intersect that with a logistic function, the points all lie along a straight line. In other words, using logistic regression gives a linear decision boundary between the classes as shown here. If you imagine looking straight down on the 3D logistic function on the left, you get the view that looks something like something on the right. Here. The points with y greater or equal to 0.5 on the logistic function, lie in a region to the right of the straight line, which is the dash line on the right here. And the points with y less than 0.5 on the logistic function would form a region to the left of that dash line. Let's look at an example with real data in Scikit-Learn. To perform logistic, regression in Scikit-Learn, you import the logistic regression class from the sklearn.linear model module, then create the object and call the fit method using the training data just as you did for other class files like k nearest neighbors. " title= "Logistic regression with binary boundary" width="180">
    </a>


+ Simple logistic regression problem: two-class, two-feature version of the fruit dataset
    + Demo: Logistic regression for binary classification on fruits dataset using height, width features (positive class: apple, negative class: others) (below left)
        ```python
        from sklearn.linear_model import LogisticRegression
        from adspy_shared_utilities import (
        plot_class_regions_for_classifier_subplot)

        fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
        y_fruits_apple = y_fruits_2d == 1   # make into a binary problem: apples vs everything else
        X_train, X_test, y_train, y_test = (
        train_test_split(X_fruits_2d.as_matrix(),
                        y_fruits_apple.as_matrix(),
                        random_state = 0))

        clf = LogisticRegression(C=100).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, \
            'Logistic regression for binary classification\nFruit dataset: Apple vs others', subaxes)

        h = ; w = 8
        print('A fruit with height {} and width {} is predicted to be: {}'
            .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

        h = 10; w = 7
        print('A fruit with height {} and width {} is predicted to be: {}'
            .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
        subaxes.set_xlabel('height')
        subaxes.set_ylabel('width')

        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
            .format(clf.score(X_train, y_train)))
        print('Accuracy of Logistic regression classifier on test set: {:.2f}'
            .format(clf.score(X_test, y_test)))
        # A fruit with height 6 and width 8 is predicted to be: an apple
        # A fruit with height 10 and width 7 is predicted to be: not an apple
        # Accuracy of Logistic regression classifier on training set: 0.77
        # Accuracy of Logistic regression classifier on test set: 0.73
        ```
        <img src="images/plt2-11.png" alt="The data set we're using here is a modified form of our fruits data set, using only height and width as the features, the features space, and with the target class value modified into a binary classification problem predicting whether an object is an apple, a positive class, or something other than an apple, a negative class. Here is a graphical display of the results. The x-axis corresponds to the height feature and the y-axis corresponds to the width feature. The black points represent the positive apple class training points. And the yellow points are instances of all the other fruits in the training set. The gray decision region represents that area of the height and width feature space, where a fruit would have an estimated probability greater than 0.5 of being an apple. And thus classified as an apple according to the logistic regression function. The yellow decision region corresponds to the region of feature space for objects that have an estimated probability of less than 0.5 of being an apple. You can see the linear decision boundary where the grey region meets the yellow region, that results applying logistic regression. In fact, logistic regression results are often quite similar to those you might obtain from a linear support vector machine, another type of linear model we explore for classification. " title= "caption" width="350">
        <img src="images/plt2-12.png" alt="text" title= "caption" width="350">
    
    + Demo: Logistic regression on simple synthetic dataset (above right)
        ```python
        from sklearn.linear_model import LogisticRegression
        from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

        X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

        fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
        clf = LogisticRegression().fit(X_train, y_train)
        title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                None, None, title, subaxes)

        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
            .format(clf.score(X_train, y_train)))
        print('Accuracy of Logistic regression classifier on test set: {:.2f}'
            .format(clf.score(X_test, y_test)))
        # Accuracy of Logistic regression classifier on training set: 0.80
        # Accuracy of Logistic regression classifier on test set: 0.80
        ```

+ Logistic Regression: Regularization
    + L2 regularization is 'on' by default (like ridge regression)
    + Parameter C controls amount of regularization (default 1.0)
    + As with regularized linear regression, it can be important to normalize all features so that they are on the same scale.
    + Demo: Logistic regression regularization: C parameter
        ```python 
        X_train, X_test, y_train, y_test = (
            train_test_split(X_fruits_2d.as_matrix(), y_fruits_apple.as_matrix(),
            random_state=0))

        fig, subaxes = plt.subplots(1, 3, figsize=(13, 4))

        for this_C, subplot in zip([0.1, 1, 100], subaxes):
            clf = LogisticRegression(C=this_C).fit(X_train, y_train)
            title ='Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)
            
            plot_class_regions_for_classifier_subplot(
                clf, X_train, y_train, X_test, y_test, title, subplot)
        plt.tight_layout()
        ```
        <img src="images/plt2-13.png" alt="Like ridge and lasso regression, a regularization penalty on the model coefficients can also be applied with logistic regression, and is controlled with the parameter C. In fact, the same L2 regularization penalty used for ridge regression is turned on by default for logistic regression with a default value C = 1. Note that for both Support Vector machines and Logistic Regression, higher values of C correspond to less regularization. With large values of C, logistic regression tries to fit the training data as well as possible. While with small values of C, the model tries harder to find model coefficients that are closer to 0, even if that model fits the training data a little bit worse. You can see the effect of changing the regularization parameter C for logistic regression in this visual. Using the same upper class of fire we now vary C to take on values from 0.1 on the left to 1.0 in the middle, and 100.0 on the right. Although the real power of regularization doesn't become evident until we have data that has higher dimensional feature spaces. You can get an idea of the trade off that's happening between relying on a simpler model, one that puts more emphasis on a single feature in this case, out of the two features, but has lower training set accuracy. And that's an example as shown on the left with C = 0.1. Or, better training data fit on the right with C = 100. You can find the code that created this example in the accompanying notebook. " title= "Logistic regression regularization: C parameter" width="700">

    + Demo: Application to real dataset
        ```python
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

        clf = LogisticRegression().fit(X_train, y_train)
        print('Breast cancer dataset')
        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
            .format(clf.score(X_train, y_train)))
        print('Accuracy of Logistic regression classifier on test set: {:.2f}'
            .format(clf.score(X_test, y_test)))
        # Breast cancer dataset
        # Accuracy of Logistic regression classifier on training set: 0.96
        # Accuracy of Logistic regression classifier on test set: 0.96
        ```

+ Logistic and linear functions
    <a href="https://www.saedsayad.com/logistic_regression.htm">
        <br/><img src="https://www.saedsayad.com/images/LogReg_1.png" alt="Logistic regression predicts the probability of an outcome that can only have two values (i.e. a dichotomy). The prediction is based on the use of one or several predictors (numerical and categorical). A linear regression is not appropriate for predicting the value of a binary variable for two reasons: 1)A linear regression will predict values outside the acceptable range (e.g. predicting probabilities outside the range 0 to 1); 2) Since the dichotomous experiments can only have one of two possible values for each experiment, the residuals will not be normally distributed about the predicted line.  On the other hand, a logistic regression produces a logistic curve, which is limited to values between 0 and 1. Logistic regression is similar to a linear regression, but the curve is constructed using the natural logarithm of the “odds” of the target variable, rather than the probability. Moreover, the predictors do not have to be normally distributed or have equal variance in each group." title= "Logistic Regression" width="300">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/c_rluz6DEee4_A7ezGAgwg.processed/full/360p/index.mp4?Expires=1536364800&Signature=XiNruM5NmiNHvjmKhBpfVD~RyzSa3QAiX-AwMnpz9~d6WjmKkmUQyteS4xAINTNQiHqqfjL8hXz2cFjImhMbm9yrhfZEfjNKgcsF0Clo0AHH14DOTOgbdLzSlquvLw9F9Zt8gQQ4hNGTGkPUOkxTNAiwA-cNTIUvX49CxLAWaE4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Logistic Regression" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Linear Classifiers: Support Vector Machines

+ Linear classifiers: how would you separate these two groups of training examples with a straight line?
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/uClaN/linear-classifiers-support-vector-machines">
        <br/><img src="images/fig2-18.png" alt="So let's take a look at a simple linear classification problem, a binary problem that has two different classes, and where each data instance is represented by two informative features. So we're going to call these features x1 on the x-axis and x2 on the y-axis. Now, how do we get from taking a data instance that's described by a particular combination of x1, x2 to a class prediction. So that represents the x1 and x2 feature values for this particular instance, let's say, and so that corresponds to the input here. And then we take those two x1, x2 values and put them through a linear function f here. And the output of this f needs to be a class value. So either we want it to return +1, if it's in one class and -1 if it's in the other class. So the linear classifier does that by computing a linear function of x1, x2 that's represented by this part of the equation here. So this w is a vector of weights. This x represents the vector of feature values. And then b is a bias term that gets added in. So this is a dot product, this circle here is a dot product. Which means that, let's say, the simple case where we had (w1, w2), and a future vector of (x1, x2). The dot product of those is simply the linear combination of w1 x1 + w2 x2. We take the X values, and we have certain weights that are learned for the classifier. So we compute w1 x1 + w2 x2. Then we feed that, and then plus a bias term, if there is one. And we feed the output of that through the sign function that converts the value in here. If it's above 0, it'll convert to +1, and if it's below 0, it'll convert it to -1 and that's what the equation represents here. So that is the very simple rule that we use to convert a data instance with its input features to an output prediction. Okay, let's take a look using a specific linear function to see how this works in practice. So what I've done here is drawn a straight line through the space that is defined by the equation x1- x2 = 0. In other words, every point on this line satisfies this equation. So you can see, for example, that x1 = -1 and x2 = -1 and if you subtract them you get 0. Essentially, it's just the line that represents all the points where x1 and x2 are equal. So this corresponds to, so we can rewrite this x1- x2 = 0 into a form that uses a dot product of weights with the input vector x. So in order to get x1- x2 = 0 that's equivalent writing a linear function where you have a weight vector of 1 and -1 and a bias term of 0 So for example, so if we have weights (w1, w2) dot (x1, x2). We call that this is just the same as computing w1 x1 + w2 x2. And so in this case, if w is 1 and -1, that's equivalent to x1- x2. " title= "Linear Classifier: how would you separate these two groups of training examples with a straight line" width="450">
    </a>
    + A linear classifier is a function that maps an input data point $x$ to an output class value $y$ (+1 or -1) using a linear function (with weight parameters $w$ of the input point's features.
    + `sign` function: $+1 \text{ if } \hat{y} > 0$ and $-1 \text{ if } \hat{y} < 0$
    + General linear classifier equation: $f(x, w, b) = sign(w \circ x + b) = sign(\sum w[i]x[i] + b)$ where $w$ is a vector of weights, $x$ is a vector of feature values, and $b$ is a bias term
    + operator $\circ$: $[w_1, w_2] \circ [x_1, x_2] = w_1 x_1 + w2 x_2$

+ Linear classifiers: how would you separate these two groups of training examples with a line?
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/uClaN/linear-classifiers-support-vector-machines">
        <br/><img src="images/fig2-19.png" alt="So all I've done here then is just convert the description of this line into a form that can be used as a decision rule for classifier. So I might have to look at a specific point. So suppose that we wanted to classify the point here that had coordinates -0.75 and -2.25. So this point right here. So all we would do to have a classifier make a decision with this decision boundary would be to plug in those coordinates into this part of the function, apply the weights and the bias term that describe the decision boundary. So here we're computing. So this expression here corresponds to this part of the equation. And so, if we compute w1 times x1 + w2 times x2 plus 0, we get a value of 0.15 that's inside the sign function. And then the sign function will output 1 if this value is greater than zero, which it is or minus 1, if the value is less than zero. So in this case, because it's greater than zero, the output from the decision function would be plus 1. So this has classified this point as being class one. If we look at a different point, let's take a look at the point -1.75 and -0.25. So again, we're just going to up, so this is corresponding to the value of x1 and x2. And again, if we just take these values and plug them into this part of the equation. Apply the weights and the bias term that describe the decision boundary as you did before. We do this computation, we find out that, in fact, classifier predicts a class of -1 here. So you see that by applying a simple linear formula, we've been able to produce a class value for any point in this two dimensional features space. " title= "caption" width="450">
    </a>
    + Suppose $w=[1, -1]$ and $b = 0$, it is depicted as diagonal line where $[w_1, w_2] = [1, -1]$, therefore $x_1 - x_2 = 0$
    + Suppose to classify $[-0.75, -2.25]$, 

        $$f([-0.75, -2.25], w, b) = sign(1 \cdot -0.75 + (-1) \cdot (-2.25) + 0) =  sign(-0.75 + 2.25 = 1.50) = +1$$
    + Suppose to classify $[-1.75, -0.25]$

        $$f([-1.75, -0.25], w, b) = sign(1 \cdot -1.75 + (-1) \cdot (-0.25) + 0) =  sign(-1.75 + 0.25 = -1.50) = -1$$


+ Linear Classifiers & Classifier Margin
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/uClaN/linear-classifiers-support-vector-machines">
        <br/><img src="images/fig2-20.png" alt="So one way to define a good classifier is to reward classifiers for the amount of separation that can provide between the two classes. And to do this, we need define the concept of classifier margin. So informally, for our given classifier, The margin is the width that the decision boundary can be increased before hitting a data point. So what we do is we take the decision boundary, and we grow a region around it. Sort of in this perpendicular to the line in this direction and that direction, and we grow this width until we hit a data point. So in this case, we were only able to grow the margin a small amount here, before hitting this data point. So this width here between the decision boundary and nearest data point represents the margin of this particular classifier. Now you can imagine that for every classifier that we tried, we can do the same calculation or simulation to find the margin. " title= "caption" width="350">
    </a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/uClaN/linear-classifiers-support-vector-machines">
        <img src="images/fig2-21.png" alt="And so among all possible classifiers that separate these two classes then, we can define the best classifier as the classifier that has the maximum amount of margin which corresponds to the one shown here. So you recall that the original classifier on the previous slide had a very small margin. This one manages to achieve a must larger margin. So again, the margin is the distance the width that we can go from the decision boundary perpendicular to the nearest data point. So you can see that by defining this concept of margin that sort of quantifies the degree to which the classifier can split the classes into two regions that have some amount of separation between them. We can actually do a search for the classifier that has the maximum margin. This maximum margin classifier is called the Linear Support Vector Machine, also known as an LSVM or a support vector machine with linear kernel. Now we'll explain more about what the concept of a kernel is and how you can define nonlinear kernels as well as kernels, and why you'd want to do that. We'll cover those shortly in a continuation of our support vector machine lecture later in this particular week. Here's an example in the notebook on how to use the default linear support vector classifier in scikit-learn, which is defined in the sklearn SVM library. 
" title= "Classifier margin" width="350">
    </a>
    + __Classifier margin__: Defined as the maximum width the decision boundary area can be increased before hitting a data point.

+ Maximum margin linear classifier: Linear Support Vector Machines
    + $f(x, w, b) = sign(w \circ x + b)$
    + Maximum margin classifier: The linear classifier with maximum margin is a linear Support Vector Machine (LSVM).
    + Demo: Linear Support Vector Machine
        ```python
        from sklearn.svm import SVC
        from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

        X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

        fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
        this_C = 1.0
        clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
        title = 'Linear SVC, C = {:.3f}'.format(this_C)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)
        ```
        <img src="images/plt2-14.png" alt="The linear SVC class implements a linear support vector classifier and is trained in the same way as other classifiers, namely by using the fit method on the training data. Now in the simple classification problem I just showed you, the two classes were perfectly separable with a linear classifier. In practice though, we typically have noise or just more complexity in the data set that makes a perfect linear separation impossible, but where most points can be separated without errors by linear classifier. And our simple binary classification dataset here is an illustration of that. So how tolerant the support vector machine is of misclassifying training points, as compared to its objective of minimizing the margin between classes is controlled by a regularization parameter called C which by default is set to 1.0 as we have here. " title= "Maximum margin linear classifier: Linear Support Vector Machines" width="350">


+ Regularization for SVMs: the $C$ parameter
    + The strength of regularization is determined by $C$
    + Larger values of $C$: less regularization
        + Fit the training data as well as possible
        + Each individual data point is important to classify correctly
    + Smaller values of $C$: more regularization
        + More tolerant of errors on individual data points
    + Demo: Linear Support Vector Machine: C parameter
        ```python
        from sklearn.svm import LinearSVC
        from adspy_shared_utilities import plot_class_regions_for_classifier

        X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
        fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

        for this_C, subplot in zip([0.00001, 100], subaxes):
            clf = LinearSVC(C=this_C).fit(X_train, y_train)
            title = 'Linear SVC, C = {:.5f}'.format(this_C)
            plot_class_regions_for_classifier_subplot(clf, X_train, y_train, 
                None, None, title, subplot)
        plt.tight_layout()
        ```
        <img src="images/plt2-15.png" alt="Larger values of C represent less regularization and will cause the model to fit the training set with these few errors as possible, even if it means using a small immersion decision boundary. Very small values of C on the other hand use more regularization that encourages the classifier to find a large marge on decision boundary, even if that decision boundary leads to more points being misclassified. Here's an example in the notebook showing the effect of varying C on this basic classification problem. On the right, when C is large, the decision boundary is adjusted so that more of the black training points are correctly classified. While on the left, for small values of C, the classifier is more tolerant of these errors in favor of capturing the majority of data points correctly with a larger margin. " title= "Linear Support Vector Machine: C parameter" width="550">

+ Demo: Application to real dataset
    ```python
    from sklearn.svm import LinearSVC
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    
    clf = LinearSVC().fit(X_train, y_train)
    print('Breast cancer dataset')
    print('Accuracy of Linear SVC classifier on training set: {:.2f}'
         .format(clf.score(X_train, y_train)))
    print('Accuracy of Linear SVC classifier on test set: {:.2f}'
         .format(clf.score(X_test, y_test)))
    # Breast cancer dataset
    # Accuracy of Linear SVC classifier on training set: 0.90
    # Accuracy of Linear SVC classifier on test set: 0.92
    ```

+ Linear Models: Pros and Cons
    + Pros:
        + Simple and easy to train.
        + Fast prediction
            + linear nature of prediction function
            + LSVM effectly performed on high dimensional dataset, in particular, sparse data instances
        + Scales well to very large datasets.
            + LSVM only using subset of training points (support vectors) and decision function
        + Works well with sparse data.
        + Reasons for prediction are relatively easy to interpret.
    + Cons:
        + For lower-dimensional data, other models may have superior generalization performance.
        + For classification, data may not be linearly separable (more on this in SVMs with non-linear kernels)

+ linear_model: Important Parameters
    + Model complexity
        + __alpha__: weight given to the L1 or L2 regularization term in regression models
        + default = 1.0
    + __C__: regularization weight for `LinearSVC` and `LogisticRegression` classification models
        + default = 1.0


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/3thhjT6DEeeR4AqenwJvyA.processed/full/360p/index.mp4?Expires=1536364800&Signature=WQwyPHi2C1RMpNioL4xmscAFMLvoymVrEG3CeWUSUm6HTlCkOAKBgJgFcSA1ffjum~caEbRzIn-0YfRQye2hz~3-HAD9CH99Kx97DBd9c8PR2qDgUTgcV2Xnmialwy6nsdQcYIjtZgSmlNf9Blxvu-AH71E7~PhKbkDLWeORc7g_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Linear Classifiers: Support Vector Machines" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Multi-Class Classification

+ Multi-Class classification with linear Models
    + Converting multi-class  classification problem into a series of binary problems
    + Binary class: one class selected and others as another class
    + Suppose $(height, weight) = (2, 6)$ for apple, $y_apple = -0.23401135 * height + 0.72246123 * weight - 3.31753728 = 0.549$, therefore, $(0.549) = +1$
    ```python
    from sklearn.svm import LinearSVC

    X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)

    clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
    print('Coefficients:\n', clf.coef_)
    print('Intercepts:\n', clf.intercept_)
    # Coefficients: (4 sets)
    #  [[-0.23  0.72]   apple vs others
    #  [-1.63  1.15]    
    #  [ 0.08  0.31]
    #  [ 1.26 -1.68]]
    # Intercepts: (4 sets)
    #  [-3.32  1.2  -2.75  1.16] (apple  )

    # Multi-class results on the fruit dataset
    plt.figure(figsize=(6,6))
    colors = ['r', 'g', 'b', 'y']
    cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

    plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
            c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

    x_0_range = np.linspace(-10, 15)

    for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
        # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
        # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
        # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
        plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
        
    plt.legend(target_names_fruits)
    plt.xlabel('height')
    plt.ylabel('width')
    plt.xlim(-2, 12)
    plt.ylim(-2, 15)
    plt.show()
    ```
    <img src="images/plt2-16.png" alt="Here, we simply pass in the normal dataset that has the value from one to four as the category of fruit to be predicted. And we fit it exactly the same way that we would fit the model as if it were a binary problem. And in general, if we're just, you know, fitting, and then predicting, all of this would be completely transparent. Scikit-learn would simply do the right thing and it would learn multiple classes, and it would predict multiple classes, and we wouldn't really have to do much else. However, we can get access to what's happening under the hood as it were, if we look at the coefficients and the intercepts of the linear models that result from fitting to the training data. And this is what this example shows. So, what we're doing here is fitting a linear support vector machine to the fruit training data. And if we look at the coefficient values, we'll see that instead of just one pair of coefficients for a single linear model, a classifier, we actually get four values. And these values correspond to the four classes of fruit in the training set. And so, what scikit-learn has done here is it's created four binary classifiers, one for each class. And so, you can see there are four pairs of coefficients here and there are also four intercept values. So, in this case the first pair of coefficients corresponds to a classifier that classifies apples versus the rest of the fruit, and so, these pair of coefficients and this intercept define a straight line. " title= "Multi-class results on the fruit dataset" height="350">



### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/OBB7qD6LEeeHpAqQsW8qwg.processed/full/360p/index.mp4?Expires=1536451200&Signature=huODEN23nuyNbAIfAYczsZvToxIs7jP49viV1anUdYxMjfwH23wxq6BNcCZZA5Ek8HmlNxBt34~YJhNHmwBhWoKryMF9E-i7NbXuNegAVlMCfxEy6EGT85HwVvhAzuPfoTOM3eTYr4u45~bZziLysjxNF0GvTvHy9V2SU9hT7Js_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Multi-Class Classification" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Kernelized Support Vector Machines

+ Complex binary classification problems
    + We saw how linear support vector classifiers could effectively find a decision boundary with maximum margin
    + But what about more complex binary classification problems?
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lCUeA/kernelized-support-vector-machines">
        <br/><img src="images/fig2-22.png" alt="Linear support vector machines worked well for simpler kinds of classification problems, where the classes were linearly separable or close to linearly separable like this example on the left. But with real data, many classification problems aren't this easy. With the different classes located in future space in a way that a line or hyperplane can't act as an effective classifier. Here's an example on the right. These dataset is difficult, or impossible for a linear model, a line or hyperplane, to classify well. So to help address the situation, we're now going to turn to our next type of supervised learning model. A very powerful extension of linear support vector machines called kernelized support vector machines. " title= "Easy for a linear classifier vs. Difficult/impossible for a linear classifier" height="200">
    </a>

    + Kernalized Supported Vector Machine a.k.a. Supported Vector Machine (SVM)
    + SVM applied for classification and regression, but only classification cocered here
    + Take original dataset and transform it to a new higher dimensional feature space, where it becomes much easier to classify the transform to data using a linear classifier

+ Way to deal with Complex classification problems
    + A simple 1-dimensional classification problemfor a linear classifier (fig.1)
    + A more perplexing 1-d classification problem for a linear classifier (fig.2)
    + Let's transform the data by adding a second dimension/feature(set to the squared value of the first feature) (fig.3)
    + The data transformation makes it possible to solve this with a linear classifier (fig.4)
    + What does the linear decision boundary in feature space correspond to in the original input space? (fig.5)
    + What does the linear decision boundary correspond to in the original input space? (fig.6) <br/>
        <img src="images/fig2-23.png" alt="Here's a binary classification problem in 1-dimension with a set of points that lie along the x-axis. Color black for one class and white for the second class. Each data point here has just one feature. It's positioned on the x-axis. If we gave this problem to a linear support vector machine, it would have no problem finding the decision boundary. It gives the maximum margin between points of different classes.  " title= "fig.1" width="200">&nbsp;&nbsp;&nbsp;&nbsp;
        <img src="images/fig2-24.png" alt="Here I've engineered the data points, so that the maximum margin decision boundary happens to be at x = 0. Now suppose we gave the linear support vector machine a harder problem, where the classes are no longer linearly separable. A simple linear decision boundary just doesn't have enough expressive power to classify all these points correctly." title= "fig.2" width="200">&nbsp;&nbsp;&nbsp;&nbsp;
        <img src="images/fig2-25.png" alt="One very powerful idea is to transform the input data from a 1-dimensional space to a 2-dimensional space. We can do this, for example, by mapping each 1-dimensional input data instance xi to a corresponding 2-dimensional ordered pair xi, xi squared, whose new second feature is the squared value of the first feature. We're not adding in any new information in the sense that all we need to obtain this new 2-dimensional version is already present in the original 1-dimensional data point. This might remind you of a similar technique that we saw when adding polynomial features to a linear regression problem earlier in the course. " title= "fig.3" width="150"><br/>
        <img src="images/fig2-26.png" alt="We can now learn a linear support vector machine in this new, 2-deminsional feature space, whose maximum margin decision boundary might look like this here to correctly classify the points. Any future 1-dimensional points for which we'd like to predict the class, we can just create the 2-deminsional transformed version and predict the class of the 2-deminsional point, using this 2-deminsional linear SVM.  " title= "fig.4" width="150">&nbsp;&nbsp;&nbsp;&nbsp;
        <img src="images/fig2-27.png" alt="If we took the inverse of the transformation we just applied to bring the data points back to our original input space, we can see that the linear decision boundary in the 2-deminsional space corresponds to the two points where a parabola crosses the x-axis. Now just so that this very important idea is clear, let's move from a 1-dimensional problem to a 2-deminsional classification problem. You can see the same powerful idea in action here." title= "fig.5" width="200">&nbsp;&nbsp;&nbsp;&nbsp;
        <img src="images/fig2-28.png" alt="Here we have two classes represented by the black and the white points. Each of which has two features, x0 and x1. The points of both classes are scattered around the origin 00 in a 2-deminsional plane. The white points form a cluster right around 00, that's completely surrounded by the black points. Again, this looks to be impossible for a linear classifier, which in 2-deminsional space is a straight line, to separate the white points from the black points with any degree of accuracy. " title= "fig.6" width="200">


+ Example of mapping a 2D classification problem to a 3D feature space to make it linearly separable

    <img src="images/fig2-29.png" alt="Here we have two classes represented by the black and the white points. Each of which has two features, x0 and x1. The points of both classes are scattered around the origin 00 in a 2-deminsional plane. The white points form a cluster right around 00, that's completely surrounded by the black points. Again, this looks to be impossible for a linear classifier, which in 2-deminsional space is a straight line, to separate the white points from the black points with any degree of accuracy. " title= "original to feature" width="220">&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="images/fig2-30.png" alt="But just as we did in the 1-dimensional case, we can map each 2-deminsional point (x0,x1) to a new 3-deminsional point by adding a third feature. Mathematically 1-(x0 squared+x1 squared), and this transformation acts to shape the points into a parabaloid around (0,0). Now the wide points since they're close to (0,0), get mapped to points with higher vertical z values, that new third feature that are close to 1. While the black points which are farther from (0,0) get mapped to points with z values that either close to 0 or even negative. With this transformation, it makes it possible to find a hyperplane. Say, z = 0.9, that now easily separates the white data points that are near z = 1 from most or all of the black data points. Finally, the decision boundary consists of the set of points in 3-deminsional space where the paraboloid intersects the maximum margin hyperplane decision boundary. " title= "original to future" width="220">&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="images/fig2-31.png" alt="This corresponds to an ellipse-like decision boundary in 2-deminsional space that separates the white points from the black points in the original input space. This idea of transforming the input data points to a new feature space where a linear classifier can be easily applied, is a very general and powerful one. " title= "future to original" width="230">

    + Transforming the data can make it much easier for a linear classifier. (fig.7)
        <a href="https://en.wikipedia.org/wiki/Kernel_method">
            <br/><img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png" alt="This idea of transforming the input data points to a new feature space where a linear classifier can be easily applied, is a very general and powerful one. There are lots of different possible transformations we could apply to data. And the different kernels available for the kernelized SVM correspond to different transformations. Here we're going to focus mainly on what's called the radial basis function kernel, which we'll abbreviate as RBF." title= "fig.7" width="350">
        </a>

+ Redial Basis Function Kernel
    + Mapping function: $K({\bf x, x^{\prime}}) = exp [-\gamma \cdot \| {\bf x} - {\bf x^{\prime}} \|^2])$
    + A __kernel__ is a similarity measure (modified dot product) between data points<br/>
    <img src="images/fig2-32.png" alt="The radial basis function kernel, the similarity between two points and the transformed feature space is an exponentially decaying function of the distance between the vector and the original input space as shown by the formula here. Using the radial basis function kernel in effect, transforms all the points inside a certain distance of the circle class to one area of the transformed feature space. And all the points in the square class outside a certain radius get moved to a different area of the feature space. The dark circles and squares represents the points that might lie along the maximum margin for a support vector machine in the transformed feature space. And also, it shows the corresponding points in the original input space. So just as we saw with the simple 1D and 2D examples earlier, the kernelized support vector machine tries to find the decision boundary with maximum margin between classes using a linear classifier in the transformed feature space not the original input space. The linear decision boundary learn feature space by linear SVM corresponds to a non-linear decision boundary In the original input space. So in this example, an ellipse like closed region in the input space. Now, one of the mathematically remarkable things about kernelized support vector machines, something referred to as the kernel trick, is that internally, the algorithm doesn't have to perform this actual transformation on the data points to the new high dimensional feature space. Instead, the kernelized SVM can compute these more complex decision boundaries just in terms of similarity calculations between pairs of points in the high dimensional space where the transformed feature representation is implicit. This similarity function which mathematically is a kind of dot product is the kernel in kernelized SVM. " title= "Redial Basis Function Kernel" width="300">

+ Applying the SVM with RBF kernel
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lCUeA/kernelized-support-vector-machines">
        <br/><img src="images/fig2-33.png" alt="Here's the result of using a support vector machine with RBF kernel, on that more complex binary classification problem we saw earlier. You can see that unlike a linear classifier, the SVM with RBF kernel finds a more complex and very effective set of decision boundaries that are very good at separating one class from the other. Note that the SVM classifier is still using a maximum margin principle to find these decision boundaries. But because of the non-linear transformation of the data, these boundaries may no longer always be equally distant from the margin edge points in the original input space. " title= "Applying the SVM with RBF kernel" width="450">
    </a>
    + Demo: Radial Basis Kernel vs Polynomial Kernel
        ```python
        from sklearn.svm import SVC
        from adspy_shared_utilities import plot_class_regions_for_classifier

        X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

        # The default SVC kernel is radial basis function (RBF)
        plot_class_regions_for_classifier(SVC().fit(X_train, y_train),
            X_train, y_train, None, None, 'Support Vector Classifier: RBF kernel')

        # Compare decision boundries with polynomial kernel, degree = 3
        plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3)
            .fit(X_train, y_train), X_train, y_train, None, None,
            'Support Vector Classifier: Polynomial kernel, degree = 3')
        ```
        <img src="images/plt2-17.png" alt="The default SVC kernel is radial basis function (RBF).  By default, the SVM will use the radial base's function, but a number of other choices are supported." title= "Support Vector Classifier: RBF kernel" width="300">
        <img src="images/plt2-18.png" alt="The polynomial kernel, using the kernel poly setting, essentially represents a future transformation similar to the earlier quadratic example. In the lecture, this future space represented in terms of futures that are polynomial combinations of the original input features, much as we saw also for linear regression. The polynomial kernel takes additional parameter degree that controls the model complexity and the computational cost of this transformation. " title= "Support Vector Classifier: Polynomial kernel, degree = 3" width="300">


+ Radial Basis Function kernel: Gamma Parameter
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/lCUeA/kernelized-support-vector-machines">
        <br/><img src="images/fig2-34.png" alt="Gamma controls how far the influence of a single trending example reaches, which in turn affects how tightly the decision boundaries end up surrounding points in the input space. Small gamma means a larger similarity radius. So that points farther apart are considered similar. Which results in more points being group together and smoother decision boundaries. On the other hand for larger values of gamma, the kernel value to K is more quickly and points have to be very close to be considered similar. This results in more complex, tightly constrained decision boundaries." title= "Radial Basis Function kernel: Gamma Parameter" width="450">
    </a>
    + Demo: The effect of the RBF gamma parameter on decision boundaries
        ```python
        # Support Vector Machine with RBF kernel: gamma parameter
        from adspy_shared_utilities import plot_class_regions_for_classifier

        X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
        fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

        for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
            clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
            title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
            plot_class_regions_for_classifier_subplot(
                clf, X_train, y_train, None, None, title, subplot)
            plt.tight_layout()
        ```
        <img src="images/plt2-19.png" alt="Small values of gamma give broader, smoother decision regions. While larger values of gamma give smaller, more complex decision regions. " title= "Support Vector Machine with RBF kernel: gamma parameter." width="600">

+ Demo: Effect of C and gamma parameters (horizontal: increasing C, vertical: increasing $\gamma$)
    ```python
    # Support Vector Machine with RBF kernel: using both C and gamma parameter 
    from sklearn.svm import SVC
    from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

    for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
        for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
            title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
            clf = SVC(kernel = 'rbf', gamma = this_gamma, C = this_C)
                .fit(X_train, y_train)
            plot_class_regions_for_classifier_subplot(
                clf, X_train, y_train, X_test, y_test, title, subplot)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    ```
    <img src="images/plt2-20.png" alt="You can set the gamma parameter when creating the SVC object to control the kernel width in this way, as shown in this code. You may recall from linear SVMs that SVMs also have a regularization parameter, C, that controls the tradeoff between satisfying the maximum margin criterion to find the simple decision boundary, and avoiding misclassification errors on the training set. The C parameter is also an important one for kernelized SVMs, and it interacts with the gamma parameter. If gamma is large, then C will have little to no effect. Well, if gamma is small, the model is much more constrained and the effective C will be similar to how it would affect a linear classifier. Typically, gamma and C are tuned together, with the optimal combination typically in an intermediate range of values. For example, gamma between 0.0001 and 10 and see between 0.1 and 100. Though the specifical optimal values will depend on your application. Kernelized SVMs are pretty sensitive to settings of gamma. The most important thing to remember when applying SVMs is that it's important to normalize the input data, so that all the features have comparable units that are on the same scale. We saw this earlier with some other learning methods like regularized regression. "  title= "Support Vector Machine with RBF kernel: using both C and gamma parameter" width="600">

+ Reminder: Using a scaler object: fit and transform methods
    ```python
    from sklearn.preprocessingimport MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled= scaler.transform(X_train)
    X_test_scaled= scaler.transform(X_test)
    clf= SVC().fit(X_train_scaled, y_train)
    accuracy = clf.score(X_test_scaled, y_test)

    # Tip: It can be more efficient to do fitting and transforming together on the training set using the `fit_transform` method.
    scaler = MinMaxScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    ```

    + Demo: Application of SVMs to a real dataset: unnormalized data -> overfitting
        ```python
        from sklearn.svm import SVC
        X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

        clf = SVC(C=10).fit(X_train, y_train)
        print('Breast cancer dataset (unnormalized features)')
        print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
            .format(clf.score(X_train, y_train)))
        print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
            .format(clf.score(X_test, y_test)))
        # Breast cancer dataset (unnormalized features)
        # Accuracy of RBF-kernel SVC on training set: 1.00
        # Accuracy of RBF-kernel SVC on test set: 0.63
        ```

    + Demo: Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling
        ```python
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = SVC(C=10).fit(X_train_scaled, y_train)
        print('Breast cancer dataset (normalized with MinMax scaling)')
        print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
            .format(clf.score(X_train_scaled, y_train)))
        print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
            .format(clf.score(X_test_scaled, y_test)))
        # Breast cancer dataset (normalized with MinMax scaling)
        # RBF-kernel SVC (with MinMax scaling) training set accuracy: 0.98
        # RBF-kernel SVC (with MinMax scaling) test set accuracy: 0.96
        ```

+ Kernelized Support Vector Machines: pros and cons
    + Pros:
        + Can perform well on a range of datasets.
        + Versatile: different kernel functions can be specified, or custom kernels can be defined for specific data types.
        + Works well for both low-and high-dimensional data.
    + Cons:
        + Efficiency (runtime speed and memory usage) decreases as training set size increases (e.g. over 50000 samples).
        + Needs careful normalization of input data and parameter tuning.
        + Does not provide direct probability estimates (but can be estimated using e.g. Platt scaling).
        + Difficult to interpret why a prediction was made.


+ Kernelized Support Vector Machines (SVC): Important parameters <br/>
    Model complexity
    + kernel: Type of kernel function to be used
        + Default = 'rbf' for radial basis function
        + Other types include 'polynomial'
    + kernel parameters
        + `gamma`($\gamma$): RBF kernel width
    + `C`: regularization parameter
    + Typically `C` and `gamma` are tuned at the same time.
    + [RBF SVM parameters](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
        + Intuitively, the `gamma` parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. The `gamma` parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.
        + The `C` parameter trades off misclassification of training examples against simplicity of the decision surface. A low `C` makes the decision surface smooth, while a high `C` aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/0F-tskyCEeeGww6XbaTymg.processed/full/360p/index.mp4?Expires=1536451200&Signature=OelcTkr3mv6FwSRwSKqNcVjTw6txXZlDVNFdvOjG~G2MJ5EGApSg3gF9skDE-XVF4NjsWJdRhp4TfZy9NDG6HFXNRoHg~1iXUcOzgEAZ3j3kA2UPTTPpYkQr-MkVTLqG6oHnCwfnoKk0c9Qf7enkEt4wCwI2OK5UtFoz0Ffrs9s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Kernelized Support Vector Machines" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Cross-Validation

+ Cross-validation
    + Uses multiple train-test splits, not just a single one
    + Each split used to train & evaluate a separate model
    + Why is this better?
        + The accuracy score of a supervised learning method can vary, depending on which samples happen to end up in the training set.
        + Using multiple train-test splits gives more stable and reliable estimates for how the classifier is likely to perform on average.
        + Results are averaged over multiple different training sets instead of relying on a single model trained on a particular training set.
    + Accuracy of k-NN classifier (k=5) on fruit data test set for different random_statevalues in train_test_split.

        | random_state | Test set accuracy |
        |--------------|-------------------|
        |  0 | 1.00 |
        |  1 | 0.93 |
        |  5 | 0.93 |
        |  7 | 0.67 |
        | 10 | 0.87 |

+ Cross-validation Example (5-fold)
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Vm0Ie/cross-validation">
        <br/><img src="images/fig2-35.png" alt="text" title= "Cross-validation Example (5-fold)" width="350">
    </a>
    + Demo: Example based on k-NN classifier with fruit dataset (2 features)
        ```python
        from sklearn.model_selection import cross_val_score

        clf = KNeighborsClassifier(n_neighbors = 5)
        X = X_fruits_2d.as_matrix()
        y = y_fruits_2d.as_matrix()
        cv_scores = cross_val_score(clf, X, y)

        print('Cross-validation scores (3-fold):', cv_scores)
        print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))
        # Cross-validation scores (3-fold): [ 0.77  0.74  0.83]
        # Mean cross-validation score (3-fold): 0.781
        ```
    + A note on performing cross-validation for more advanced scenarios.<br/> In some cases (e.g. when feature values have very different ranges), we've seen the need to scale or normalize the training and test sets before use with a classifier. The proper way to do cross-validation when you need to scale the data is not to scale the entire dataset with a single transform, since this will indirectly leak information into the training data about the whole dataset, including the test data (see the lecture on data leakage later in the course). Instead, scaling/normalizing must be computed and applied for each cross-validation fold separately. To do this, the easiest way in scikit-learn is to use pipelines. While these are beyond the scope of this course, further information is available in the scikit-learn documentation here: <br/> http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

+ Stratified Cross-validation
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Vm0Ie/cross-validation">
        <br/><img src="images/fig2-36.png" alt="In the default cross-validation set up, to use for example five folds, the first 20% of the records are used as the first fold, the next 20% for the second fold, and so on. One problem with this is that the data might have been created in such a way that the records are sorted or at least show some bias in the ordering by class label." title= "Stratified Cross-validation" width="325">
    </a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Vm0Ie/cross-validation">
        <img src="images/fig2-37.png" alt="When scikit-learn doing cross-validation for a classification task, it actually does instead what's called 'Stratified K-fold Cross-validation'. The Stratified Cross-validation means that when splitting the data, the proportions of classes in each fold are made as close as possible to the actual proportions of the classes in the overall data set as shown here. " title= "Stratified Cross-validation" width="310">
    </a>
    + Stratified folds each contain a proportion of classes that matches the overall dataset. Now, all classes will be fairly represented in the test set.

+ Leave-one-out cross-validation (with N samples in dataset)
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Vm0Ie/cross-validation">
        <br/><img src="images/fig2-38.png" alt="For regression, scikit-learn uses regular k-fold cross-validation since the concept of preserving class proportions isn't something that's really relevant for everyday regression problems. At one extreme we can do something called 'Leave-one-out cross-validation', which is just k-fold cross-validation, with K sets to the number of data samples in the data set. In other words, each fold consists of a single sample as the test set and the rest of the data as the training set. Of course this uses even more computation, but for small data sets in particular, it can provide improved proved estimates because it gives the maximum possible amount of training data to a model, and that may help the performance of the model when the training sets are small." title= "Leave-one-out cross-validation (with N samples in dataset)" width="350">
    </a>

+ Validation curves show sensitivity to changes in an important parameter
    ```python
    # Validation curve example
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    param_range = np.logspace(-3, 3, 4)
    train_scores, test_scores = validation_curve(
        SVC(), X, y, param_name='gamma', param_range=param_range, cv=3)

    print(train_scores)
    # [[ 0.49  0.42  0.41]
    #  [ 0.84  0.72  0.76]
    #  [ 0.92  0.9   0.93]
    #  [ 1.    1.    0.98]]
    print(test_scores)
    # [[ 0.45  0.32  0.33]
    #  [ 0.82  0.68  0.61]
    #  [ 0.41  0.84  0.67]
    #  [ 0.36  0.21  0.39]]
    ```
    + One row per parameter sweep value, One column per CV fold.

+ Validation Curve Example
    + The validation curve shows the mean cross-validation accuracy (solid lines) for training (orange) and test (blue) sets as a function of the SVM parameter (gamma). It also shows the variation around the mean (shaded region) as computed from k-fold cross-validation scores.
    + Demo: scikit-learn validation_plot example
        ```python
        #  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
        plt.figure()

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title('Validation Curve with SVM')
        plt.xlabel('$\gamma$ (gamma)')
        plt.ylabel('Score')
        plt.ylim(0.0, 1.1)
        lw = 2

        plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=lw)
        plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
                    color='navy', lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=lw)

        plt.legend(loc='best')
        plt.show()
        ```
        <img src="images/plt2-21.png" alt="You can plot these results from validation curve as shown here to get an idea of how sensitive the performance of the model is to changes in the given parameter. The x axis corresponds to values of the parameter and the y axis gives the evaluation score, for example the accuracy of the classifier. Finally as a reminder, cross-validation is used to evaluate the model and not learn or tune a new model. " title= "Validation Curve Example" width="350">


### LEcture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/uHP16UGREeeR4AqenwJvyA.processed/full/360p/index.mp4?Expires=1536451200&Signature=JgcWFHJDdoYJKqb8s-G3Aclm4012Y~b0Xoqoren5qFhHLIOsq8u9W2f8Tf7cdsV9H5liu9RinW2Q-dRRUXeRJLXHo0Dgr6Fb-Ad6CoXlrd7UDwrSj6TOpZA5mcfQVNvWV8fFv-bXGLqx7FlJVcvkkkcXxrUJtWj5ieestVBVlzY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Cross-Validation" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Decision Trees

+ Decision Trees
    + popular supervised learning method
    + learn a series of explicit if then rules on feature values that result in a decision that predicts the target value

+ Decision Tree Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Zj96A/decision-trees">
        <br/><img src="images/fig2-39.png" alt="If we think of the property of being alive as a binary feature of an object and the property of having orange fur with black stripes as another feature, we can say that the is a live feature is more informative at an early stage of guessing and thus would appear higher in our tree of questions." title= "Specific question to narrow down the selection" height="150">&nbsp;&nbsp;&nbsp;&nbsp;
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Zj96A/decision-trees">
        <img src="images/fig2-40.png" alt="Form these questions into a tree with a node representing one question and the yes or no possible answers as the left and right branches from that node that connect the node to the next level of the tree. One question being answered at each level. At the bottom of the tree are nodes called leaf nodes that represent actual objects as the possible answers. For any object there's a path from the root of the tree to that object that is determined by the answers to the specific yes or no questions at each level." title= "Tree structure of Decision Tree - Root node & Leaf node" height="150">
    </a>

+ The [Iris Daatset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
    + The dataset contains a set of 150 records under 5 attributes - Petal Length , Petal Width , Sepal Length , Sepal width and Class.
    + Species: Iris setosa, Iris versicolor, Iris virginica
    + 50 examples/species

+ Decision Tree Splits
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Zj96A/decision-trees">
        <br/><img src="images/fig2-41.png" alt="The goal when building a decision tree is to find the sequence of questions that has the best accuracy at classifying the data in the fewest steps. Looking at a decision tree, each decision splits the data into two branches based on some feature value being above or below a threshold. Using that split leaves a pool of instances that are a combination of virginica and versicolor that still need to be distinguished further. So we can further improve the accuracy of the classification by continuing this process of finding the best split for the remaining subsets with the iris dataset. " title= "caption" height="350">
    </a>

+ Informativenessof Splits
    + The _value_ list gives the number of samples of each class that end up at this leaf node during training.
    + The iris dataset has 3 classes, so there are three counts.
    + Sample = 37, value = [37, 0, 0], class = setosa:
        + This leaf has 37 setosa samples, zero versicolor, and zero virginicasamples.
    + Sample = 36, value = [0, 33, 3], class = versicolor
        + This leaf has 0 setosa, 33 versicolor, and 3 virginicasamples.
    + Sample = 39, value = [0, 1, 38], class = virginica
        + This leaf has 0 setosa, 1 versicolor, and 38 virginicasamples.
        <!---
        <img src="images/fig2-42.png" alt="There are a number of mathematical ways to compute the best split. One criterion that's widely used for decision trees is called information game, for example. So to build the decision tree, the decision tree building algorithm starts by finding the feature that leads to the most informative split. For any given split of the data on a particular feature value, even for the best split it's likely that some examples will still be incorrectly classified or need further splitting. " title= "First node decision" width="200">
        <img src="images/fig2-43.png" alt="Trees whose leaf nodes each have all the same target value are called pure, as opposed to mixed where the leaf nodes are allowed to contain at least some mixture of the classes. To predict the class of a new instance given its feature measurements, using the decision tree we simply start at the root of the decision tree and take the decision at each level based on the appropriate feature measurement until we get to a leafnode." title= "Second level decision node" width="200">
        --->
        <img src="images/fig2-44.png" alt="The prediction is then just the majority class of the instances in that leafnode. So for the iris data for example, a flower that has a petal length of three centimeters, a petal width of two centimeters and a sepal width of two centimeters would end up at this leafnode, whose instances are all of the virginica class. So the prediction would be virginica. " title= "Complete decision tree for Iris dataset" width="450">
    + Demo: Decision Trees
        ```python
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeClassifier
        from adspy_shared_utilities import plot_decision_tree
        from sklearn.model_selection import train_test_split

        iris = load_iris()

        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)
        clf = DecisionTreeClassifier().fit(X_train, y_train)

        print('Accuracy of Decision Tree classifier on training set: {:.2f}'
            .format(clf.score(X_train, y_train)))
        print('Accuracy of Decision Tree classifier on test set: {:.2f}'
            .format(clf.score(X_test, y_test)))
        # Accuracy of Decision Tree classifier on training set: 1.00
        # Accuracy of Decision Tree classifier on test set: 0.97
        ```

+ Decision tree for regression
    + using the same process of testing the future values at each node and predicting the target value based on the contents of the leafnode
    + the leafnode prediction would be the mean value of the target values for the training points in that leaf

+ Controlling the Model Complexity of Decision Trees
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Zj96A/decision-trees">
        <br/><img src="images/fig2-45.png" alt="Typically such trees are overly complex and essentially memorized the training data. So when building decision trees, we need to use some additional strategy to prevent this overfitting. One strategy to prevent overfitting is to prevent the tree from becoming really detailed and complex by stopping its growth early. This is called pre-pruning. Another strategy is to build a complete tree with pure leaves but then to prune back the tree into a simpler form. This is called post-pruning or sometimes just pruning. The decision tree implementation and scikit-learn only implements pre-pruning. We can control tree complexity via pruning by limiting either the maximum depth of the tree using the max depth parameter or the maximum number of leafnodes using the max leafnodes parameter. We could also set a threshold on the minimum number of instances that must be in a node to consider splitting it. And this would be using the min samples leaf parameter we can see the effect of pre-pruning by setting max depth to three on the iris dataset. " title= "Controlling the Model Complexity of Decision Trees - prunning" height="250">
    </a>
    + Demo:Setting max decision tree depth to help avoid overfitting
        ```python
        clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

        print('Accuracy of Decision Tree classifier on training set: {:.2f}'
             .format(clf2.score(X_train, y_train)))
        print('Accuracy of Decision Tree classifier on test set: {:.2f}'
             .format(clf2.score(X_test, y_test)))
        # Accuracy of Decision Tree classifier on training set: 0.98
        # Accuracy of Decision Tree classifier on test set: 0.92
        ```

+ Visualizing Decision Trees
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/Zj96A/decision-trees">
        <br/><img src="images/fig2-46.png" alt="Plot decision tree, that takes the classifier object, the feature names, and the class names as input. And uses the graphics library to visualize the tree. It works by calling the export graph its function in the scikit-learn tree module to create a dot file which is a text file description of the tree, and then using the graphics library to visualize the dot file creating an image. Here's the resulting plot for the unpruned Iris dataset tree. The first line in a node indicates the decision rule being applied for that node. The second line indicates the total number of data instances for that node. The third line shows the class distribution among those instances. And the fourth line shows the majority class of that nodes data instances. " title= "caption" height="250">
    </a>
    + See: `plot_decision_tree()` function in `adspy_shared_utilities.py` code
    + Demo: 
        ```python
        # Visualizing decision trees
        plot_decision_tree(clf, iris.feature_names, iris.target_names)

        # #### Pre-pruned version (max_depth = 3)
        plot_decision_tree(clf2, iris.feature_names, iris.target_names)
        ```

+ Feature Importance: How important is a feature to overall prediction accuracy?
    + A number between 0 and 1 assigned to each feature.
    + Feature importance of 0 --> the feature was not used in prediction.
    + Feature importance of 1 --> the feature predicts the target perfectly.
    + All feature importancesare normalized to sum to 1.


+ Feature Importance Chart
    </a>
    + See: `plot_feature_importances()` function in `adspy_shared_utilities.py` code
    + Demo: # Feature importance
        ```python
        from adspy_shared_utilities import plot_feature_importances

        plt.figure(figsize=(10,4), dpi=80)
        plot_feature_importances(clf, iris.feature_names)
        plt.show()

        print('Feature importances: {}'.format(clf.feature_importances_))

        from sklearn.tree import DecisionTreeClassifier
        from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
        fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

        pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        tree_max_depth = 4

        for pair, axis in zip(pair_list, subaxes):
            X = X_train[:, pair]
            y = y_train
            
            clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
            title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
            plot_class_regions_for_classifier_subplot(
                clf, X, y, None, None, title, axis, iris.target_names)
            
            axis.set_xlabel(iris.feature_names[pair[0]])
            axis.set_ylabel(iris.feature_names[pair[1]])
            
        plt.tight_layout()
        plt.show()
        ```
    <img src="images/plt2-24.png" alt="In scikit-learn, feature importance values are stored as a list in an estimated property called feature_importances_. And note the underscore at the end of the name which indicates it's a property of the object that's set as a result of fitting the model and not say as a user defined property. The shared utilities python file contains a function called plot feature importances, that you can import and use to visualize future importance. It plots a horizontal bar chart with the features listed along the y axis by name and feature importance along the x axis. " title= "Feature importances" height="200"> <br/>
    <img src="images/plt2-25.png" alt="Here's the feature importance chart for the iris decision tree and this example for this particular train/test split of the iris data set. The pedal length feature easily has the largest feature importance weight. We can confirm this by looking at the decision tree that this is indeed corresponds to that features position at the top of the decision tree, showing that this first level just using the petal length feature does a good job splitting the turning data into separate classes. Note that if a feature has a low feature importance value, that doesn't necessarily mean that the feature is not important for prediction. It simply means that the particular feature wasn't chosen at an early level of the tree and this could be because the future may be identical or highly correlated with another informative feature and so doesn't provide any new additional signal for prediction. Feature importance values don't tell us which specific classes a feature might be especially predictive for, and they also don't indicate more complex relationships between features that may influence prediction. " title= "Feature importances for cross valiadtions" height="200">

+ Demo: # Decision Trees on a real-world dataset
    ```python
    from sklearn.tree import DecisionTreeClassifier
    from adspy_shared_utilities import plot_decision_tree
    from adspy_shared_utilities import plot_feature_importances

    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

    clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8,
        random_state = 0).fit(X_train, y_train)

    plot_decision_tree(clf, cancer.feature_names, cancer.target_names)

    print('Breast cancer dataset: decision tree')
    print('Accuracy of DT classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of DT classifier on test set: {:.2f}'
        .format(clf.score(X_test, y_test)))

    plt.figure(figsize=(10,6),dpi=80)
    plot_feature_importances(clf, cancer.feature_names)
    plt.tight_layout()

    plt.show()
    ```
    <img src="images/plt2-25.png" alt="Finally let's apply decision trees to the breast cancer data set that we've been using across multiple supervised learning algorithms. Here we're controlling them all the complexity by setting the max depth and min samples leaf parameters. And we've included the visualization of the resulting tree here followed by a feature importance plot. As an exercise, try removing these two parameters to just use the default settings to see the effect on test set accuracy and the increase in overfitting that occurs. For this training set, the mean concave points feature gives the most informative initial split, followed by the worst area parameter." title="Decission Tree - Breast Cancer" height="250">


+ Decision Trees: Pros and Cons
    + Pros:
        + Easily visualized and interpreted.
        + No feature normalization or scaling typically needed.
        + Work well with datasets using a mixture of feature types (continuous, categorical, binary)
    + Cons:
        + Even after tuning, decision trees can often still overfit.
        + Usually need an ensemble of trees for better generalization performance.


+ Decision Trees: DecisionTreeClassifierKey Parameters
    + `max_depth`: controls maximum depth (number of split points). Most common way to reduce tree complexity and overfitting.
    + `min_samples_leaf`: threshold for the minimum # of data instances a leaf can have to avoid further splitting.
    + `max_leaf_nodes`: limits total number of leaves in the tree.
    + In practice, adjusting only one of these (e.g. `max_depth`) is enough to reduce overfitting.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/kIeJZUG_EeedLxJ0mGjb9g.processed/full/360p/index.mp4?Expires=1536451200&Signature=LxQjbCiWUOn2-o4lzUBh~JjSBUcEL5FfUG9aoDwpx-rzOjsRAqBhQTQj61GjHIVNDil5xYU2QUu1m9h-KT4F7ivg3GUuE69RfcR8k6dt8QF3~3jD4ndauZM9ppD~6afpHl3gFUaA-KBLC78SB1kEZzWREEnBWDmlD0~Yumjx46w_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Decision Trees" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## A Few Useful Things to Know about Machine Learning

This article by Prof. Pedro Domingos provides a bit more background and discussion of the essential concepts in machine learning covered in Modules 1 and 2. It covers topics such as overfitting, the role of data vs model vs features, and the use of ensembles, where many models are learned instead of just one (something we look at with random forests).

Domingos, P. (2012). [A few useful things to know about machine learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf). Communications of the ACM, 55(10), 78. doi:10.1145/2347736.2347755

## Ed Yong: Genetic Test for Autism Refuted (optional)

This article by Ed Yong in The Scientist is included because it describes a real-world example of a prediction problem in the health/medical sciences domain - training a classifier to predict risk of autism spectrum disorder (ASD) based on genetic markers - as well as including discussion of potential overfitting of the classifier (by training and testing on the same data) as one possible issue, among other factors, by researchers attempting to replicate the study.

[Genetic Test for Autism Refuted](http://www.the-scientist.com/?articles.view/articleNo/38030/title/Genetic-Test-for-Autism-Refuted/)


## Quiz: Module 2 Quiz

Q1. After training a ridge regression model, you find that the training and test set accuracies are 0.98 and 0.54 respectively. Which of the following would be the best choice for the next ridge regression model you train?

    a. You are overfitting, the next model trained should have a lower value for alpha
    b. You are overfitting, the next model trained should have a higher value for alpha
    c. You are underfitting, the next model trained should have a lower value for alpha
    d. You are underfitting, the next model trained should have a higher value for alpha

    Ans: b


Q2. After training a Radial Basis Function (RBF) kernel SVM, you decide to increase the influence of each training point and to simplify the decision surface. Which of the following would be the best choice for the next RBF SVM you train?

    a. Decrease C and gamma
    b. Increase C and gamma
    c. Increase C, decrease gamma
    d. Decrease C, increase gamma

    Ans: a, xc, xd, xb
    The trick here is to simplify the decision surface. Remember when c is low, the SVM will allow more misclassified pts.


Q3. Which of the following is an example of multiclass classification? (Select all that apply)

    a. Classify a set of fruits as apples, oranges, bananas, or lemons
    b. Predict whether an article is relevant to one or more topics (e.g. sports, politics, finance, science)
    c. Predicting both the rating and profit of soon to be released movie
    d. Classify a voice recording as an authorized user or not an authorized user.

    Ans: a, xabc
    Normally, multi stands for >=3. For two labels, it's called binary classification in most situations.


Q4. Looking at the plot below which shows accuracy scores for different values of a regularization parameter lambda, what value of lambda is the best choice for generalization?

<img src="images/fig2-q1.png" alt="text" title= "caption" width="250">

    Ans: 10


Q5. Suppose you are interested in finding a parsimonious model (the model that accomplishes the desired level of prediction with as few predictor variables as possible) to predict housing prices. Which of the following would be the best choice?

    a. Lasso Regression
    b. Logistic Regression
    c. Ridge Regression
    d. Ordinary Least Squares Regression

    Ans: a


Q6. Match the plots of SVM margins below to the values of the C parameter that correspond to them.

<img src="images/fig2-q2.png" alt="text" title= "caption" width="450">

    a. 1, 0.1, 10
    b. 0.1, 1, 10
    c. 10, 0.1, 1
    d. 10, 1, 0.1


    Ans: b


Use Figures A and B below to answer questions 7, 8, 9, and 10.

<img src="images/fig2-q3.png" alt="text" title= "Figure A" width="300">
<img src="images/fig2-q4.png" alt="text" title= "Figure B" width="300">

Q7. Looking at the two figures (Figure A, Figure B), determine which linear model each figure corresponds to:

    a. Figure A: Ridge Regression, Figure B: Lasso Regression
    b. Figure A: Lasso Regression, Figure B: Ridge Regression
    c. Figure A: Ordinary Least Squares Regression, Figure B: Ridge Regression
    d. Figure A: Ridge Regression, Figure B: Ordinary Least Squares Regression
    e. Figure A: Ordinary Least Squares Regression, Figure B: Lasso Regression
    f. Figure A: Lasso Regression, Figure B: Ordinary Least Squares Regression

    Ans: a


Q8. Looking at Figure A and B, what is a value of alpha that optimizes the $R^2$ score for the Ridge Model?

    Ans: 3


Q9. Looking at Figure A and B, what is a value of alpha that optimizes the $R^2$ score for the Lasso Model?

    Ans: 11, x20, x110


Q10. When running a LinearRegression() model with default parameters on the same data that generated Figures A and B the output coefficients are:

    Coef 0      -19.5
    Coef 1      48.8
    Coef 2      9.7
    Coef 3      24.6
    Coef 4      13.2
    Coef 5      5.1

For what value of Coef 3 is $R^2$ score maximized for the Ridge Model?


    Ans: 0, x40, x35
    1. you need to identify which figure correspond to the Lasso model (Q7)
    2. Find the alpha that maximize the R2 of the Lasso model (Q8/9)
    3. Identify the lines at default value (1.0): Coef 1 > Coef 3 > Coef 4 > Coef 2 > Coef 5 > Coef 0 (Orange, Red, Green, Purple, Brown, Blue)
    3. find the value of Coef 3 at the alpha that maximized the R2 of the Lasso model. (Red @ 11)



Q11. Which of the following is true of cross-validation? (Select all that apply)

    a. Increases generalization ability and reduces computational complexity
    b. Removes need for training and test sets
    c. Helps prevent knowledge about the test set from leaking into the model
    d. Increases generalization ability and computational complexity
    e. Fits multiple models on different splits of the data

    Ans: cde, xace-0.6, xac-0.4
    There is no pm functionality. You could always think about that and resubmit the quiz in a few hours.



## Classifier Visualization Playspace




<a href="url">
    <br/><img src="url" alt="text" title= "caption" width="350">
</a>

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

