# Assignment 2

## Notebook

+ [Launching web page](https://www.coursera.org/learn/python-machine-learning/notebook/BAqef/assignment-2)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=v4kpnNfCSRKJKZzXwlkSaw&next=%2Fnotebooks%2FAssignment%25202.ipynb)
+ [Local Notebook](notebooks/Assignment02.ipynb)
+ [Local Python Code](notebooks/Assignment02.py)

## Useful Links

### [Q1 and Q2 - answer not matching grader answer](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/2/threads/ahAIGlc4Eee7qw4C9eIhHA)

The data is already split to train/ test, you don't need to do that in the function. so you need to remove the steps that contains data splitting

answer_one is a function that uses X_train, y_train which are defined in the first cell of the notebook

here is how I would approach this question:

res = np.zeros(4,100)

for each degree in [1,3,6,9]

1. create polynomial features object for that degree, lets call it poly
2. reshape X_train (original shape is (11,) needs to be (11,1)) and convert it to polynomial features using poly, let us call the result X_train_poly (and its shape= (11,))
3. define and fit a LinearRegression model lr using X_train_poly and y_train
4. create test_x = np.linspace(0,10,100).reshape(-1,1)(shape is (100,1)), transform test_x to polynomial features using poly.tranform. let's call the result test_x_poly (shape =(100,degree+1)
5. apply lr predictto test_x_poly, let's call the result y_predict.
6. make sure y_predict is a 1 dimensional array, of shape (100,) you can do so using y_predict.flatten() and store in res using
    ```python
    res[i,:] = y_predict.flatten() 
    ```

Checking code is done only when a grader problem is suspected :(. Following the steps above should generate the correct answer. and plot the results to make sure you're on the right track

Edit:

P.S. The length of the data does not change as the degree changes, the number of rows is the number of items, the number of columns is the number of features, so X_train_poly will always have the same number of rows as y_train, only the number of columns will change.

P.S.S.

The following table may help you digest the information above
<br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/MFDYTlf4Eee7Tw73tY0PgA_9e429450c6baa32583813458dd0bc7b7_Screen-Shot-2017-06-23-at-10.41.33.png?expiry=1536624000000&hmac=bwJE24K-KYps8h6MIrWIsRf2tTLU_AbaHnaRI0If2MI" alt="text" title= "caption" width="450">


## [Assignment 2, Q 6, which dataset to use?](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/2/threads/sydl3EcyEeeqVwpT36CBzg)

For performance reasons, Q6 is based on a subset of the whole dataset. The assignment was updated (version 1.2) to create new variables X_subset and y_subset along with a new explanation that these should be used as input, and with a description of how to call validation_curve that matches what's in the lecture (and these happen to be defined as X_test2, y_test2 in the setup block so that nothing about the results changes.).

The update version of the assignment can be found in the readonly folder, you may need to logout /login of coursera to be able to view it.

You can access the readonly folder by clicking File->Open from the menu of the notebook, this will load in the home folder page, clicking on the read only icon you'll be able to browse to the note book you're interested in

<a href="https://www.coursera.org/learn/python-machine-learning/discussions/weeks/2/threads/sydl3EcyEeeqVwpT36CBzg">
    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/WOtPAUe4Eeem0wpMSN0I3g_ae7d4eb7186a3595b56b0e32b3a9833b_readonly-copy.png?expiry=1536710400000&hmac=90bp1gcHWLh6YnsoTe_Hy2ru1kH9O9WbfcTe-ffp25o" alt="Web Notebook Use instruction" title= "Home folder screenshot" height="200">
</a>

alternatively, if you change the name of of the notebook by clicking on it's name and entering a different name, logging out/in or restarting server would automatically retrieve a fresh/ most up-to-date copy of the assignment. More information in [Resources/ Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa)

meanwhile here is a screen shot of the update

<a href="https://www.coursera.org/learn/python-machine-learning/discussions/weeks/2/threads/sydl3EcyEeeqVwpT36CBzg">
    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/KXAoX0e4EeeJ9Aq7pT6nQA_511d80966def1a3abf13377fca586bba_Screen-Shot-2017-06-02-at-18.22.55.png?expiry=1536710400000&hmac=C11ukZo-XsrTwhL7OINFV50ypu4q0Y8u7Dvtk03XWls" alt="Question Revised" title= "Question Revised" height="200">
</a>


## Solution

### Regression

```python
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);

# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
# part1_scatter()

# import matplotlib.pyplot as plt
# plt.savefig('../images/asgn2-1.png')
```

+ Question 1: Write a function that fits a polynomial LinearRegression model on the *training data* `X_train` for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. `np.linspace(0,10,100)`) and store this in a numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.

    ```python
    def answer_one(debug=False):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        rlt = np.zeros((4, 100))

        for i, deg in enumerate([1, 3, 6, 9]):

            poly = PolynomialFeatures(degree=deg)

            X_poly = poly.fit_transform(X_train.reshape(-1,1))
            X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
            linreg = LinearRegression().fit(X_poly, y_train)

            if debug:
                print("\nDegree: {:02d}".format(deg))
                print("linear model coeff (w): {}".format(linreg.coef_))
                print("linear model intercept (b): {}".format(linreg.intercept_))
                print("R-squared score (train): {}".format(linreg.score(X_poly, y_train)))
                print("R-squared scire (test): {})".format(linreg.score(X_test_poly, y_test)))

            result = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)));
            if debug: 
                print("result = \n{}".format(result))

            rlt[i, :] = result

        return rlt

    answer_one(debug=False)
    # array([[  2.53040195e-01,   2.69201547e-01,   2.85362899e-01,
    #           3.01524251e-01,   3.17685603e-01,   3.33846955e-01,
    #           3.50008306e-01,   3.66169658e-01,   3.82331010e-01,
    #           3.98492362e-01,   4.14653714e-01,   4.30815066e-01,
    #           4.46976417e-01,   4.63137769e-01,   4.79299121e-01,
    #  ...

    # feel free to use the function plot_one() to replicate the figure 
    # from the prompt once you have completed question one
    def plot_one(degree_predictions):
        import matplotlib.pyplot as plt
        %matplotlib notebook
        plt.figure(figsize=(10,5))
        plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
        plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
        for i,degree in enumerate([1,3,6,9]):
            plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, 
                    lw=2, label='degree={}'.format(degree))
        plt.ylim(-1,2.5)
        plt.legend(loc=4)

    # plot_one(answer_one())

    # plt.savefig('../images/asgn2-2.png')
    ```


+ Question 2: Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9. For each model compute the  R2R2 (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.

    ```python
    def answer_two(debug=False):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics.regression import r2_score

        r2_train = np.zeros(10)
        r2_test = np.zeros(10)

        for deg in range(10):

            poly = PolynomialFeatures(degree=deg)

            X_train_poly = poly.fit_transform(X_train.reshape(-1,1))
            X_test_poly  = poly.fit_transform(X_test.reshape(-1, 1))

            linreg = LinearRegression().fit(X_train_poly, y_train)

            r2_train[deg] = linreg.score(X_train_poly, y_train)
            r2_test[deg]  = linreg.score(X_test_poly, y_test)

        if debug:
            print("\nR2 Train: \n{}".format(r2_train))
            print("R2 Test: \n {}".format(r2_test))

        return (r2_train, r2_test)

    answer_two(debug=True)
    # (array([ 0.        ,  0.42924578,  0.4510998 ,  0.58719954,  0.91941945,
    #          0.97578641,  0.99018233,  0.99352509,  0.99637545,  0.99803706]),
    #  array([-0.47808642, -0.45237104, -0.06856984,  0.00533105,  0.73004943,
    #          0.87708301,  0.9214094 ,  0.92021504,  0.6324795 , -0.64524602]))
    ```

+ Question 3: Based on the $R^2$ scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset?

    ```python
    def answer_three(debug=False):

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics.regression import r2_score

        rlt = np.zeros((10, 2))
        for deg in np.arange(10):

            poly = PolynomialFeatures(degree=deg)

            X_poly = poly.fit_transform(X_train.reshape(-1,1))
            X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
            linreg = LinearRegression().fit(X_poly, y_train)

            result = [linreg.score(X_poly, y_train), linreg.score(X_test_poly, y_test)]          
            rlt[deg, :] = result

        good_value = rlt[:, 1].max()
        good_idx = np.where(rlt == good_value)[0][0]

        if debug:
            print("Result= \n{}".format(rlt))
            print("Goodfit value: {}".format(good_value))
            print("Index of goodfit: {}".format(good_idx))    

        return (0, 9, good_idx)
        
    answer_three(debug=False)
    # (0, 9, 6)
    ```

+ Question 4: Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.

    ```python
    def answer_four():
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Lasso, LinearRegression
        from sklearn.metrics.regression import r2_score

        poly = PolynomialFeatures(degree=12)

        X_train_poly = poly.fit_transform(X_train.reshape(-1,1))
        X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))

        # Non-regularized Linear Regression Model
        linreg = LinearRegression().fit(X_train_poly, y_train)

        # Regularized Lasso Regression Model
        linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_train_poly, y_train)

        return (linreg.score(X_test_poly, y_test), linlasso.score(X_test_poly, y_test))

    answer_four()
    # (-4.3119651811521678, 0.8406625614750356)
    ```

### Classification

Here's an application of machine learning that could save your life! For this section of the assignment we will be working with the [UCI Mushroom Data Set](http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io) stored in `mushrooms.csv`. The data will be used to train a model to predict whether or not a mushroom is poisonous. The following attributes are provided:

*Attribute Information:*

1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
4. bruises?: bruises=t, no=f 
5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
6. gill-attachment: attached=a, descending=d, free=f, notched=n 
7. gill-spacing: close=c, crowded=w, distant=d 
8. gill-size: broad=b, narrow=n 
9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
10. stalk-shape: enlarging=e, tapering=t 
11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
16. veil-type: partial=p, universal=u 
17. veil-color: brown=n, orange=o, white=w, yellow=y 
18. ring-number: none=n, one=o, two=t 
19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

<br>

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2
```

+ Question 5: Using X_train2 and y_train2 from the preceding cell, train a DecisionTreeClassifier with default parameters and random_state=0. What are the 5 most important features found by the decision tree?

    ```python
    def answer_five(debug=False):
        from sklearn.tree import DecisionTreeClassifier

        # Your code here
        tree_clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)

        feature_names = []

        for index, importance in enumerate(tree_clf.feature_importances_):
            # Add importance so we can further order this list, and add feature name with index
            feature_names.append([importance, X_train2.columns[index]])

        if debug:
            print("Feature importance and name - unsorted: \n{}".format(feature_names))

        # sort importances in decnding ordor
        feature_names.sort(reverse=True)

        if debug:
            print("\nFeature importance and name - sorted: \n{}".format(feature_names))

        top5_features =[]
        for idx in range(5):
            top5_features.append(feature_names[idx][1])

        return top5_features # Your answer here

    answer_five(debug=False)
    # ['odor_n', 'stalk-root_c', 'stalk-root_r', 'spore-print-color_r', 'odor_l']
    ```

+ Question 6: For this question, we're going to use the `validation_curve` function in `sklearn.model_selection` to determine training and test scores for a Support Vector Classifier (`SVC`) with varying parameter values.  Recall that the validation_curve function, in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal train-test splits to compute results.

    ```python
    def answer_six(debug=False):
        from sklearn.svm import SVC
        from sklearn.model_selection import validation_curve

        rlt= np.zeros((6, 3))

        # Your code here
        # initial SVC with default, kernel='rbf', C=1
        svc = SVC(kernel='rbf', C=1, random_state=0)

        gamma = np.logspace(-4, 1, 6)

        train_scores, test_scores = validation_curve(
            svc, X_subset, y_subset, 
            param_name = 'gamma',
            param_range = gamma,
            scoring = 'accuracy'
        )
        if debug:
            print("train scores = \n{}".format(train_scores))
            print("test scores = \n{}".format(test_scores))

        return (train_scores.mean(axis=1), test_scores.mean(axis=1))

    answer_six(debug=False)
    # (array([ 0.56647847,  0.93155951,  0.99039881,  1.        ,  1.        ,  1.        ]),
    #  array([ 0.56768547,  0.92959558,  0.98965952,  1.        ,  0.99507994,
    #          0.52240279]))

    def part2_line():
        import matplotlib.pyplot as plt
        %matplotlib notebook

        train_scores, test_scores = answer_six()

        plt.figure()
        plt.semilogx(np.logspace(-4, 1, 6), train_scores, label='training scores')
        plt.semilogx(np.logspace(-4, 1, 6), test_scores, label='test scores')
        plt.legend(loc=2);

        plt.show()

    # part2_line()
    ```

+ Question 7: Based on the scores from question 6, what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with good generalization performance on this dataset (high accuracy on both training and test set)? 

    ```python
    def answer_seven():

        # Your code here
        # based oion the figure ploted in the last cell

        return (0.0001, 10, 0.1) # Return your answer
    ```

