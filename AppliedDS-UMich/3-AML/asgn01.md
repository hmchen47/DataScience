# Assignment 1

## Notebook Links

+ [Launching Web Page](https://www.coursera.org/learn/python-machine-learning/notebook/oxndk/assignment-1)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=8yJcS-rfTAeiXEvq3_wHmg&next=%2Fnotebooks%2FAssignment%25201.ipynb)
+ [Local Notebook](notebooks/Assignment01.ipynb)

## Useful Info Links

### [Assignment 1 FAQs](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/1/threads/KG7q2GIhEeeg5QoVkEgmCg)

+ Q1 Basic Check script

    ```python
    #test 
    def test_one():
        columns =cancer.feature_names.tolist() +['target']
        index =pd.RangeIndex(start=0, stop=569, step=1).tolist()
        df = answer_one()
        res = 'Type Test: '
        res += ['Failed\n','Passed\n'][type(df)==pd.DataFrame]
        
        res += 'dtypes Test: '
        res += ['Failed; all columns should have float64(last column can be int64) \n',
                'Passed\n'][all(df[df.columns[:-1]].dtypes=='float64') and df[df.columns[-1]].dtype in ['float64','int64']]
        
        res += 'df shape Test: '
        res += ['Failed\n','Passed\n'][df.shape==(569,31)]

        res += 'Columns Names Test: '
        res +=  ['Failed\n','Passed\n'][all(df.columns == columns)]

        res += 'Index Test: '
        res +=  ['Failed\n','Passed\n'][df.index.tolist()==index]
        
        res += 'Values test: '
        res +=  ['Failed\n','Passed\n'][(df[df.columns[:-1]].values==cancer.data).all().all()]
        
        res += 'target test: '
        try:
            res +=  ['Failed\n','Passed\n'][(df[df.columns[-1]].values*1.0==cancer.target*1.0).all()]
        except:
            res += 'Falied: target must be numercial\n'
        return res
    print(test_one())
    ```

+ the first step will be to look at the grader output, you can view the grader output in your "My submission page" by clicking on the latest submission and clicking "show grader output"

    scrolling down the output you will be able to see the result of grading each question. or an error message

    Check the error message. the notebook is graded as script, i.e. all the cells in the notebook need to run without error for the grading to take place.
    
    to find the issue, go to your notebook and click Kernel->Restart and Run all. check the output of each cell, and fix any issues, save and resubmit.

    I've explained a bit more about how the grader works and best way to debug/troubleshoot, [here](https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng)

+ There is an indentation problem that caused your notebook to not be converted to python script. you'll see the error if you try to download the notebook as python .py file
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/weeks/1/threads/KG7q2GIhEeeg5QoVkEgmCg">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Pz818HqsEeeOygpRbdVQKg_8951cea0acb8aab84a422d2cfc95594f_Screen-Shot-2017-08-06-at-14.36.08.png?expiry=1535760000000&hmac=Cx1IXzT7COXfnnLchEbgU9uUCMVl7NcETDky_TqCE0Q" alt="text" title= "iPython File Menu" width="450">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/weeks/1/threads/KG7q2GIhEeeg5QoVkEgmCg">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/S-bN4nqsEeeOygpRbdVQKg_b34151568a5ccd90fab20aa3e04d76ad_Screen-Shot-2017-08-06-at-14.35.10.png?expiry=1535760000000&hmac=Z9HRBrPub_mPMiz8om5D1bc-Emd5XQGhixKr9Llv-pI" alt="500: Internal Server Error" title= "caption" width="450">
    </a>

    finally, I found its the first line in answer_two(), make sure the indentation of the first line is the same as the rest of the function,


+ Q3 test

    ```python
    #test 
    def test_three():
        columnsX =cancer.feature_names
        shapeX = cancer.data.shape
        namey ='target'
        shapey = cancer.target.shape
        X, y = answer_three()

        res  = 'X type Test: '
        res +=  ['Failed\n','Passed\n'][type(X)==pd.DataFrame]

        res += 'y type Test: '
        res +=  ['Failed\n','Passed\n'][type(y)==pd.Series]


        res += 'X shape Test: '
        res += ['Failed\n','Passed\n'][int(X.shape==shapeX)]

        res += 'y shape Test: '
        res +=  ['Failed\n','Passed\n'][int(y.shape==shapey)]
        try:
            res += 'X columns Test: '
            res +=  ['Failed\n','Passed\n'][int(all(X.columns==columnsX))]

            res += 'y name Test: '
            res +=  ['Failed\n','Passed\n'][int(y.name==namey)]

            res += 'X data Test: '
            res +=  ['Failed\n','Passed\n'][int(all(X.values.reshape(1,-1)[0]==cancer.data.reshape(1,-1)[0]))]

            res += 'y data Test: '
            res +=  ['Failed\n','Passed\n'][int(all(y.values==cancer.target))]

            res += 'X columns length Test: '
            res +=  ['Failed\n','Passed\n'][X.shape[1] ==len(cancer.feature_names)]

            res += 'X column names Test: '
            res +=  ['Failed\n','Passed\n'][all(X.columns==cancer.feature_names)]     
        except:
            print('Data type problem, X should be a pandas DataFrame and y should be a pandas Series')
            return res
        return res

    print(test_three())
    ```





### [More on the Grader](https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng)

+ Before submitting the assignment notebook in this course/specialisation, comment/delete the following
    + any calls to any plot functions precoded in the assignment or any plot code you write yourself
    + any magic functions (anything starting with %)
    + any unassigned linux commands (anything that starts with !),
    + any global import statements of __matplotlib, seaborn or adspy_shared_utilities__

+ A word of advice about the grader; a lot of issues and confusion arise due to difference between the interactive nature of notebooks and how you run them as an author and the way the grader executes the code; i.e.. notebook converted to python script then script is run then the functions are called one by one for evaluation. The best way to catch these issues is to run the notebook in a way that simulate how the grader works; i.e. test the functions in the last cell of the notebook rather than after each function definition and also Run the notebook with a fresh kernel. here is how I test my notebooks before I submit them for grading

    ```python
    fDict =globals()
    res = {k:v() for k,v in fDict.items() if k.startswith('answer')}
    res['answer_...']
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/JObvSVPwEeeroBJvqbisdg_5218f5c89da9d846b5689e501dca3ae1_Screen-Shot-2017-06-18-at-07.32.27.png?expiry=1535760000000&hmac=tcA-3-SG0Me0Kr1j6Qq1q3kXZNnmEfCqnmCK3_RnSs4" alt="Inspection" title= "caption" width="450">
    </a>

    and then I inspect individual functions, e.g.
    <a href="https://www.coursera.org/learn/python-machine-learning/discussions/forums/G2azJEn1EeeZ0AqTqdDjpg/threads/ngzbpVPdEeeqKBJHVk6djA/replies/Kzii_1P-EeeuixKUKINPng">
        <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ROIJvFPwEeeroBJvqbisdg_96f5c564855119bf4a98c17cd76c7f63_Screen-Shot-2017-06-18-at-07.33.43.png?expiry=1535760000000&hmac=4xqocnGItKTjleojh7o9IVtiPbBeJ96yAUwcudwYtzg" alt="Inspection" title= "caption" width="450">
    </a>


## Solution

### Introduction to Machine Learning

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# print(cancer.DESCR) # Print the data set description

cancer.keys()
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
```

+ Question 1 (Example)
    ```python
    # You should write your whole answer within the function provided. The autograder will call
    # this function and compare the return value against the correct solution value
    def answer_zero():
        # This function returns the number of features of the breast cancer dataset, which is an integer. 
        # The assignment question description will tell you the general format the autograder is expecting
        return len(cancer['feature_names'])

    # You can examine what your function returns by calling it in the cell. If you have questions
    # about the assignment formats, check out the discussion forums for any FAQs
    answer_zero()       # 30
    ```

+ Question 1: Convert the sklearn.dataset `cancer` to a DataFrame.
    ```python
    def answer_one():
        
        data = np.c_[cancer.target, cancer.data]
        columns = np.append(["target"], cancer.feature_names)
        
        return pd.DataFrame(data, columns=columns)

    answer_one()
    # column labes: target, mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, ..., worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension
    # 0  0.0  17.990  10.38  122.80  1001.0  0.11840  0.27760  0.300100  0.147100  0.2419  ...  0.11890
    # 1  0.0  20.570  17.77  132.90  1326.0  0.08474  0.07864  0.086900  0.070170  0.1812  ...  0.08902
    # 2  0.0  19.690  21.25  130.00  1203.0  0.10960  0.15990  0.197400  0.127900  0.2069  ...  0.08758
    # 3  0.0  11.420  20.38   77.58   386.1  0.14250  0.28390  0.241400  0.105200  0.2597  ...  0.17300
    # 4  0.0  20.290  14.34  135.10  1297.0  0.10030  0.13280  0.198000  0.104300  0.1809  ...  0.13740
    # <entries omitted>
    ```

+ Question 2: What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
    ```python
    def answer_two():
        cancerdf = answer_one()
        
        target= cancerdf['target']
        
        return pd.Series({'malignant': cancerdf.where(cancerdf['target'] == 0.0)['target'].count(),
                        'benign': cancerdf.where(cancerdf['target'] == 1.0)['target'].count()
                        })

    answer_two()
    # benign       357
    # malignant    212
    # dtype: int64
    ```

+ Question 3: Split the DataFrame into `X` (the data) and `y` (the labels).
    ```python
    def answer_three():
        cancerdf = answer_one()
        
        X = cancerdf[cancerdf.columns[1:]]
        y = cancerdf['target']
        
        return X, y
    ```

+ Question 4: Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
    ```python
    from sklearn.model_selection import train_test_split

    def answer_four():
        X, y = answer_three()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = answer_four()

    print("X_train size: {}".format(len(X_train.index)))
    print("X_test size: {}".format(len(X_test.index)))
    print("y_train size: {}".format(len(y_train.index)))
    print("y_test size: {}".format(len(y_test.index)))
    # X_train size: 426
    # X_test size: 143
    # y_train size: 426
    # y_test size: 143
    ```

+ Question 5: Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1)
    ```python
    from sklearn.neighbors import KNeighborsClassifier

    def answer_five():
        X_train, X_test, y_train, y_test = answer_four()
        
        knn = KNeighborsClassifier(n_neighbors = 1)
        
        model = knn.fit(X_train, y_train)
        
        return model

    model = answer_five()
    model
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #            metric_params=None, n_jobs=1, n_neighbors=1, p=2,
    #            weights='uniform')
    ```

+ Question 6: Using your knn classifier, predict the class label using the mean value for each feature.
    ```python
    def answer_six():
        cancerdf = answer_one()
        means = cancerdf.mean()[:-1].values.reshape(1, -1)
            
        model = answer_five()
        
        return model.predict(means)

    means = answer_six()
    print("Means= {}".format(means))
    # Means= [ 1.]
    ```

+ Question 7: Using your knn classifier, predict the class labels for the test set X_test.
    ```python
    def answer_seven():
        X_train, X_test, y_train, y_test = answer_four()
        knn = answer_five()
        
        return knn.predict(X_test)

    test = answer_seven()
    print("Test set result= {}".format(test))
    # Test set result= 
    # [ 1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  0.  0.  1.  0.
    #   0.  0.  0.  1.  1.  1.  0.  1.  1.  1.  1.  0.  1.  0.  1.  0.  1.  0.
    #   1.  0.  1.  0.  0.  1.  0.  1.  0.  0.  1.  1.  1.  0.  0.  1.  0.  1.
    #   1.  1.  1.  1.  1.  0.  0.  0.  1.  1.  0.  1.  0.  0.  0.  1.  1.  0.
    #   1.  1.  0.  1.  1.  1.  1.  1.  0.  0.  0.  1.  0.  1.  1.  1.  0.  0.
    #   1.  0.  1.  0.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  0.  1.  0.  1.
    #   0.  1.  1.  0.  0.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  0.  1.
    #   1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  0.  1.  1.  1.  0.]
    ```

+ Question 8: Find the score (mean accuracy) of your knn classifier using X_test and y_test.
    ```python
    def answer_eight():
        X_train, X_test, y_train, y_test = answer_four()
        knn = answer_five()
        
        return knn.score(X_test, y_test)

    score = answer_eight()
    print("Score = {}".format(score))
    # Score = 0.916083916083916
    ```

+ Optional plot: 
    ```python
    def accuracy_plot():
        import matplotlib.pyplot as plt

        %matplotlib notebook

        X_train, X_test, y_train, y_test = answer_four()

        # Find the training and testing accuracies by target value (i.e. malignant, benign)
        mal_train_X = X_train[y_train==0]
        mal_train_y = y_train[y_train==0]
        ben_train_X = X_train[y_train==1]
        ben_train_y = y_train[y_train==1]

        mal_test_X = X_test[y_test==0]
        mal_test_y = y_test[y_test==0]
        ben_test_X = X_test[y_test==1]
        ben_test_y = y_test[y_test==1]

        knn = answer_five()

        scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
                knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

        plt.figure()

        # Plot the scores as a bar chart
        bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

        # directly label the score onto the bars
        for bar in bars:
            height = bar.get_height()
            plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                        ha='center', color='w', fontsize=11)

        # remove all the ticks (both axes), and tick labels on the Y axis
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

        # remove the frame of the chart
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
        plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    ```




