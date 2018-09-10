# Assignment 2

## Notebook

+ [Launching web page](https://www.coursera.org/learn/python-machine-learning/notebook/BAqef/assignment-2)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=v4kpnNfCSRKJKZzXwlkSaw&next=%2Fnotebooks%2FAssignment%25202.ipynb)
+ [Local Notebook](notebooks/Assignment02.ipynb)
+ [Local Python Code](notebooks/Assignment02.py)

## Useful 

###  [Q1 and Q2 - answer not matching grader answer](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/2/threads/ahAIGlc4Eee7qw4C9eIhHA)

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


