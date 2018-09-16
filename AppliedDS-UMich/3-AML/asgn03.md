# Assignment 3

## Note Book links

+ [Launching Web Page](https://www.coursera.org/learn/python-machine-learning/notebook/eqnV3/assignment-3)
+ [Web Notebook](https://hub.coursera-notebooks.org/user/elkljxyoytcwjbmkgctrtg/notebooks/Assignment%203.ipynb)
+ [Local Notebook](notebooks/Assignment03.ipynb)


## Useful links

### [Question 4 & Question 6](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/3/threads/X-g9oWVkEeeNUQ7djXWc7g)

Question 4 asks for the confusion matrix, by just looking at your numbers I know that's not the correct answer because, the sum of items in the confusion matrix should add up to the number of elements in X_test

I.e.

```python
res = answer_four()
res.flatten().sum()==len(X_test) #this should be True
```

len(X_test) is 5424, the sum of values in your result is much lower than that.

the steps to take in Q4

1. initialise the learning model clf using the values provided in the description
2. fit the model clf on X_train and y_train
3. get y_predict using the decision function on X_test
4. convert the y_predict to values of 0 or 1 based on whether the y_predict is above (1)or below(0) the threshold (y_pred_thres)
5. use y_test and y_pred_thres to find the confusion matrix

I hope this helps and Good luck

Sophie


first C should be 1e9 and gamma 1e7

step 3-get y_predict using the decision function on X_test not using predict

so y_predict will contain large negative numbers, using broadcasting , y_predict_threshold = y_predict>threshold.


### [Meaning of evaluation metrics like precision score per class](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/3/threads/ikXu94tGEee-2hJt6mnFXg)

+ Thread 1

    The support line, contains the weights, i.e. the number of samples belonging to each class in y_true,


    the average is computed using

    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/wFPIKI4wEeekTw4xHnjhxA_a0cee795b4b2b3bb8b5c29f9ea05c485_Screen-Shot-2017-08-31-at-10.42.16.png?expiry=1537228800000&hmac=5qCQF00p0xDu349cZ0j0cXLwVbVrycwRlm9dazhlxUU" alt="Image for y_test_mc Series" title= "caption" height="200">

    so this average is not the the precision with macro averaging.

    ```python
    s = [37, 43, 44, 45, 38, 48, 52, 48, 48, 47]
    np.average([1.,1.,1.,1.,.14,1.,1.,1.,1.,1.],s)
    ```

+ Thread 2

    classification report is generated using the confusion matrix, and it will contain rows corresponding to each class, i.e. of you have 5 classes, the report will have 5 rows

    in multiclass case considering a case with n classes, we have n binary classification problems i.e.

    class 1 / not class 1

    class 2/ not class 2

    ...

    class n/ not class n

    when computing precision we need TP and FP and hence we need to look at the predicted values i.e. the columns of the confusion matrix

    when computing recall we need TP and FN and hence we need to look at the actual values/classes i.e. the rows of the confusion matrix

    lets clarify the above with the example from the notebook, below is the confusion matrix(CM) for the digit classification

    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/sujHiovEEeedowrbq0BbtA_f07ceb1fe09c37886980d58a48a0ed25_Screen-Shot-2017-08-28-at-08.44.03.png?expiry=1537228800000&hmac=wujdljiYD_kWtoxUw0NijHyERhrl0aXkgl1K49aTC38" alt="Multi-class confusion matrix" title= "caption" height="200">

    and here is the classification report for the matrix above

    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Edkgz4vFEeeo6wrD2KBiEg_5ee7352979263749af5dc9924e8d470a_Screen-Shot-2017-08-28-at-08.46.19.png?expiry=1537228800000&hmac=3hEExxZn75QO9u17sJ18AlOEZI1uCC9Pc3Jy_ZsZEmU" alt="Multi-class calssification report" title= "caption" height="200">

    let's calculate a few of the items in the report

    to calculate the precision for digit zero, we look at the predicted label for 0 i.e. the first column in CM,

    ```python
    TP =24(intersection of row 0 and column 0), 

    FP=sum of all the values in column 0 except at row 0 i.e. 0+0+0+0+0+0+0+0+0=0

    precision for digit 0 = TP/TP+FP = 24/24 = 1
    ```

    to calculate the precision for digit 4, we look at the predicted label for 4 i.e. the fifth column in CM,

    ```python
    TP =38(intersection of row 4 and column 4), 

    FP=sum of all the values in column 4 except at row 4 i.e. 13+33+27+3+32+24+31+47+21=231

    precision for digit 4 =TP/TP+FP = 38/(38+231) = .14
    ```

    to calculate the recall for digit zero, we look at the true label for 0 i.e. the first row in CM,

    ```python
    TP =24(intersection of row 0 and column 0), 

    FN=sum of all the values in row 0 except at column 0 i.e. 0+0+0+13+0+0+0+0+0=13

    recall for digit 0 =TP/TP+FN =24/(24+13) =.65
    ```

    to calculate the recall for digit 4, we look at the true label for 4 i.e. the fifth row in CM,

    ```python
    TP =38(intersection of row 4 and column 4), 

    FN=sum of all the values in row 4 except at column 4 i.e. 0+0+0+0+0+0+0+0+0=0

    recall for digit 4 =TP/TP+FP =38/(38+0) = 1
    ```

    Notice that the results match these in the classification repot

    for multi-class classification the overall precision of a classification model, is calculated using macro averaging of the values you see in the classification report


### [Assignment Q 5]

+ Issue

    Is there anyone else who struggled with this problem? I was able to draw two curves easily, but finding recall_value for a given precision value and tpr for given fpr was difficult.

    I tried looking at curve and was able to guesstimate approximate values, but grader marked in incorrect. I even tried fitting vertical lines and finding intersection of two lines to pin-point the intersection location.

    The values I interpreted by looking at the graph were approximately (0.83, 0.82).


+ Thread 1

    If you have plots like Sourabh Jain shows in the original posting, you're doing it wrong. I had the same plots, I read values from them, and the autograder kept counting my answer wrong.

    If your plots look like those, your problem is that you are getting your y_predicted values using the .predict() method. You should be using the .predict_proba() method instead. You need to use the second column of the return value of .predict_proba() instead of the return value of the .predict() method. Then your plots will look like the plots from the lectures and you will be able to read the correct values off the plots.

    That's what Zijian Wang is trying to say in his post that begins with "When I did the quiz at the very first attempt..." 

+ Thread 2

    I think that I too got the same plot but it should have more than 3 points? I looked at the documentation for precision_recall and it says it returns array of size "len(unique(values passed))". How did others get the jagged graph? https://www.coursera.org/learn/python-machine-learning/discussions/weeks/3/threads/GfmnBlEpEeeuixKUKINPng/replies/zGjBeVFFEeexEgqqsTMOtA/comments/ByqU61JNEee3RwoqcUym2A

+ Thread 3

    All of you can take a look at this post from Stack Overflow. You can get your y-axis points from a corresponding x-axis point very easily from the information contained in the return type of plot()

    https://stackoverflow.com/questions/9850845/how-to-extract-points-from-a-graph

+ Thread 4

    My plots look similar to the plot from Thanh and the values I get using Zijian's code look right compared to the plot, but the autograder still reports it as incorrect. I even made sure it was a tuple of floats instead of numpy floats if that matters... A little stuck here...

    Change your float() to np.float64() and then it should be resolved!

    I tried changing it to np.float64() and then just removed it. Grader did not accept it either time.

    Just commented out the import of the matplotlib library and that was it

+ Thread 5

    When I did the quiz at the very first attempt I also got a graph like yours. The reason of this, as pointed by people above(thanks!), is when drawing graph like this, we're using the probability instead of the prediction(e.g., 0/1). If we go with prediction, we will get a very clean graph, but this is incorrect. Instead, you may switch using the probability and you will get some stair-like graphs(which will lead you to the correct answer).

    I agree with Jesse and Pini. They have got the correct graph.

    I made a mistake. Instead of using probability of prediction, I simple used prediction, which is why the graphs I got are wrong. In P-R and ROC curve function, you need to use y_test and y_prob_prediction values.

+ Thread 6

    Actually you do not need to plot the curve. This piece of code might be helpful: tp_queried = tp[np.argmin(abs(fp - 0.16))], where tp, fp are from roc_curve

    I have tried using the tp_queried = tp[np.argmin(abs(fp - 0.16))] code, and I am still not getting anywhere. I used the above line of code for both the recall and true positive rate values and I'm getting values that look like what they should be given the plots of the two curves. Someone mentioned rounding and/or decimal places earlier, would the grader not accept an answer if it went to around 15 decimal places? I have tried using the round(recall, 3) function, but it still returns recall to 15 decimal places.

    The reason is that we could not make sure whether there is an exact value which we are looking for. If you directly check the table, what about this situation: let's find the corresponding value for 0.31415926. If so, the argmin method will work but looking table will not work. Also, if there is exactly the same value in the table, looking table and argmin should give us the same value.









