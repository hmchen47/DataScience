# Assignment 3

## NoteBook links

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


### Assignment Q 5

+ Issue

    Is there anyone else who struggled with this problem? I was able to draw two curves easily, but finding recall_value for a given precision value and tpr for given fpr was difficult.

    I tried looking at curve and was able to guesstimate approximate values, but grader marked in incorrect. I even tried fitting vertical lines and finding intersection of two lines to pin-point the intersection location.

    The values I interpreted by looking at the graph were approximately (0.83, 0.82).


+ Thread 1

    If you have plots like Sourabh Jain shows in the original posting, you're doing it wrong. I had the same plots, I read values from them, and the autograder kept counting my answer wrong.

    If your plots look like those, your problem is that you are getting your y_predicted values using the .predict() method. You should be using the .predict_proba() method instead. You need to use the second column of the return value of .predict_proba() instead of the return value of the .predict() method. Then your plots will look like the plots from the lectures and you will be able to read the correct values off the plots.

    That's what Zijian Wang is trying to say in his post that begins with "When I did the quiz at the very first attempt..." 

+ Thread 2

    I think that I too got the same plot but it should have more than 3 points? I looked at the documentation for precision_recall and it says it returns array of size "len(unique(values passed))". [How did others get the jagged graph?](https://www.coursera.org/learn/python-machine-learning/discussions/weeks/3/threads/GfmnBlEpEeeuixKUKINPng/replies/zGjBeVFFEeexEgqqsTMOtA/comments/ByqU61JNEee3RwoqcUym2A)

+ Thread 3

    All of you can take a look at this [post from Stack Overflow](https://stackoverflow.com/questions/9850845/how-to-extract-points-from-a-graph). You can get your y-axis points from a corresponding x-axis point very easily from the information contained in the return type of plot()

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



## Solution

In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).

```python
import numpy as np
import pandas as pd
```

+ Question 1: Import the data from fraud_data.csv. What percentage of the observations in the dataset are instances of fraud?

    ```python
    def answer_one(debug=False):
        
    #     # Your code here
    #     fraud_df = pd.read_csv("fraud_data.csv ") 
        
    #     fraud_cnt = fraud_df['Class'].sum()
        
    #     return fraud_cnt / fraud_df.shape[0] # Return your answer

        # Your code here
        data_frame = pd.read_csv('fraud_data.csv')
        X, y = data_frame.drop('Class', axis=1), data_frame.Class;
        
        result = len(y[y==1]) / (len(y[y==1]) + len(y[y==0]))
        
        return result # Return your answer

    answer_one(debug=True)
    # 0.016410823768035772

    # Use X_train, X_test, y_train, y_test for all of the following questions
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('fraud_data.csv')

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    ```


+ Question 2: Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?

    ```python
    def answer_two(debug=False):
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import recall_score, accuracy_score
        
        # Your code here
        dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
        # Therefore the dummy 'most_frequent' classifier always predicts class 0
        y_dummy_predictions = dummy_majority.predict(X_test)
        
        # metric scores
        dummy_acc_score = accuracy_score(y_test, y_dummy_predictions)
        dummy_recall_score = recall_score(y_test, y_dummy_predictions)
        
        return (dummy_acc_score, dummy_recall_score) # Return your answer

    answer_two(debug=True)
    # (0.98525073746312686, 0.0)
    ```


+ Question 3: Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?

    ```python
    def answer_three(debug=False):
        from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
        from sklearn.svm import SVC

        # Your code here
        svm = SVC().fit(X_train, y_train)
        svm_predict = svm.predict(X_test)
        
        prec_score = precision_score(y_test, svm_predict)
        recall_score = recall_score(y_test, svm_predict)
        acc_score = accuracy_score(y_test, svm_predict)
        
        if debug:
            print(classification_report(y_test, svm_predict, target_names=['Neg', 'Post']))
        
        return (acc_score, recall_score, prec_score)  # Return your answer

    answer_three(debug=False)
    # (0.99078171091445433, 0.375, 1.0)
    ```


+ Question 5: Train a logisitic regression classifier with default parameters using X_train and y_train.

    ```python
    def answer_five(debug=False):
        
        # Your code here
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression().fit(X_train, y_train).decision_function(X_test)
        
        def precision_recall():
            """Plot Precision-Recall Curve to view the result with Precision=0.8"""
            from sklearn.metrics import precision_recall_curve
            import matplotlib.pyplot as plt
            
            precision, recall, thresholds = precision_recall_curve(y_test, lr)
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
            
            print("Precision \t Recall:")
            for cnt in range(len(precision)):
                if 0.75 < precision[cnt] < 0.85:
                    print("({:.6f}, {:.6f})".format(precision[cnt], recall[cnt]))
            
            return None
        
        def roc_curve():
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr)
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
            plt.show()
            
            print("ROC Curve around 0.16")
            for idx in range(len(fpr_lr)):
                if 0.14 < fpr_lr[idx] < 0.22:
                    print("({:.6f}, {:.6f})".format(fpr_lr[idx], tpr_lr[idx]))
            
        if debug:
            import matplotlib.pyplot as plt
            
            precision_recall()
            roc_curve()
        
        return (0.825, 0.944) # Return your answer

    answer_five(debug=True)
    # (0.825, 0.944)
    ```


+ Question 6: Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.

    ```python
    def answer_six(debug=False):    
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression

        # Your code here
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression().fit(X_train, y_train)
        
        grid_values = {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100]}

        grid_clf_custom = GridSearchCV(lr, param_grid=grid_values, scoring='recall')
        grid_clf_custom.fit(X_train, y_train)

        predicted = grid_clf_custom.predict(X_test)

        if debug:
            print('Grid best parameter (max. recall): ', grid_clf_custom.best_params_)
            print('Grid best score (recall): ', grid_clf_custom.best_score_)
            print("CV Results: \n{}".format(grid_clf_custom.cv_results_))
            print("\n\n")
        
        return np.reshape(grid_clf_custom.cv_results_['mean_test_score'], (5, 2)) # Return your answer

    answer_six(debug=True)
    # array([[ 0.66666667,  0.76086957],
    #        [ 0.80072464,  0.80434783],
    #        [ 0.8115942 ,  0.8115942 ],
    #        [ 0.80797101,  0.8115942 ],
    #        [ 0.80797101,  0.8115942 ]])
    ```

    ```python
    # Use the following function to help visualize results from the grid search
    def GridSearch_Heatmap(scores):
        %matplotlib notebook
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure()
        sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
        plt.yticks(rotation=0);

    GridSearch_Heatmap(answer_six())
    ```
