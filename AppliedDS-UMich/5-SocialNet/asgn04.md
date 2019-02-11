# Module 4 Assignment

## Notebooks

+ [Notebook Launching Web Page](https://www.coursera.org/learn/python-social-network-analysis/notebook/rMoj0/assignment-4)
+ [Web Notebook](https://bajwjsbbpcxhnmzzoyjrrp.coursera-apps.org/notebooks/Assignment%204.ipynb)
+ [Local Notebook](notebooks/Assignment04.ipynb)


## Discussion Forum

### [Week 4 Part 2b, need more feedback (no AUC from grader)](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/CrGwkeqAEeeUNw523nnMcghttps://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/CrGwkeqAEeeUNw523nnMcg)


+ Dale Schouten init

    For part 2b, it just tells me that the "answer is incorrect". It doesn't give any sort of AUC score or anything, even though I've checked that it returns a pandas series with the correct length and numbers between 0 and 1, i.e. all probabilities for the right number of edges.

    In part 2a,it seemed that "answer incorrect" usually meant something more drastic than just the wrong probabilities, i.e. wrong length, wrong types, stuff like that. Once I got it to report an AUC score for me it was easy to see where the problem was and I could fix the problem. With part 2b, I don't understand if there's something basic I'm doing wrong or if I just don't have a high enough AUC score. I've checked the AUC score on a train/test split and have gotten up to .9, so I'm not sure why the grader is marking it incorrect. Is there any way I can find more info?

    Thanks!

    Dale

+ Uwe F Mayer reply 

    Dale, the autograder should report back your AUC regardless of whether it passes the threshold or not. If you don't get an AUC reported then there is something that keeps the autograder from computing it, such as import matplotlib, exceptions thrown, wrong return format, and so on.

    You might want to start with a clean copy of the original unmodified assignment notebook and enter a solution with just random scores. Here is some code that does that.

    ```python
    def salary_predictions():
        # next statement is one long line
        nan_mgr_sal = [i[0] for i in nx.get_node_attributes(G, 'ManagementSalary').items() if np.isnan(i[1])]
        import random
        random.seed(0)
        predictions = [random.uniform(0,1) for m in nan_mgr_sal]
        return pd.Series(predictions, index=nan_mgr_sal)

    def new_connections_predictions():
        # next statement is one long line
        nan_future_connections = list(future_connections['Future Connection'].loc[future_connections['Future Connection'].isnull()].index)
        import random
        random.seed(0)
        predictions = [random.uniform(0,1) for e in nan_future_connections]
        return pd.Series(predictions, index=nan_future_connections)
    ```

    I left the other part unanswered. When submitted the above answers it resulted in the following grader output:

    > This assignment was graded with Grader version
    > 2017.09.22a. If you don't understand the grader
    > message, or believe the message to be inaccurate,
    > please see the discussion forums for details on
    > the grader limitations. See the course resource
    > Jupyter Notebook FAQ page for more details on
    > limitations of the autograder.  NOTE: If you have
    > the line %matplotlib notebook or you are using
    > import matplotlib.pyplot as plt in your solution
    > you MUST REMOVE IT or comment it out before
    > submitting for grading. This is the number one
    > issue with autograding errors, please check your
    > work closely.  Detected Jupyter notebook
    > submission.   ----------  Function
    > graph_identification was answered incorrectly, 0.2
    > points were not awarded. Incorrect. For the salary
    > predictions your AUC 0.5868271758682717 was
    > awarded a value of 0.0 out of 1.0 total grades For
    > the new connections predictions your AUC
    > 0.49570810206465743 was awarded a value of 0.0 out
    > of 1.0 total grades


    Once you get the same output from the grader, copy your solution without your return statement into the function definitions right at the top of the function bodies, but always leave that code I gave you here in it. Submit again and see if you still get an AUC reported. If not then the autograder has a problem with running your code, and you have to start digging to find out what the autograder cannot handle by removing / commenting out pieces of it and resubmitting until the grader reports an AUC. Then you need to figure out how to change the code that the grader cannot handle. If you still get an AUC then it's your return value that the autograder cannot handle, and you need to look there to make sure it's the right format, length, nan-free, and so on.

    This procedure takes some time, but should help to figure out what's broken in your notebook.


+ Uwe F Mayer reply

    Kayla, mentors are unpaid volunteers, who do not have access to your submissions, nor access to the backend that does the grading. You might want to go about it a bit differently. Presumably your code is error free as far as you can tell. So why don't you submit half of it, either the autograder agrees with you or it doesn't. Then submit the other half. And so on. You should find where you and the autograder disagree in the order of log(n) steps, n being the number of code lines in your notebook. Even with 1000 lines you should be done in 10 submissions.


+ Uwe F Mayer reply

    Jean, here's some reading on indexing:
        + [Pandas doc on Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/indexing.html)
        + [Stockoverflow discussion on pandas indexing](https://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation-how-are-they-different)

    And in case anybody wonders (I did), the first documentation link clearly states that you can use boolean masks with either .loc or .iloc (I've always only used .loc for that).


### [Networkx version issue](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/mHXy-we9Eei6gArYkvS6gA)

+ Rajesh Rajendran init

    Hi, I am unable to read the input files provided using pickle package. The following is what i get.

    P1_Graphs = pickle.load(open('A4_graphs','rb'))

    P1_Graphs

    Out[3]:

    [<networkx.classes.graph.Graph at 0x2550d0415c0>,

    <networkx.classes.graph.Graph at 0x2550d05c198>,

    <networkx.classes.graph.Graph at 0x2550d05c1d0>,

    <networkx.classes.graph.Graph at 0x2550d05c208>,

    <networkx.classes.graph.Graph at 0x2550d05c240>]

    P1_Graphs.nodes()

    Traceback (most recent call last):

    File "<ipython-input-4-a9264211af7a>", line 1, in <module>

    P1_Graphs.nodes()

    AttributeError: 'list' object has no attribute 'nodes'

    Anybody faced similar issues? Is it incompatibility issues between networkx and pickle? If yes, how do we solve this?

+ Uwe F Mayer reply

    Rajesh, apparently there was a major update to networkx from 1.11 to 2.x, and it broke lots of things. My local installation behaves just like yours and trows the same exception. I am running an Anaconda installation and downgraded networkx back to 1.11 with

    > conda install networkx=1.11

    If you don't use conda you might want to try pip with (and yes, there are two equal signs):

    > pip install -I networkx==1.11

    You can check which version you are running with:

    > import networkx as nx
    > print(nx.__version__)

    I had networkx version 2.1, downgraded to 1.11, and now the code works fine.


### [Assignment 4 updated version (v1.2) is released](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/qPTjZaMDEeedIRL4mTW7Xg)

+ Ruihan Wang init

    Hi everyone,

    A new version of Assignment 4 (v1.2) is released. Some typos like "Managment" which may lead to confusions are fixed. Sorry for the inconvenience and we appreciate your understanding. If you find any other issues about course contents, feel free to post in the forum.

    Also, learners who opened this assignment first time will directly see the latest version. For those who have viewed previous versions and want to get the latest version, you may need to reset your notebook. Please go to "Resources"-"Jupyter Notebook FAQ Page"-"6. How do I reset my notebook?" if you have no idea on how to do that.

    Thanks,

    Team of Applied Social Network Analysis in Python

    --------------------------- Below this line is the update information of v1.1-------------------

    Hi everyone,

    We just released an updated version of assignment 4 which revised the wrong grading criteria in the instructions for part 2A and 2B. We sincerely apologize for the confusion caused by this mistake.

    The grader's grading criteria for both part 2A and part 2B is based on the following: A model which with an AUC of 0.88 or higher will receive full points, and with an AUC of 0.82 or higher will pass (get 80% of the full points).

    The benchmark 0.75 in the previous version of assignment 4 was set tentatively at the beginning of designing this assignment. We increased the benchmark after conducting a lot of testings with different models and features of these two questions before the course was launched. Unfortunately, the instructions were not updated at the same time. Therefore, the grader was grading assignment 4 under a different standard all the time. That's why a lot of learners found that the grader output was confusing.

    Also, some learners reported that they got grades more than the full points assigned to part 2A or 2B, this issue has also been fixed. Please resubmit your assignment to check whether you can pass the latest autograder now.

    Learners who opened this assignment first time will directly see the latest version. For those who have viewed previous versions and want to get the latest version, you may need to reset your notebook. Please go to "Resources"-"Jupyter Notebook FAQ Page"-"6. How do I reset my notebook?" if you have no idea on how to do that.

    We are sorry for the inconvenience and thank you for your understanding.

    Best,

    Team of Applied Social Network Analysis in Python


### [Asg 4 question 1 pseudocode](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/jMXF0EP1Eei4xBLcWCNWTA)

+ Gint Butenas init

    Can staff please provide what is expected in pseudocode and point to sources in the materials that provides more detail in terms of parameters. I have been struggling with this final assignment for months and am tired of paying the monthly Coursera fees.

    This is my final assignment for the certificate, but I can't get over the finish line!

+ Uwe F Mayer reply

    Gint, this question is about understanding of the network generation algorithms (Preferential Attachment, Small World). It is up to you to study the networks provided, and to judge what algorithm was likely used to generate it. With study I mean print them out, or plot them, or whatever you want to do. Then you can simply hardcode your answers and return a list of 5 classifications, one of 'PA', 'SW_L', or 'SW_H' per network. (For extra credit you could code an algorithm to do that classification automatically, but that's not really required.)

+ Uwe F Mayer reply

    You can generate a PA network with the barabasi_albert_graph constructor, and a Small World network with the watts_strogatz_graph, that's correct.

    However, that's not the point of this assignment. You ought to look at the number of neighbors each node has, and more precisely the distribution of those neighbor counts.

    Preferential Attachment networks have a distinct distribution of the neighbor counts.

    For Small World networks again that distribution is rather distinctive, and depends on the re-wiring probability.


### [Part 2A](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/V6oG1JwbEeefoQoT04yVmg)

+ Futaya Yamazaki init

    I still cannot get this right. But I don't have any clue. AUC was 0.9 in test set.

    I got the following.

    1 0.167128
    ...
    Length: 252, dtype: float64

+ Henry Woody reply

    If your ROC AUC Score is less than 0.5, it is likely that your model is predicting labels backwards. Consider the fact that if you just choose classes randomly, you'll likely end up with a ROC AUC Score of 0.5. So if you get a score of, say, 0.15, that means you're doing reliably worse than just random, which means you're actually doing okay, because if you flip the classes you assign you'd have a ROC AUC Score of 0.85.

    Suggestion: if you are getting a score much less than 0.5, consider what in your model is making the classes be predicted in reverse. Or, for a quick fix, swap the predicted classes before submission.

+ Badaruddin Chachar reply

    ```python
        mdl.fit(X, y)
        
    #     mdl_pred = mdl.predict(X)
    #     auc = roc_auc_score(y, mdl_pred)
    #     print(auc)
        mdl_prob = mdl.predict_proba(test[features])[:, 1]

        return pd.Series([p for p in mdl_prob], index=test.index.values)
    ```

+ Sidney Antonio A. Viana reply

    Hi folks,

    Dealing with the Coursera Autograder is really tricky. I just passed Assignment 4 with full credit. But I suffered to discover the right manner to submit my code.

    I originally did my assignment locally on my machine, and got AUC = 0,91 for Part 2A (management salary prediction), and AUC = 0,92 for Part 2B (future edges prediction). When I submitted the assigment by uploading my ".ipynb" file to the Coursera platform, the Autograder stated the it was not able to load the file.

    Then, I opened the Assignment notebook on the Cousera platform and copied all my code, and executed it on-line. I worked fine and returned the same results I got when running it on my local machine. Then I submitted it to the Autograder. Part 1 and Part 2B were graded, but Part 2A was not...

    THEN, after doing (another!) detailed check in my results to confirm that they were ok, I TOOK ALL MY CODE OUTSIDE THE FUNCTIONS PROVIDED IN THE NOTEBOOK ASSIGNMENT AND MOVED THEM TO INSIDE THOSE FUNCTIONS.

    And the Autograder finally brought me 100% grade...

    Hence, I suggest that you put all your code inside the functions ( def xxxx() ) provided in the notebook, to avoid external computations, because this seems to impair the grading process.

    Hope this hint be helpful to all you who are in the path to finish Assignment 4 with full credit. (But more important is all we have learned along the four courses of this specialization!).


### [part 2A - data leakage](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/2D0JVZ5FEeeCGg4L_00Nzg)

+ Paul Roggeveen init

    Hi there,

    I'm curious to hear your opinion. When calculating clustering coefficients and particularly when calculating centrality measures, data from the training data and the test data (and even more when you split the training data in two sets as well) will be combined in order to calculate the measures. This will lead to data leakage in my opinion. Therefore the generalization power of the model is reduced.

    Do you agree and if so, did you take care of it by splitting the graph in several parts and calculate all the measures separately (which is a "P-in-the-A" with graphs as far as I know)?

    Cheers, Paul

+ Alberto Ramírez reply

    In Part 2B I took care to only find the measures in my feature engineering on the learning set and the hold out set (those where Future Connection is NaN), by specifying the index of the data frame as the 'ebunch' parameter in the function, that is, just finding the given parameters for the pairs of nodes in the future_connections df; and also did this on the whole network G.

    I find that I get the exact same dame data frames in both cases (using the pd.DataFrame.equals() method). So, there is likely not a concern of data leakage (at least in part 2B). I doubt that this would be a concern for part 2A as well.

    Not only that, creating a data frame with the features from the whole network is faster than doing it according to the specific pairing in the index of the futures_connection df

    ```python
    features = ['JC','RA','AAI','PA','CCN','CRA']

    methods = [nx.jaccard_coefficient,nx.resource_allocation_index,
            nx.adamic_adar_index,nx.preferential_attachment,
            nx.cn_soundarajan_hopcroft,nx.ra_index_soundarajan_hopcroft]
    ```

    This codes takes 55 seconds to run

    ```python
    for frame in [hold_out,learning]:
        for var,func in zip(features,methods):
    #         #Uncomment below if it is desired to add closest neighbors metric
    #         if var is 'CN' and func is None:
    #             frame[var] = [len(list(nx.common_neighbors(G,e[0],e[1]))) for e in frame.index]
    #         else:
            frame[var] = [e[2] for e in func(G,frame.index)]
    ```

    While this code takes only 44 seconds to run

    ```python
    df = pd.DataFrame(index=list(nx.non_edges(G)))

    for var,func in zip(features,methods):
    #         #Uncomment below if it is desired to add closest neighbors metric
    #         if var is 'CN' and func is None:
    #             df[var] = [len(list(nx.common_neighbors(G,e[0],e[1])))
    #                            for e in df.index]
    #         else:
        df[var] = [e[2] for e in func(G)]

    for frame in [learning,hold_out]:
        frame = frame.merge(df,how='inner',left_index=True,right_index=True)
    ```

+ Takua Liu reply

    I think the model is in practice impossible, and no need, to be generalized to any new data, so we don't have to worry about the data leakage.

    The data is derived from a long period of correspondence record among employees. For anyone new to this company, it's impossible for them to have such data. When a new employee has accumulated much correspondence with other coworkers, the graph might have also changed greatly anyway, the model we derived here will be completely useless then.

    In short, this model is only used to analyze accumulated data post hoc, and is applicable only to the current dataset.


### [Part 2A Help Needed](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/xOMcCK0mEee7eQoPPhIfbA)

+ Virender Jamnal init

    I am getting AUC 0.947 in test set. Using Logisticregression with default parameters.

    I am using Clustering, Degree, Closeness_Centrality and betweenness_centrality as features.

    Autograder is giving me score 0.050 or at best 0.060.

    My sample code is below , please suggest what I am doing wrong. I am really frustrated with trying all possible things.

    ```python
    clf = LogisticRegression()

    df = pd.DataFrame(index=G.nodes())
    df['ManagementSalary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
    df['clustering'] = pd.Series(nx.clustering(G))
    df['degree'] = pd.Series(G.degree())
    df['closeness'] = pd.Series(nx.closeness_centrality(G))
    df['between'] = pd.Series(nx.betweenness_centrality(G))
    df1 = df[np.isfinite(df['ManagementSalary'])]

    X_train = df1.iloc[:,1:5].values
    y_train = df1.iloc[:,0:1].values

    X_train, X_test, y_train, y_test = train_test_split(df1.iloc[:,1:5].values, df1.iloc[:,0:1].values, test_size=0.20, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)

    df2 = df[np.isnan(df['ManagementSalary'])]
    x_test = df2.iloc[:,1:5].values
    x_test = scaler.transform(x_test)

    df2['ManagementSalary'] = clf.predict_proba(x_test)

    ser = pd.Series(data=df2['ManagementSalary'].map('{:,.4f}'.format), index=df2.index)

    return ser
    ```

+ Javier Ruano reply

    Your hiphotesis is not right because you shouldn't split the train data, you have the test data (missing Salary ), your strategy is right

    ```python
    df1 = df[np.isfinite(df['ManagementSalary'])]

    X_train = df1.iloc[:,1:5].values
    ```

    after you have a confusion between prevent overfitting and x_test.

    (I do the foundations of data science from edx.org too, ;-) )

    But thanks your forum message is very helpful.


### [Help with Part 2A - Salary Prediction](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/LLvTJ5sTEeeKqQqmdZSVhg)

+ Alessandro Corradini init

    I can't pass 2A, salary prediction. I tried different algorithms (SVC, tree, random forest, extratree, Neural network) without success. I tried without any data transformation and with a department sparse matrix transformation. Both of cases I can't pass the grader. Max auc of 0.64******

    Any suggestion? Thanks

+ Han Cho reply

    Marc,

    Did you scale your data? I also tried scale vs non-scale, and the results are quite different. With scaling, even decision tree gives slightly better result (which is not generally supposed to).

    And what features did you choose? With 4 features, most models gives > 0.9 AUC. And department feature is not a good choice for salary prediction. Considering manager's role, it needs to connect people, and being hub in communication. After scaling, I saw some feature values are pretty much correlated, so I took out each one of such duplicates. Since you need to find best fit, first add all features you can think of and then remove duplicates.

    Also, it's better to split data into train and test set to evaluate multiple models and to find best parameter values (C, gamma, alpha etc) for each model.

    Once you got right combination, you can re-train with whole set for submission. This will get you better score.

+ Marc Toma reply

    @AllWhoPassed2A

    I am close to giving up. I completed other ML courses, but here, I cannot get past the grading demon. Can't tell, what I am doing wrong. Then again, it can only be me, since others passed. So - Here my 2c :

    1. Create a graph G in networkx from the text file.
    2. Create multiple dictionaries in networkx, all having to do with clustering, degree, centrality, etc.
    3. Iterate over all nodes of G, adding node attributes from 2.
    4. Create a pandas DataFrame df, indexed with the original nodes from G.
    5. Add multiple Series to df by extracting the attributes from 3.
    6. Select feature matrix X_train and result vector y_train from df, for those observations, where 'ManagementSalary' is not null. X_pred likewise only where 'ManagementSalary' is null.
    7. Select classifier model from sklearn. Feed it X_train and y_train, fit it and apply to X_pred, to predict probability, that observation belongs to class 'ManagementSalary'.
    8. Extract result, convert into pandas Series, index with node numbers from X_pred and return.

    Now, I wouldn’t go so far to claim, that I have tried all possible combinations of features and all possible classifiers, but I am getting close - if you know what I mean.

    What is it, that I am not seeing ? Please help ! ! !

    Downhearted, disconsolate and dejected I remain

    Truly Yours

    Marc

+ Han Cho reply

    Marc,

    Before submitting it, you should try to split data into train and test set (75%, 25%) to test your AUC score. That way you will find the path to correct combination.

    Here's my step;

    load data info df

    Add 4 features to the frame

    Data scale (except managementsalary)

    Split into X and Xu(nan data)

    Split X into X_train, X_test

    run various model with various parameters to find the best model by evaluating AUC score (>0.9)

    Once you reach this step,

    Retrain best model/parameter with data X (not X_train nor Xu)

    Find predict probability with data Xu

    Hope this help.

+ Vivek Krishnan reply

    Hi Marc, most classifier models provide two functions - one to predict the target class and one to predict the target class probability. In the case of the logisticregression model for example, the functions are predict() and predict_proba() - hope you are using the latter as the former will output the actual predicted class and will impact your auc score.

    If that's what you are doing, then the next thing to check would be if you are scaling the feature values or not. For models like LogisticRegression and SVM, you would need to scale them.

    Also, you could check if your hyperparameters are fine - you can try using GridSearchCV to select the optimum hyperparameter values though for this particular data set, the default parameters seem to be fine.

    The steps you provided above are more or less same as what most of us have probably done except for a few minor differences:

    1. The graph G is already provided to you, so we don't need to read the graph from the file.

    2. The graph can be converted into a dataframe using the graph's nodes and then the features can directly be added to the dataframe by calling the respective networkx functions that return dicts as explained in the week 4 tutor's notebook. We don't really need to add the node attributes to the nodes and then extract them from the node attributes to add to the dataframe - this difference however only reduces the code involved, but shouldn't make any difference to the model or results

    3. If you are using classifiers like LogisticRegression or SVM etc., you need to use a scaler before calling fit on the model. If you scale, you need to use the scaler to transform the test set/prediction set as well (without fitting the scaler again)

    Finally, please check the training score that you get - if you are getting a good training score but a low test score, then you may need to recheck the above steps and also recheck the hyperparameters.

+ Marc Toma reply

    @Han, @Vivek

    Maybe it is not meant to be.

    + I tried scaling, that is using a scaler from preprocessing
    + I tried scaling in pandas ( x - x_mean ) / std_dev
    + I tried both HubScore and AuthorityScore as features
    + I tried SVM, LogistigRegression, KNN
    + In my salary_predictions function I get the score : print(svmClassifier.score(X_test, y_test))
        ```python
        salaryPredictions = salary_predictions() #0.860927152318
        ```
    + After applying the model to X_pred here are the last 50 predictions: `salaryPredictions[-50:]`

    Nothing but crap. Just wasting time. Nothing works. Congrats to all who passed !

    Take care ...

    Marc

+ Marc Toma reply

    @Vivek

    Hi Vivek - My sincere Thanks in advance ! ! !

    Please find attached code snippets ...

    My Dataframe has 8 columns, the first 2 being 'Department' and 'ManagementSalary', which I am not using in the classifier model training data.

    ```python
        # ...
        X_notnull = df[df.ManagementSalary.notnull()].iloc[:,2:8].values.tolist()
        y_notnull = df[df.ManagementSalary.notnull()].iloc[:,1].values.tolist()
        X_pred = df[df.ManagementSalary.isnull()].iloc[:,2:8].values.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X_notnull, y_notnull, test_size=0.2)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)  
        X_pred = scaler.transform(X_pred)  

        logRegClassifier = LogisticRegression()
        logRegClassifier.fit(X_train, y_train)
        print('Training Score :\t', logRegClassifier.score(X_train, y_train))
        print('Test Score :\t\t', logRegClassifier.score(X_test, y_test))
        y_pred = [a[0] for a in list(logRegClassifier.predict_proba(X_pred))]

        # ...
    ```
    Model score :
    ```python
    Training Score :     0.897009966777
    Test Score :           0.913907284768
    ```

    Thanks a million again my friend - I am starting to feel really bad ;)

    Truly Yours

    Marc

+ Han Cho reply

    Marc, you'd need to return [:,1] of predict_proba. Don't need to use list() for that. Just predict_proba[:,1] and convert it to Series with correct index.

    Also, you don't need to use tolist() for data set. Any model will work with DataFrame, you just need to be aware of return datatype.

    Your X_pred can be rewritten

    X_pred = df[ np.isnan(df.ManagementSalary) ]

    And using below, you can calculate AUC with your X_test before submitting. You also need to have y_test.
    ```python
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, logRegClassifier.decision_function(X_test))
    roc_auc = auc(fpr, tpr)
    ```

+ Marc Toma reply

    @Han

    Tyvm sir !

    I returned the wrong part of the tuple the whole time. I simply did not see it, although I went over the code like a 1000 times ...

    And thanks for the super cool coding tips - very useful ! ! !

    While we are at it : Is it only my perception or does the code run slightly faster, when in implemented in this more 'vectorized' style ?

    Thx a bunch again ;)

    Marc

+ Han Cho reply

    ok, I just have tested your code and it works for me. I just used 2 features using your code, I continue to get auc >0.85

    2 things, first you need to make sure about y value. In your code, you extract y value using .iloc[:,1]. This means y (managementsalary) is located at second column. Make sure the value of this column (or y value) is either 0 or 1.

    second, your proba return should be [:,1]

+ Marc Toma reply

    @Han

    Thank You my friend !

    I have corrected already and all is good now. The mistake that you found - by the way - resolved my problems in 2B as well

    Now, I am testing your coding tips. Not only is my code a lot cleaner, and really slick. I have the impression, that it is dramatically faster too.

    Is the ‘vectorized’ version implemented under the hood in a more efficient style ? Just wondering ...

    Truly Yours

    Marc

+ Marc Toma reply

    @David

    I've been there, so I understand very well, what you are going through - just hang tough ;)

    A few hints from someone, who went through the suffering :

    + The number of the department someone works in, might be as good an indicator as the color of his shirt, or the size of his shoes.
    + An important node connects other nodes - directly or indirectly - and supports other important nodes.
    + Recall, that every dictionary generated with networkx can be passed into the pd.Series constructor. I have 6 metrics. Watch the videos from Week 4 again. Just follow the instructions.
    + Be aware, that when you scale your data, you want to use the very same scaler that you fitted on your X_train for X_test and X_pred.
    + No need to try fancy ML. Logistic Regression will do the trick for both 2A and 2B. You can always try to get better later.
    + Standard hyper parameters work just fine. I get AUC well over 0.9 for both questions.
    + Han posted some super cool coding tips, on how to handle the returned data from the predict_proba() functions. Find his post. You will find me there too. One can return the wrong values for a very long time …

    Hope this help ;)

    Marc

+ David Calloway reply

    Marc:

    I've taken most of your suggestions, reverting to a simple/default params LogisticRegression classifier, dropping Department from my feature set, etc. I'm using the StandardScaler, fit to all the X data and applied to X_train, X_test, and X (for submission).

    When I do a train_test_split (80/20) and train with X_train & y_train and then test with X_test and y_test, I get a train_score of 0.90, a test_score of 0.89, and an AUC of 0.90. These all look good to me.

    When I fit to all of X & y for submission, my first few results look like:
    ```
    1       0.134524
    2       0.568210
    5       0.999825
    8       0.092230
    14      0.234590
    18      0.350949
    27      0.222425
    30      0.244749
    31      0.123970
    34      0.054526
    ```

    Still...the AutoGrader give me 0 points with no feedback on what it thinks my AUC is.

    Any ideas, anybody???

+ David Calloway reply

    Good morning to you, too, Marc.

    I am returning the results of predict_proba [:,1] which I think is correct. But the autoGrader disagrees (with something)

    Thanks for your contributions to this forum, BTW. I can see that you have gone through the same pains I'm experiencing and came out the other end with eventual success. Gives me hope...

    Also...do the example result values shown in the instructions correlate to something we might actually expect to see (I.e. Do they identify the actual probable managers for the listed nodes?)?

+ Marc Toma reply

    @David

    Been there ;)

    OK, let's go through my sets step by step. Then compare to what you came up with :

    1. X_notnull is my set of observations for the training data. Select rows by applying a Boolean mask to the Data Frame. Recall, that there is a numpy method, which you can apply to a Data Frame column, which only returns those rows, where the supplied columns is not null. See Han's post. Select the columns with your features. I used : .iloc[...].
    2. y_notnull is the result vector for your training data. Follow the same rules to select rows, but you have only one column.
    3. From X_notnull select X_train and X_test. From y_notnull select y_train and y_test. I used : train_test_split()
    4. X_pred is the set of observations, where the filter column from 1. is null. Use the same features ( = columns ) as in 1.
    5. Create a scaler, fit on X_train, transform X_train, X_test, X_pred. Recall that transform does not change in place. You want to use it like X_train = scaler.transform(X_train)
    6. Select a classifier. Do NOT get fancy. Fit your classifier using the now scaled X_train and y_train. Call the predict_proba() method on your classifier, pass scaled X_pred as input. This will return y_pred, that is your predictions for X_pred.
    7. Convert y_pred to a pd.Series. Be sure to return the correct part AND the correct index ( = which rows hat null in them to begin with ? )
    8. The sample output in the assignment is just an example for the format ...
    ```python
    1     0.067718
    2     0.748885
    5     0.999989
    8     0.121369
    14    0.218906
    18    0.300001
    27    0.285913
    30    0.293113
    31    0.098008
    34    0.103298
    dtype: float64

    974     0.106509
    984     0.002799
    987     0.109866
    989     0.089301
    991     0.104336
    992     0.000353
    994     0.000217
    996     0.000097
    1000    0.035582
    1001    0.079134
    dtype: float64
    ```

    Hope this helps ...

    Marc

+ David Calloway reply

    With some help from Marc, I was finally able to pass this part of the assignment. The two key things I changed had to do with how I initialized the dataframes with the various data items. Marc & Han did things slightly different than my earlier methods, and after changing my approach to use their techniques, things started working. For example, to initialize my main dataframe that I subsequently added to with various features (like centrality measures, etc.), Marc used this technique:

    df = pd.DataFrame(index=G.nodes())

    To add columns to the df directly from the graph, this technique worked:

    df['dept' ] = pd.Series(nx.get_node_attributes(G, 'Department' ))

    Those simple hints got me on the right track to success.

    Thanks, Marc!

+ Marc Toma reply

    @Paul

    Hi Paul,

    the autograder works, but I think there is a time constraint for evaluation. So you want to make your code efficient and fast.

    Make sure you understand the notebook 'Extracting Features from Graphs' in Week 4. You can build a DataFrame directly from the graph.
    You can add Series to a DataFrame by passing a dict to the Series constructor. Avoid loops and list comprehensions. Again, take a look at the notebook.
    A feature matrix can be a DataFrame. No loops, no lists. Find Han Chos post. He suggests a very cool way to create X and y.
    When extracting the correct part of the predict_proba() function, avoid expensive computations. Slicing will do the trick.
    If you return a Series of length 253, there is something wrong. Make sure, you return a Series of proper lengh, as required in the assignment.
    As far as scaling is concerned : I am pretty sure, that you want to scale your data, but I may be wrong.
    Hope this helps ...

    Regards

    Marc


### [Part 2B](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/dThNU55sEeeuFgojn_NhSg)

+ Futaya Yamazaki init

    It takes time to make graph from DataFrame. Is there better way to do that? Or should I write to a file and new_connections_predictions() read it?

    I did this.

    ```python
        G = nx.Graph()
        for index, row in future_connections.iterrows():
            A, B = index
            G.add_edge(A, B, weight = row['Future Connection'])
    ```

+ Han Cho reply

    You may use G.add_edges_from()

    It takes a list of edge tuples.

    for nodes, use G.add_nodes_from()

    Or nx.from_pandas_dataframe() too.

    There are examples in week1's worksheet.


### [Part 2B Grading Difficulties](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/dd3S_a6zEeeANQrOhBtvUA)

+ Maria Corsaro init

    I'm having a lot of trouble with this question. I've tried a couple different variations, but the autograder always times out when I submit an answer for question 2B. If I leave question2B blank, then I get a 60/100 from the other two questions.

    My current approach is:
    + For each edge in Future Connections, both training edges and predicting edges, add it to the graph G, calculate the Jaccard coefficient and Adamic Adar Index, remove the edge from G
    + take the dataframe with all these values, separate the training edge rows from the predicting edge rows
    + use the training data to train an svm.SVC classifier
    + predict the values for the predicting data

    The last two steps are nearly the same as what I did for part 2A, and I got full credit there. If there's anyone that has any suggestions I'd really appreciate it!!

+ Maria Corsaro reply

    I'm pretty sure it's the jaccard coefficient and adamic adar index calculations that are slowing my program down. When I test my code separately, that's what's causing problems. This is that part of my code:

    ```python
    for i in range(0, df.shape[0]):
        curr_u = df.index[i][0]
        curr_v = df.index[i][1]
        G.add_edge(curr_u, curr_v)

        df['jaccard coefficient'][i] = ([j[2] for j in nx.jaccard_coefficient(G, [(curr_u, curr_v)])])[0]
        df['adamic adar'][i] = ([j[2] for j in nx.adamic_adar_index(G, [(curr_u, curr_v)])])[0]

        G.remove_edge(curr_u, curr_v)
    ```

+ Ruihan Wang reply

    If your assignment achieved a high score on your local machine but cannot be completed in the autograder, you can return the data frame of your result directly. Since the server resource is limited, it cannot accept some time-consuming models which may take hours to run. Thanks for your understanding.

+ Ruihan Wang reply

    We do not recommend you submit your answer this way, because there are a lot of efficient methods allowing you to achieve a high accuracy. If you choose to do it, my guess is that you need to code the results into the data frame manually. Here are my tips: You can try to generate a formatted string in your local machine first and copy that string into your notebook and doing some manipulations to create the data frame. Hope it works.


### [Part 2B Grader Timeout](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/Lg4ct6Q4EeeaaAq6Nz_GxA)

+ Samir Bajaj init

    I am running following line of code for entire future_connections df:
    ```python
    Future_connections['Edgefeature'][i]=((list(nx.preferential_attachment(G, [Future_connections.index[i]])))[0])[2]
    ```
    But it takes forever and auto grader times out, How can I optimize the code?

+ Jesus Adrian Perez reply

    I used a for loop on the generator returned from preferential_attachment, and appended those scores onto a list. Then I added this list to a dataframe.

    ```python
    import networkx as nx
    G = nx.complete_graph(5)
    preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
    scores = list()
    for u, v, p in preds:
    scores.append(p)
    ```

+ Sidney Antonio A. Viana reply

    I used "list comprehension" to record a entire list of values into a dataframe column. Here is an example:

    FUTURE_EDGES["Common_Neighbors"] = [len(list(nx.common_neighbors(G,edge[0],edge[1]))) for edge in FutureEdges]

    This is a single line of code that do the same as a "for loop", but performs faster. This approach saved me about 1.5 minutes of processing time for the comutation of one feature. For two or more features, the time savings will be better.

    See also this post, about the features I chose:

    https://www.coursera.org/learn/python-social-network-analysis/discussions/all/threads/_w__VtEbEeeT2RIs_W-FMg/replies/-MAjtNGAEeexuwrR6wbv6A/comments/gVcX-9hjEeeerg6JA-5x3A

    best!

### [2B help working with generators](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/iFZmPqM_Eee1fxJ_AEil0g)

+ Isabel Camilla Hutchison reply

    I'm stuck on how to add features to the future_connections dataframe.

    I've tried generating each feature from G using e.g.

    nx.jaccard_coefficient(G, ebunch=future_connections.index) and adding this as a new column in the dataframe, but I end up with a generator in each row! I assumed that setting the ebunch parameter would create actual coefficients for each tuple (edge) in future_connections. What am I doing wrong?

    Thanks in advance!

+ Takua Liu reply

    You can do this:
    ```python
    idx = nx.oooooo_coefficient(G,'''node pairs''')
    for node1, node2, value in idx:
    # do something with node1
    # do something with node2
    # do something with value
    ```

    Or use list comprehension:
    ```python
    ['''whatever expression you like''' for node1, node2, value in nx.oooooo_coefficient(G, '''node pairs''')]
    ```

    But when you construct feature Series to add to the original dataframe, please be careful:
    ```python
    future_connections['feature'] = pd.Series('''the list of values''', index='''node pairs''')
    ```

    if the index is not specified, the Series can't match the existing dataframe because the index names are different


### [My take on Part 2B](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/WqNCjeZFEeeXQhKQwLMgHA)

+ Giorgio Clauser init

    Hi everyone,

    I found this assignment interestng and fun and I particularly liked part 2B as it was challenging too. This is the approach I used to achieve a ROC AUC score of 0.93.

    I put the most of my effort in feature engineering. I built 8 features, namely:

    1. Preferential attachment and common neighbours, as suggested in the lesson.
    2. A dummy variable which equals 0 when the two nodes belong to different departments, 1 if they belong to the same one.
    3. Three dummies to catch the effect of the "ManagementSalary" node attribute: one dummy identifies if both the nodes have a management salary, another one identifies if both the nodes do not have it, the third one identifies when the attribute of the first node is different from the attribute of the second one. To properly compute these variables I updated the graph G with the results from the model computer for part 2A.
    4. The lenght of the shortest path between the two nodes. When the nodes are not connected, I set this value to max(value obtained among connected nodes) + 2.
    5. Nodes connectivity, which I unfortunately had to remove from my model because it's computationally heavy and crashes the grader. Dropping this variable caused a estimated decrease of my ROC AUC score of 0.01.

    Afterwards I fitted a logistic regression model with standard parameters, and that achieved already the ROC AUC score of 0.93.

    I hope this can be interesting for other fellow students!

+ James McDermott reply

    This is an unconventional approach, but I can understand the logic behind your feature extraction.

    But I think the extra engineering you did to design these features was unnecessary.

    Simply performing the various measures from 'Link-Prediction' on the network G, and then running a correlation analysis (pd.DataFrame.corr()) to see which features are strongly correlated with 'Future Connection' would also be enough for this assignment.

+ Чижов Владимир Борисович reply

    Your approach is too unconventional. I used the skills received during the fourth week of training in the topic "Link-Prediction". These properties are just designed to predict the veracity of new compounds. Using Logistic Regression with matched parameters gave ROC AUC score of 0.91


### [Part 2B feature selection](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/Cs8ii5_FEeeFSg7SmKEJwg)

+ Yang Fu reply

    Hi all who passed 2B,

    I'm just curious about the edge based features you used to train your classifiers.

    I was trying to include the 5 measurements included in the lecture video, but was only able to get the preferential attachment scores. The rest 4 just took forever to calculate and I could never saw the output data.

    Am I missing anything?

+ Sidney Antonio A. Viana reply

    Maybe those posts can help you:

    + [Post 1](https://www.coursera.org/learn/python-social-network-analysis/discussions/all/threads/_w__VtEbEeeT2RIs_W-FMg/replies/-MAjtNGAEeexuwrR6wbv6A/comments/gVcX-9hjEeeerg6JA-5x3A)
    + [Post 2 (which also as a reference to Post 1)](https://www.coursera.org/learn/python-social-network-analysis/discussions/all/threads/OpGTxqxmEeeB2hJL-yunuA/replies/VszoUayAEee7eQoPPhIfbA/comments/NXIQithqEeex6A7lu81O-A)

+ Sergio Vignali reply

    I have exactly the same problem, the Preferential Attachment score runs pretty fast, for the others measures it runs ages and I don't get any result...

    @Lucas could you please explain us how did you manage it?

+ Lucas Goldstone reply

    In the "extracting features from graphs" workbook, they extract preferential attachment scores using this list comprehension:
    ```python
    df['preferential attachment'] = [i[2] for i in nx.preferential_attachment(G, df.index)]
    ```
    where df is your dataframe, and G is your graph. What worked for me was to use this for all the features I was interested in, but replacing
    ```python
    nx.preferential_attachment(G, df.index)
    ```
    with whatever nx.____ method I was interested in.

    Hope this helps!

+ Christian Tellkamp reply

    Thanks for your reply! I used exactly the same line. So this does not seem to be the reason as far as I can see.

    Maybe it has to do with the way I constructed the graph? I used the following approach:
    ```python
    fc = future_connections.reset_index()
    fc[['source', 'target']] = pd.DataFrame(fc['index'].values.tolist())
    g = nx.from_pandas_dataframe(fc, 'source', 'target')
    ```
    Potentially, this is the "wrong" approach which makes the later calculations very infefficient?

+ Yang Fu reply

    @Sergio

    I finally figured it out. If the features are directly extracted from graph G, they can be calculated pretty fast. Previously I was trying to extract them from a graph I created using the future_connections dataframe.

    I still do know the reason why my previous method failed, but G can definitely give you all five features.


### [Construction of Graph for part 2B](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/38KCVKcJEee6bw62IA80dA)

+ Jeffrey Schafer init

    I'm really struggling with deciding what should exist in the the graph G for this problem. Let me try to be as specific as possible
    + Does G refer to the graph G created previously in the notebook or a new graph G? I find it confusing the way the problem is worded (or maybe I am reading too much in to it)
    + If I am starting with the previous graph G, do I need to add the appropriate edges from Future Connections or not before I do my calculations?

+ jeremy886 reply

    Yes, you use the same graph G from 2A for 2B.

    Interestingly, you want to add edges from Future Connections to G while I was thinking of creating a complete graph (fully connected graph) to work on.

    Neither is needed for this question as I have passed it this way: Treat the future connections as the question you want to solve for graph G; i.e. predict future connections and only used edges from there for calculation. (However, you still need G to create your features.)

    I was probably lucky but I tried many models and once I passed 2A with a model, I used it in 2B and it worked like a charm too. (pass means 100%).


### [Assignment 4 - Part 2B](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/Gzhpnd5JEeeKGAoMjlehBg)

+ Carlos Alberto Duque init

    When I read first this question and I did not understand what was this question asking for. After I read a few posts, and I got a light which I followed. So, now I am going to write what I did in order to pass this question.

    1. I built a DataFrame with those nodes which will were linked in the future.
    2. I looked for this measures in link prediction, ['cn_soundarajan_hopcroft', 'preferential_attachment', 'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index']
    3. I wrote those measures in the DataFrame
    4. I devided the DataFrame in two new DataFrames. One with 'future_connection' values equals to NaN, and the second with the rest.
    5. I fitted a model with that DataFrame which had non-NaN values in 'future_connection'
    6. After teste a model with a good roc_auc_score, I looked for predict_proba for those values for that DataFrame with NaN values in 'future_connection
    7. I got a grade.

    I hope this help anyone in the future.


### [Part 2B: The confusion & realisation](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/RehORv4VEeeRYhJzSSdfKA)

+ Krishna Ramalingam init

    I spent a lot of time trying to understand what needs to be done. For the benefit of others, here is what is expected from the question

    1. Use the existing Graph G and find 3 to 4 metrics of G from the Link Prediction video. Each of these metrics are your features
    2. Write the output of each of these metrics to a single data frame
    3. Merge the future connections data frame with this
    4. Then do the model fitting which you learnt in Machine Learning module

+ Mike Newman reply

    or you could use the existing 'Future Connections' dataframe and add the 3 4 features as new columns.


### [Autograder does not grade 2b](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/IuGoriXQEemaRQ4MbceaAA)

+ Uwe F Mayer reply

    No, 2A and 2B do not have the same format.

    For 2A the index is a list of nodes, that is, integers. For 2B the index is a list of edges, that is, 2-tuples of integers:
    ```python
    (107, 348) 0.028654
    (542, 751) 0.022871
    (20, 426)  0.744657
    ...
    (939, 940) 0.024305
    (555, 905) 0.023074
    (75, 101)  0.037086
    Length: 122112, dtype: float64
    ```

    Of course the scores depend on your model. That's why I suggested to first get the result to be accepted, regardless of model performance. This means in particular you have to figure out those 122112 edges that need a score in the first place.

+ Uwe F Mayer reply

    Let's check if your index is the same as mine (this is a rough check):
    ```python
    # provide a checksum on the index
    # remove this before submission as the grader does not allow to print
    import numpy as np
    nc = new_connections_predictions()
    print("sum 1st nodes", np.sum(list(map(lambda e:e[0], nc.index.values))))
    print("sum 2nd nodes", np.sum(list(map(lambda e:e[1], nc.index.values))))
    print("sum all nodes", np.sum(list(map(np.sum, nc.index.values))))
    ```

    For me this prints:
    ```python
    sum 1st nodes 41359954
    sum 2nd nodes 82700555
    sum all nodes 124060509
    ```

### [Potential solution for problems with no grade for 2B](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/Zm2eqsWyEeemvAoyMa4lng)

+ Tom Miller

    My code added features to the future_connection DataFrame and executed correctly but, I always had to reread the future_connection DataFrame again prior to execution. I didn't really have problems with the code timing out and I added three features to the DataFrame to make the prediction.

    I added a line to reread the future_connection data inside the new_connections_predictions() and everything worked fine. Many of the other suggestion found in the Discussion Forum didn't work for me. Final AUC 0.9074830815094399 and full credit.

    I hope this helps someone.


### [2B help regarding feature extraction](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/_w__VtEbEeeT2RIs_W-FMg)

+ GNANA GRACY init

    I have tried most of the methods listed in forums, for getting the feature values, still all takes so long time. Is there any method to speed it up ? Can any mentor please help me. Its really killing my time.

+ Briag Dupont reply

    It became easier when I realized that it is better not to waste time on features that take a long time to extract. Preferential attachment is a quick to extract and also the number of common neighbors for each of edge helped. Then I used a gradient boosting classifier with a little parameter tuning I achieved 0.91.

+ Sidney Antonio A. Viana reply

    Good choice of attributes.

    Here, the attributes I chose were: Common Neighbors, Jaccard Distance, and Adamic-Adar Index. The model I used was Logistic Regression. I got AUC = 0.908.

    I agree that it's not necessary to use many features. Three features would be enough to attain an AUC of 0.9+, depending on the model used.

    When computing the features, I strongly recommend to avoid the use of "for loops". Use "list comprehension" intead. This will speed-up the computations and save you about a minute of computation time, depending of your machine. My code requires about 1.5 minutes to compute all the three features I mentioned above.


### [Q 2B graded 0 points without a reason](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/eiUPqb42EeezMQ4NzSCjvA)

+ Wu Jingxianinit

    My output prediction is a series consistent to the question description as below. I looked at previous threads in this forum and look like my predictions are similar to those who had got full marks. I do not have any command that calls local files.

    However, the audograder gave me 0 points without explanation of why. How can I diagnose my code?

    ```python
    predict_pred.head()
    (107, 348)    0.026059
    (542, 751)    0.011457
    (20, 426)     0.505379
    (50, 989)     0.011407
    (942, 986)    0.011353
    dtype: float64
    ```

+ Uwe F Mayer reply

    Jinagxian, that code is correct. Specifically, I inserted my model and my computation of predict_X where you have "......" and the result then matches what I get with my code (which passes the autograder).

    At this point you likely should check the basics.

    1. Does your notebook cause the grader to fail because it prints or plots something? (I think this is unlikely because that usually generates some kind of error message.)
    2. Does your code modify an essential variable further down in the notebook. Remember, the autograder first loads (and executes) the entire notebook, and only then calls one assignment function after the other to compute your point score. You can check if this is the problem by first restarting your notebook's kernel, and then you execute all cells up to and including the function definition, then call your function right after the definition and store the output in a variable, then execute all cells to the end of the notebook, and finally also call your function in a new cell at the very end of the notebook, and compare what you get there with what's stored in your variable. It should be the same. It's easiest to check with print(all(stored_result == new_result)).
    3. Are there any NaN scores in your submission?
    4. Are you writing anything to a file and are trying to read it back? That will not work because the autograder runs in its own directory.
    5. Maybe your model is not getting the necessary AUC to get points for the result. For that you'll have to do the usual model work. Take your training set, split into a subset train/test, and train only on the train subset, and evaluate on the test subset. What's the AUC? My model had an AUC > 0.9 on both train and test set, so I was quite confident it's a good model of sufficient generalization power. The instructions say one needs to get AUC >= 0.75 on the unlabeled submitted test set for full points. Of course I don't know what my model achieved there, but it must have passed that bar since I got full points.

    Checks 1-4 are easily done, and should be done right away to clear that up. If I was a guessing man I would say the problem lies in step 5 with an overfit model, but then again, that might not be the case. Maybe it's something else, but that's where I would start to look.


+ Uwe F Mayer reply

    One more follow-up, what precisely is the grader saying? I just submitted a random-score model:
    ```python
    import random
    def new_connections_predictions():
        # TODO: create a data set df_edge_test_submit with only
        # the test records

        # compute random model scores
        random.seed(123123123)
        y_edge_test_submit_predicted = [random.random() for i in range(0,len(df_edge_test_submit))]
        return pd.Series(data=y_edge_test_submit_predicted,
            name = None, index=df_edge_test_submit.index,
            dtype='float64')
    ```

    While I am getting 0 points for that, at least the autograder is telling me my AUC:
    ```python
    For the new connections predictions your AUC 0.49904177798494853 was
    awarded a value of 0.0 out of 1.0 total grades
    ```

    So the question is, do you get something like that, or do you get 0 without any explanation?

+ Uwe F Mayer reply

    Kyle, it appears that the autograder runs into trouble executing your code. I am saying this because you are saying it doesn't report back an AUC value with your model.

    The autograder not being able to run your code successfully, or not producing a valid score output, could have lots of reasons.

    The usual strategy to find the source of the trouble is the following: Start with the random model and add your code to it line by line (or block by block). Make sure you leave the return value to be the random model scores. Submit after each addition. At some point the autograder will stop reporting back the AUC and will simply say you got no points. That's when you know that the last addition of your code is what the autograder has problems with.

    Yes, I know this is an arduous way of figuring out what's not working, but it is fairly guaranteed to find the part of your code causing problems. How to fix that code is then your next task ;-).


+ Uwe F Mayer reply

    My submission records started with the same edges as Jingxian's original post, so I don't think the order is the issue here. The probabilities listed in the original post are similar to what I had in my submission that passed, so that looks reasonable as well.

    Maybe the submission set was incorrectly assembled by Jingxian, there should be 122112 records. The other thing to pay attention to is the index, when I print it I get

    ```python
    Index([(107, 348), (542, 751),  (20, 426),  (50, 989), (942, 986), (324, 857),
        (13, 710),  (19, 271), (319, 878), (659, 707),
       ...
       (144, 824), (742, 985), (506, 684), (505, 916), (149, 214), (165, 923),
       (673, 755), (939, 940), (555, 905),  (75, 101)],
      dtype='object', length=122112)
    ```

### [Module 4 Assignment part 2B - long running feature population](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/4/threads/V0c05r92EeeA8BKEefIB9g)

+ Nattapon Sub-Anake init

    I have a problem with my long running script when I add more features for prediction model.

    # features = ['Preferential Attachment','Jaccard Coefficient','Resource Allocation']

    is it possible to upload my future_connectioned dataframe with all populated features?

    Here is the error I got when I tried to load my data and use it in the script.

    Unable to load student solution datafile for autograding. Please see the course resources autograding FAQ for common issues associated with this error. Function graph_identification was answered correctly, 0.2 points were awarded. Correct. For the salary predictions your AUC 0.9443752594437526 was awarded a value of 0.4 out of 1.0 total grades Function new_connections_predictions was answered incorrectly, 0.4 points were not awarded.

    Here is the error because of long running script.

    We encountered the following warnings when grading this part:

    Grader timed out while grading your submission. Please try submitting an optimized solution. If you think your solution is correct, please visit the Discussion forum to see if your peers are experiencing similar errors. If the issue isn't resolved in 24 hours, please reach out to Coursera through our Help Center

+ Uwe F Mayer reply

    Logistic regression is what I suggest as a first attempt (and if you do it right, last attempt). This is a message to remember, always start with a simple model architecture first.

+ Uwe F Mayer reply

    Hmm, I used the same features, and it ran fine. It did take quite a while to grade, but much less than an hour. You might want to retry at a less busy hour (the autograder is a shared resource obviously).

    As to uploading a feature dataset, I don't think that's possible.

    What is possible is to essentially hard-code your final scores. The problem is that there are 122112 scores, so you cannot just copy-and-paste the scores, the notebook will be too large and the grader won't accept it. The following approach works to get around that limitation, but it takes a bit of work on figuring out the details (and no, I won't provide them). Here you go: You compute the scores of your submission in your notebook on your computer, limit the scores to 3 digits (or whatever is reasonable), transform the score list into a single (very long) string, compress that string, and finally uuencode it. Print that out. All this is in a workbook you will not submit.

    Now create your function in the notebook that you want to submit and hardcode and assign that uuencoded compressed value to a string variable, then uudecode and decompress, and parse into a list of numbers, that gives you the scores. Then you read the provided data file future_connections, filter to the right records, get the index and add that index to your scores, and return that series. It took me a while to figure that out, but it does work.

    If you really want to go down that road you can, but you might be better off to look at your code, make it as efficient as you can, and submit at a less busy hour. My submission has always been successfully graded (as long as I didn't have an error of my own, which did happen a few times).


## Solution

### Part 1 - Random Graph Identification

    P1_Graphs is a list containing 5 networkx graphs. Each of these graphs were generated by one of three possible algorithms:
    + Small World with low probability of rewiring ('SW_L')
    + Small World with high probability of rewiring ('SW_H')
    + Preferential Attachment ('PA')

    ```python
    import networkx as nx
    import pandas as pd
    import numpy as np
    import pickle

    P1_Graphs = pickle.load(open('A4_graphs','rb'))
    # [<networkx.classes.graph.Graph at 0x1b8f547f5c0>,
    #  <networkx.classes.graph.Graph at 0x1b8f547f780>,
    #  <networkx.classes.graph.Graph at 0x1b8f547f7b8>,
    #  <networkx.classes.graph.Graph at 0x1b8f547f7f0>,
    #  <networkx.classes.graph.Graph at 0x1b8f547f828>]

    def graph_identification(debug=False):

        # Your Code Here
        if debug:
            print(nx.info(P1_Graphs[0]), "\n")
            print(nx.info(P1_Graphs[1]), "\n")
            print(nx.info(P1_Graphs[2]), "\n")
            print(nx.info(P1_Graphs[3]), "\n")
            print(nx.info(P1_Graphs[4]), "\n")

            for cnt, G in enumerate(P1_Graphs):
                print("G[{}]: Avg. clu= {}\n      Avg. SPL= {}".format(
                    cnt, nx.average_clustering(G), nx.average_shortest_path_length(G)))

        return ['PA', 'SW_L', 'SW_L', 'PA', 'SW_H']       # Your Answer Here

    graph_identification(True)

    # Name: barabasi_albert_graph(1000,2)
    # Type: Graph
    # Number of nodes: 1000
    # Number of edges: 1996
    # Average degree:   3.9920 
    # 
    # Name: watts_strogatz_graph(1000,10,0.05)
    # Type: Graph
    # Number of nodes: 1000
    # Number of edges: 5000
    # Average degree:  10.0000 
    # 
    # Name: watts_strogatz_graph(750,5,0.075)
    # Type: Graph
    # Number of nodes: 750
    # Number of edges: 1500
    # Average degree:   4.0000 
    # 
    # Name: barabasi_albert_graph(750,4)
    # Type: Graph
    # Number of nodes: 750
    # Number of edges: 2984
    # Average degree:   7.9573 
    # 
    # Name: watts_strogatz_graph(750,4,1)
    # Type: Graph
    # Number of nodes: 750
    # Number of edges: 1500
    # Average degree:   4.0000 
    # 
    # G[0]: Avg. clu= 0.03167539146454044
    #       Avg. SPL= 4.099161161161161
    # G[1]: Avg. clu= 0.5642419635919628
    #       Avg. SPL= 5.089871871871872
    # G[2]: Avg. clu= 0.4018222222222227
    #       Avg. SPL= 9.378702269692925
    # G[3]: Avg. clu= 0.03780379975223251
    #       Avg. SPL= 3.1048046283934134
    # G[4]: Avg. clu= 0.0033037037037037037
    #       Avg. SPL= 5.0785509568313305
    # ['PA', 'SW_L', 'SW_L', 'PA', 'SW_H']
    ```


### Part 2 - Company Emails

The network also contains the node attributes Department and ManagmentSalary.

Department indicates the department in the company which the person belongs to, and ManagmentSalary indicates whether that person is receiving a managment position salary.

```python
G = nx.read_gpickle('email_prediction.txt')

print(nx.info(G))
# Name: 
# Type: Graph
# Number of nodes: 1005
# Number of edges: 16706
# Average degree:  33.2458
```

+ Part 2A - Salary Prediction

    Using network G, identify the people in the network with missing values for the node attribute ManagementSalary and predict whether or not these individuals are receiving a managment position salary.

    ```python
    def salary_predictions(debug=False):

        # Your Code Here
        from sklearn import preprocessing
        from sklearn import metrics
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        # generate dataset
        df = pd.DataFrame(index=G.nodes())

        df['ManagementSalary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
        df['Department'] = pd.Series(nx.get_node_attributes(G, 'Department'))

        # Assign different measurements into dataframe
        df['degree'] = pd.Series(nx.degree_centrality(G))
        df['clustering'] = pd.Series(nx.clustering(G))
        df['closeness'] = pd.Series(nx.closeness_centrality(G))
        df['betweenness'] = pd.Series(nx.betweenness_centrality(G))

        # Split the dataframe into two sub-dataframes: df_train & df_pred
        df_train = df.dropna()
        df_pred = df[np.isnan(df['ManagementSalary'])]

        selected_attrs = ['Department', 'degree', 'closeness', 'betweenness', 'clustering']
        # selected_attrs = ['degree', 'closeness', 'betweenness', 'clustering']

        X = df_train[selected_attrs]
        y = df_train['ManagementSalary']

        # split into training and test data w/ given training dataframe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale transformation
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_pred = scaler.transform(df_pred[selected_attrs])

        # Traing data with Logistic Regression Model
        clf = LogisticRegression()

        clf.fit(X_train, y_train)

        # check with training dataset with various metrics
        if debug:
            # predict the X_test
            pred1 = clf.predict(X_test)
            pred2 = clf.predict_proba(X_test)

            print("\nAccuracy: {}".format(metrics.accuracy_score(y_test, pred1)))
            print("\nROC_AUC: {}".format(metrics.roc_auc_score(y_test, pred2[:, 1])))
            print("\nConfusiion matrix: \n{}".format(metrics.confusion_matrix(y_test, pred1)))
            print("\nCalssification report: \n{}".format(metrics.classification_report(y_test, pred1)))

        # Generte result for autograder
        pred = clf.predict_proba(X_pred)

        rlt = pd.Series(data=pred[:, 1], index=df_pred.index)

        return rlt # Your Answer Here

    salary_predictions(debug=True)
    # Accuracy: 0.9337748344370861
    # 
    # ROC_AUC: 0.9355311355311355
    # 
    # Confusiion matrix: 
    # [[130   0]
    #  [ 10  11]]
    # 
    # Calssification report: 
    #              precision    recall  f1-score   support
    # 
    #         0.0       0.93      1.00      0.96       130
    #         1.0       1.00      0.52      0.69        21
    # 
    # avg / total       0.94      0.93      0.92       151
    # 
    # 1       0.067045
    # 2       0.733487
    # 5       0.999975
    # 8       0.123629
    # 14      0.206789
    #           ...   
    # 
    # 1000    0.037435
    # 1001    0.092543
    # Length: 252, dtype: float64
    ```

+ Part 2B - New Connections Prediction

    For the last part of this assignment, you will predict future connections between employees of the network. The future connections information has been loaded into the variable future_connections. The index is a tuple indicating a pair of nodes that currently do not have a connection, and the Future Connection column indicates if an edge between those two nodes will exist in the future, where a value of 1.0 indicates a future connection.

    ```python
    future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})
    future_connections.head(10)
    # Future Connection
    # (6, 840)      0.0
    # (4, 197)      0.0
    # (620, 979)    0.0
    # (519, 872)    0.0
    # (382, 423)    0.0
    # (97, 226)     1.0
    # (349, 905)    0.0
    # (429, 860)    0.0
    # (309, 989)    0.0
    # (468, 880)    0.0
    ```

    Using network G and future_connections, identify the edges in future_connections with missing values and predict whether or not these edges will have a future connection.

    ```python
    def new_connections_predictions(debug=False):

        # Your Code Here
        from sklearn import preprocessing
        from sklearn import metrics
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        # Preprocessing to add various Link Prediction measures
        # Not all of them used due to exution time limitation
        future_connections['comm_neigh'] = [len(list(
            nx.common_neighbors(G, edge[0], edge[1]))) for edge in future_connections.index]
        future_connections['jaccard'] = [list(
            nx.jaccard_coefficient(G, [edge]))[0][2] for edge in future_connections.index]
        future_connections['adamic_adar'] = [list(
            nx.adamic_adar_index(G, [edge]))[0][2] for edge in future_connections.index]
        future_connections['res_alloc'] = [list(
            nx.resource_allocation_index(G, [edge]))[0][2] for edge in future_connections.index]
        future_connections['pref_attach'] = [list(
            nx.preferential_attachment(G, [edge]))[0][2] for edge in future_connections.index]

        # split whole dataset into training and predict datasets via NaN in Future Connection column
        df_train = future_connections.dropna()
        df_pred = future_connections[np.isnan(future_connections['Future Connection'])]

        # Feature selection
        selected_attrs = ['comm_neigh', 'jaccard', 'adamic_adar', 'res_alloc', 'pref_attach']
        # selected_attrs = ['comm_neigh', 'adamic_adar', 'pref_attach']

        # Processing training dataset into X and y datasets
        X = df_train[selected_attrs]
        y = df_train['Future Connection']

        # Geneerate model model training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Traing data with Logistic Regression Model
        clf = LogisticRegression()

        clf.fit(X_train, y_train)

        if debug: 
            # predict the X_test
            pred1 = clf.predict(X_test)
            pred2 = clf.predict_proba(X_test)

            print("\nAccuracy: {}".format(metrics.accuracy_score(y_test, pred1)))
            print("\nROC_AUC: {}".format(metrics.roc_auc_score(y_test, pred2[:, 1])))
            print("\nConfusiion matrix: \n{}".format(metrics.confusion_matrix(y_test, pred1)))
            print("\nCalssification report: \n{}".format(metrics.classification_report(y_test, pred1)))

        # Generate prediction data
        X_pred = df_pred[selected_attrs]

        # Generte result for autograder
        pred = clf.predict_proba(X_pred)

        rlt = pd.Series(data=pred[:, 1], index=df_pred.index)

        return rlt # Your Answer Here

    new_connections_predictions(True)
    # Accuracy: 0.9577162979240312
    # 
    # ROC_AUC: 0.9071000899915729
    # 
    # Confusiion matrix: 
    # [[66752   619]
    # [ 2479  3417]]
    # 
    # Calssification report: 
    #             precision    recall  f1-score   support
    # 
    #         0.0       0.96      0.99      0.98     67371
    #         1.0       0.85      0.58      0.69      5896
    # 
    # avg / total       0.95      0.96      0.95     73267
    # 
    # (107, 348)    0.037508
    # (542, 751)    0.013362
    # (20, 426)     0.585797
    #                 ...   
    # (555, 905)    0.013212
    # (75, 101)     0.020787
    # Length: 122112, dtype: float64
    ````

