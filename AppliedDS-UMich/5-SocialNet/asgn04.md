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
    Training Score :	 0.897009966777
    Test Score :		   0.913907284768
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




















## Solution



