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




## Solution



