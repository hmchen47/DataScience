# Assignment 1 - Creating and Manipulating Graphs

## Notebooks

+ [Launch Notebook Web Page](https://www.coursera.org/learn/python-social-network-analysis/notebook/hvNc1/assignment-1)
+ [Web Notebook](https://bajwjsbbpcxhnmzzoyjrrp.coursera-apps.org/notebooks/Assignment%201.ipynb#Assignment-1---Creating-and-Manipulating-Graphs)
+ [Local Notebook](notebooks/Assignment01.ipynb)

## Discussion Forum

### [Issue with NetworkX: 'Graph' object has no attribute 'edge'](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/LWpy24krEeiCDgqalKfDcA)

+ Adrian Curtin

    The switch from NetworkX 1 to 2 has caused multiple incompatibility issues with the course material as well as the juypter notebook.

    For instance the example code from the documentation for Network X throws an error

    ```python
    G = nx.Graph([(1, 2), (1, 3)])

    nx.set_node_attributes(G, name='label', values={1: 'one', 2: 'two', 3: 'three'})

    nx.set_edge_attributes(G, name='label', values={(1, 2): 'path1', (2, 3): 'path2'})

    G.edges(data=True)
    ```

    Also using G.edge[1],[2] returns

    AttributeError: 'Graph' object has no attribute 'edge'

    This issue affects at least partially how you might solve HW 1 Problem 2

+ Uwe F Mayer reply

    In a nutshell, one cannot read a network 1.x graph object that was pickled to a file if one is using networkx 2.x because the graph class definition change in a not backwards compatible way. As Ruihan says in order to work on this course's assignment you need to downgrade networkx. If you use Anaconda then run:

    `conda install -y networkx=1.11`

    If you use pip then run: `pip install networkx==1.11`

    And yes, the conda command has 1 equal sign, the pip command has 2.

    Alternatively, you could do all the assignments using the online Coursera system (I realize this is not the best option for many out of various reasons),

+ Uwe F Mayer reply

    Würfel please note that you will not be able to successfully submit anything expecting networkx 2.0 or higher, that is, the autograder will not be able to run it.

    The Graph attributes G.node and G.edge have been removed in favor of using G.nodes[n] and G.edges[u, v] in the version change of networkx from 1.X to 2.0.

    For more about the version changes of networkx see the Announcement: [NetworkX 2.0](https://networkx.github.io/documentation/networkx-2.0/release/release_2.0.html) and [Migration guide from 1.X to 2.0](https://networkx.github.io/documentation/stable/release/migration_guide_from_1.x_to_2.0.html).


### [Q2 output failing, unknown reason, but type is set](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/DopcG3xwEeiYXw7jJimx_A)

+ Mike Taylor init

    Q1 is correct, (tried it both hard coded and from the df), but my Q2 answer is apparently not....

    The graph yields the following from G.nodes(data=True), without a sample of what it should look like this is intensely frustrating. Can someone show me where I'm going wrong here? I don't need algo help, but not knowing what the output should look like makes debugging all but impossible

    Also, as an aside, using type isn't great python practice as it's already assigned. Appreciate this is a string but it is still sub optimal.

    ```python
    [('Andy', {'type': 'employee'}),
    ('Anaconda', {'type': 'movie'}),
    ('Mean Girls', {'type': 'movie'}),
    ('The Matrix', {'type': 'movie'}),
    ...
    ('The Dark Knight', {'type': 'movie'}),
    ('Vincent', {'type': 'employee'}),
    ('The Godfather', {'type': 'movie'})]
    ```

+ Uwe F Mayer reply

    Mike, you might be running into an autograder issue. A while back there was a learner who assigned the type to each node in a loop. Another approach is to assign the type for all employees in one statement using add_nodes_from(), and then doing the same for the movies. Logically both approaches are obviously correct. That is, from what I could tell at the time the graphs were identical, but the autograder accepted only the second approach and not the first. Is this maybe what's going on here as well?

    On a different note, in accordance with the Honor Code please do not post the entire solution to a question. I have edited your post correspondingly.


### [DataFrame constructor not properly called](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/IsxQ0ADrEeiceA60MlquvA)

+ Kenny Lin init

    Here is my question:

    df = pd.DataFrame(employee_with_common_movies.edges(data=True), columns=['node1', 'node2', 'weights'])

    when I run above code, I got the following error message

    DataFrame constructor not properly called!

    my Pandas version is 0.20.3 and NetworkX is 2.0, is it version compatiable problem?

+ Uwe F Mayer reply

    Maxym, you can certainly use that function. Just make sure you know what this function does when two nodes do not have an edge, and how weights are handled.

+ Uwe F Mayer reply

    Kenny, your code used to work just fine, but apparently after a recent pandas upgrade it no longer runs. Now the object returned by the edges() call needs to be explicitly transformed into a list. Try the following:

    ```python
    df = pd.DataFrame(list(employee_with_common_movies.edges(data=True)), columns=['node1', 'node2', 'weights'])
    ```

### [Announcement for NetworkX versions](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/IRFQzdicEeeiYRK-GDlqxA)

+ Ruihan Wang init

    Hi everyone,

    Many learners reported that they encountered problems when they were using NetworkX 2.0. NetworkX 2.0 was releases very recent with a lot of adjustments and improvements. We are working on updating our course materials to migrate from v1.0 to v2.0.

    For now, we highly recommend you use v1.11 to pass this course smoothly because the grader is using this NetworkX version to execute your code for assignment. We will update the grader as well as other course materials like videos and quizzes at the same time to ensure the content consistency.

    You can use “pip install networkx==1.11” to install or up/down-grade your NetworkX version.

    Thank you for your understanding.

    Best,

    Team of Applied Social Network Analysis in Python


### [Tips for passing autograder [Updated for Week1 Q4]](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/hi7Omq3AEeerBA5wa_HeHg)

+ Ruihan Wang

    Hi everyone,

    Here are several tips for submitting assignments successfully:

    1. Make sure that you make some specific statements commented out. These statements are clearly noted with instructions. If you don't comment these lines of code (like "%matplotlib" or calling matplotlib.pyplot function), you may not pass the grader successfully. Also, Do NOT rename or move the data files provided to you. It is because only your code will be submitted to the server, you need to ensure the inconsistency between the files on the server and your local machine.

    2. No error in your code. Before submitting the notebook to the grader, Try "Kernel"-"Restart & Run all" to see whether any error is raised. This process may avoid you calling variables stored on your local notebook but cannot be recognized by the grader.

    3. Check the data type of your answer for each question, and ensure they are consistent with the requirements.

    4. Currently there are some unknown issues for some learners to pass Q4 in Assignment 1. If you believe your result is correct, you may return the value of your result directly within answer_four function to check whether you can pass the grader like this:

    ```python
    def answer_four():
        result = #the value of your result
        return result
    ```

    We are trying our best to solve this problem now. Sorry for the inconvenience.

    Best,

    Team of Applied Social Network Analysis in Python

+ Uwe F Mayer reply

    Is there any way that we can send our code to a TA, so that they may manually grade and give marks for this questions?: No

    If you have hardcoded the correct answer you will pass.

    More details: I submitted with 3 digits after decimal period, it was not accepted. I submitted with 6 and with all 17 digits, both answers were accepted. This is for grader 2017.09.22a, which is the current grader version as of 2018-11-12.

+ Uwe F Mayer reply

    That of course means your code was the problem, at least if you ask the autograder. That often happens when learners write code using a newer version of pandas and other libraries on their system as compared to the grader (which hasn't been updated since 2017), or if the learner's code is not repeatable (meaning if you call the function twice it fails). However for this specific question there really seems to be something broken with the grader, even some code running just fine on the online Coursera system fails to pass. That's why the original post of this thread shows how to hard-code the answer without using any other code in the function.

    So, I'm glad to see you got it done!

+ Uwe F Mayer

    Sang, make sure all your code is inside of the assigned functions, and that you don't write over any global variables, in case you have some. There have been lot's of learners who assign to a global variable G over and over again, this won't work. The autograder runs first the entire notebook top to bottom, and only then calls your assigned functions. You can simulate this by adding a cell at the bottom of your notebook and calling your function(s) from there. Do you get the expected answer(s)?

    For Q1 the graph should look like:
    <a href="https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/hi7Omq3AEeerBA5wa_HeHg"> <br/>
        <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H7ciPs_REeifhhLjQokuMg_793c4e184c07e52891e5df48cdd17182_Capture.PNG?expiry=1548892800000&hmac=3VmN1ElwQMGuYUfV5uPSRcAunZuYdkto_HoXWfDV9A8" alt="text" title="caption" height="200">
    </a>

    This was generated with plot_graph(answer_one()), of course you need to comment this out before submission, the grader cannot handle plotting. Also, the appearance of the nodes on the plot is somewhat random, it's the structure that matters.

+ Brian R. von Konsky reply

    <a href="https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/1/threads/hi7Omq3AEeerBA5wa_HeHg"> <br/>
        <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/AzmCPK80Eees5QofbDYJWg_fa288d683aa495c2ca658a9bc1ff3670_code.png?expiry=1548892800000&hmac=DYe7m42nIEI6QSNQhP1pQSLQEk8LzTq_wvAhb45LEW0" alt="text" title="caption" height="200">
    </a>

    There' are definitely still problems with the autograder as of 5:50PM on 12 October 2017 (GMT+8).

    When developing my solution off line, I was always careful to restart and clear output and then restart and run all before saving and submitting. There were no run time errors anywhere in the notebook.

    I convinced myself that my answer was correct by carefully checking my answer manually, and based on information supplied by peers in the discussion forum.

    As was recommended in one thread, I ended up cutting and pasting the answer that was computed by my actual solution into a "dummy" answer_four() function. Note that I verified that this approach returned the same type AND value as the actual answer_fourBeta() as shown above.

    EDIT: The dummy answer_four() passed the autograder whereas ts computed version that returns the same value and type does not.

    While I would have much preferred to find out why my original solution did not pass the autograder, it's time to get on with the rest of this class!

    I've wasted more than two full days on this :-(

    Now on to learn Applied Social Network Analysis :-)

    Brian



## Solution





