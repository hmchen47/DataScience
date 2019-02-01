# Module 2 Assignment

## Assignment Notebooks

+ [Launch Notebook Web Page](https://www.coursera.org/learn/python-social-network-analysis/notebook/7RsKp/assignment-2)
+ [Web Notebook](https://bajwjsbbpcxhnmzzoyjrrp.coursera-apps.org/notebooks/Assignment%202.ipynb)
+ [Local Notebook](notebooks/Assignment02.md)


## Discussion Forum

### [Announcement for NetworkX versions](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/Ne9qldicEeeBthLtuqY5zA)

+ Ruihan Wang init

    Many learners reported that they encountered problems when they were using NetworkX 2.0. NetworkX 2.0 was releases very recent with a lot of adjustments and improvements. We are working on updating our course materials to migrate from v1.0 to v2.0.

    For now, we highly recommend you use v1.11 to pass this course smoothly because the grader is using this NetworkX version to execute your code for assignment. We will update the grader as well as other course materials like videos and quizzes at the same time to ensure the content consistency.

    You can use “pip install networkx==1.11” to install or up/down-grade your NetworkX version.

### [Assignment 2 updated version (v1.2) is released](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/rNS4sZ2GEeeImxLZqEy9Mg)

+ Ruihan Wang init

    We just revised the wording of Question 11 of Assignment 2 to make it clear. The question has been reworded to the following (The major differences are marked in bold below).

    Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?

    How many nodes are connected to this node?

    This function should return a tuple (name of node, number of satisfied connected nodes).

    Please note that this modification aims to direct you to figure out the correct answer that is consistent with the desired result in grader. If you have already passed the grader for this question, you don't need to resubmit it.

    Learners who opened this assignment first time will directly see the latest version. For those who have viewed previous versions and want to get the latest version, you may need to reset your notebook. Please go to "Resources"-"Jupyter Notebook FAQ Page"-"6. How do I reset my notebook?" if you have no idea on how to do that.


### [Some Important notes: Clear your previous outputs before submission](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/8yJdjZsaEee3MA7n8NBemA)

+ Ruihan Wang init

    We just noticed that some learners submitted notebooks with previous outputs remained in the notebook files. Sometimes these submissions will be graded incorrectly due to some unknown issues. So We highly recommend you clear all your previous outputs before submission and not call your functions outside the function definition scope. Here is an example.

    ```python
    def answer_two():
        G = answer_one() # It is fine to call other questions' functions within the def scope 
        # Some code for this question
        return # Your answer here 

    answer_two() # Try Not to call it outsied the def scope in your submission, the grader will do it for you
    ```

    Here are some important notes for you to make you pass the autograder successfully:

    1. To clear previous outputs before submission, click "Kernel"-"Restart & Clear outputs", assuming there is no error of your code.

    2. Make sure that you make some specific statements commented out. These statements are clearly noted with instructions. If you don't comment these lines of code (like plot function), you may not pass the grader successfully.

    3. No errors of code. Before submitting the notebook to the grader, Try "Kernel"-"Restart & Run all" to see whether any error is raised. This process may avoid you calling variables stored on your local notebook but cannot be recognized by the grader.

    4. Check the data type of your answer for each question, and ensure they are consistent with the requirements. For example, you need to return sets instead of lists for Q9 and Q10. All node names of assignment 2 should be strings.


### [Week 2 Question about Q12 (solved)](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/8FTSIgD8EemxYQougy73TA)

+ Yang Ruijing init

    I got all other questions right except for Q12.

    I use the function `nx.node_connectivity(G_sc, center_node, Q11_node)`, but it just doesn't work. I also change the order of the two nodes in this function, but it doesn't work either. Could someone help me with that?

    Thanks so much.

    [Uwe: I edited the title so other learners might find this more easily]

+ Uwe F Mayer reply

    nx.node_connectivity really ought to give the right answer. The issue might be how this function handles the case when the nodes that are specified are having a direct edge between them. This case is strictly speaking not an allowed case if you think about it. If two nodes have a direct edge, then no matter how many other nodes you are removing they still will have a path.

    As it turns out for Question 12 in the assignment there unfortunately is a direct edge between a center node and the edge node from Question 11, and hence the question is actually a nonsensical question. This exception situation apparently is handled differently by node_connectivity() and minimum_node_cut() [which is another approach for getting the answer], with node_connectivity() returning a value 1 larger that the size of the set returned by minimum_node_cut().

    That is, you might want to try: `return nx.node_connectivity(G_sc, center_node, Q11_node) - 1`


### [Problem with Q11](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/qnR5GGo5Eei4mw7zD8vsPA)

+ Anna Korzhenko init

    This is my algorithm of this question:

    1. Use G_sc
    2. Use diameter from Q8
    3. Use shortest_path()
    4. I create a dictionary. Go through to each node. If the short path is equal to the diameter, I increase the counter of a particular node and save it in the dictionary.
    5. In the dictionary I take the node with the maximum number of values returns a tuple ('number of node', sum of short path with equal to the diameter) What am I doing wrong?
    
    I can send my code for review. All previous assignments are correct. here are no extra cells.

+ Uwe F Mayer reply

    Anna, in step 4, for each node in your dictionary you need to go through each node in G_sc and find the length of the shortest path, and then compare to the diameter. That is, you need two loops. Not sure if that's what you are doing. Also, when you get the shortest path for a pair of nodes from shortest_path() you get a list of nodes that include the source and the target node, and len(shortest_path()) is NOT the length of the path because it counts the nodes and not the edges (you'll have to subtract 1 to get the length of the path). Does any of that help?

### [Q11: Potential clarification](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/n0KYdaR1EeeF0Qp3MGZafg)

+ David O'Donnell init

    I'm using 2017.09.22a. version of the Auto Grader.

    I know the question was reworded as follows:

    Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?

    How many nodes are connected to this node?

    This function should return a tuple (name of node, number of satisfied connected nodes).

    My understanding of the question and also from reading the forum posts is:

    We calculate the shortest path length for all node combinations. I used nx.shortest_path_length, flattened the result and put it in a DataFrame.
    Filter the full list of source/target/path_length to remove any node combinations where the path length is not equal to the diameter (i.e. nodes considered to be on the periphery of the network. I filtered the DataFrame based on path_length == Diameter
    Select the source node that has the greatest number of connections (from the reduced list). I summarised my DataFrame based on n1 to give me a row count and selected n1 with the highest row count.
    Return the node name and the count of connected nodes that are connected by a shortest path length equal to the diameter (returned as a tuple).
    The node is '97' which from looking at the forums appears to be correct, and I've double checked the results returned by shortest_path_length for nodes connected to '97' with path length equal to the diameter (i.e. 3).

    I also tried an alternative which was to return a count of all the connected nodes (not just those with path length = 3) as the question above says "How many nodes are connected to this node?"...but I figure this is not correct either as the wording also says "the number of satisfied connected nodes" and I assume "satisfied" means satisfies the conditions stated although this wasn't entirely clear.

    If someone could provide help/guidance I would really appreciate it.


### [Q11 wording still unclear](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/bM6wEMX6EeeVSBJr3SXqFA)

+ Niklas Fricke init

    The autograder does not give me points, and I'm not sure whether I understand the task correctly. I have two questions in particular:

    1. Does 'other nodes' refer to nodes in G_sc or in all G?
    2. Does 'How many nodes are connected to this node?' ask for the number of neighbors or the number of nodes connected to the node in question via a directed (in or out?) path?

    Any help appreciated,

+ Ruihan Wang reply

    1. "other nodes" refers to nodes in G_sc
    2. "this node" is the node you found in the previous step. "How many nodes are connected" is asking you to give the exact number of nodes are connecting to "this node" which satisfy the condition in the previous step.

+ Alberto Ramírez reply

    The problem with this question, that also took me time to figure out by having to look at the forums to understand it, is that

    "How many nodes are connected to this node?"

    can easily be interpreted to mean the degree (as the definition of a network degree of a node in graph theory is the number of connection it has to other nodes, at least in this course and when I took network theory in graduate econ).

    I would suggest that this be clarified to something like:

    "How many instances does this node connect to other nodes where the path to those nodes is equal to the diameter?"


### [Q11 - help in logic](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/qBlTpgtNEem_CRLe3EGozA)

+ Lalitha Priyadarshini Uppaluri reply

    This is what i followed -
    ```python
    nx.diameter(G_sc)
    s=nx.shortest_path_length(G_sc)
    ```
    For above, I converted that to a dataframe and kept only those cells that had value==diameter==3

    On Above i did & groupby and found number of nodes connected to it.

    Where am i going wrong in logic? 

+ Uwe F MayerMentor reply

    Question is how you converted the dictionary s to a dataframe. You might want to print out s, and then print out the dataframe, and check if they contain what you think they should contain. Then there's the question on how you computed distance. For example, the shortest path from node 25 to node 11 is [25,11]. That's a path with 2 nodes of length 1. For this case, would your code compute length=2 or length=1? These are just starting points for your debugging exercise.

    Please let us know if this helped, and what you find. And if it didn't help, ask again, but please respect the Honor Code and don't post your code just as your original post did (and thanks for that!).

### [Q11 in assignment 2 - incorrect wording of question](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/0RPFk5lAEeeKqQqmdZSVhg)

+ Vivek Krishnan init

    The question in Q11 in assignment 2 states:

    Which node in G_sc has the most shortest paths to other nodes whose distance equal the diameter of G_sc?

    How many of these paths are there?

    Clearly, the question is asking learners to find the number of shortest paths for the node (let's call it node A) that has the most shortest paths to other nodes with distance=diameter. In the given graph, there are multiple shortest paths that have the same length between two nodes. For example, if node A is connected to node B through a shortest path A->C->D->B and is also connected to the node B through another shortest path A->E->F->B, that counts as two shortest paths from node A to B.

    However, the autograder is not grading this question on this basis, it is not expecting the count of shortest paths as the second part of the answer.

    Instead, the autograder expects the learner to return the number of nodes that are connected to node A that are at distance = diameter from node A.

    So in the above example, though there are 2 shortest paths from node A to B, the autograder's basis is to only count it once ie. count the number of such connected nodes and not the number of shortest paths.

    The autograder passes Q11 only if you return the node name and the number of nodes as mentioned above, and fails Q11 if you return the node name and the number of shortest paths to all such nodes from node A.

    Q11 should ideally then be rephrased as:

    Which node in G_sc is connected to the most number of nodes that are at a distance equal to the diameter of G_sc.

    How many such nodes are connected to this node?

    Otherwise, for the current content of the question, the autograder should be grading answers on the basis of the total number of shortest paths and not the total of number of nodes as mentioned above.

+ Victor Garzon reply

    Hi Vivek,

    Many thanks for your hint! Like you, I interpreted the question as asking for the sum of counts of shortest paths for all pairs of nodes (with distance equal to the diameter). But the grader is expecting the count of node pairs (not the sum of paths).

    I collected the number of shortest paths (for node pairs with distance equal to the diameter) in a data frame with columns node1, node2 and number of shortest paths. In my first attempt, I returned the aggregated sum of the third column for the most frequent node, but the grader did not accept it. After reading your hint I went back and submitted the count of rows and the grader accepted it.

+ Ruihan Wang reply

    "How many paths are there" means you need to give the exact value why you measure the node you found for the first half of this question has "the most shortest paths"


### [Q11 Steps / Pseudo Code (Vectorized / no for loops)](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/2/threads/EJ0A7NhTEeiaaRKj2tsbjA)

+ Niccolo Alexander Hamlin init

    Hi all, just wanted to help out some since I notice a lot of hints about q11 use for loops. You can get the correct results using numpy arrays / pandas without using any for loops.

    Here's how:

    1. G_sc = *assign the graph*
    2. diameter = *assign the graph diameter*
    3. p = *get shortest paths between all nodes* #(one nx function returns a nested dictionary)
    4. df = pd.io.json.json_normalize( *parameter*).*further processing* #(flatten the dictionary into a dataframe and process it a bit to make the following steps easier to implement)
    5. pairs = df.*column as string*.extract(*regex*) #(the output from step 4 combines the source node and target node for each pair into one column/index. You now want to create a new dataframe that separates this value into the two appropriate columns)
    6. df = pd.concat(*code*) #(combine your two dataframes)
    7. df['len'] = *code* #(now get the path lengths for each row)
    8. df = df[*code*] #(query the dataframe to return only the rows with the correct source node and path length
    9. return ans1, ans2 #(Hope this helped!)




## Solution






