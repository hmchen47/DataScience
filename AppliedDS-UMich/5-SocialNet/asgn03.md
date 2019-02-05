# Module 3 Assignment

## Notebooks

+ [Launching Web Page](https://www.coursera.org/learn/python-social-network-analysis/notebook/utvmz/assignment-3)
+ [Web Notebook](https://bajwjsbbpcxhnmzzoyjrrp.coursera-apps.org/notebooks/Assignment%203.ipynb)
+ [Local Noteook](notebooks/Assignment03.ipynb)


## Discussion Forum

+ [Assignment 3 updated version (v1.2) is released](https://www.coursera.org/learn/python-social-network-analysis/discussions/weeks/3/threads/JJ6oOp2IEeeOdRJt4xIANg)

    Hi everyone,

    We just revised the instructions for Question 2, 3 and 4 and the wording of Question 4 to make Assignment 3 clear. The instructions and questions have been reworded to the following (The major differences are marked in bold below).

    For Questions 2, 3, and 4, assume that you do not know anything about the structure of the network, except for the all the centrality values of the nodes. That is, use one of the covered centrality measures to rank the nodes and find the most appropriate candidate.

    Question 4

    Assume the restriction on the voucher’s travel distance is still removed, but now a competitor has developed a strategy to remove a person from the network in order to disrupt the distribution of your company’s voucher. Your competitor is specifically targeting people who are often bridges of information flow between other pairs of people. Identify the single riskiest person to be removed under your competitor’s strategy?

    Please note that these modifications aim to direct you to figure out the correct answer that is consistent with the desired result in grader. If you have already passed the grader for this question, you don't need to resubmit it.

    Learners who opened this assignment first time will directly see the latest version. For those who have viewed previous versions and want to get the latest version, you may need to reset your notebook. Please go to "Resources"-"Jupyter Notebook FAQ Page"-"6. How do I reset my notebook?" if you have no idea on how to do that.

    Thanks a lot to learners who pointed these out.

    Best,

    Team of Applied Social Network Analysis in Python


## Solution

```python
import networkx as nx

G1 = nx.read_gml('friendships.gml')
```

+ Q1: Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.

```python
def answer_one():
        
    # Your Code Here
    deg_cent = nx.degree_centrality(G1)
    close_cent = nx.closeness_centrality(G1)
    btw_cent = nx.betweenness_centrality(G1, endpoints = False, normalized=True)
    
    return  (deg_cent[100], close_cent[100], btw_cent[100])      # Your Answer Here

answer_one()    # (0.0026501766784452294, 0.2654784240150094, 7.142902633244772e-05)
```

+ Q2: The voucher can be forwarded to multiple users at the same time, but the travel distance of the voucher is limited to one step, which means if the voucher travels more than one step in this network, it is no longer valid. Apply your knowledge in network centrality to select the best candidate for the voucher.

```python
def answer_two():
        
    # Your Code Here
    import operator
    
    deg_cent = nx.degree_centrality(G1)
    
    return sorted(deg_cent.items(), key=operator.itemgetter(1), reverse=True)[0][0]   # Your Answer Here

answer_two()        # 105
```

+ Q3: How would you change your selection strategy? Write a function to tell us who is the best candidate in the network under this condition.

```python
def answer_three():
        
    # Your Code Here
    import operator
    
    cls_cent = nx.closeness_centrality(G1)
    
    return sorted(cls_cent.items(), key=operator.itemgetter(1), reverse=True)[0][0]      # Your Answer Here

answer_three()      #23
```

+ Q4: Identify the single riskiest person to be removed under your competitor’s strategy?

```python
def answer_four():
        
    # Your Code Here
    import operator 
    
    btw_cent = nx.betweenness_centrality(G1)
    
    return sorted(btw_cent.items(), key=operator.itemgetter(1), reverse=True)[0][0]      # Your Answer Here

answer_four()       # 333
```

+ Q5: Apply the Scaled Page Rank Algorithm to this network. Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.

```python
def answer_five():
        
    # Your Code Here
    spr = nx.pagerank(G2, alpha=0.85)
    
    return spr['realclearpolitics.com']      # Your Answer Here

answer_five()       # 0.004636694781649093
```

+ Q6: Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. Find the 5 nodes with highest Page Rank.

```python
def answer_six():
        
    # Your Code Here
    import operator
    
    spr = nx.pagerank(G2, alpha=0.85)
    top5 = sorted(spr.items(), key=operator.itemgetter(1), reverse=True)[0:5]
    
    return [site for (site, score) in top5]    # Your Answer Here

answer_six()    
# ['dailykos.com', 'atrios.blogspot.com', 'instapundit.com', 'blogsforbush.com', 'talkingpointsmemo.com']
```

+ Q7: Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'.

```python
def answer_seven():
        
    # Your Code Here
    hits = nx.hits(G2)
    
    return (hits[0]['realclearpolitics.com'], hits[1]['realclearpolitics.com'])   # Your Answer Here

answer_seven()  # (0.000324355614091667, 0.003918957645699856)
```

+ Q8: Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.

```python
def answer_eight():
        
    # Your Code Here
    import operator
    
    hits = nx.hits(G2)
    
    top5 = sorted(hits[0].items(), key=operator.itemgetter(1), reverse=True)[0:5]
    
    return [site for (site, score) in top5]      # Your Answer Here

answer_eight()
# ['politicalstrategy.org', 'madkane.com/notable.html', 'liberaloasis.com',
#  'stagefour.typepad.com/commonprejudice', 'bodyandsoul.typepad.com']
```

+ Q9: Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.

```python
def answer_nine():
        
    # Your Code Here
    import operator
    
    hits = nx.hits(G2)
    
    top5 = sorted(hits[1].items(), key=operator.itemgetter(1), reverse=True)[0:5]
    
    return [site for (site, score) in top5]      # Your Answer Here

answer_nine()
# ['dailykos.com', 'talkingpointsmemo.com', 'atrios.blogspot.com',
#  'washingtonmonthly.com', 'talkleft.com']
```


