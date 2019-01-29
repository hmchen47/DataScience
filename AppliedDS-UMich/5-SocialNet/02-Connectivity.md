# Module 2: [Network Connectivity](02-Connectivity.md)

## Clustering Coefficient

### Lecture Note

+ Triadic Closure
    + __Triadic closure__: The tendency for people who share connections in a social network to become connected.
    + <n style="color:cyan">How can we measure the prevalence of triadic closure in a network?</b>

+ Local Clustering Coefficient
    + Local clustering coefficient of a node: Fraction of pairs of the node's friends that are friends with each other.
    + <n style="color:cyan">Compute the local clustering coefficient of node C</b>:

        $$\frac{\text{\# of pairs of C's friends who are friends}}{\text{\# of pairs of Cʹs friends}}$$

        + $\text{\# of C's friends} = 𝑑_c = 4$ <n style="color:red">(the "degree" of C)</b>
        + $\text{\# of pairs of Cʹs friends} = \frac{d_c (d_c - 1)}{2} = 12/2 = 6$
        + $\text{\# of pairs of Cʹs friends who are friends} = 2$

        $\text{Local clustering coefficent of C} = 2/6 = 1/ 3$
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/2/"> <br/>
        <img src="https://lh3.googleusercontent.com/bB1-lboheo4WVSapzhZH9kItMmZqjF7AbGhJSI5PCsSpNpAqlv2-4gj5JOqPW6avfjAz3pJ-UzJaBJu7znBLCeQ_afbAR_8K1S7TzotmCG-2vAFpU2F20PMOQMM3r4Iyr9JwYk_95g=w2400" alt="the way it's defined is the fraction of pairs of the nodes friends that are friends with each other. The best way to show you how Local Clustering Coefficient works is by showing you an example. So, let's say, you wanted to compute the Clustering Coefficient of node C. What you would need to do is to take the ratio of the number of pairs of C's friends who are friends with each other, and the total number of pairs of C's friends." title="Triadic closure: Node C - original" height="250">
        <img src="images/m2-01.png" alt="Okay. So, C has four friends in this network. That means that C has a degree of four. That's what we refer to as degree. It's the number of connections that a node has. And, we refer to it as DC as well. So, DC here, which the degree of C is four." title="Triadic closure: Node  - # of C's friends" height="250">
        <img src="images/m2-02.png" alt="Now, how many pairs of C's friends are there? Well, there are four friends of C, and you can easily see that, if you have four pairs of four people, then there are six total possible pairs of people. And so, the total number of pairs of C's friends is six. Now, this is easy to see because there is only four friends of C, but sometimes, there are many more and it might be harder to see how many possible pairs of friends you have. So, what you can do is you can just use this formula here which tells you how many. It's dc times dc-1 over two. In this case, that number is six, which is 12 or 2. Okay. So, that will be our denominator." title="Triadic closure: Node C - # of pairs of C's friends" height="250">
        <img src="images/m2-03.png" alt="What about the numerator? The number of pairs of friends of C who are friends with each other. Well, there are only two pairs of friends of C that are friends with each other. AB and EF. So, that number is two. So then, the Local Clustering Coefficient of node C is two over six or one-third. That means that one-third of all the possible pairs of friends of C who could be friends, are actually friends." title="Triadic closure: Node C - # of pairs of C's friends who are friends" height="250">
    </a>
    + <n style="color:cyan">Compute the local clustering coefficient of node F</b>:

        $$\frac{\text{\# of pairs of Fʹs friends who are friends}}{\text{\# of pairs of Fʹs friends}}$$

        + $d_F = 3$
        + $\text{\# of pairs of Fʹs friends} = \frac{d_F (d_F - 1)}{2} = 6/2 = 3$
        + $\text{\# of pairs of Fʹs friends who are friends} = 1$

        $\text{Local clustering coefficent of F} =1/3$
    + <n style="color:cyan">Compute the local clustering coefficient of node J</b>:

        $$\frac{\text{\# of pairs of Jʹs friends who are friends}}{\text{\# of pairs of Jʹs friends}}$$

        + \# of pairs of Jʹs friends who are friends = 0 <n style="color:red"> (cannot divide by 0)</b>

        We will assume that the local clustering coefficient of a node of degree less than $2$ to be $0$.
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/2/"> <br/>
        <img src="images/m2-04.png" alt="Okay. Let's do another example. Compute the Local Clustering Coefficient of node F. Again, we need to compute the ratio of the number of pairs of F's friends who are friends with each other, and the total number of pairs of F's friends. So, we'll do the same thing here. F has a degree of three. So, the number of pairs of F's friends is three times two over two which is three. And then, there's only one pair of friends of F who are actually friends with each other. That's C and E. And so, the Local Clustering Coefficient of F is also one-third. " title="Triadic closure: Node F" height="250"> (Node F)
        <img src="https://lh3.googleusercontent.com/bB1-lboheo4WVSapzhZH9kItMmZqjF7AbGhJSI5PCsSpNpAqlv2-4gj5JOqPW6avfjAz3pJ-UzJaBJu7znBLCeQ_afbAR_8K1S7TzotmCG-2vAFpU2F20PMOQMM3r4Iyr9JwYk_95g=w2400" alt="All right. And one last example. Compute the Local Clustering Coefficient of node J. All right. So, node J has only one friend which is node I, which means that J actually has zero pairs of friends. And, because that's what we're supposed to put in the denominator, we're in trouble because we cannot divide by zero. And so, what we're going to do for cases like this, where the definition doesn't work for nodes that have less than two friends, is we're going to assume that nodes that have less than two friends have a Local Clustering Coefficient of zero. And this is consistent with what network X does." title="Triadic closure: Niode J" height="250"> (Node J)
    </a>
    + __Local clustering coefficient in NetworkX__:
        ```python
        G = nx.Graph()
        G.add_edges_from([('A', 'K'), ('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'K'),
        ('C', 'E'), ('C', 'F'), ('D', 'E'), ('E', 'F'), ('E', 'H'), ('F', 'G'), ('I', 'J')])

        nx.clustering(G, 'F')
        # 0.3333333333333333

        nx.clustering(G, 'A')
        # 0.6666666666666666

        nx.clustering(G, 'J')
        # 0.0
        ```
    + IVQ: What is the clustering coefficent of node H? Scroll down to see all answer options
        <a href="url"> <br/>
            <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Qi1pVX6nEeenPwquAsH1BA_de80dbed2e1ebc9cb9bbb2745d0c905a_Screen-Shot-2017-08-11-at-11.10.03-AM.png?expiry=1548892800000&hmac=eMxgJUcYHvLy7qufLMNt9LRXdv0Cr3AJM0qYRcFqNvw" alt="IVQ diagram" title="Graph to compute local clustering coefficient" height="200">
        </a>

        Ans: 1/2

+ Global Clustering Coefficient
    + Measuring clustering on the whole network:
    + __Approach 1__: Average local clustering coefficient over all nodes in the graph.
        ```python
        nx.average_clustering(G)
        # 0.28787878787878785
        ```
    + __Measuring clustering on the whole network (Approach 2)__:
        + Percentage of "open triads" that are triangles in a network.
        <a href="urhttps://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/2/l"> <br/>
            <img src="https://lh3.googleusercontent.com/8v48Nnrcm7gEPBcZMpZbcsRxLEgS2DuD1pamncsVljppmbt2FuZt-ZWRDJdEBvjlqr34XoBjgWLKhzYhdorHf5uwSIGTSrOLpEgOLR-TIUSz0hQe-2APJBFeAhEOfMG24enrtuj39w=w2400" alt="A second approach, it's the following. We're going to try to measure the percentage of open triads in the network that are triangles. Okay. So, what our open triads and what are triangles? Triangles are simply three nodes that are connected by three edges. And this is called a triangle because it looks like a triangle. Now, open triads are three nodes that are connected by only two edges. The thing to notice here is that a triangle actually contains three different open triads, right? So, if we consider this triangle here, you will notice that it contains three different open triads. The first open triad considers the three nodes and all the edges, these two edges but not this one. That is the first open triad. But, you can also consider the three nodes and these two edges in this one. Or, we could consider the three nodes and these two edges but not this one. Right. So, inside each triangle, there are three different open triads. So, if you go out in the network and count how many triangles it has, and then it counts how many possible open triads it has, for each time that you see a triangle, you're going to count three different open triads. And so, what we're going to do for the second approach for measuring Clustering Coefficient, which is actually called the Transitivity, is simply that it's going to take the number of closed triads, which are the triangles, multiplied times three divided by the number of open triads. And that is the percentage of open triads that are actually triangles or close triads. You can use network X to get the Transitivity of the network by using the function Transitivity. And, in this case, this network has a Transitivity of 0.41. " title="Graph for Transitivity" height="120">
        </a>
        + __Transitivity__: Ratio of number of triangles and number of "open triads" in a network.
            ```python
            nx.transitivity(G)
            # 0.409090909091
            ```

+ Transitivity vs. Average Clustering Coefficient
    + Both measure the tendency for edges to form triangles.
    + Transitivity weights nodes with large degree higher.
        <a href="urhttps://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/2/l"> <br/>
            <img src="https://lh3.googleusercontent.com/Ud9zTHLmq4yus8004YizOi-OHbgaQju2vKC2-spllXxeOJK7rEi6hz-kzoe1f8qEEJH-pCtGOo_3tGJytRjv9uZkvfc_r60zqR5opbGmRZT6O0dHAxekmzMOdXFoJVext6HWa368yg=w2400" alt="They both try to measure the tendency for the edges to form triangles, but it turns out the Transitivity weights the nodes with a larger number of connections higher. It weights the nodes with a larger degree higher. The best way to see that is by looking at examples. So, here is this graph that kind of looks like a wheel. If you look at this graph closely, you'll find that most nodes actually have a pretty high Local Clustering Coefficient. So, all the nodes that are on the outside of the wheel have a Local Clustering Coefficient of one because, each one of these nodes, you see that it has two connections. So, he has one pair friends, and that pair friend is connected. So, this node here has a Local Clustering Coefficient of one and the same is true for all the nodes on the outside of the wheel. So, most nodes have a high Local Clustering Coefficient. However, if you consider the node inside the wheel, the central node there, that one has a pretty high degree but it has a very low Clustering Coefficient. That is because it has many, many connections in many pairs of connections and only a few of those are actually connected to each other. But, most of them are not connected. For example, these two nodes are not connected. These two nodes are not connected. These two nodes are not connected and so on. Even though all of them are friends with that central node. So, in this graph, the average Clustering Coefficient is pretty high, it's 0.93 because most nodes have a very high Local Clustering Coefficient, except for one. However, the Transitivity of this network is 0.23. And that's because Transitivity weights the nodes with high degree higher. And so, in this network, there's one node with a very high degree compared to the others that has a very small Local Clustering Coefficient compared to the others, and Transitivity penalizes that. So, you get a much lower Transitivity. Okay. Can you see an example that goes the other way around? Well, here is this network. In this network, most nodes have a very low Local Clustering Coefficient. So, each one of these outer nodes here has a Local Clustering Coefficient of zero because they either have only one friend or they have two friends but those two are not connected. And, there are 15 nodes like that. The nodes inside here, there are only five nodes like that and they have high degree, and then they have high Local Clustering Coefficient. So, when you look at the average Clustering Coefficient and Transitivity, you find that the average Local Clustering Coefficient of this network is pretty low because most nodes, the 15 outer nodes, have very low Local Clustering Coefficient. However, the Transitivity is high because the nodes with high degree happen to have high Local Clustering Coefficient. So, these two graphs showed you the differences between the Local Clustering Coefficient, the average Local Clustering Coefficient, and Transitivity. One weights the nodes with a large degree higher." title="Comparison between average clustering & transitivity" height="170">
        </a>
        + Wheel: all outer nodes w/ LCC = 1; central node w/ low LCC
        + Pentagon: all outer nodes w/ LCC = 0; central 5 nodes w/ LCC = 1

+ Summary
    + __Clustering coefficient__ measures the degree to which nodes in a network tend to “cluster” or form triangles.
    + __Local Clustering Coefficient__: Fraction of pairs of the node's friends that are friends with each other.

        $$\text{LCC of C} = \frac{2}{6} = \frac{1}{3}$$
    + __Global Clustering Coefficient__
        + __Average Local Clustering Coefficient__: `nx.average_clustering(G)`
        + __Transitivity__
            + Ratio of number of triangles and number of “open triads”.
            + Puts larger weight on high degree nodes.
            + `nx.transitivity(G)`



### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/X4oOAnxFEeeR4BLAuMMnkA.processed/full/360p/index.mp4?Expires=1548892800&Signature=jaKakh9T7ryPe9CyCEO~xnGGaXzWlDf-0I1drKoChs2UTmTm0jCxlpEwvORtsgjwfJDifeIe59za0sWRVllRqtX~NpCKko59xHth4LGjLRpqnQ7pncK~ri8iVlxMc6oDd5hE7aGgMI1A3YqH35JO6fvupWBVuKWWJa9t-s-KBow_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Clustering Coefficient" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Distance Measures

### Lecture Note



+ Demonstration
    ```python

    ```

### Lecture Video

<a href="url" alt="Distance Measures" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Connected Components

### Lecture Note



+ Demonstration
    ```python

    ```

### Lecture Video

<a href="url" alt="Connected Components" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Network Robustness

### Lecture Note



+ Demonstration
    ```python

    ```

### Lecture Video

<a href="url" alt="Network Robustness" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Notebook: Simple Network Visualizations in NetworkX

### Lecture Note



+ Demonstration
    ```python

    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## TA Demonstration: Simple Network Visualizations in NetworkX

### Lecture Note



+ Demonstration
    ```python

    ```

### Lecture Video

<a href="url" alt="TA Demonstration: Simple Network Visualizations in NetworkX" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Quiz: Module 2 Quiz




