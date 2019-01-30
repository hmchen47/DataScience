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

+ Distance
    + How “far” is node A from node H?
    + Are nodes far away or close to each other in this network?
    + Which nodes are “closest” and “farthest” to other nodes?
    + We need a sense of distance between nodes to answer these questions
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/2/"> <br/>
        <img src="https://lh3.googleusercontent.com/4Ux0yK97oXdtmwr9he1a3l4a_-baBRGknIrMXZrgem9kC0SchN9q4b2d1W1g9XwKGWnZ7MQKvE8J6jlFr0Lil1EeKkOJOeC5D-qLOmXHjya7BgG2Cz-YeV5yuO1oYR1f8x41Z_jVsw=w2400" alt="So for example in this network that you see here how far is node A from node H? Or we'd like to know, for example, are some nodes far away from each other and other nodes close to each other in general in this network? And if so, which nodes are closest and which nodes are the furthest away from each other in the network? To answer all these questions, we need to develop a concept of distance between nodes. And that's what we're going to do in this video today. " title="Example graph for distance" height="200">
        <img src="images/m2-05.png" alt="the first concept we need is the concept of a path. A path is simply a sequence of nodes that are connected by an edge. So for example, we can find paths that go from the node G to the node C. Here's the path G-F-C. And you can find different paths so for example here's the path G-F-E-C." title="Graph for path definition" height="200">
        <img src="images/m2-06.png" alt="And to define the distance between two nodes, we're going to define it to be the length of the shortest possible path between the two nodes. So going back to the question of what is the distance between node A to node H, the answer is four, because the shortest path between A and H has four hops or has length 4. In network X, you can use the function shortest_path to find a distance from any node to any other node. So here in this example finding the distance between node A and H in the graph G which is the graph that you see here. And so here you get the shortest path between A and H. If you're interested in just the length of this path then you can use the function shortest_path_length, and this gives you the length of this path which is four." title="Graph for Distance calculation" height="200">
    </a>

+ Paths
    + Path: A sequence of nodes connected by an edge.
    + Find two paths from node G to node C:
        ```
        G – F – C
        G – F – E – C
        ```

+ Distance
    + How far is node A from node H?
        + Path 1: A – B – C – E – H (4 “hops”)
        + Path 2: A – B – C – F – E – H (5 “hops”)
    + __Path length__: Number of steps it contains from beginning to end.
        + Path 1 has length 4, Path 2 has length 5
    + Distance between two nodes: the length of the shortest path between them.
    + The distance between node A and H is 4
        ```python
        nx.shortest_path(G, 'A', 'H')
        # ['A', 'B', 'C', 'E', 'H']

        nx.shortest_path_length(G, 'A', 'H')
        # 4
        ```
    + Finding the distance from node A to every other node.
    + Easy to do manually in small networks but tedious in large (real) networks.

+ Breadth-First Search
    + __Breadth-first search__: a systematic and efficient procedure for computing distances from a node to all other nodes in a large network by “discovering” nodes in layers.
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/2/"> <br/>
        <img src="https://lh3.googleusercontent.com/xujB_jNGOn4bT9MpgY11o5VmvkBGixBHohBAQQtvGpNJ49uGz0lPEmpa_opT968hTt8jUtNbCF9wRraOo4zKge4h5IQkrn-iUFoZyjrYkInMow7ntPd6l_IRDHiwi_lQj4HSR5bAEg=w2400" alt="Example of how breadth-first search works. So here we have the network and we're interested in figuring out the distance from node A to all the other nodes in the network. So what we're going to do is we're going to start at A and we're going to start discovering new nodes as we kind of walk through this network. And we're going to be writing down all the nodes that we discover. So we start at A and we sort of process the node A by looking at who is connected to A. In this case, K and B are connected to A and so those are going to be a distance one away because they're the shortest path from each one of those nodes to A it's just one hop, right? A path of length one. Okay, so now we're going to process each one of the newly discovered nodes and ask which nodes are connected to this newly discovered node that we haven't discovered yet? And those nodes are going to be assigned to the next layer. So let's say we process node B. Node B is connected to K, A and C. But we've already discovered nodes A and K, so the only node that we discover here is node C. Now we're going to process node K, and node K is connected to node A and B, but we've already discovered both of those. So the only newly discovered node is node C and it's a distance two away from A. Now we process node C which is connected to B, F, and E. And here we've already discovered B so the only two nodes that we discover are F and E and those are a distance three away from A. Okay, now we're going to process node E. Okay, node E has five connections and out of those five, C and F we already discovered. So the only new ones are the other three which are D, I and H. So we assign those to the next layer. Now we process node F which is connected to three nodes G, C and E. But the only one we haven't discovered yet out of all those is G so I want this to get assigned to the next layer. And all of those nodes are a distance four away from A. Okay, now we have to process each one of those newly discovered nodes. And by now you can see that we're already almost done here. So let's process node D which is only connected to E. But we've already discovered E so D does not discover any new nodes. Now let's go with I. I is connected to E and J. And we haven't discovered J yet, so this one it's assigned to the next layer. Next we process H which is only connected to E but we already discovered E. And finally, we process G which is connected to F which you've already discovered. So, J is a distance five away. " title="Breadth-First Search" height="250">
    </a>
        ```python
        T = nx.bfs_tree(G, 'A')

        T.edges()
        # [('A', 'K'), ('A', 'B'), ('B', 'C'), ('C', 'E'), ('C', 'F'), ('E', 'I'), ('E', 'H'), ('E', 'D'), ('F', 'G'), ('I', 'J')]

        nx.shortest_path_length(G,'A')
        # {'A': 0, 'B': 1, 'C': 2, 'D': 4, 'E': 3, 'F': 3, 'G': 4, 'H': 4, 'I': 4, 'J': 5, 'K': 1}
        ```

+ Distance Measures
    + How to characterize the distance between all pairs of nodes in a graph?
    + __Average distance__ between every pair of nodes.
        ```python
        nx.average_shortest_path_length(G)    # 2.52727272727
        ```
    + __Diameter__: maximum distance between any pair of nodes.
        ```python
        nx.diameter(G)  # 5
        ```
    + The __Eccentricity__ of a node `n`: the largest distance between `n` and all other nodes.
        ```python
        nx.eccentricity(G)
        # {'A': 5, 'B': 4, 'C': 3, 'D': 4, 'E': 3, 'F': 3, 'G': 4, 'H': 4, 'I': 4, 'J': 5, 'K': 5}
        ```
    + The __radius__ of a graph: the minimum eccentricity
        ```python
        nx.radius(G)        # 3
        ```
    + The __Periphery__ of a graph: the set of nodes that have eccentricity equal to the diameter.
        ```python
        nx.periphery(G)     # ['A', 'K', 'J']
        ```
    + The __center__ of a graph: the set of nodes that have eccentricity equal to the radius.
        ```python
        nx.center(G)        # ['C', 'E', 'F']
        ```
    + IVQ: 
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/SeNEl/distance-measures"> <br/>
            <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/2g5_LX6rEeef4BI7KaessA_aef80d415fc6b75dbafe7a0d75c0383f_Screen-Shot-2017-08-11-at-11.10.03-AM.png?expiry=1548979200000&hmac=RK0tsfWkbJxAn1DO73zp5baw2Mk97roMSHYIjvn__4o" alt="IVQ Graph for diameter" title="IVQ graph for diameter" height="200">
        </a>

        + What is the diameter of this network?

            Ans: 3 <br/>
            The longest distance between any two nodes is 3. For example, the distance from node A to C is 3 (path A-H-G-C). Hence, the diameter of the network is 3.
        + What is the eccentricity of node F? 

            Ans: 2
            The distance from F to every other node is either 1 or 2. Hence the eccentricity of node F is 2.
        + Which node is in the periphery of the network?

            a. B
            b. H
            c. B & H
            d. none of above

            Ans: a
            The diameter of the network is 3 and the distance from B to C is 3 (path B-D-G-C). Hence, node B is in the periphery. The maximum distance from node H to any other node is 2, so H is not in the periphery.

+ Karate Club Network
    ```python
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G,first_label=1)
    ```
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/SeNEl/distance-measures"> 
        <img src="images/m2-07.png" alt="So, let's run an example in this using the Karate Club Network, which we had seen in a previous video. So, this is a network of friendship in a Karate Club. And as you may remember, the story here, is that node one is the instructor of the Karate Club, and this node 34 is an assistant, and they have some type of dispute, they're not friends with each other, and so the club actually splits into two groups, and sort of, this is the separation of the two groups. So one, this set of students on the left go with one of the instructors or with the assistant and the other ones go with the original instructor. So, if we take this network and apply the definitions about distances that we just covered, we can discover how far nodes are from each other and who's central and who's not. So, let's begin by loading up this network. This network is so famous that actually on network X you can simply load it by using the function karate club graph. So, that one returns this particular graph. Now, I'm converting the nodes labels to be integers, so that they match the figure I have here. So, that's what I'm doing with that command there, and then I could ask different questions about the network. So, in this case, the average shortest path between the nodes is about 2.41. The radius of the network is three, and the diameter is five. So, meaning there's a pair of nodes here, there are distance far away from each other and that's the largest that a distance can be. And then, we're going to ask who's at the center of this network? So, here the nodes are in the center, and here, I'm highlighting them in red. So, as you can see the instructor is in the center and all the other nodes are in the center also connected to the instructor, and they also tend to have high degrees, so they are easily connected to many other high degree nodes, and they just have small distances from them to all the other nodes in the network. Now, when you look at the periphery, these are the peripheral nodes, and I'm highlighting them here and in blue and as you can see, they're kind of on the outside, they tend to have small number of connections, and none of them are actually connected to the instructor. Now, you might look at this and say, okay, this make sense. But, for example, this node 34, was the assistant here, he seems pretty central. He's connected to a bunch of nodes, it seems like he could be close to all the other nodes in the graph as well. Why is 34 not showing up in the center? Well, it turns out that if you look carefully, node 34 has a distance four to node 17, right? To get from 34 to 17, you have to go 34, 32, 16, and 17, and so, it couldn't be in the center because the radius of the graph is three and this one has a node that is the distance four away from it. Now, it turns out that actually if this node 17 was just a bit closer, for example, if this node 17 was a distance three away from 34, then 34 would actually be in the center, because 34 is a distance at most three to every other node in the network. And so, this shows that this definition of center is pretty sensitive to just one node that happens to be far away." title="Friendship network in a 34-person karate club" height="200">
    </a>

    + Average shortest path = 2.41
    + Radius = 3
    + Diameter = 5
    + Center = [1, 2, 3, 4, 9, 14, 20, 32]
    + Periphery: [15, 16, 17, 19, 21, 23, 24, 27, 30]

    Node 34 looks pretty “central”. However, it has distance 4 to node 17

+ Summary
    + __Distance between two nodes__: length of the shortest path between them.
    + __Eccentricity__ of a node n is the largest distance between n and all other nodes.
    + Characterizing distances in a network:
        + __Average distance__ between every pair of nodes.
        + __Diameter__: maximum distance between any pair of nodes.
        + __Radius__: the minimum eccentricity in the graph.
    + Identifying central and peripheral nodes:
        + The __Periphery__ is the set of nodes with eccentricity = diameter.
        + The __center__ is the set of nodes with eccentricity = radius.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/VVQyoJTMEeeClxLmJhEfgA.processed/full/360p/index.mp4?Expires=1548892800&Signature=Hfx0XLHeDJI9cSXdftL2~oGunpZSkhB1qJaRCYzJwRMJEbp3hkYkddTG7oa~BG8RPG1QN~0fLqN1l1K00eeKqE1f-M5FJgEJzFN3dWGbbNLUGSQJPRzjQdrOkCfvtmxiXU56WHk-p5Mg~aHnyTeDmim7RnZ0hiueak2oRBWoZe8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Distance Measures" target="_blank">
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




