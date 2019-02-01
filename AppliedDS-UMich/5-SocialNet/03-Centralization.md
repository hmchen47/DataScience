# Module 3: Influence Measures and Network Centralization

## Degree and Closeness Centrality

### Lecture Notes

+ Node Importance
    + Based on the structure of the network, which are the 5 most important node in the Karate Club friendship network?
    + Different ways of thinking about “importance”.
        + Ex. Degree: number of friends. 5 most important nodes are: 34, 1, 33, 3, 2
        + Ex. Average proximity to other nodes. 5 most important nodes are: 1, 3, 34, 32, 9
        + Ex. Fraction of shortest paths that pass through node. 5 most important nodes are: 1, 34, 33, 3, 32
    <a href="https://anthonybonato.com/2016/04/13/the-mathematics-of-game-of-thrones/"> <br/>
        <img src="https://lh3.googleusercontent.com/OQqUIVdAO_KrEiIsfGN4mARt24rHxQzWZ9IndHfY3DEvgvYp-m7PW4BzaaKpb9Trp2w8UKvvkuW3tSN6O7pJ7L7vm9P_pBX-eLOf03QKFd9y2jVQ" alt="xxx" title="Friendship network in a 34-person karate club [Zachary 1977]" height="200">
    </a>
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/noB1S/degree-and-closeness-centrality">
        <img src="images/m3-01.png" alt="So, one way to answer the question would be to say, well, nodes who have a very high degree, nodes who have lots of friends are important nodes. And if we use that definition then we'll find that the five most important nodes are nodes 34, 1, 33, 3 and 2." title="Importance as high degrees" height="200"> <br/>
        <img src="images/m3-02.png" alt="There are other ways in which you can imagine answering this question. Another way would be to say that nodes who are important are nodes who are very close to other nodes and network, nodes who have high proximity to other nodes and network. And if we use that definition, then the five most important nodes in the network would be notes 1, 3, 34, 32 and 9.  So, instead of having node 33 we'll have node 9 and then instead of having node 2 we'll have node 32 and all the other ones stay the same. Yet, another way of thinking about importance would be to say that nodes who are important are nodes who tend to connect other nodes into network." title="Importance as high proximity" height="200">
        <img src="images/m3-03.png" alt="And so, we could imagine measuring importance by the fraction of shortest paths that pass through a particular node. And if we do that, if we define in that way, we find that the five most important nodes in the network are nodes 1,34, 33, 3 and 32. So, instead of having node number 9, we'll have node number 33 in the top five and every all the other nodes will stay the same." title="Importance as high fraction of shortest path passing the node" height="200">
    </a>

+ Network Centrality <br/>
    Centrality measures identify the most important nodes in a network:
    + Influential nodes in a social network.
    + Nodes that disseminate information to many nodes or prevent epidemics.
    + Hubs in a transportation network.
    + Important pages on the Web.
    + Nodes that prevent the network from breaking up.

+ Centrality Measures
    + __Degree centrality__
    + __Closeness centrality__
    + Betweenness centrality
    + Load centrality
    + Page Rank
    + Katz centrality
    + Percolation centrality

+ Degree Centrality
    + Assumption: important nodes have many connections.
    + The most basic measure of centrality: number of neighbors.
    + Undirected networks: use degree Directed networks: use in-degree or out-degree

+ Degree Centrality – Undirected Networks
    $$C_{deg}(v) = \frac{d_v}{|N| - 1}$$
    + $N$: the set of nodes in the network
    + $d_v$: the degree of node $v$
    <a href="https://anthonybonato.com/2016/04/13/the-mathematics-of-game-of-thrones/"> <br/>
        <img src="https://lh3.googleusercontent.com/OQqUIVdAO_KrEiIsfGN4mARt24rHxQzWZ9IndHfY3DEvgvYp-m7PW4BzaaKpb9Trp2w8UKvvkuW3tSN6O7pJ7L7vm9P_pBX-eLOf03QKFd9y2jVQ" alt="the degree centrality of a node V is going to be the ratio between its degree and the number of nodes in the graph minus one. So, in this case a node would have a centrality of one if it's connected to every single node in the network and a centrality of zero if it's connected to no node in the network. And so, this measure goes between zero and one, with one being the case where you're most connected. Let's see how we can use network X to find the degree centrality of nodes in this network. So, first let me load up the karate club graph and then let me convert the labels of the nodes so that they match what we see here in this figure. And now, we can use the function degree centrality to measure the centrality of all the nodes in the network. And so here, these returns a dictionary of centralities of every node and we can for example, look at the degree centrality of node number 34 which is 0.515 and that's because node 34 has 17 connections and there are 34 nodes in the network, so that is 17 or 33. The Degree Centrality of note 33 is 0.364 and that is 12 over 33." title="Friendship network in a 34-person karate club [Zachary 1977]" height="200">
    </a>
    ```python
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G,first_label=1)
    degCent = nx.degree_centrality(G)

    degCent[34]
    # 0.515 # 17/33

    degCent[33]
    # 0.182 # 6/33
    ```

+ Degree Centrality – Directed
    + In-bound:

        $$C_{indeg}(v) = \frac{d_v^{in}}{|N| - 1}$$
        + $N$: the set of nodes in the network
        + $d_v^{in}$: the in-degree of node $v$
        <a href="url"> <br/> 
            <img src="images/m2-13.png" alt="directed networks, we have the choice of using the in-degree centrality or the out-degree centrality of a node and everything else is defined in the same way. So, the in-degree centrality of a node V is going to be its in-degree divided by the number of nodes in the graph minus one. And we can use the function in-degree centrality network X to find the in-degree centrality of all the nodes in a directed network. And so in this case, A will have in-degree of 0.143 which is two or 14. There are 15 nodes in this network and L will have a in-degree centrality of 0.214 which is three over 14 and that's because node L has an in-degree of three. And the very same way we can define not using the out-degree instead of the in-degree centrality using the function out-degree centrality and the out-degree centrality of A is 0.214, because it has an out-degree of three and the out-degree centrality of node L is 0.071, because L has an out-degree of one." title="Directed graph: weakly connected component" height="250">
        </a>
        ```python
        indegCent = nx.in_degree_centrality(G)
        
        indegCent[‘A’]
        # 0.143 # 2/14

        indegCent[‘L’]
        # 0.214 # 3/14
        ```
    + Outbound:

        $$C_{outdeg}(v) = \frac{d_v^{out}}{|N| - 1}$$
        + $N$: the set of nodes in the network
        + $d_v^{out}$: the out-degree of node $v$

        ```python
        outdegCent = nx.out_degree_centrality(G)
        
        outdegCent[‘A’]
        # 0.214 # 3/14

        indegCent[‘L’]
        # 0.071 # 1/14
        ```

+ Closeness Centrality <br/>
    Assumption: important nodes are close to other nodes.

    $$C_{close}(v) = \frac{|N| - 1}{\sum_{u \in N \backslash \{v\}} d(v, u)}$$
    + $N$: set of nodes in the network
    + $d(v, u)$: length of shortest path from 𝑣 to 𝑢.
    <a href="https://anthonybonato.com/2016/04/13/the-mathematics-of-game-of-thrones/"> <br/>
        <img src="https://lh3.googleusercontent.com/OQqUIVdAO_KrEiIsfGN4mARt24rHxQzWZ9IndHfY3DEvgvYp-m7PW4BzaaKpb9Trp2w8UKvvkuW3tSN6O7pJ7L7vm9P_pBX-eLOf03QKFd9y2jVQ" alt="closeness centrality. And the assumption here is that nodes that are important are going to be a short distance away from all other nodes in the network. Recall that we measure distance between two nodes by looking at the shortest path, the length of the shortest path between them. And so, the way we're going to define the closeness centrality of node V is going to be by taking the ratio of the number of nodes in the network minus one divided by the sum over all the other nodes in the network, and the distance between node V and those nodes. So, that's the sum and the denominator in the definition of centrality. Now, so let's say that we want to use network X to find the closeness centrality of this node 32. We can use the function closeness centrality which returns the dictionary of the centrality of the closeness centrality of all the nodes. And here, we find node 32 has a closeness centrality of 0.541. So, using the definition of closeness centrality let's see how this 0.541 comes about. So, first of all let me look at the sum of the length of the shortest path from node number 32 to all the other nodes in the network. I'm using here the shortest path length function, which you've seen before, which gives you the length of all the shortest path from the node number 32 to all the other nodes and that sum here is 61. And so, then if we take the number of nodes in the graph minus one divided by 61 that's how we get this 0.541." title="Friendship network in a 34-person karate club [Zachary 1977]" height="200">
    </a>
    ```python
    closeCent = nx.closeness_centrality(G)

    closeCent[32]                                   # 0.541
    sum(nx.shortest_path_length(G,32).values())     # 61
    (len(G.nodes())-1)/61.                          # 0.541
    ```

+ Disconnected Nodes
    + How to measure the closeness centrality of a node when it cannot reach all other nodes?
    + <n style="color:red">What is the closeness centrality of node L?</n>
    <a href="url"> <br/> 
        <img src="images/m2-13.png" alt="the first option we can simply only consider the nodes that L can actually reach in order to measure its closeness centrality. So, the way this would work is that we define this set RL to be the set of nodes that L can actually reach and we define the closeness centrality of L to be the ratio of the number of nodes that L can reach divided by the sum of the distances from L to all the nodes that L can actually reach. And so, for node L here, this would be simply one, because L can only reach node M so RL here is just the set M is just the node M and L can reach M in just one step. So, the closeness centrality of L here, would be one over one, which is one and this is the highest possible centrality a node can have in a network, which seems a little bit problematic because as node L can only reach one node and we're saying that it has the highest possible centrality than any node can have, this seems an intuitive, right? So, here's where option 2 comes in. In option 2, we again only consider the nodes that L can reach, but then, we normalize by the fraction of nodes that L can reach. So, the way this looks here is that when we compute the closeness centrality of L, we have the same ratio of RL over the sum. But now, we're going to multiply that ratio, the fraction of nodes that L can reach, RL, divided by the number of nodes in the graph minus one. So basically, we're normalizing by the fraction of nodes that L can reach. And so, if L cannot reach many nodes we're going to be multiplying these other fraction by a very small number. And so, in this case if we do that we find that L has a closeness centrality of 0. 071 which is more reasonable than defined to be one. One thing to note here is that in this new definition when we're normalizing, we're not changing the definition of closeness centrality when the graph is connected, where in every node can reach every other node. That's because when that's the case RL for node L would equal M minus one and this formula that you see here would be the exact same formula that we had before. use the function closeness centrality. And here, you get the option of normalizing or not normalizing. And so for example, if we choose not to normalize then the closeness centrality of node L would be one, as we saw before, and if we choose to normalize then it's closeness centrality would be 0.071." title="Directed graph: weakly connected component" height="250">
    </a>
    + Option 1: Consider only nodes that $L$ can reach:

        $$C_{close}(L) = \frac{|R(L)|}{\sum_{u \in R(L)} d(L, u)}$$
        + $R(L)$: the set of nodes L can reach.
        + $C_{close}(L) = 1/1 = 1$, since $L$ can only reach $M$ and it has a shortest path of length 1.
        + __Problem__: centrality of 1 is too high for a node than can only reach one other node!
    + Option 2: Consider only nodes that $L$ can reach and normalize by the fraction of nodes $L$ can reach:

        $$C_{close} (L) = [\frac{|R(L)|}{|N -1|}] \frac{|R(L)|}{\sum_{u \in (L)} d(L, u)}$$

        $$C_{close} (L) = [\frac{1}{14}] \frac{1}{1} = 0.071$$
        + Note that this definition matches our definition of closeness centrality when a graph is connected since $R(L) = N − 1$
    + Programming
        ```python
        closeCent = nx.closeness_centrality(G, normalized = False)
        closeCent[‘L’]      # 1

        closeCent = nx.closeness_centrality(G, normalized = True)
        closeCent[‘L’]      # 0.071
        ```
    + IVQ: Which node has the highest closeness centrality under option 1, where we only consider the distance to reachable nodes? Note: the closeness centrality of nodes that cannot reach any other nodes is always zero.
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/noB1S/degree-and-closeness-centrality"> <br/>
            <img src="images/m3-q01.png" alt="closeness centrality" title="cloaseness centrality graph" height="50">    
        </a>

        Ans: C <br/>
        Under option 1, node C has closeness centrality of 1, the highest of all nodes, because it can only reach D and it reaches in one step.
    + IVQ: Which node has the highest closeness centrality under option 2, where we normalize by the fraction of nodes a node can reach ? Note: the closeness centrality of nodes that cannot reach any other nodes is always zero. (same diagram)

        Ans: A
        Under option 2, node A has closeness centrality of ½, the highest of all nodes. A can reach all other nodes in the network: B in 1 step, C in 2 steps, and D in 3 steps. Hence, A’s closeness centrality is $(3/3)(3/6) = ½$.

+ Summary <br/>
    Centrality measures identify the most important nodes in a network:
    + Degree Centrality <br/>
        Assumption: important nodes have many connections.

        $$C_{deg} (v) = \frac{d_v}{|N| - 1}$$
        ```python
        nx.degree_centrality(G)
        nx.in_degree_centrality(G)
        nx.out_degree_centrality(G)
        ```
    + Closeness Centrality <br/>
        Assumption: important nodes are close to other nodes.

        $$C_{close} (L) = [\frac{|R(L)|}] \frac{|R(L)|}{\sum_{u \in R(L)}d(L, u)}$$
        ```python
        nx.closeness_centrality(G, normalized = True)
        ```


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/gut-1JTMEeeRmQ5TE1Qolg.processed/full/360p/index.mp4?Expires=1549152000&Signature=IaL9f58-joxTFQ5t6xL40ICRAV8MGmc~-AYffdEPOEe33nht-WQGf6JRnKvEZ64OFRhPSizfbr3-c~bk4beAO7Gg6yJXT857VSajVDenRnzjxZktlHEmi9hCTcJb1dFcEQghiM8UbR39CfE~b5sYwwIj5P2YK7xkjJhKBleLql4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Degree and Closeness Centrality" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Betweenness Centrality

### Lecture Notes



+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Betweenness Centrality" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Basic Page Rank

### Lecture Notes



+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Basic Page Rank" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Scaled Page Rank

### Lecture Notes



+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Scaled Page Rank" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Hubs and Authorities

### Lecture Notes



+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Hubs and Authorities" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Centrality Examples

### Lecture Notes



+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Centrality Examples" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Quiz: Module 3 Quiz




## PageRank and Centrality in a real-life network





