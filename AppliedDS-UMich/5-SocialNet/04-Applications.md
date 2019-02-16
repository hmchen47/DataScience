# Module 4: Applications

## Preferential Attachment Model

### Lecture Notes

+ Degree Distributions
    + The __degree__ of a node in an undirected graph is the number of neighbors it has. Eg., A(3), B(2), C(3), D(2), E(1_, F(3), G(4), H(2), I(2)
    + The __degree distribution__ of a graph is the probability distribution of the degrees over the entire network.
    + The degree distribution, $𝑃(𝑘)$ of this network has the following values:

        $$P(1) = \frac{1}{9}, P(2) = \frac{4}{9}, P(3)=\frac{1}{3}, P(4) = \frac{1}{9}$$
        + $k$: the degree of a given node
    + Plot of the degree distribution of this network:
        ```python
        degrees = G.degree()
        degree_values = sorted(set(degrees.values()))
        histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in degree_values]

        import matplotlib.pyplot as plt
        plt.bar(degree_values,histogram)
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.show()
        ```
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/abipd/preferential-attachment-model">
            <img src="https://lh3.googleusercontent.com/P_mxP3SNySiu36dL3jwWq06zuEVuB1jtC3xSQ0cIJCh50TlZH7DgS4D-6Pgm5f8LGVZSCONtIg65n3IeZ_7f_69MEpm8tx32SKUFBzEPvcIiLOll-MY1c7q9R9pGM9u-6QH-VyhDbQ=w2400" alt="let P(k), where k is degree, be the degree distribution of this network, we'll find that P(k) has the following values: P(1) would be 1/9 because only one node, node E, has degree one out of all nine nodes; P(2) will be 4/9 because there are four nodes that have degree two over the nine nodes; and then P(3) is 3/9 or one-third because three nodes have degree three." title="Example of undirectedd graph" height="200">
            <img src="https://lh3.googleusercontent.com/1v-beDXVxIqEEj17jLP13jycyPNuok2BY1jR1KB28c6Wt6caiozyCVK6hgIeTYcpou3u8AWOqJguMqwtehy690_Hi2ippr-iGt7mtc31NE2L4eHyiUHvN7aSBBnrQPklH9yDNgCsIQ=w2400" alt="plot the degree distribution of a network using Network X in the following way: first, we use the function degrees which returns a dictionary where the keys are nodes and the values are the degree of the nodes; and then we construct a list of sorted degrees (so this would just have the degrees of all the nodes in the network in a sorted list); then we construct a histogram that tells us how many nodes of a particular degree we have; and then we can just plot a bar plot of these histogram." title="Histogram of degree distribution" height="200">
        </a>

+ In-Degree Distributions
    + The __in-degree__ of a node in a directed graph is the number of in-links it has.
    + E.g. graph above - A(3), B(1), C(2), D(1), E(0), F(1), G(1), H(0), I(2)
    + The in-degree distribution, $P_{in}(k)$, of this network has the following values:

        $$P_{in}(0)=\frac{2}{9}, P_{in}(1)=\frac{4}{9}, P_{in}(2)=\frac{2}{9}, P_{in}(3)=\frac{1}{9}$$
    + Plot of the degree distribution of this network:
        ```python
        in_degrees = G.in_degree()
        in_degree_values = sorted(set(in_degrees.values()))
        histogram = [list(in_degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in in_degree_values]

        plt.bar(in_degree_values,histogram)
        plt.xlabel('In Degree')
        plt.ylabel('Fraction of Nodes')
        plt.show()
        ```
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/abipd/preferential-attachment-model">
            <img src="images/m4-01.png" alt="in-degree distribution, say P_in(k), that degree distribution would have these values: we have P_in(0) is 2/9 because two nodes have degree zero, P_in(1) is 4/9 because four of the nodes have in-degree one and so on." title="Example of directed graph" height="200">
            <img src="https://lh3.googleusercontent.com/1v-beDXVxIqEEj17jLP13jycyPNuok2BY1jR1KB28c6Wt6caiozyCVK6hgIeTYcpou3u8AWOqJguMqwtehy690_Hi2ippr-iGt7mtc31NE2L4eHyiUHvN7aSBBnrQPklH9yDNgCsIQ=w2400" alt="text" title="Histogram of in-degree distribution" height="200">
        </a>

+ Degree Distributions in Real Networks
    + A – __Actors__: network of 225,000 actors connected when they appear in a movie together.
    + B – __The Web__: network of 325,000 documents on the WWW connected by URLs.
    + C – __US Power Grid__: network of 4,941 generators connected by transmission lines.
    + Degree distribution looks like a straight line when on a log-log scale. __Power law__: $P(k)=Ck^{-\alpha}$, where $\alpha$ and C are constants. $\alpha$ values: A: 2.3, B:2.1, C:4.
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> <br/>
        <img src="https://lh3.googleusercontent.com/rZEP7WDAdwWkHjlFWGRdmnUZueZsL5uG_71R7HYoj87gicbjACxwr3FFDkekP0XWOSto-el8GSPFsCx9heQitepqWTlIYTcfrqWqWhp_0F0MRacvi6hE_twNh_7SGXLfY-XQaYHOgg=w2400" alt="the degree distributions of three different networks, and let me tell you what the networks are: Network A is a network of 225,000 actors connected when they appear in a movie together, B is a network of the World Wide Web so it has about 325,000 documents that are connected by URLs, and then C is a network of the US Power Grid so it's a network of about 5,000 generators connected by transmission lines. There are two things to notice about these degree distributions. The first thing is that they're all in log-log scale, meaning that the x-axis and the y-axis are both on log scale rather than linear scale. The second thing to notice is that, for at least part of these distributions, they tend to look like straight lines for all three cases. When you put those two things together when you have a degree distribution on a log-log scale and it looks kind of like a straight line, then we say that this degree distribution looks kind of like a power law. A power law degree distribution would have the form P(k) equals C times k to the negative alpha, where C and alpha are constant, and the alpha values for these three distributions that we have here are 2.3 for A, 2.1 for B, and four for C" title="Examples of power law distribution" height="250">
    </a>

+ Modeling Networks
    + Networks with power law distribution have many nodes with small degree and a few nodes with very large degree.
    + What could explain power law degree distribution we observe in many networks?
    + Can we find a set of basic assumptions that explain this phenomenon?

+ Preferential Attachment Model
    + Start with two nodes connected by an edge.
    + At each time step, add a new node with an edge connecting it to an existing node.
    + Choose the node to connect to at random with probability proportional to each node's degree.
    + The probability of connecting to a node $u$ of degree $k_u$:

        $$k_u = k_u/\sum_j k_j$$
    + As the number of nodes increases, the degree distribution of the network under the preferential attachment model approaches the power law $P(k) = Ck^{-3}$ with constant $C$.
    + The preferential attachment model produces networks with degree distributions similar to real networks.
    + `barabasi_albert_graph(n, m)` returns a network with $n$ nodes. Each new node attaches to $m$ existing nodes according to the Preferential Attachment model.
        ```python
        G = nx.barabasi_albert_graph(1000000, 1)
        degrees = G.degree()
        degree_values = sorted(set(degrees.values()))
        histogram = [list(degrees.values().count(i))/float(nx.number_of_nodes(G)) for i in degree_values]

        plt.plot(degree_values,histogram, 'o')
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        ```
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/abipd/preferential-attachment-model"> <br/>
            <img src="images/m4-03.png" alt="start with these two nodes connected by an edge, and then at each time step we're going to add a new node. Node three is going to come in and is going to attach to a single node, either node one and node two, but it's going to do so with probability proportional to their degree. But right now, node one and two both have degree of one, so the probability of choosing one and two is 50/50 for node three." title="Preferential Attachment Model: 2 nodes" height="150">
            <img src="images/m4-04.png" alt="node two has degree of two, and nodes one and three have degree one, and so the probability of attaching to node two is 0.5, and the probability of attaching to nodes one or three is 0.25. Node four attaches to node three, which had a lower probability." title="Preferential Attachment Model: 3 nodes" height="150">
            <img src="images/m4-05.png" alt="node five comes in, and we have to recompute all the probabilities. Well, now, nodes two and three both have degree two, and nodes one and four have degree one. So nodes two and three are going to have a higher probability of attaching, and so that's 0.33 for nodes two and three, and 0.17 for nodes one and four." title="Preferential Attachment Model: 4 nodes" height="150">
            <img src="images/m4-06.png" alt="continue this node six, again, we recompute the probabilities. Now, node two has the highest degree, so we will have the highest probability of getting that new attachment. So six, let's say, attaches to two." title="Preferential Attachment Model: 5 nodes" height="150">
            <img src="images/m4-07.png" alt="five attaches to node two, and we  Now, seven comes in, we recompute the probabilities. Again, now, node two has an even higher degree so it has an ever higher probability of getting that new node. Second is node three that has a degree or two with probability of getting that new node edge at 0.2, and seven attaches to two. Node eight comes in, we recompute the probabilities." title="Preferential Attachment Model: 6 nodes" height="200">
            <img src="images/m4-02.png" alt="text" title="Preferential Attachment Model: 7 nodes" height="200">
        </a>
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> 
            <img src="https://lh3.googleusercontent.com/psmc0XAoiIJH0Nt-Ih35ZjtYxSY-nYdImizg2yW3G0AvJzwYbI_7HSJFJwowI-OzDKNHkA0jJl1xZ86vtPy01btFD2_JAegkFjZ4zBa_-8Ko9lJIF7YFwZqeODcaI84rioaPm_o6zQ=w2400" alt="as node two started to get larger and larger degree, its probability of getting a new edge became larger and larger as well. There is this sort of rich get richer phenomenon, where as the nodes get larger and larger degree, they also start to become more and more likely to increase their degree." title="caption" height="200">
        </a>
    + IVQ: What is the probability that node 8 attaches to node 3?
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/abipd/preferential-attachment-model"> <br/>
            <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/BMmbx4wWEee9xRJas1XV1A_5bc5f7c50053380f9ceccea18a941082_PrefAttachIVQ.png?expiry=1549497600000&hmac=O-i_CXNm-_BhhVJbPTRLxEEbxi7PQAYRrsiI7blF3Po" alt="In Network X, you can use the function, barabasi_albert_graph, which is named after the researchers that came up with this model, with input powers n and m, where n is the number of nodes and m is the number of new nodes that an arriving node would attach to. In our example, the way we define it, this m parameter would be one because we said that every new node would attach to only a single existing node, but you can generalize this and have it so that every node attaches to m existing nodes, and that m will not change the fact that you still get a power law degree distribution. You can use this function in Network X to create networks that follow these Preferential Attachment Model. Let's create one here with a million nodes and an m(1), so every node attaches to a single existing node. Now, let's plot the degree distribution in the same way that we did before except that now I'm not using a bar plot, I'm using a scatter plot, and now I'm setting the scales, the y-scale and x-scales to be logged so we can see that straight line that gets formed on the log-log scale for the degree distribution." title="IVQ: Preferential Attachment Model w/ 8 nodes" height="200">
        </a>

+ Summary
    + The degree distribution of a graph is the probability distribution of the degrees over the entire network.
    + Many real networks have degree distributions that look like power laws ($P(k) = Ck^{-\alpha}$).
    + Models of network generation allow us to identify mechanisms that give rise to observed patterns in real data.
    + The Preferential Attachment Model produces networks with a power law degree distribution.
    + Use `barabasi_albert_graph(n,m)` to construct a n-node preferential attachment network, where each new node attaches to m existing nodes.

+ `G.in_degree` method
    + Signature: `nx.DiGraph.in_degree(nbunch=None, weight=None)`
    + Docstring: Return the in-degree of a node or nodes. The node in-degree is the number of edges pointing in to the node.
    + Parameters
        + `nbunch` (iterable container, optional (default=all nodes)): A container of nodes.  The container will be iterated through once.
        + `weight` (string or None, optional (default=None)): The edge attribute that holds the numerical value used as a weight.  If None, then each edge has weight 1. The degree is the sum of the edge weights adjacent to the node.
    + Returns: `nd` (dictionary, or number): A dictionary with nodes as keys and in-degree as values or a number if a single node is specified.

+ `nx.barabasi_albert_graph` function
    + Signature: `nx.barabasi_albert_graph(n, m, seed=None)`
    + Docstring: Returns a random graph according to the Barabási–Albert preferential attachment model.
    + Note: A graph of `n` nodes is grown by attaching new nodes each with `m` edges that are preferentially attached to existing nodes with high degree.
    + Parameters
        + `n` (int): Number of nodes
        + `m` (int): Number of edges to attach from a new node to existing nodes
        + `seed` (int, optional): Seed for random number generator (default=None).
    + Returns: `G`: Graph
    + References: A. L. Barabási and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/lNURBZTMEeeOmgqEJWRlfA.processed/full/360p/index.mp4?Expires=1549497600&Signature=ZbRdvAcpqHNU0nMzfycqCvR4CKg5EohROkgICPmo5lpA0gHueOlj0q-Gx6XPn3g1Y2kac33geKa5kBntgmGYTtaqB0XrIMXL5XSs48jXxsQu2~NATd0DVRQaHqCh1t8c~00H36bVh3A-x7EbW-I7u-egb1buQCx1L55kH9rjSa4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Preferential Attachment Model" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Power Laws and Rich-Get-Richer Phenomena (Optional)

Read [Chapter 18]((http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch18.pdf)) from "Networks, Crowds, and Markets: Reasoning about a Highly Connected World" By David Easley and Jon Kleinberg. Cambridge University Press, 2010 for an interesting read on Power Laws and Rich-Get-Richer phenomena such as the preferential attachment model.

+ [Popularity as a Network Phenomenon](p1-PowerLaw.md#popularity-as-a-network-phenomenon)
+ [Power Laws](p1-PowerLaw.md#power-laws)
+ [Rich-Get-Richer Models](p1-PowerLaw.md#rich---get---richer-effects)
+ [The Unpredictability of Rich-Get-Richer Effects](p1-PowerLaw.md#the-unpredictability-of-rich---get---richer-effects)
+ [The Long Tail](p1-PowerLaw.md#the-long-tail)
+ [The Effect of Search Tools and Recommendation Systems](p1-PowerLaw.md#the-effect-of-search-tools-and-recommendation-systems)
+ [Advanced Material: Analysis of Rich-Get-Richer Processes](p1-PowerLaw.md#advanced-material-analysis-of-rich---get---richer-processes)


## Small World Networks

### Lecture Notes

+ The Small-World Phenomenon
    + The world is small in the sense that "short" paths exists between almost any two people.
    + How short are these paths?
    + How can we measure their length?

+ Milgram Small World Experiment: 
    + Set up (1960s)
        + 296 randomly chosen "starters" asked to forward a letter to a "target" person.
        + Target was a stockbroker in Boston.
        + Instructions for starter:
            + Send letter to target if you know him on a first name basis.
            + If you do not know target, send letter (and instructions) to someone you know on a first name basis who is more likely to know the target.
        + Some information about the target, such as city, and occupation, was provided.
    + Results:
        + 64 out of the 296 letters reached the target.
        + Median chain length was 6 (consistent with the phrase "six degrees of separation")
    + Key points:
        + A relatively large percentage (>20%) of letters reached target.
        + Paths were relatively short.
        + People were able to find these short paths.
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/Iv4e8/small-world-networks"> <br/>
        <img src="images/m4-08.PNG" alt="these were the results. Out of all 296 starters, 64 of them reached the target. So the median chain length, the number of hops that it took for these letters to get there was 6, which was consistent with this phrase of six degrees of separation. Here in this plot, we can see a histogram of the chain length for the letters that actually arrived, and we can see that they took anywhere from 1 to 10 hops to get there, but the median was 6. And so the key points here are the following. So first of all, a relatively large percentage of these letters actually arrived. If you think about the fact that these were kind of randomly selected, and that people could drop out of this and say I'm not even going to try to forward this letter to anyone. And so despite all of these things, more than 20% of the letters actually reached the target. And the second is that for the ones that reached, these paths were relatively short, right. So median of 6, a single digit, in a network of millions and millions of people, that seems pretty small. The other thing that's interesting, although we're not going to focus much in this part of it, is that people actually are able to find this short paths." title="Small-World Phenomenon" height="250">
    </a>

+ Small World of Instant Message
    + Nodes: 240 million active users on Microsoft Instant Messenger.
    + Edges: Users engaged in two-way communication over a one-month period.
    + Estimated median path length of 7.
    + Leskovec and Horvitz, 2008
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/Iv4e8/small-world-networks"> <br/>
        <img src="images/m4-09.png" alt="looking at instant message communication among people. So researchers took a network of 240 million active users on Microsoft Instant Messenger. And then they connected them if two users were engaged in a two-way communication over a period of a month. And so these defined a network, a very large network. And the estimated path length in this network, so if you take any two people at random and check what their distance between them in the network, so the shorter the length of the shortest path, the medium is estimated to be 7. Which is very close to what Milgram had found in the 1960s which was 6, and it's also very small. Here is the histogram of the distances of this particular network." title="Small World of Instant Message [Leskovec and Horvitz, 2008]" height="200">
    </a>

+ Small World of Facebook
    + Global network: average path length in 2008 was 5.28 and in 2011 it was 4.74.
    + Path are even shorter if network is restricted to US only.
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/Iv4e8/small-world-networks"> <br/>
        <img src="images/m4-10.png" alt="the full Facebook network versus just a subset of it in a particular region, like in the United States. And what they found is that, if you take the global network, the average path length in 2008 was about 5.28, and three years later in 2011, it was 4.74. So it seems like these path lengths are short and they seem to be getting smaller over time. And as you may expect, if you take a smaller region like United States, then this tend to be even smaller." title="Small World of Facebook [Backstrom et al. 2012]" height="250">
    </a>

+ Clustering Coefficient
    + __Local clustering coefficient of a node__: Fraction of pairs of the node's friends that are friends with each other.
        + Facebook 2011: High average CC (decreases with degree)
        + Microsoft Instant Message: Average CC of 0.13.
        + IMDB actor network: Average CC 0.78
    + In a random graph, the average clustering coefficient would be much smaller.
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/Iv4e8/small-world-networks"> <br/>
        <img src="images/m4-11.png" alt="Facebook in 2011, we find that the average clustering coefficient tends to be pretty high. So here is a plot that has on the x axis the degree of a node, and then on the y axis we have the average clustering coefficient. And we find that, it decreases as degree increases. So for people that have lots and lots of friends, their clustering coefficient tends to be smaller. But on average, it's still pretty large. So if you look at nodes that have say, 20 to 50 friends, they're clustering coefficient is somewhere in the 30s or so.  In the Microsoft Instant Message network, the average clustering coefficient was 0.13. And in the actor network that we talked about before, the average clustering coefficient is even higher at 0.78. And so the thing to note here is that, I say that these clustering coefficients are high because, if you imagine that these graphs were completely random." title="Clustering Coefficient[Ugander et al. 2012]" height="250">
    </a>

+ Path Length and Clustering
    + Social networks tend to have high clustering coefficient and small average path length. Can we think of a network generative model that has these two properties?
    + How about the Preferential Attachment model?
        ```python
        G = nx.barabasi_albert_graph(1000,4)
        print (nx.average_clustering(G))
        # 0.0202859273671
        print (nx.average_shortest_path_length(G))
        # 4.16942942943
        ```
    + What if we vary the number of nodes ($n$) or the number of edges per new node ($m$)?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/Iv4e8/small-world-networks"> <br/>
        <img src="images/m4-12.png" alt="create one of a 1,000 nodes and parameter m of 4, that means that each new node is going to attach to 4 existing nodes. And let's see what its average clustering is. Well in this case, it's 0.02 which is pretty small compared to the networks that we had seen before. Now for the average shortest path length, it is 4.16, which is pretty small, it's pretty decent. So it seems that it get's one of the properties but not the other. Let's see what happens if we vary the number of nodes and the number of edges per new node perimeter m. And let's see what happens to the path link in the clustering. On the x axis, I have the number of nodes. And on the y axis, I have the average clustering coefficient. And we have different curves that represent different values of the perimeter m. And then, I have the same thing for the average shortest path. Well we serve as these networks have a small average shortest path, and the reason why is that, if you remember, these power law degree distributions have the property that some nodes have a very, very high degree. And so these high degree nodes act as hubs that connect many pairs of nodes, and make sort of bridges between them. So this is why we see that these average shortest paths tend to be kind of small. However, for the clustering side, we see that as the number of nodes increases, the average clustering coefficient also decreases and it becomes very small. And so even at 2,000 nodes, we see that the average clustering coefficient is very small, at like 0.05 or so. And we have seen networks of millions and millions of nodes that had a much larger clustering coefficient." title="Small World of Facebook [Backstrom et al. 2012]" height="250">
    </a>
        + Small average shortest path (characteristics):
            + high degree nodes act as hubs
            + connect many pairs of nodes.
        + For a fixed $m$, clustering coefficient becomes very small as the number of nodes increases.
        + No mechanism in the Preferential Attachment model favors triangle formation.

+ Small World Model
    + Motivation: Real networks exhibit high clustering coefficient and small average shortest paths. Can we think of a model that achieves both of these properties?
    + Small-world model:
        + Start with a ring of 𝑛 nodes, where each node is connected to its $k$ nearest neighbors.
        + Fix a parameter $p \in [0,1]$
        + Consider each edge $(u, v)$. With probability $p$, select a node $w$ at random and rewire the edge $(u, v)$ so it becomes $(u, w)$.
    + Example: $k = 2, p = 0.4$
    <a href="url"> <br/>
        <img src="images/m4-13.png" alt="12 nodes, and in the example, k will be 2, so each node is connected to its 2 nearest neighbors and p will be 0.4. I will say that these parameters k = 2 and p = 0.4 are not the typical parameters you would use for this particular model. Typically, you have a k that's much larger, and a p that's much smaller." title="Small World Model: 0. original" height="130">
        <img src="images/m4-14.png" alt="the edge LA and we have to decide whether to rewire or not. And let's say we use some type of random number generator and we decide that we don't rewire it, okay" title="Small World Model: 1. no rewired" height="130">
        <img src="images/m4-15.png" alt="go to the next one and this one, we won't rewire either." title="Small World Model: 2. no rewired" height="130">
        <img src="images/m4-16.png" alt="text" title="Small World Model: 3. rewired" height="130">
        <img src="images/m4-17.png" alt="go to the next one and this one we will rewire. So now we have to pick another node at random, and in this case, we would pick G. And so this edge instead of connecting K and J, is going to connect K and G." title="Small World Model: 4. rewired" height="130">
        <img src="images/m4-18.png" alt="go to the next edge, GI and this one we're not going to rewire." title="Small World Model: 5. no rewired" height="130">
        <img src="images/m4-19.png" alt="text" title="Small World Model: 6. rewired" height="130">
        <img src="images/m4-20.png" alt="The next edge, we will rewire, so we pick a node at random and rewire it." title="Small World Model: 7. rewired" height="130">
        <img src="images/m4-21.png" alt="Go to the next edge, don't rewire it." title="Small World Model: 8. no rewired" height="130">
        <img src="images/m4-22.png" alt="no rewire" title="Small World Model: 9. no rewired" height="130">
        <img src="images/m4-23.png" alt="Go to the next one, rewire it. It becomes FJ." title="Small World Model: 10. rewired" height="130">
        <img src="images/m4-24.png" alt="text" title="Small World Model: 11. rewired" height="130">
        <img src="images/m4-25.png" alt="Go to the next one no rewire" title="Small World Model: 12. no rewired" height="130">
        <img src="images/m4-26.png" alt="Go to the next one no rewire" title="Small World Model: 13. no rewired" height="130">
        <img src="images/m4-27.png" alt="go to the next one, rewire so it becomes the DH." title="Small World Model: 14. rewired" height="130">
        <img src="images/m4-28.png" alt="Rewired" title="Small World Model: 15. rewired" height="130">
        <img src="images/m4-29.png" alt="go to the next one we don't rewire it" title="Small World Model: 16. rewired" height="130">
        <img src="images/m4-30.png" alt="no rewired" title="Small World Model: 17. no rewired" height="130">
    </a>
    + __Regular Lattice__ ($p = 0$): no edge is rewired.
    + __Random Network__ ($p = 1$): all edges are rewired.
    + __Small World Network__ ($0 < p < 1$): Some edges are rewired. Network conserves some local structure but has some randomness.
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> <br/>
        <img src="https://lh3.googleusercontent.com/-Wk8hCym28VKsTEGwZ_MO11cE51jmVIZ25cQ6caj0vgyFAJYn1g3tGZPwfle_Y3jRXFbvJE1OnVxjil5UmICb5cWuTBsAgJxQhRN22OdxwpBleZBOgLos9sNj6WADY6qfAtnsg64mQ=w2400" alt="think about first what happens in the extreme cases. So when p is 0, so we're looking at this type of network. What we have is a completely regular lattice. So there is no rewiring, and because every node is connected to k of its neighbors, then there are lots of triangles that get formed locally, right? Because, well it depends on the value of k, but if k is sort of large enough, then you start to form many triangles. And so this network will have pretty high clustering coefficient because it purposely going to form triangles in the beginning. And then nothing gets rewire, nothing gets changed, so it has a pretty high clustering coefficient. However, if you imagine there's been a very large network, you can imagine that the distances between nodes can get very large, right. So if you're in one side of the ring to get to the other side of the ring, you can hop a little bit but there is no long bridge that can take you there. And so, we expect that the distances would be long. Now let's think at the other extreme where we're going to rewire every single one of the edges, and so that would be this network. And so what's happening here is that, we're going to rewire every single edge. And so this network is now completely random. So we've created a bunch of long bridges. And so presumably, distances are pretty small between different nodes. But now, we kind of destroyed the sort of local structure that we had before. And so now we probably don't have many triangles there. And so while the distances are small, now the clustering also got very small. If you're In between, if p is between 0 and 1, then what you have is that some edges get rewire, so you create some long bridges. And so the distances between nodes, the average distance gets reduced. But the local structure depending on p can be maintained. So you can maintain that high clustering as well. So there's a sweet spot for the parameter p, where you can maintain both the short distances as well as the high clustering" title="caption" height="250">
    </a>
    + What is the average clustering coefficient and shortest path of a small world network? <br/> It depends on parameters $k$ and $p$.
    + As $p$ increases from $0$ to $0.09$:
        + average shortest path decreases rapidly.
        + average clustering coefficient deceases slowly.
    + An instance of a network of 1000 nodes, $k = 6$, and $p = 0.04$ has:
        + $8.99$ average shortest path.
        + $0.53$ average clustering coefficient.
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> <br/>
        <img src="https://lh3.googleusercontent.com/JZw6rImwOYdoN4Qw1kCkB72_WA9u8E1u09JjbNdLOjdSR0xm5MFFPbEXM64Hj4UF3OidvpkP8apYdIMLyeGTRcIkQm2fnDn-iEeVXMTaLDkXZDkiZ1CyUBxSzAi2IJZR0llJZyeSmw=w2400" alt="as p increases from 0 to 0.1, notice here that we don't get anywhere close to p = 1. So, this is staying with very small values of p. What we see happening is that, the average shortest path decrease rapidly right after sort of p is away from 0, it just drops down. Whereas, the average clustering coefficient while it also decreases as p increases, it decreases much slower. So for example, an instance of a network with 1,000 nodes and k = 6 and p = 0.04 has the following values. It has a value of 8.99 average shortest path, and 0.53 average clustering coefficient. So for these types of values of p, we can achieve both of the properties that we wanted. The average shortest path being small, single digit, and the average clustering coefficient being pretty large." title="caption" height="350">
    </a>

+ Small World Model in NetworkX
    + `watts_strogatz_graph(n, k, p)` returns a small world network with $n$ nodes, starting with a ring lattice with each node connected to its $k$ nearest neighbors, and rewiring probability $p$.
    + Small world network degree distribution:
        ```python
        G = nx.watts_strogatz_graph(1000,6,0.04)
        degrees = G.degree()
        degree_values = sorted(set(degrees.values()))
        histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in degree_values]

        plt.bar(degree_values, histogram)
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.show()
        ```
        + Small world network: 1000 nodes, $k = 6$, and $p = 0.04$
        + No power law degree distribution.
        + Since most edges are not rewired, most nodes have degree of $6$.
        + Since edges are rewired uniformly at random, no node accumulated very high degree, like in the preferential attachment model
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/Iv4e8/small-world-networks"> <br/>
            <img src="images/m4-31.png" alt="use the same code that we used before to visualize the degree distributions of network. This time using a small world network with 1,000 nodes, k = 6, and p = 0.04. What we is that, it looks like this. So most nodes have degree 6. A few of them have 5 and 7, and I think maybe 1 or various small number of them has degree 4 and 8, and that's about it." title="Histogram of Small World Model" height="200">
        </a>
    + Variants of the small world model in NetworkX: Small world networks can be disconnected, which is sometime undesirable.
    + `connected_watts_strogatz_graph(n, k, p, t)` runs `watts_strogatz_graph(n, k, p)` up to t times, until it returns a connected small world network.
    + `newman_watts_strogatz_graph(n, k, p)` runs a model similar to the small world model, but rather than rewiring edges, new edges are added with probability $p$.
    + IVQ: Is the degree distribution of small world network a power law distribution?

        Ans: No
        The degree distribution of small world network is not a power law because the degree of most nodes lie in the middle.

+ Summary
    + Real social networks appear to have small shortest paths between nodes and high clustering coefficient.
    + The preferential attachment model produces networks with small shortest paths but very small clustering coefficient.
    + The small world model starts with a ring lattice with nodes connected to $k$ nearest neighbors (high local clustering), and it rewires edges with probability $p$.
    + For small values of $p$, small world networks have small average shortest path and high clustering coefficient, matching what we observe in real networks.
    + However, the degree distribution of small world networks is not a power law.
    + On NetworkX, you can use `watts_strogatz_graph(n, k, p)` (and other variants) to produce small world networks. <br/><br/>

    | Model | Real World Network | Preferential Attachment Model | Small-World Model |
    |-------|:------:|:-----:|:-----:|
    | Shortest Paths | Small | Small | Small |
    | Clustering Coefficient | High | Low | High |
    | Power Law | Yes | Yes | No |

+ `nx.watts_strogatz_graph` function
    + Signature: `nx.watts_strogatz_graph(n, k, p, seed=None)`
    + Docstring: Return a Watts–Strogatz small-world graph.
    + Parameters
        + `n` (int): The number of nodes
        + `k` (int): Each node is joined with its ``k`` nearest neighbors in a ring topology.
        + `p` (float): The probability of rewiring each edge
        + `seed` (int, optional): Seed for random number generator (default=None)
    + Notes
        + First create a ring over $n$ nodes.  Then each node in the ring is joined to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd). Then shortcuts are created by replacing some edges as follows: for each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors" with probability $p$ replace it with a new edge $(u, w)$ with uniformly random choice of existing node $w$.
        + In contrast with `newman_watts_strogatz_graph`, the random rewiring does not increase the number of edges. The rewired graph is not guaranteed to be connected as in `connected_watts_strogatz_graph`.
    + References: Duncan J. Watts and Steven H. Strogatz, Collective dynamics of small-world networks, Nature, 393, pp. 440--442, 1998.

+ `connected_watts_strogatz_graph` function
    + Signature: `nx.connected_watts_strogatz_graph(n, k, p, tries=100, seed=None)`
    + Docstring: Returns a connected Watts–Strogatz small-world graph.
    + Note: Attempts to generate a connected graph by repeated generation of Watts–Strogatz small-world graphs.  An exception is raised if the maximum number of tries is exceeded.
    + Parameters
        + `n` (int): The number of nodes
        + `k` (int): Each node is joined with its $k$ nearest neighbors in a ring topology.
        + `p` (float): The probability of rewiring each edge
        + `tries` (int): Number of attempts to generate a connected graph.
        + `seed` (int, optional): The seed for random number generator.


+ `newman_watts_strogatz_graph` function
    + Signature: `nx.newman_watts_strogatz_graph(n, k, p, seed=None)`
    + Docstring: Return a Newman–Watts–Strogatz small-world graph.
    + Parameters
        + `n` (int): The number of nodes.
        + `k` (int): Each node is joined with its $k$ nearest neighbors in a ring topology.
        + `p` (float): The probability of adding a new edge for each edge.
        + `seed` (int, optional): The seed for the random number generator (the default is `None`).
    + Notes: First create a ring over $n$ nodes.  Then each node in the ring is connected with its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).  Then shortcuts are created by adding new edges as follows: for each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors" with probability $p$ add a new edge $(u, w)$ with randomly-chosen existing node $w$.  In contrast with `watts_strogatz_graph`, no edges are removed.
    + References: M. E. J. Newman and D. J. Watts, [Renormalization group analysis of the small-world network model](http://dx.doi.org/10.1016/S0375-9601(99)00757-4), Physics Letters A, 263, 341, 1999.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/pJ8eFZTMEeeClxLmJhEfgA.processed/full/360p/index.mp4?Expires=1549497600&Signature=iTfGk0ABVH-VCOu8GPJ68KtOmP9Mwn1YwrtG4klpWxd05VkrXz~fxHBrmrhIO88pprQ29MABvBjJZ~JvMGhn4qftCC2isynvKkv8AwNysoKC1FxB~gtvtXJK04XOeThVu0ervNzfG~yUk3X08NFVmJK3x87IkKqjvWHpbhWAfKw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Small World Networks" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Link Prediction

### Lecture Notes

+ Link Prediction Problem
    + Modeling network evolution:
        + Preferential attachment model
        + Small world model
    + Link prediction: Given a network, can we predict which edges will be formed in the future?

+ Link Prediction
    + What new edges are likely to form in this network? B
    + Given a pair of nodes, how to assess whether they are likely to connect?
    + __Triadic closure__: the tendency for people who share connections in a social network to become connected.
    + Measure 1: number of common neighbors.

+ Measure 1: Common Neighbors
    + The number of common neighbors of nodes $X$ and $Y$ is

        $$\text{comm\_neigh}(X, Y) = | N(X) \cap N(Y) |$$
        + $N(X)$: the set of neighbors of node $𝑋$
    + E.g., $\text{comm\_neigh}((A, C) = |\{B, D\}| = 2$
        ```python
        common_neigh = [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1])))) for e in nx.non_edges(G)]
        sorted(common_neigh,key=operator.itemgetter(2), reverse = True)

        print (common_neigh)
        # [('A', 'C', 2), ('A', 'G', 1), ('A', 'F', 1), ('C', 'E', 1), ('C', 'G', 1),
        #  ('B', 'E', 1), ('B', 'F', 1), ('E', 'I', 1), ('E', 'H', 1), ('E', 'D', 1),
        #  ('D', 'F', 1), ('F', 'I', 1), ('F', 'H', 1), ('I', 'H', 1), ('A', 'I', 0),
        #  ('A', 'H', 0), ('C', 'I', 0), ('C', 'H', 0), ('B', 'I', 0), ('B', 'H', 0),
        #  ('B', 'G', 0), ('D', 'I', 0), ('D', 'H', 0), ('D', 'G', 0)]
        ```
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> <br/>
        <img src="https://lh3.googleusercontent.com/LTuRz23kE61hfzKzyvX8a-y0-i4LQkwagjFYi9kIwtj3PmTnxaQl0gSF8UbtdZi6EBdXCVKV5OSfE5YYHgpxg_eekSFPZGU9yGWeNKslTIowMFWWzgwzRIgqTZ3GTJhTpqWvC3gM8w=w2400" alt="the number of common neighbors of nodes X and Y is going to be the size of a set, which is the intersection of the sets N(X) and N(Y), where N(X) defines the set of neighbors of the node X. And so for example, the common neighbor measure for nodes A, C in this network is going to be 2 because nodes A and C have two common neighbors. That is node B and node D. So A, C have two common neighbors." title="Example Graph for Common Neighbors" height="250">
    </a>
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hvFPZ/link-prediction"> 
        <img src="images/m4-32.png" alt="In network X, we can use the function common neighbors, which takes in as input the graph to nodes. And it outputs an iterator of all the common neighbors of the two nodes. And so here, what I'm doing is I'm creating a list of tuples which have the two nodes and the number of common neighbors. And I'm only including the nodes that are not connected with each other. The ones that don't have an edge between them by using the function non_edges. And if I sort this list, we can see the pairs of nodes that have the most common neighbors between them. And we see that the pair A, C has two neighbors in common, there are many others that have only one neighbor in common. And then many others that have zero neighbors in common. And so if we start to compare between different edges, for example we look at the pair A, G and the pair H, I, and ask, which one of these two is more likely to become connected? Then by looking at the number of common neighbors, we actually can't tell, because both of these have exactly one neighbor in common. And so let's look at other measures that may potentially give us different answers for these particular pairs of nodes." title="Example Graph of (A, G) & (H, I)" height="250">
    </a>

+ Measure 2: Jaccard Coefficient
    + Number of common neighbors normalized by the total number of neighbors.
    + The Jaccard coefficient of nodes 𝑋 and 𝑌 is

        $$\text{jacc\_coeff}(X, Y) = \frac{|N(X) \cap N(Y)|}{|N(X) \cup N(Y)}$$
    + E.g., $\text{jacc\_coeff}(A, C) = \frac{|\{B, D\}|}{|\{B, D, E, F\}|}$
    + IVQ: What is the Jaccard Coefficient between node A and F?  Enter your answer as a fraction below the graphic.

        Ans: 1/5 <br/>
        $\text{J\_coef}(F) = |{E}|/|{B, C, D, E, G}|= \frac{1}{5} = 0.2$
    + Number of common neighbors normalized by the total number of neighbors.
        ```python
        L = list(nx.jaccard_coefficient(G))
        L.sort(key=operator.itemgetter(2), reverse = True)
        print(L)
        # [('I', 'H', 1.0), ('A', 'C', 0.5), ('E', 'I', 0.3333333333333333),
        #  ('E', 'H', 0.3333333333333333), ('F', 'I', 0.3333333333333333),
        #  ('F', 'H', 0.3333333333333333), ('A', 'F', 0.2), ('C', 'E', 0.2),
        #  ('B', 'E', 0.2), ('B', 'F', 0.2), ('E', 'D', 0.2), ('D', 'F', 0.2),
        #  ('A', 'G', 0.16666666666666666), ('C', 'G', 0.16666666666666666),
        #  ('A', 'I', 0.0), ('A', 'H', 0.0), ('C', 'I', 0.0), ('C', 'H', 0.0),
        #  ('B', 'I', 0.0), ('B', 'H', 0.0), ('B', 'G', 0.0), ('D', 'I', 0.0),
        #  ('D', 'H', 0.0), ('D', 'G', 0.0)]
        ```
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hvFPZ/link-prediction"> 
            <img src="images/m4-32.png" alt="The next measure we're going to look at is called the Jaccard coefficient. And what it does is that it looks at the number of common neighbors but it normalizes it by the total number of neighbors of the two nodes. So the way that we write it down is we say the Jaccard coefficient of nodes X and Y is going to be the fraction of the number of common neighbors. That's in the numerator, so it's the intersection of the sets N(X) and N(Y), divided by the number of neighbors of X and Y which would be the union of N(X) and N(Y). And so here, the pair of nodes of A and C, have a Jaccard coefficient of one-half because they have two common neighbors, B and D. And they have four total neighbors, nodes B, D, E, and F. And so that's 2 over 4 which is one-half." title="Example graph for Jaccard Coefficient" height="250">
        </a>

+ Measure 3: Resource Allocation
    + Fraction of a "resource" that a node can send to another through their common neighbors.
    + The Resource Allocation index of nodes $X$ and $Y$ is

        $$\text{res\_alloc}(X, Y) = \sum_{u \in N(X) \cap N(Y)} \frac{1}{|N(u)|}$$
    + Basic Principle: $Z$ has $n$ neighbors, $X$ sends 1 unit to $Z$, $Z$ distributes the unit evenly among all neighbors $\rightarrow Y$ receives $1/n$ of the unit.
    + E.g., $\text{res\_alloc}(A, C) = \frac{1}{3} + \frac{1}{3}$
    + Fraction of a "resource" that a node can send to another through their common neighbors.
    ```python
    L = list(nx.resource_allocation_index(G))
    L.sort(key=operator.itemgetter(2), reverse = True)
    print(L)
    # [('A', 'C', 0.6666666666666666), ('A', 'G', 0.3333333333333333),
    #  ('A', 'F', 0.3333333333333333), ('C', 'E', 0.3333333333333333),
    #  ('C', 'G', 0.3333333333333333), ('B', 'E', 0.3333333333333333),
    #  ('B', 'F', 0.3333333333333333), ('E', 'D', 0.3333333333333333),
    #  ('D', 'F', 0.3333333333333333), ('E', 'I', 0.25), ('E', 'H', 0.25),
    #  ('F', 'I', 0.25), ('F', 'H', 0.25), ('I', 'H', 0.25), ('A', 'I', 0),
    #  ('A', 'H', 0), ('C', 'I', 0), ('C', 'H', 0), ('B', 'I', 0),
    #  ('B', 'H', 0), ('B', 'G', 0), ('D', 'I', 0), ('D', 'H', 0), ('D', 'G', 0)]
    ```
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hvFPZ/link-prediction"> <br/>
        <img src="images/m4-32.png" alt="the Resource Allocation index of the nodes X, Y is going to be the sum over all the common neighbors of X and Y of one over the degree of the nodes. So in this case, if X and Y have a lot of common neighbors and they're going to have a large Resource Allocation index. But if they have a lot of neighbors that have low degree, then they're going to have an even larger Resource Allocation index. Now what's the intuition behind this? Let's consider two nodes X and Y, and let's say that we're measuring the Resource Allocation index between these two nodes." title="Example graph for Resource Allocation" height="250">
        <img src="images/m4-33.png" alt="Imagine that X is trying to send to Y a unit of something, let's say information or something else. And is going to do so by passing it for X to Z and then hoping that Z will pass this unit to Y. But actually what Z does is that when it receives this unit from X is going to distribute this unit evenly among all the neighbors of Z. Then in that case, well Y is only going to get a fraction of that unit. And which fraction depends on what the degree of Z is. So if Z has degree N, then Y is only going to get 1 over N of that unit. And so if Z is the only common neighbor of X and Y, and Z has a lot of neighbors, a very large degree. Then X is going to be able to send less to Y, than if Z had a very few neighbors. that's why this resource allocation index penalizes pairs of nodes that have common neighbors that themselves have lots of other neighbors." title="Intution of Resource Allocation" height="125">
        <img src="images/m4-34.png" alt="for example, when we measure the Resource Allocation index of nodes A and C, we would have one-third for node B. because node B is the common neighbor of A and C and has degree of 3, plus 1 over 3 which is for node D, which also has degree 3 and it's also a common neighbor of A and C. So the Resource Allocation index of A, C is going to be two-thirds." title="Example graph for Resource Allocation w/ Node D" height="250">
    </a>
    + IVQ: What is the Resource Allocation index of Node I and H?

        Ans: 0.25
        Node I and H have only one common neighbor: G. The degree of node G is 4. Hence the Resource Allocation index is $\frac{1}[4} = 0.25$.

+ Measure 4: Adamic-Adar Index
    + Similar to resource allocation index, but with log in the denominator.
    + The Adamic-Adar index of nodes $X$ and $Y$ is

        $$\text{adamic\_adar}(X, Y) = \sum_{u \in N(X) \cap N(Y)} \frac{1}{\log(|N(u)|)}$$
    + E.g., $\text{admic\_adar}(A, C) = \frac{1}{\log(3)} + \frac{1}{\log(3)} = 1.82$
    + Similar to resource allocation index, but with log in the denominator.
        ```python
        L = list(nx.adamic_adar_index(G))
        L.sort(key=operator.itemgetter(2), reverse = True)
        print(L)
        # [('A', 'C', 1.8204784532536746), ('A', 'G', 0.9102392266268373),
        #  ('A', 'F', 0.9102392266268373), ('C', 'E', 0.9102392266268373),
        #  ('C', 'G', 0.9102392266268373), ('B', 'E', 0.9102392266268373),
        #  ('B', 'F', 0.9102392266268373), ('E', 'D', 0.9102392266268373),
        #  ('D', 'F', 0.9102392266268373), ('E', 'I', 0.7213475204444817),
        #  ('E', 'H', 0.7213475204444817), ('F', 'I', 0.7213475204444817),
        #  ('F', 'H', 0.7213475204444817), ('I', 'H', 0.7213475204444817),
        #  ('A', 'I', 0), ('A', 'H', 0), ('C', 'I', 0), ('C', 'H', 0),
        #  ('B', 'I', 0), ('B', 'H', 0), ('B', 'G', 0), ('D', 'I', 0),
        #  ('D', 'H', 0), ('D', 'G', 0)]
        ```

+ Measure 5: Preferential Attachment
    + In the preferential attachment model, nodes with high degree get more neighbors.
    + Product of the nodes' degree.
    + The preferential attachment score of nodes $X$ and $Y$ is

        $$\text{pref\_attach}(X, Y) = |N(X)||N(Y)|$$
    + E.g., $\text{pref\_attach}(A, C) = 3 * 3 = 9$
    + Product of the nodes' degree.
        ```python
        L = list(nx.preferential_attachment(G))
        L.sort(key=operator.itemgetter(2), reverse = True)
        print(L)
        # [('A', 'G', 12), ('C', 'G', 12), ('B', 'G', 12), ('D', 'G', 12),
        #  ('A', 'C', 9), ('A', 'F', 9), ('C', 'E', 9), ('B', 'E', 9), ('B', 'F', 9),
        #  ('E', 'D', 9), ('D', 'F', 9), ('A', 'I', 3), ('A', 'H', 3), ('C', 'I', 3),
        #  ('C', 'H', 3), ('B', 'I', 3), ('B', 'H', 3), ('E', 'I', 3), ('E', 'H', 3),
        #  ('D', 'I', 3), ('D', 'H', 3), ('F', 'I', 3), ('F', 'H', 3), ('I', 'H', 1)]
        ```

+ Community Structure
    + Some measures consider the community structure of the network for link prediction.
    + Assume the nodes in this network belong to different communities (sets of nodes).
    + Pairs of nodes who belong to the same community and have many common neighbors in their community are likely to form an edge.
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hvFPZ/link-prediction"> <br/>
        <img src="images/m4-35.png" alt="a network of communication among employees in a company. Then you may think that the department for which an employee works for would define a community. So for example, there's HR, and there's legal and so on, and so you could imagine thinking of those as communities. So if you had that type of structure, then you could use different measures for determining whether two nodes are likely to connect to each other or not. So in this case, let's assume that these network has two communities. So, these are the two communities. So there's A, B, D and C belong to Community 1, and the other nodes belong to Community 2. And what these two measures do, is they make the assumption that if two nodes belong to the same community, and they have many neighbors that also belong to the same community. Then they're more likely to form an edge, than if they had neighbors that belong to different communities, or if the two nodes themselves were in different communities." title="Example graph with two communities" height="250">
    </a>

+ Measure 6: Community Common Neighbors
    + Number of common neighbors with bonus for neighbors in same community.
    + The Common Neighbor Soundarajan-Hopcroft score of nodes $X$ and $Y$ is:

        $$\text{cn\_soundarajan\_hopcroff}(X, Y) = |N(X) \cap N(Y)| + \sum_{u \in N(X) \cap N(Y)} f(u)$$

        $$f(u) = \left\{ \begin{array}{ll}
            1, & u \text{ in same comm. as } X \text{and } Y \\
            0, & \text{ otherwise}
            \end{array} \right.$$
    + Number of common neighbors with bonus for neighbors in same community.
        + $\text{cn\_soundarajan\_hopcroft}(A, C) = 2 + 2 = 4$
        + $\text{cn\_soundarajan\_hopcroft}(E, I) = 1 + 1 = 2$
        + $\text{cn\_soundarajan\_hopcroft}(A, G) = 1 + 0 = 1$
    + IVQ: What is the Common Neighbor Soundarajan-Hopcroft score of node I and H?

        Ans: 2 <br/>
        Node I and H have only one common neighbor G. G is in the same community hence $f(u)=1$. The result is $1+1=2$.
    + Assign nodes to communities with attribute node "community"
        ```python
        # assign community
        G.node['A']['community'] = 0
        G.node['B']['community'] = 0
        G.node['C']['community'] = 0
        G.node['D']['community'] = 0
        G.node['E']['community'] = 1
        G.node['F']['community'] = 1
        G.node['G']['community'] = 1
        G.node['H']['community'] = 1
        G.node['I']['community'] = 1

        L = list(nx.cn_soundarajan_hopcroft(G))
        L.sort(key=operator.itemgetter(2), reverse = True)
        print(L)
        # [('A', 'C', 4), ('E', 'I', 2), ('E', 'H', 2), ('F', 'I', 2),
        #  ('F', 'H', 2), ('I', 'H', 2), ('A', 'G', 1), ('A', 'F', 1), ('C', 'E', 1),
        #  ('C', 'G', 1), ('B', 'E', 1), ('B', 'F', 1), ('E', 'D', 1), ('D', 'F', 1),
        #  ('A', 'I', 0), ('A', 'H', 0), ('C', 'I', 0), ('C', 'H', 0), ('B', 'I', 0),
        #  ('B', 'H', 0), ('B', 'G', 0), ('D', 'I', 0), ('D', 'H', 0), ('D', 'G', 0)]
        ```
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hvFPZ/link-prediction"> 
            <img src="images/m4-36.png" alt="the Common Neighbor Soundarajan-Hopcroft score. And if we're looking at nodes X and Y is simply going to be the number of common neighbors. So the size of intersection of N(X) and N(Y), plus some bonus, which is going to be the sum over all the common neighbors of this function, f(u). And f(u) is simply an indicator that tells us whether the u, which is a common neighbor of X and Y, belongs to the same community as X and Y, or not. And if it does, then it's a 1. If not, it's a 0.  And now we can use the function cn_soundarajan_hopcroft(G) which outputs an iterator of the tuples with the nodes and the score for each one, each pair, that is not already connected by an edge. And we can sort it and find which nodes have the highest score. And if we look at the two edges that we've been following, then we find that I, H has a score of 2, because their common neighbor G belongs to the same community as A do, whereas A, G has a score of 1." title="Example graph for Community Common Neighbors" height="250">
        </a>

+ Measure 7: Community Resource Allocation
    + Similar to resource allocation index, but only considering nodes in the same community
    + The Resource Allocation Soundarajan-Hopcroft score of nodes 𝑋 and 𝑌 is:

        $$\text{ra\_soundarajan\_hopcroff}(X, Y) = |N(X) \cap N(Y)| + \sum_{u \in N(X) \cap N(Y)} \frac{f(u)}{|N(u)|}$$

        $$f(u) = \left\{ \begin{array}{ll}
            1, & u \text{ in same comm. as } X \text{and } Y \\
            0, & \text{ otherwise}
            \end{array} \right.$$
    + Similar to resource allocation index, but only considering nodes in the same community
        + $\text{ra\_soundarajan\_hopcroft}(A, C) = \frac{1}{3} + \frac{1}{3} = \frac{2}{3}$
        + $\text{ra\_soundarajan\_hopcroft}(E, I) = \frac{1}{4}$
        + $\text{ra\_soundarajan\_hopcroft}(A, G) = 0$
    + Similar to resource allocation index, but only considering nodes in the same community
        ```python
        L = list(nx.ra_index_soundarajan_hopcroft(G))
        L.sort(key=operator.itemgetter(2), reverse = True)
        print(L)
        # [('A', 'C', 0.6666666666666666), ('E', 'I', 0.25), ('E', 'H', 0.25),
        #  ('F', 'I', 0.25), ('F', 'H', 0.25), ('I', 'H', 0.25), ('A', 'I', 0),
        #  ('A', 'H', 0), ('A', 'G', 0), ('A', 'F', 0), ('C', 'I', 0), ('C', 'H', 0),
        #  ('C', 'E', 0), ('C', 'G', 0), ('B', 'I', 0), ('B', 'H', 0), ('B', 'E', 0),
        #  ('B', 'G', 0), ('B', 'F', 0), ('E', 'D', 0), ('D', 'I', 0), ('D', 'H', 0),
        #  ('D', 'G', 0), ('D', 'F', 0)]
        ```
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hvFPZ/link-prediction"> 
            <img src="images/m4-36.png" alt="the Resource Allocation index but it only takes into account nodes that are in the same community as the two nodes we're looking at. So if we're computing this measure which is called the Resource Allocation Soundarajan-Hopcroft score. And this is after the researchers that came up with this measure of X and Y. Then what we do is we sum over all the neighbors of X and Y. And rather than summing just one over the degree of the nodes of the common neighbors like we did in the standard Resource Allocation index. We now have this f(u) in the denominator of the fraction. And this function f(u) again is the same as before is 1, if u belongs to the same community as X and Y, and 0 otherwise. So in the case where you have a common neighbor that does not belong to the same community as X and Y, then that neighbor is not contributing anything to the sum because you have a 0 in the numerator.  And if we look at for example, nodes E and I, we find that well, they have one common node which is node G. And node G has a degree of 4, so it has a score of 1 over 4 because G belongs to same community as E and I. However, if we look at two nodes that are not in the same community like A and G. Then they would have score 0 because their common neighbor, while they have one, namely E, belongs to a different community as at least one of the two nodes." title="Example graph for Community Resource Allocation" height="250">
        </a>

+ Summary
    + Link prediction problem: Given a network, predict which edges will be formed in the future.
        + 5 basic measures:
            + Number of Common Neighbors
            + Jaccard Coefficient
            + Resource Allocation Index
            + Adamic-Adar Index
            + Preferential Attachment Score
        + 2 measures that require community information:
            + Common Neighbor Soundarajan-Hopcroft Score
            + Resource Allocation Soundarajan-Hopcroft Score

+ `nx.non_edges` function
    + Signature: `nx.non_edges(graph)`
    + Docstring: Returns the non-existent edges in the graph.
    + Parameters
        + `graph` (NetworkX graph.): Graph to find non-existent edges.
    + Returns: `non_edges` (iterator)

+ `nx.common_neighbors` function
    + Signature: `nx.common_neighbors(G, u, v)`
    + Docstring: Return the common neighbors of two nodes in a graph.
    + Parameters
        + `G` (graph): A NetworkX undirected graph.
        + `u`, `v` (nodes): Nodes in the graph.
    + Returns: `cnbors` (iterator): Iterator of common neighbors of u and v in the graph.

+ `nx.jaccard_coefficient` function
    + Signature: `nx.jaccard_coefficient(G, ebunch=None)`
    + Docstring: Compute the Jaccard coefficient of all node pairs in `ebunch`.
    + Note: Jaccard coefficient of nodes `u` and `v` is defined as

        $$\frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}$$
        + $\Gamma(u)$: the set of neighbors of `u`
    + Parameters
        + `G` (graph): A NetworkX undirected graph.
        + `ebunch` (iterable of node pairs, optional (default = None)): Jaccard coefficient will be computed for each pair of nodes given in the iterable. The pairs must be given as 2-tuples `(u, v)` where `u` and `v` are nodes in the graph. If `ebunch` is None then all non-existent edges in the graph will be used. Default value: None.
    + Returns: `piter` (iterator): An iterator of 3-tuples in the form `(u, v, p)` where `(u, v)` is a pair of nodes and p is their Jaccard coefficient.

+ `nx.resource_allocation_index` function
    + Signature: `nx.resource_allocation_index(G, ebunch=None)`
    + Docstring: Compute the resource allocation index of all node pairs in `ebunch`.
    + Note: Resource allocation index of `u` and `v` is defined as

        $$\sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{|\Gamma(w)|}$$
        + $\Gamma(u)$: the set of neighbors of `u`.
    + Parameters
        + `G` (graph): A NetworkX undirected graph.
        + `ebunch` (iterable of node pairs, optional (default = None)): Resource allocation index will be computed for each pair of nodes given in the iterable. The pairs must be given as 2-tuples `(u, v)` where `u` and `v` are nodes in the graph. If `ebunch` is None then all non-existent edges in the graph will be used. Default value: None.
    + Returns: `piter` (iterator): An iterator of 3-tuples in the form `(u, v, p)` where `(u, v)` is a pair of nodes and `p` is their resource allocation index.
    + References: T. Zhou, L. Lu, Y.-C. Zhang. [Predicting missing links via local information](http://arxiv.org/pdf/0901.0553.pdf). Eur. Phys. J. B 71 (2009) 623.

+ `nx.adamic_adar_index` function
    + Signature: `nx.adamic_adar_index(G, ebunch=None)`
    + Docstring: Compute the Adamic-Adar index of all node pairs in `ebunch`.
    + Note: Adamic-Adar index of `u` and `v` is defined as

        $$\sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}$$
        + $\Gamma(u)$: the set of neighbors of `u`.
    + Parameters
        + `G` (graph): NetworkX undirected graph.
        + `ebunch` (iterable of node pairs, optional (default = None)): Adamic-Adar index will be computed for each pair of nodes given in the iterable. The pairs must be given as 2-tuples `(u, v)` where `u` and `v` are nodes in the graph. If ebunch is None then all non-existent edges in the graph will be used. Default value: None.
    + Returns: `piter` (iterator): An iterator of 3-tuples in the form `(u, v, p)` where `(u, v)` is a pair of nodes and `p` is their Adamic-Adar index.
    + References: D. Liben-Nowell, J. Kleinberg. [The Link Prediction Problem for Social Networks](http://www.cs.cornell.edu/home/kleinber/link-pred.pdf) (2004).

+ `nx.perferential_attachment` function
    + Signature: `nx.preferential_attachment(G, ebunch=None)`
    + Docstring: Compute the preferential attachment score of all node pairs in `ebunch`.
    + Note: Preferential attachment score of `u` and `v` is defined as

        $$|\Gamma(u)| |\Gamma(v)|$$
        + $\Gamma(u)$: the set of neighbors of `u`
    + Parameters
        + `G` (graph): NetworkX undirected graph.
        + `ebunch` (iterable of node pairs, optional (default = None)): Preferential attachment score will be computed for each pair of nodes given in the iterable. The pairs must be given as 2-tuples `(u, v)` where `u` and `v` are nodes in the graph. If ebunch is None then all non-existent edges in the graph will be used. Default value: None.
    + Returns: `piter` (iterator): An iterator of 3-tuples in the form `(u, v, p)` where `(u, v)` is a pair of nodes and `p` is their preferential attachment score.

+ `nx.cn_soundarajan_hopcroft` function
    + Signature: `nx.cn_soundarajan_hopcroft(G, ebunch=None, community='community')`
    + Docstring: Count the number of common neighbors of all node pairs in ebunch using community information.
    + Note: For two nodes `u` and `v`, this function computes the number of common neighbors and bonus one for each common neighbor belonging to the same community as `u` and `v`. Mathematically,

        $$|\Gamma(u) \cap \Gamma(v)| + \sum_{w \in \Gamma(u) \cap \Gamma(v)} f(w)$$
        + $f(w)$:
            + 1 if `w` belongs to the same community as `u` and `v`
            + 0 otherwise
        + $\Gamma(u)$: the set of neighbors of `u`
    + Parameters
        + `G` (graph): A NetworkX undirected graph.
        + `ebunch` (iterable of node pairs, optional (default = None)): The score will be computed for each pair of nodes given in the iterable. The pairs must be given as 2-tuples `(u, v)` where `u` and `v` are nodes in the graph. If ebunch is None then all non-existent edges in the graph will be used. Default value: None.
        + `community` (string, optional (default = 'community')): Nodes attribute name containing the community information. `G[u][community]` identifies which community `u` belongs to. Each node belongs to at most one community. Default value: 'community'.
    + Returns: `piter` (iterator): An iterator of 3-tuples in the form `(u, v, p)` where `(u, v)` is a pair of nodes and p is their score.
    + References: Sucheta Soundarajan and John Hopcroft. [Using community information to improve the precision of link prediction methods](http://doi.acm.org/10.1145/2187980.2188150). In Proceedings of the 21st international conference companion on World Wide Web (WWW '12 Companion). ACM, New York, NY, USA, 607-608.

+ `nx.ra_index_soundarajan_hopcroft` function
    + Signature: `nx.ra_index_soundarajan_hopcroft(G, ebunch=None, community='community')`
    + Docstring: Compute the resource allocation index of all node pairs in ebunch using community information.
    + Note: For two nodes `u` and `v`, this function computes the resource allocation index considering only common neighbors belonging to the same community as `u` and `v`. Mathematically,

        $$\sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{f(w)}{|\Gamma(w)|}$$
        + $f(w)$:
            + 1 if `w` belongs to the same community as `u` and `v`
            + 0 otherwise 
        + $\Gamma(u)$: the set of neighbors of `u`.
    + Parameters
        + `G` (graph): A NetworkX undirected graph.
        + `ebunch` (iterable of node pairs, optional (default = None)): The score will be computed for each pair of nodes given in the iterable. The pairs must be given as 2-tuples `(u, v)` where `u` and `v` are nodes in the graph. If ebunch is None then all non-existent edges in the graph will be used. Default value: None.
        + `community` (string, optional (default = 'community')): Nodes attribute name containing the community information. `G[u][community]` identifies which community u belongs to. Each node belongs to at most one community. Default value: 'community'.
    + Returns: `piter` (iterator): An iterator of 3-tuples in the form (u, v, p) where (u, v) is a pair of nodes and p is their score.
    + References: Sucheta Soundarajan and John Hopcroft. [Using community information to improve the precision of link prediction methods](http://doi.acm.org/10.1145/2187980.2188150). In Proceedings of the 21st international conference companion on World Wide Web (WWW '12 Companion). ACM, New York, NY, USA, 607-608.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/T2Y25ZTLEeeRmQ5TE1Qolg.processed/full/360p/index.mp4?Expires=1549584000&Signature=B-HnMCFqSq0itHv2cnUluG6Zh4d2HKl11W14Y038LHmUuQMhKccn6glaOITcSyt47fzBmZRmoxmNpLUnryt9kKP0m6w9qRQ52JXU76kgb22fY15ieBxGCvfGDIpfdEsSCZkeO9lCGQxpBuVfvj8ziZ7yvd8b-9CvOqhfzhmC5PA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Link Prediction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Notebook: Extracting Features from Graphs

### Notebook Sources

+ [Launchiong Web Page](https://www.coursera.org/learn/python-social-network-analysis/notebook/ntrdh/extracting-features-from-graphs)
+ [Web Notebook](https://bajwjsbbpcxhnmzzoyjrrp.coursera-apps.org/notebooks/Graph%20Features.ipynb)
+ [Local Notebook](notebooks/04-GraphFeatures.ipynb)
+ [Python Code](notebooks/04-GraphFeatures.py)


### Python Code with Results

```python
# In this notebook we will look at a few ways to quickly create a 
# feature matrix from a networkx graph.
import networkx as nx
import pandas as pd

G = nx.read_gpickle('major_us_cities')

# ## Node based features
G.nodes(data=True)
# [('El Paso, TX', {'location': (-106, 31), 'population': 674433}),
#  ('Long Beach, CA', {'location': (-118, 33), 'population': 469428}),
#  ('Dallas, TX', {'location': (-96, 32), 'population': 1257676}),
#  ...
#  ('Jacksonville, FL', {'location': (-81, 30), 'population': 842583})]

# Initialize the dataframe, using the nodes as the index
df = pd.DataFrame(index=G.nodes())

# ### Extracting attributes
# 
# Using `nx.get_node_attributes` it's easy to extract the node attributes in the graph into DataFrame columns.
df['location'] = pd.Series(nx.get_node_attributes(G, 'location'))
df['population'] = pd.Series(nx.get_node_attributes(G, 'population'))

df.head()
#                 location    population
# El Paso, TX     (-106, 31)    674433
# Long Beach, CA  (-118, 33)    469428
# Dallas, TX      (-96, 32)    1257676
# Oakland, CA     (-122, 37)    406253
# Albuquerque, NM (-106, 35)    556495

# ### Creating node based features
# 
# Most of the networkx functions related to nodes return a dictionary, which can also easily be added to our dataframe.
df['clustering'] = pd.Series(nx.clustering(G))
df['degree'] = pd.Series(G.degree())

df
#                 location    population    clustering    degree
# El Paso, TX     (-106, 31)    674433        0.700000        5
# Long Beach, CA  (-118, 33)    469428        0.745455        11
# Dallas, TX      (-96, 32)    1257676        0.763636        11
# ...

# # Edge based features
G.edges(data=True)
# [('El Paso, TX', 'Albuquerque, NM', {'weight': 367.88584356108345}),
#  ('El Paso, TX', 'Mesa, AZ', {'weight': 536.256659972679}),
#  ('El Paso, TX', 'Tucson, AZ', {'weight': 425.41386739988224}),
#  ...
#  ('Columbus, OH', 'Virginia Beach, VA', {'weight': 701.8766661783677})]


# Initialize the dataframe, using the edges as the index
df = pd.DataFrame(index=G.edges())

# ### Extracting attributes
# 
# Using `nx.get_edge_attributes`, it's easy to extract the edge attributes in the graph into DataFrame columns.
df['weight'] = pd.Series(nx.get_edge_attributes(G, 'weight'))

df
# weight
# (El Paso, TX, Albuquerque, NM)  367.885844
# (El Paso, TX, Mesa, AZ)         536.256660
# (El Paso, TX, Tucson, AZ)       425.413867
# ...

# ### Creating edge based features
# 
# Many of the networkx functions related to edges return a nested data structures. We can extract the relevant data using list comprehension.
df['preferential attachment'] = [i[2] for i in nx.preferential_attachment(G, df.index)]

df
#                                 weight      preferential attachment
# (El Paso, TX, Albuquerque, NM)  367.885844    35
# (El Paso, TX, Mesa, AZ)         536.256660    40
# (El Paso, TX, Tucson, AZ)       425.413867    40
# ...

# In the case where the function expects two nodes to be passed in, we can map the index to a lamda function.
df['Common Neighbors'] = df.index.map(lambda city: len(list(nx.common_neighbors(G, city[0], city[1]))))

df
#                                 weight      preferential    Common
#                                             attachment      Neighbors
# (El Paso, TX, Albuquerque, NM)  367.885844    35            4
# (El Paso, TX, Mesa, AZ)         536.256660    40            3
# (El Paso, TX, Tucson, AZ)       425.413867    40            3
# ...
```

## Quiz: Module 4 Quiz

Q1: Suppose P(k) denotes the degree distribution of the following network, what is the value of P(2) + P(3)?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-1.png" alt="Q1 Graph" title="Q1 Graph" height="200">
    </a>

    a. 1/6
    b. 1/3
    c. 1/2
    d. 5/6

    Ans: c
    P(2) = 1/6, P(3) = 2/6


Q2: Let P(k) denote the in-degree distribution of the given network below. What value of k gives the highest value of P(k)?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-2.png" alt="Q2 Graph" title="Q2 Graph" height="150">
    </a>

    a. 1
    b. 2
    c. 3
    d. 0

    Ans: a


Q3: Select all that apply

    a. Networks with a power law distribution have many nodes with large degree and a few nodes with very small degree.
    b. If we draw a power law distribution in log-log scale, the distribution will look like a straight line.
    c. In the Preferential Attachment Model, a new node always connects to the node with highest in-degree.
    d. The Preferential Attachment Model generates a network with a power law degree distribution.

    Ans: bd
    a. False; d: True


Q4: Select all that apply

    a. The degree distribution of small-world networks follows power-law distribution.
    b. Some Small-world networks have high local clustering coefficient and small average shortest path.
    c. The Preferential Attachment Model generates a small-world network.
    d. Small-world networks are always connected.
    e. In the small-world model starting with k nearest neighbors, increasing the rewiring probability p generally decreases both the average clustering coefficient and average shortest path.

    Ans: be
    a: False; c: False; d: False


Q5: Suppose we want to generate several small-world networks with $k$ nearest neighbors and rewiring probability $p$. If $p$ remains the same and we increase k, which best describes the variation of average local clustering coefficient and average shortest path?

    a. Both of them will increase.
    b. Both of them will decrease.
    c. Average local clustering coefficient will increase and average shortest path will decrease.
    d. Average local clustering coefficient will decrease and average shortest path will increase.

    Ans: c, xd


Q6: Based on the network below, suppose we want to apply the common neighbors measure to add an edge from node H, which is the most probable node to connect to H?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-6.png" alt="Q6 Graph" title="Q6 Graph" height="200">
    </a>

    a. A
    b. B
    c. C
    d. G

    Ans: a


Q7: Based on the network below, what is the Jaccard coefficient of nodes D and C?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-6.png" alt="Q7 Graph" title="Q7 Graph" height="200">
    </a>

    a. 0.29
    b. 0.33
    c. 0.40
    d. 0.50

    Ans: c
    jacc_coeff(D, C) = |{A, G}| / |{AB, E, G, H}| = 2/5 = 0.4


Q8: Based on the network below, if we apply Resource Allocation method to predict the new edges, what is the value of Resource Allocation index of nodes C and D?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-6.png" alt="Q8 Graph" title="Q8 Graph" height="200">
    </a>

    a. 0.20
    b. 0.33
    c. 0.70
    d. 0.83

    Ans: d
    res_alloc(C, D) = 1/3 + 1/2 = 5/6


Q9: Based on the network below, what is the preferential attachment score of nodes C and D?
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-6.png" alt="Q9 Graph" title="Q9 Graph" height="200">
    </a>

    a. 5
    b. 8
    c. 10
    d. 15

    Ans: c
    pref_attach(C, D) = 2 * 5 = 10


Q10: Assume there are two communities in this network: `{A, B, C, D, G}` and `{E, F, H}`. Which of the following statements is(are) True? Select all that apply.
    <a href="https://www.coursera.org/learn/python-social-network-analysis/exam/CgIV0/module-4-quiz"> <br/>
        <img src="images/q4-6.png" alt="Q9 Graph" title="Q9 Graph" height="200">
    </a>

    a. The Common Neighbor Soundarajan-Hopcroft score of node C and node D is 2.
    b. The Common Neighbor Soundarajan-Hopcroft score of node A and node G is 4.
    c. The Resource Allocation Soundarajan-Hopcroft score of node E and node F is 0.
    d. The Resource Allocation Soundarajan-Hopcroft score of node A and node G is 0.7

    Ans: bd
    a. cn_soundarajan_hopcroft(C, D) = 2 + 2 = 4
    c. ra_index_soundarajan_hopcroft(E, F) = 1/3




## The Small-World Phenomenon (Optional)

Read chapters 2 and 20 from "[Networks, Crowds, and Markets: Reasoning about a Highly Connected World](http://www.cs.cornell.edu/home/kleinber/networks-book/)" By David Easley and Jon Kleinberg. Cambridge University Press, 2010 for a more in-depth take on the Small World Phenomenon.

+ [Chapter 2: Graphs](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch02.pdf)
    + [Basic Definitions](p2-Graphs.md#basic-definitions)
    + [Paths and Connectivity](p2-Graphs.md#paths-and-connectivity)
    + [Distance and Breadth-First Search](p2-Graphs.md#distance-and-breadth---first-search)
    + [Network Datasets: An Overview](p2-Graphs.md#network-datasets-an-overview)

+ [Chapter 20:](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch20.pdf)




