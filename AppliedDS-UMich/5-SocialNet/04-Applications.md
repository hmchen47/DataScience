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
    + Choose the node to connect to at random with probability proportional to each node’s degree.
    + The probability of connecting to a node $u$ of degree $k_u$ is $k_u/\sum_j k_j$.
    + As the number of nodes increases, the degree distribution of the network under the preferential attachment model approaches the power law $𝑃(k) = Ck^{-3}$ with constant $𝐶$.
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
            <img src="images/m4-04.png" alt="text" title="Preferential Attachment Model: 3 nodes" height="150">
            <img src="images/m4-05.png" alt="text" title="Preferential Attachment Model: 4 nodes" height="150">
            <img src="images/m4-06.png" alt="text" title="Preferential Attachment Model: 5 nodes" height="150">
            <img src="images/m4-07.png" alt="text" title="Preferential Attachment Model: 6 nodes" height="200">
            <img src="images/m4-02.png" alt="text" title="Preferential Attachment Model: 7 nodes" height="200">
        </a>
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> 
            <img src="https://lh3.googleusercontent.com/psmc0XAoiIJH0Nt-Ih35ZjtYxSY-nYdImizg2yW3G0AvJzwYbI_7HSJFJwowI-OzDKNHkA0jJl1xZ86vtPy01btFD2_JAegkFjZ4zBa_-8Ko9lJIF7YFwZqeODcaI84rioaPm_o6zQ=w2400" alt="text" title="caption" height="200">
        </a>
    + IVQ: What is the probability that node 8 attaches to node 3?
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/abipd/preferential-attachment-model"> <br/>
            <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/BMmbx4wWEee9xRJas1XV1A_5bc5f7c50053380f9ceccea18a941082_PrefAttachIVQ.png?expiry=1549497600000&hmac=O-i_CXNm-_BhhVJbPTRLxEEbxi7PQAYRrsiI7blF3Po" alt="text" title="IVQ: Preferential Attachment Model w/ 8 nodes" height="200">
        </a>


  Let's say, it chooses node two. Now, node four comes in, but now, node two has degree of two, and nodes one and three have degree one, and so the probability of attaching to node two is 0.5, and the probability of attaching to nodes one or three is 0.25. Node four attaches to node three, which had a lower probability. That could happen. So it attaches to node three. Now, node five comes in, and we have to recompute all the probabilities. Well, now, nodes two and three both have degree two, and nodes one and four have degree one. So nodes two and three are going to have a higher probability of attaching, and so that's 0.33 for nodes two and three, and 0.17 for nodes one and four. Let's say, five attaches to node two, and we continue this node six, again, we recompute the probabilities. Now, node two has the highest degree, so we will have the highest probability of getting that new attachment. So six, let's say, attaches to two. Now, seven comes in, we recompute the probabilities. Again, now, node two has an even higher degree so it has an ever higher probability of getting that new node. Second is node three that has a degree or two with probability of getting that new node edge at 0.2, and seven attaches to two. Node eight comes in, we recompute the probabilities. Node two probability becomes even larger, but this node eight attaches to node three, and so on. You can see how this continues. The thing to notice here is that as node two started to get larger and larger degree, its probability of getting a new edge became larger and larger as well. There is this sort of rich get richer phenomenon, where as the nodes get larger and larger degree, they also start to become more and more likely to increase their degree. What we can prove about this particular mechanism is that it gives rise to a power law. It approaches a power law as the number of nodes gets larger, and larger, and larger. So it matches the kind of degree distribution that we see in these real networks. This type of modeling technique allows us to explain or at least have some hypothesis for what kind of mechanism could give and rise to this shape of the degree distribution that we observe. For example, if we believe that a very popular actor that has appeared with many other actors in movies has a higher likelihood of getting an additional actor to co-appear in a movie than a maybe less popular actor that has not appeared with many other actors in movies, then this is the kind of mechanism that could be explaining the sort of very skewed power law distribution that we observed. In Network X, you can use the function, barabasi_albert_graph, which is named after the researchers that came up with this model, with input powers n and m, where n is the number of nodes and m is the number of new nodes that an arriving node would attach to. In our example, the way we define it, this m parameter would be one because we said that every new node would attach to only a single existing node, but you can generalize this and have it so that every node attaches to m existing nodes, and that m will not change the fact that you still get a power law degree distribution. You can use this function in Network X to create networks that follow these Preferential Attachment Model. Let's create one here with a million nodes and an m(1), so every node attaches to a single existing node. Now, let's plot the degree distribution in the same way that we did before except that now I'm not using a bar plot, I'm using a scatter plot, and now I'm setting the scales, the y-scale and x-scales to be logged so we can see that straight line that gets formed on the log-log scale for the degree distribution. Here it is. We see that it forms the straight line that characterizes the power law degree distribution. So in summary, we have the degree distribution of a graph is the probability distribution of the degrees over the entire network, and in many real networks that we see, this degree distribution tends to look sort of like a power law because it looks sort of like a straight line on a log-log scale. We also discussed that models of networks generation allows to identify mechanisms that give rise to patterns that we observe in real networks. In this case, we observed that many real networks tend to have these power law degree distribution, and then we covered a model that gives rise to this particular property which allows us to have some insight about the kinds of things that could be given rise to these properties in real networks. In this case, this was the Preferential Attachment Model that produces these networks with power law degree distribution. In Network X, you can use this function, barabasi_albert_graph to construct networks that have n nodes and where every single node attaches to m existing nodes following the Preferential Attachment Model. That's all for today. I hope to see you next time.

+ Summary
    + The degree distribution of a graph is the probability distribution of the degrees over the entire network.
    + Many real networks have degree distributions that look like power laws ($P(k) = Ck^{-\alpha}$).
    + Models of network generation allow us to identify mechanisms that give rise to observed patterns in real data.
    + The Preferential Attachment Model produces networks with a power law degree distribution.
    + Use `barabasi_albert_graph(n,m)` to construct a n-node preferential attachment network, where each new node attaches to m existing nodes.



### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/lNURBZTMEeeOmgqEJWRlfA.processed/full/360p/index.mp4?Expires=1549497600&Signature=ZbRdvAcpqHNU0nMzfycqCvR4CKg5EohROkgICPmo5lpA0gHueOlj0q-Gx6XPn3g1Y2kac33geKa5kBntgmGYTtaqB0XrIMXL5XSs48jXxsQu2~NATd0DVRQaHqCh1t8c~00H36bVh3A-x7EbW-I7u-egb1buQCx1L55kH9rjSa4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Preferential Attachment Model" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Reading: ReadingPower Laws and Rich-Get-Richer Phenomena (Optional)

Read [Chapter 18]((http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch18.pdf)) from "Networks, Crowds, and Markets: Reasoning about a Highly Connected World" By David Easley and Jon Kleinberg. Cambridge University Press, 2010 for an interesting read on Power Laws and Rich-Get-Richer phenomena such as the preferential attachment model.


## Small World Networks

### Lecture Notes



### Lecture Video

<a href="url" alt="Small World Networks" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Link Prediction

### Lecture Notes



### Lecture Video

<a href="url" alt="Link Prediction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Notebook: Extracting Features from Graphs




## Quiz: Module 4 Quiz




## Reading: ReadingThe Small-World Phenomenon (Optional)

Read chapters 2 and 20 from "Networks, Crowds, and Markets: Reasoning about a Highly Connected World" By David Easley and Jon Kleinberg. Cambridge University Press, 2010 for a more in-depth take on the Small World Phenomenon.

+ [Chapter 2](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch02.pdf)
+ [Chapter 20:](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch20.pdf)




