# Module 4: Applications

## Preferential Attachment Model

### Lecture Notes

+ Degree Distributions
    + The __degree__ of a node in an undirected graph is the number of neighbors it has. Eg., A(3), B(2), C(3), D(2), E(1_, F(3), G(4), H(2), I(2)
    + The __degree distribution__ of a graph is the probability distribution of the degrees over the entire network.
    + The degree distribution, $𝑃(𝑘)$, of this network has the following values:

        $$P(1) = \frac{1}{9}, P(2) = \frac{4}{9}, P(3)=\frac{1}{3}, P(4) = \frac{1}{9}$$
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
            <img src="https://lh3.googleusercontent.com/P_mxP3SNySiu36dL3jwWq06zuEVuB1jtC3xSQ0cIJCh50TlZH7DgS4D-6Pgm5f8LGVZSCONtIg65n3IeZ_7f_69MEpm8tx32SKUFBzEPvcIiLOll-MY1c7q9R9pGM9u-6QH-VyhDbQ=w2400" alt="text" title="caption" height="200">
            <img src="https://lh3.googleusercontent.com/1v-beDXVxIqEEj17jLP13jycyPNuok2BY1jR1KB28c6Wt6caiozyCVK6hgIeTYcpou3u8AWOqJguMqwtehy690_Hi2ippr-iGt7mtc31NE2L4eHyiUHvN7aSBBnrQPklH9yDNgCsIQ=w2400" alt="text" title="caption" height="200">
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
            <img src="images/m4-01.png" alt="text" title="caption" height="200">
            <img src="https://lh3.googleusercontent.com/1v-beDXVxIqEEj17jLP13jycyPNuok2BY1jR1KB28c6Wt6caiozyCVK6hgIeTYcpou3u8AWOqJguMqwtehy690_Hi2ippr-iGt7mtc31NE2L4eHyiUHvN7aSBBnrQPklH9yDNgCsIQ=w2400" alt="text" title="caption" height="200">
        </a>

+ Degree Distributions in Real Networks
    + A – __Actors__: network of 225,000 actors connected when they appear in a movie together.
    + B – __The Web__: network of 325,000 documents on the WWW connected by URLs.
    + C – __US Power Grid__: network of 4,941 generators connected by transmission lines.
    + Degree distribution looks like a straight line when on a log-log scale. __Power law__: $P(k)=Ck^{-\alpha}$, where $\alpha$ and C are constants. $\alpha$ values: A: 2.3, B:2.1, C:4.
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> <br/>
        <img src="https://lh3.googleusercontent.com/rZEP7WDAdwWkHjlFWGRdmnUZueZsL5uG_71R7HYoj87gicbjACxwr3FFDkekP0XWOSto-el8GSPFsCx9heQitepqWTlIYTcfrqWqWhp_0F0MRacvi6hE_twNh_7SGXLfY-XQaYHOgg=w2400" alt="text" title="caption" height="250">
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
            <img src="images/m4-02.png" alt="text" title="caption" height="200">
        </a>
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/4/"> 
            <img src="https://lh3.googleusercontent.com/psmc0XAoiIJH0Nt-Ih35ZjtYxSY-nYdImizg2yW3G0AvJzwYbI_7HSJFJwowI-OzDKNHkA0jJl1xZ86vtPy01btFD2_JAegkFjZ4zBa_-8Ko9lJIF7YFwZqeODcaI84rioaPm_o6zQ=w2400" alt="text" title="caption" height="200">
        </a>

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




