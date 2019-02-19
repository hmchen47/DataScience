# Small-World Phenomenon

+ [Networks, Crowds, and Markets: Reasoning About a Highly Connected World](http://www.cs.cornell.edu/home/kleinber/networks-book/http://www.cs.cornell.edu/home/kleinber/networks-book/)
+ [Chapter 20. The Small-World Phenomenon](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch20.pdf)

## Six Degrees of Separation

+ Social networks: serve as conduits by which ideas and innovations flow through groups of people

+ Basic structure issue
    + these groups can be connected by very short paths through the social network
    + Using short paths to reach others who are social distant, engaging in a kind of "focused" search that is much more targeted than the broad spreading pattern exhibited by the diffusion of information or a new behavior

+ Small-worl phenomenon or "Six-degree of separation"
    + Stanley Milgram: randomly chosen "starter" individuals to each try forwarding aa letter to a designated "target" person living in the town of Sharon, MA, a suburb of Boston
    + Roughly $1/3$ letters eventually arrived at the target, in a median of six steps
    A basic experimental evidence for the existence of short paths in the global friendship network, linking all (or almost all) of us together in society.
    + Two striking facts:
        1. abundant short paths
        2. effective at collectively finding these short paths without any sort of global "map" of this network

+ Forwarding strategies
    + The real global friendship network contains enough clues about how people fit together in the larger structures to allow the preocess of search to focus in on distant targets.
    + People employ for choosing how to forward a message toward a targets.
    + Killworth and Bernard: a mixture of primarily geographical and occupational features being used, with different features being favored depending on the characteristics of the target in relation to the sender.
    + People are most successful at finding paths when the target is high-status and socially accessible.



## Structure and Randomness

+ Models for the existence of short paths
    + short paths are at lease compatible with intitution
    <a href="http://www2.unb.ca/~ddu/6634/Lecture_notes/Lec8_smallworld.pdf"> <br/>
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRUXYVumoSkeiIjtNz4GtzOgz6r44A2fAfiadfADLfGo7kgIyES" alt="Network grows exponentially, leading to the the existence of short paths! The average person has between 500 and 1500 acquaintances, leading to 5002 = 25K in one step, 5003 = 125M in two steps, 5004 = 62.5B in four (Figure (a)). However, the effect of triadic closure works to limit the number of people you can reach by following short paths (Figure (b)). Triadic closure: If two people in a social network have a friend in common, then there is an increased likelihood that they will become friends themselves at some point in the future. Question: Can we make up a simple model that exhibits both of the features: many closed triads (high clustering), but also very short path (small-world)?" title="Social networks expand to reach many people in only a few steps." height="300">
    </a>
    + Exponential growth: the numbers are growing by powers of 100 with each step, bringing us to 100 million after four steps, and 10 billion after five steps.
    + The effect of triadic closure in social networks works to limit the number of people when reaching by following short paths.
    + The social network appears from the local perspective of any one individual to be highly clustered, not the kind of massively branching structure that would more obviously reach many nodes along very short paths.

+ The Watts-Strogatz Model
    + A model follows naturally from a combination of two basic social-network idea: (Chapters 3 & 4)
        1. __homophily__: the principle that connect to others who are like ourselves
        2. __weak ties__: the links to acquaintances that connect us to parts of the network that would otherwise be far away
    + Homophily creates many triangles while the weak ties still produce the kind of widely branching structure that reaches many nodes in a few steps.
    <a href="https://slideplayer.com/slide/8594393/"> <br/>
        <img src="https://images.slideplayer.com/26/8594393/slides/slide_4.jpg" alt="the set of nodes arranged on a grid; we say that two nodes are one grid step apart if they are directly adjacent to each other in either the horizontal or vertical direction." title="The Watts-Strogatz model arises from a highly clustered network (such as the grid), with a small number of random links added in." height="250">
        <img src="https://player.slideplayer.com/26/8594393/data/images/img4.jpg" alt="Suppose, for example, that instead of allowing each node to have k random friends, we only allow one out of every k nodes to have a single random friend — keeping the proximitybased edges as before," title="The general conclusions of the Watts-Strogatz model still follow even if only a small fraction of the nodes on the grid each have a single random link." height="250">
    </a>
    + The set of nodes arranged on a grid: two nodes are one grid step apart if they are directly adjacent to each other in either the horizontal or vertical direction.
    + Two kinds of links:
        + Homophily captured by having each node form a link to all other nodes that lie within radius of up to $r$ grid steps away, for some constant value of $r$.
        + The links formed to similar people.
    + For some other constant value $k$, each node also forms a link to $k$ other nodes selected uniformly at random from the grid.
    + Fig.20.2 (b), a schematic picture of the resulting network: a hybrid structure consisting of a small amount of randomness (the weak ties) sprinkled onto an underlying structured pattern (the homophilous links).
    + Obgservation: the network has many triangles - any two neighboring nodes (or nearby nodes) will have many common friends, where their neighborhoods of radius $r$ overlap, and this produces many triangles.
    + With high probability, very short paths connecting every pair of nodes in the network.
    + Start tracing paths outward from a starting node $v$, using only the $k$ random weak ties out of each node.
    + Since these edges link to nodes chosen uniformly at random, very unlikely to ever see a node twice in the first few steps toward from $v$.
    + Only allow one out of every $k$ nodes to have a _single_ random friend -- keeping the proximity-based edges as before.
    + Conceptually group $k \times k$ subsquares of the grid into "towns".
    + Consider the small-world phenomenon at the level of towns.  Each town contains approximately $k$ people who each have a random friend.
    + The town collectively has $k$ links to other towns selected uniformly at random.
    + Find a short path between any two people: 
        1. find a short path between the two towns they inhabit
        2. use the proximity-based edges to turn this into an actual path in the network on individual people
    + The crux of the Watts-Strogatz model: introducing a tiny amount of randomness, in the form of long-range weak-ties, is enough to make the world "small," with short paths between every pair of nodes.



## Decentralized Search

+ Basic aspect of the Milfram small-world experiment
    + people were actually able to collectively find short paths to the designated target
    + Social search task: a necessary consequence of the way Milgram formulated the experiment for participants
    + the shortest path:
        + instruct the starter to forward a letter to _all_ of his/her friends, who in turn should have forwarded the letter to all of their friends, and so forth
        + flooding: reach the target as rapid as possible
        + breadth-first search: not a feaible option
    + "Tunneling" through the network: a process that could well hav failed to reach the target, even if a short path existed

+ The power of collective search
    + Q: why should it have been structured so as to make this type of _decentralized search_ so effective?
    + The Watts-Strogatz model: provide a simple framework for thinking about short path in highly clustered networks
    + Q: Can we construct a random network network in which decentralize routing succeeds, and if so, what are the qualitative properties that are crucial for success?

+ A model for decentralized search
    + Grid-based model of Watts-Strogatz: s starting node $s$ is given a message that it must forward to a target node $t$, passing it along edges of the network
    + Initially $s$ only knows the location of $t$ on the grid, but not knowing the random edges out of any node other than itself.
    + __delivery time__: the expected number of steps required to reach the target, over a randomly generated set of long-range contacts, and randomly chosen starting and target nodes
    + Decentralized search in the Watts-Strogatz model: require a large number of steps to reach a target
    Mathematical mode: effect at capturing the density of triangles and the existence of short paths, but not the ability of people, working together in the network, to actually find the paths
    + Problem: the weak ties that make the world small are "too random" in this model $\; \longrightarrow \;$ completely unrelated

+ To reach a far-away target:
    + Use long-range weak ties in a fairly structured, methodical way, constantly reducing the distance tot he target.
    + a progressive closing in on the target area as each new person is added to the chain
    + not enough to have a network model in which weak ties span only the very long ranges


## Modeling the Process of Decentralized Search

+ Generalizing the network model
    + nodes on a grid and edges to each other nodes within $r$ grid steps
    + each of its $k$ random edges is generated in a way that decays with distance, controlled by a __clustering exponent $q$__
    + $d(v, w)$: the number of grid steps between nodes $v$ and $w$
    + In generating a random edge out of $v$, edge link to $w$ with probability proportional to $d(v, w)^{-q}$
    + Different value $q$
        + $q=0$: original grid-based model
        + $q$ very small: the long-range links are "too random"; not used effectively for decentralized search
        + $q$ large: the long-range links are "not random enough"; not provide enough of the long-distance jumps that are needed to create a small world
        + optimal operating point?
    + decentralized search is most efficient with $q=2$
    <a href="https://slideplayer.com/slide/8594393"> <br/>
        <img src="https://player.slideplayer.com/26/8594393/data/images/img6.jpg" alt="Pictorially, this variation in q can be seen in the difference between the two networks in Figure 20.5." title="With a small clustering exponent, the random edges tend to span long distances on the grid; as the clustering exponent increases, the random edges become shorter." height="200">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlK5DrK_6PJPiuHF22m7mImGznEKhkehNB3BDyS5V6zelANU65zQ" alt="the performance of a basic decentralized search method across different values of q, for a network of several hundred million nodes." title="Simulation of decentralized search in the grid-based model with clustering exponent q. Each point is the average of 1000 runs on (a slight variant of) a grid with 400 million nodes. The delivery time is best in the vicinity of exponent q = 2, as expected; but even with this number of nodes, the delivery time is comparable over the range between 1.5 and 2" height="200">
    </a>
    + network size $\rightarrow \infty$: decentralized search has not about the same efficiency on networks of this size all exponents $q \in [1.5, 2.0]$
    + network size $\uparrow \implies$ the best performance occurs at exponents $q \rightarrow 2$

+ A Rough Calculation Motivating the Inverse-Square Network
    + decentralized search is efficient when $q=2$ and sketch why search is more efficient with $q=2$ than with any other exponent
    + effecive decentralized search "funnels inward" through these different scales of resolution
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img9.jpg" alt="So now let’s look at how the inverse-square exponent q = 2 interacts with these scales of resolution. We can work concretely with a single scale by taking a node v in the network, and a fixed distance d, and considering the group of nodes lying at distances between d and 2d from v," title="The concentric scales of resolution around a particular node." height="250">
    </a>
    + what is the probability that $v$ forms a link to some node inside this group?
    + With the sqaure of the radius, the total number of nodes in this group is proportional to $d^2$
    + The probability that $v$ links to any one node in the group varies depending on exactly how far out it is, but each individual probability is proportional to $d^{-2}$.
    + The probability that a random edge links into some node in this ring is approximately independent of the value of $d$.
    + $q=2$: long-range weak ties are being formed in a way that's spread roughly uniformly over all different scales of resolution.


## Empirical Analysis and Generalized Models

+ Geographic Data on Friendship
    + Liben-Nowell et al.: the blogging site LiveJournal of roughly 500,000 users w/ ZIP code
    + The population of the users is extremely non-uniform

+ Rank-Based Friendship
    + As a node $v$ looks out at all other nodes, it _ranks_ them by proximity
    + $\text{rank}(w)$: the rank of a node $w$
    + $\text{rank}(w)$ = the number of other nodes that are closerto $v$ than $w$ is
    + rank-based friendship with exponent $p$: for som eexponent $p$, node $v$ creates a random link that chooses a node $w$ as the other end with probability proportional to $\text{rank}(w)^{-q}$
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img13.jpg" alt="When the population density is non-uniform, it can be useful to understand how far w is from v in terms of its rank rather than its physical distance. In (a), we say that w has rank 7 with respect to v because it is the 7th closest node to v, counting outward in order of distance. In (b), we see that for the original case in which the nodes have a uniform population density, a node w at distance d from v will have a rank that is proportional to d2, since all the nodes inside the circle of radius d will be closer to v than w is." title="Figure 20.9(a), node w would have rank seven, since seven others nodes (including v itself) are closer to v than w is. Figure 20.9(b) shows, if a node w in a uniformly-spaced grid is at distance d from v, then it lies on the circumference of a disc of radius d, which contains about $d^2$ closer nodes — so its rank is approximately $d^2$." height="250">
    </a>
    + Fig (b): a node $w$ in a uniformly-spaced grid is at distance $d$ from $v \; \implies \;$ node $w$ lies on the circumference of a disc of radius $d$, which contains about $d^2$ closer node $\;\;\; \therefore$ its rank is approximately $d^2$
    + linking to $w$ with probability proportional to $d^{-2}$ is approximately the same as linking with probability $\text{rank}(w)^{-1} \; \implies \;$ exponent $p = 1$ is the right generalization of the inverse-sqaure distribution
    + For essentially any population density, if random links are constructed using rank-based friendship with exponent $1$, the resulting network allows for efficient decentaslized search with high probability.
    + quantative summary: to construct a network that is efficienly searchable, create a link to each node with probability that is inversely proportional to the number of closer nodes
    + Pairs of nodes where one assigns to other a rank of $r$
    + What fraction $f$ of these pairs are actually friends, as a function of $r$?
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img14.jpg" alt="Figure 20.10(a) shows this result for the LiveJournal data; we see that much of the body of the curve is approximately a straight line sandwiched between slopes of −1.15 and −1.2, and hence close to the optimal exponent of −1. It is also interesting to work separately with the more structurally homogeneous subsets of the data consisting of West-Coast users and East-Coast users, and when one does this the exponent becomes very close to the optimal value of −1. Figure 20.10(b) shows this result: The lower dotted line is what you should see if the points followed the distribution rank−1, and the upper dotted line is what you should see if the points followed the distribution rank−1.05. The proximity of the rankbased exponent on real networks to the optimal value of −1 has also been corroborated by subsequent research." title="The probability of a friendship as a function of geographic rank on the blogging site LiveJournal." height="250">
    </a>
    + Figure shows the close alignment of theory and measurement.
    + Why real social networks have arranged themselves in a pattern of friendships across distance that is close to optimal for forwarding messages to far-away targets.
    + dynamic forces or selective pressures driving the network toward this shape
    + Open problem: determine wether such forces exist and how they might operate

+ Socail Foci and Social Distance
    + social foci: provide a flexible and general way to produce models of networks exhibiting both an abundance of short paths and efficient decentralized search
    + a socail focus is any type of community, occupational pursuit, neighborhood, shared interest, or activity that serves to organize social life around it.
    + Foci are a way of summarizing the many possible reasons that two people can know each other or become friends - live on the same block, work at the same company, frequent the same cafe, or attend the same kinds of concerts.
    + a natural way to define the social distance between two people is to declare it to be the size of the smallest focus that includes both of them.
    + $\text{dist}(v, w)$: the social distance between node $v$ and $w$
    + Shared foci: $\text{dist}(v, w)$ is the size of the smallest focus that contains both $v$ and $w$
    + construct a link between each pair of nodes $v$ and $w$ with probability proportional to $\text{dist}(v, w)^{-p}$
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img18.jpg" alt="When nodes belong to multiple foci, we can define the social distance between two nodes to be the smallest focus that contains both of them. In the figure, the foci are represented by ovals; the node labeled v belongs to five foci of sizes 2, 3, 5, 7, and 9 (with the largest focus containing all the nodes shown)." title="The node labeled v construct links to three other nodes at social distances 2, 3, and 5." height="250">
    </a>
    + the node labeled $v$ construct links to three other nodes at social distances 2, 3, and 5.
    + Subject to some technical assumptions on the structure of the foci, that when links are generated this way with exponent $p=1$, the resulting network supports efficient decentralized search with high probability.
    + Aspects of result
        1. with rank-based friendship, a simple description of underlying principle: when nodes link to each other with probability inversely proportional to their social distance, the resulting network is efficiently searchable.
        2. the exponent $p=1$ is again the natural generalization of the inverse-square law for the simple grid model.
    + For each location $v$ on the grid, and each possible radius $r$ around that location, there is a focus consisting of all nodes who are within distance $r$ of $v$.
    + For two nodes who are a distance $d$ apart, their smallest shared focus has a number of nodes proportional to $d^{-2}$.
    + linking with probability proportional to $d^{-2}$ is essentially the same as linking with probability inversely proportional to their social distance.

+ Adamic and Adar
    + a social network on employees of Hewlett Parkard Research Lab
    + defined a focus for each of the groups within the organizational structure (i.e. a group of employees all reporting a common manager)
    + the probability of a link between two employees at social distance $d$ within the organization scaled proportionally to $d^{-3/4}$
    + The best exponent for making decentralized search within the network efficient

+ Search as an Instance of Decentralized Problem-Solving
    + designated to test the hypothesis that people are connected by short paths in the global social network
    + using only very local information, and by communicating only with their neighbors in the social network
    + social networks can be effective at this type of decentralized problem-solving is an intriguing and general premise that applied more broadly than just to the problem of path-finding that Milgram considered.
    + their effectiveness will depend both on the difficulty of the problem being solved and on the network that connects them.



## Core-Periphery Structures and Difficulties in Decentralized Search

+ Critique of the Milgram experiment:
    + Judith Kleinfeld: the success rate at finding targets in recreations of the experiment has often been much lower than it was in the original work.
    + 18 different targets were used, drawn from a wide range of backgrounds.
    + Completion rates to all targets were small, due to lack of participation in the e-mail based forwarding messages.

+ Core-Periphery Structures
    + Social networks are structured to make high-status individuals much easier to find than low-status ones.
    + Homophily suggests that high-status people will mainly know other high-status people, and low-status people will mainly know other low-status people.
    + __Core-periphery structure__
        + large social networks tend to be organized
        + the high-status people are linked in a densely-connected _core_, while the low-status people aer atomized around the _periphery_ of the network
    + As moving toward a high-status target, the link structure tends to become richer, based on connections with an increasing array of underlying social reasons. $\; \rightarrow \;$ find a low-status target.
    + The link structure becomes structurally more impoverished as you move toward the periphery.


## Advanced Material: Analysis of Decentralized Search

+ The best exponent for seach is equal to the dimension.

+ choice of $q$ is in fact the best for decentralized search in the limit of increasing network size

### The Optimal Exponent in One Dimension

+ Analysis of decentralized search
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img3.jpg" alt="The analysis of decentralized search is a bit cleaner in one dimension than in two, although it is conceptually easy to adapt the arguments to two dimensions. As a result, we focus most of the discussion on a one-dimensional ring augmented with random long-range links." title="(a) A set of $n$ nodes are arranged on a one-dimensional ring. (b) a ring augmented with random long-range links" height="250">
    </a>
    + Fig. (a)
        + A set of $n$ nodes are arranged on a one-dimensional ring.
        + With each node connected by directed edges to the two others immediately adjacent to it.
    + Fig. (b)
        + Each node $v$ alos has a single directed edge to some other node on the ring
        + $d(v, w)$: the distance between nodes $v$ and $w$ on the ring
        + the probability that $v$ links to any particular node $w$ is proportional to $d(v, w)^{-1}$
        + contacts: an edge existed between nodes
            + _local contact_: two nodes adjacent on the ring
            + _long-range contact_: two nodes with edge other than adjacent

+ Myopic Search
    + Random start and target nodes on this argument ring network: $s$ and $t$
    + goal: to forward a message from the start to the target, with each intermediate node on the way only knowing the locations of its own neighbors, and the location of $t$, but nothing else about the full network
    + myopic search
        + forwarding strategy
        + work well on the ring when $q=1$
        + when a node $v$ is holding the message, it passes it to the contact that lies as close to $t$ on the ring as possible
        + a reasonable approximation to the strategies used by most people in Milgram-style experiments
    + Myopic path
        <a href="https://slideplayer.com/slide/4169089/"> <br/>
            <img src="https://player.slideplayer.com/13/4169089/data/images/img5.jpg" alt="In myopic search, the current message-holder chooses the contact that lies closest to the target (as measured on the ring), and it forwards the message to this contact." title="The myopic path that would be constructed if we chose a as the start node and i as the target node in the network." height="300">
        </a>
        1. Node $a$ first sends the message to node $d$, since among $a$'s contact $p$, $b$, and $d$, node $d$ lies closest to $i$ on the ring.
        2. The $d$ passes the message to its local contact $e$, and $e$ likewise passes the message to its local contact $f$, since the long-range contacts of both $d$ and $e$ lead away from $i$ on the ring, not closer to it.
        3. Node $f$ has a long-range contact $h$ that proves useful, so it passes it to $h$. Node $h$ actally has the target as a local contact , so it hands it directly to $i$, completing the path in five steps.
    + The myopic path is not the shortest path from $a$ to $i$ but a three-step $a-b-h-i$ path.

+ Analyzing Mypoic Search: The Basic Plan
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img21.jpg" alt="We analyze the progress of myopic search in phases." title="We analyze the progress of myopic search in phases. Phase j consists of the portion of the search in which the message’s distance from the target is between $2^j$ and $2^{j+1}$." height="300">
    </a>
    + generate a random network by adding long-range edges to a ring as above
    + $X$: a random variable as the number of steps required by myopic search
    + The expected value of $X$, $E[X]$, is relatively small
    + track how long it takes for the message to reduce its distance by factors of two as it closes in on the target
    + As the message moves from $s$ to $t$, it's in _phase j_ of the search if its distance from the target is between $2^j$ and $2^{j+1}$.
    + Notice that the number of different phases is at most $\log_2 n$, that is, the number of doublings needed to go from $1$ to $n$.
    + $X$, the number of steps taken by the full search

        $$X = X_1 + X_2 + \cdots + X_{\log n} \; \Longrightarrow \; E[X] = E[X_1 + X_2 + \cdots + X_{\log n}] = E[X_1] + E[X_2] + \cdots + E[X_{\log n}]$$
    + The expected value of each $X_j$ is at most proportional to $\log n$ and $E[X]$ is at most proportional to $(\log n)^2$
    + The full network has $n$ nodes, but myopic search constructs a path that is _exponentially smaller_: proportional to $(\log n)^2$

+ Intermediate Step: The Normalizing Constant
    + $v$ forms its long-range link to $w$ with probability _proportional to_ $d(v, w)^{-1}$
    + Q: what is the constant of proportionality?
    + a set of probabilities up to a missing constant of proportionality $1/Z$
    + $Z = \sum_{v} d(v, w)^{-1} \; \forall \; v \neq w$ on the ring
    + Normalizing constant $Z$, the probability of $v$ linking to $w$ is equal to $\frac{1}{Z} d(v, w)^{-1}$
    + The value of $Z$
        + there are two nodes at distance $1$ from $v$, two at distance $2$ an two at each distance $d$ up to $n/2$
        + Assuming $n$ is even there is also a single node at distance $n/2$ from $v$, i.e., the node diametrically opposite it on the ring.

        $$Z \leq 2(1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \cdots + \frac{1}{n/2})$$
    + The quantity inside parentheses on the right is a common expression in the probabilistic calculation: the sum of the first $k$ reciprocials, for some $k$, in the case $n/2$.
    + Determining the normalizing constant for the probability of links involves evaluating the sum of the first $n/2$ reciprocals. An upper bound on the value of this sum can be determined from the area under the curve $y = 1/x$.
    + A sequence of rectabngles of unit widths and heights $1/2, 1/3, 1/4, \ldots, 1/k$ fits under the curve $y=1/x, \; x = 1, 2, \ldots, k$

        $$1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \cdots + \frac{1}{k} \leq 1 + \int_1^k \frac{1}{x} dx = 1 + \ln k$$
    + Let $k = n/2$ and known $\ln x \leq \log_2 x$

        $$ Z \leq 2 (1+\ln(n/2)) = 2 + 2 \ln(n/2) \; \Longrightarrow \; Z \leq 2 + 2 \log_2 (n/2) = 2 + 2 (\log_2 n) - 2(log_2 2) = 2\log_2 n$$
    + The probability $v$ links to $w$

        $$\frac{1}{Z} d(v, w)^{-1} \geq \frac{1}{2 \log n} d(v, w)^{-1}$$

+ Analyzing the Time Spent in One Phase of Myopic Search
    <a href="url"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img24.jpg" alt="At any given point in time, the search is in some phase j, with the message residing at a node v at distance d from the target. The phase will come to an end if v’s longrange contact lies at distance $ d/2 from the target t, and so arguing that the probability of this event is large provides a way to show that the phase will not last too long." title="A particular phase j of the search, when the message is at a node v whose distance to the target t is some number d between $2^j$ and $^{j+1}$." height="300">
        <img src="https://player.slideplayer.com/13/4169089/data/images/img25.jpg" alt="Showing that, with reasonable probability, v’s long-range contact lies within half the distance to the target." title="Let I be the set of nodes at distance $\leq d/2$ from t; this is where we hope v’s long-range contact will lie." height="300">
    </a>
    + The time spent by the search in any one phase is not very large.
    + Choose a particular phase $j$ of the search, when the message is at node $v$ whose distance to the target $w$ is some number $d \in [2^j, 2^{j+1}]$.
    + The phase will come to an end once the distance to the target decreases below $2^j$.
    + One way for the phase to come to an end immediately would be for $v$'s long-range contact $w$ to be at $d(v, tw \leq d/2$ from $t$.
    + $v$ would necessarily be the last node to belong to phase $j$.
    + The immediate halving of the distance in fact happens with reasonably large probability.
    + $I$: the set of nodes at $d(v, t) \leq \frac{d}{2} \; \forall v \not\in\{s, t\}$ where $v$'s long-range contact lies hopefully
    + $d+1$ nodes in $I$: include node $t$ and $d/2$ nodes consecutively on each side of it
    + Each node $w \in I$ has distance at most $3d/2$ from $v$: the farthest one is on the "far side" of $t$ from $v$, at distance $d + d/2$.
    + Each node $w \in I$ has probability of being the long-range contact of $v$

        $$\frac{1}{2\log n} d(v, w)^{-1} \geq \frac{1}{2\log n} \cdot \frac{1}{3d/2} = \frac{1}{3d\log n}$$
    + There more than $d$ nodes in $I$, the probability that one of thenm is the long-range contact of $v$ is at least

        $$d \cdot \frac{1}{3d\log n} = \frac{1}{3 \log n}$$
    + If one of these nodes is the long-range contact of $v$, then phase $j$ ends immediately in this step.
    + In each step that it proceeds, phase $j$ has a probability of at least $1/(3\log n)$ of coming to an end, independetly of what has happened so far. 
    + To run for at least $i$ steps, phase $j$ has to fail to come to an end $i-1$ times in a row, and so the probability that phase $j$ runs for at least $i$ steps is at most

        $$(1 - \frac{1}{3 \log n})^{i-1}$$
    + Substitute with

        $$Pr[X_j \geq i] \leq (1 - \frac{1}{3 \log n})^{i-1}$$

        The expect value of a random variable

        $$E[X_j] = 1 \cdot Pr[X_j = 1] + 2 \cdot Pr[X_j = 2] + 3 \cdot Pr[X_j = 3] + \cdots = Pr[X_j \geq 1] + Pr[X_j \geq 2] + Pr[X_j \geq 3] + \cdots$$

        Therefore, 

        $$E[X_j] \leq 1 + (1 - \frac{1}{3 \log n}) + (1 - \frac{1}{3 \log n})^2 +  (1 - \frac{1}{3 \log n})^3 + \cdots$$

        The right-hand side is a geometric sum with multiplier $(1 - \frac{1}{3 \log n})$ and converges to

        $$\frac{1}{1 - (1 - \frac{1}{3 \log n})} = 3 \log n$$

        Thus,

        $$ E[X_j] \leq 3 \log n$$
    + $E[X]$ = a sum of the $\log n$ terms $E[X_1] + E[X_2] + \cdots + E[X_{\log n}]$ and each of them at most $3 \log n$.
    + Therefore, th eexpected value of $X$

        $$ E[X] \leq 3(\log n)^2 \; \propto \; (\log n)^2$$


### Higher Dimensions and Other Exponents

+ Issues
    + sketch how it can be used to analyze networks built by adding long-range contacts to nodes arranged in two dimensions
    + more efficient search when $q$ is equal to the underlying dimension that when it it equal to other value

+ The Analysis in Two Dimensions
    + Two distinct places used in one dimension for the analysis
        + determine the normalizing constant $Z$
        + ay lease $d$ nodes within distance $d/2$ of the target $t$
    + The factor of $d$ canceled the $d^{-1}$ in the link probability and the probability of halving the distance to the target in any given step was at least proportional tp $1/(\log n)$, regardless of the value $d$.
    + quantative level: with link probability $d^{-1}$ on the ring, the probability of linking to any one exactly offsets the number of nodes close to $t$, and so myopic search makes at every possible distance away from the target.
    + Two dimensional analysis: the number of nodes within distance $d/2$ of the target will be proportional to $d^2$.
    + cancellation property: $v$ link to each node $w$ with probability proportional to $d(v, w)^{-2}$ and this exponent $-2$ is what we will use.
    + The analysis for two dimensions is almost exactly the same as what we just saw for the one-dimensional ring.
    + the normalizing constant $Z$ is still proportional to $\log n$ when the probability of $v$ linking to $w$ is proportional to $d(v, w)^{-2}$
    + consider $\log n$ different phases. the probability that at any given moment, the current message holder $v$ has a long-range contact $w$ that halves the distance to the target, ending the phase immediately.
    + The number of nodes within distance $d/2$ of the target is proportional to $d^2$, and the probability that $v$ links to each is proportional to $1/(d^2 \log n)$.
    + The probability that the message halves its distance to the target in this step is at least $d^2/(d^2 \log n) = 1/(\log n)$, and the rest of the analysis then finishes as before.
    + decentralized search is efficient for networks built by adding long-range contacts to grids in $D > 2$ dimensions, when the exponent $q = D$.

+ Why Search is Less Efficicent with Other Exponents
    + why search doesn't work well when $q=0$: the original Watts-Strogatz model when long-range links are chosen uniformly at random
    + Using one-dimensional ring rather than the two-dimensional grid
    + key idea: consider the set of all nodes within some distance of the target $t$
    + with $q=1$ it is easy to enter smaller and smaller sets centered around $t \; \Longrightarrow \;$ identify a set of nodes centered at $t$ that is somehow "impenetrable", a set hard for the search to enter
    <a href="https://slideplayer.com/slide/4169089/"> <br/>
        <img src="https://player.slideplayer.com/13/4169089/data/images/img27.jpg" alt="To show that decentralized search strategies require large amounts of time with exponent q = 0, we argue that it is difficult for the search to cross the set of &n nodes closest to the target. Similar arguments hold for other exponents q < 1." title="Decentralized seach requires time" height="400">
    </a>
    + $K$: a set of all nodes within distance less than $\sqrt{n}$ of the target $t$.
    With high probability, the starting point of the search lies outside $K$.
    + Because long-range contacts are created unformly at random ($q=0$), the probability that any one node has a long-range contact inside $K$ is equal to the size of $K$ divided by $n$: less than $2\sqrt{n}/n = 2/\sqrt{n}$
    + any decentralized search strategy will need at least $\sqrt{n}/2$ steps in expectation to find a node with a long-range contact $K$.
    + as long as it doesn't find a long-range link leading into $K$, it can't reach the target in less than $\sqrt{n}$ steps, since it would take this long to "walk" step-by-step through $K$ using only the connection among local contacts.
    + The expected time for any decentralized search strategy to reach $t$ must be at least proportional to $\sqrt{n}$.
    + $q \in [0, 1]$: a version of argument above works, with a set $K$ centered at $t$ whose width depnds on the value of $q$
    + $q > 1$, decentralized search is inefficient: even with relative short long-range links, it takes a long time for decentralized search to find links that span sufficiently long distances $\implies$ hard to quickly traverse the distance from the starting node to the target
    + $q \neq 1$, $\exists \: c>0 \: \ni$ at least proportional to $n^c$ steps in expectation for any decentralized search strategy to search the target in the network generated with exponent $q$
    + $n \: \rightarrow \infty$: decentralized search time $\mathcal{t}$
        + $q=1$: $\mathcal{t} \propto \log n$
        + $q \neq 1$: $\mathcal{t} \propto n$ (exponentially worse)
    + Optimal balance between producing network that are "too random" for search and those that are not rondom enoughy:
        + ring: $q=1$
        + plane: $q=2$



