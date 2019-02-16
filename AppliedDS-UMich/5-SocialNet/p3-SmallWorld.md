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



## Empirical Analysis and Generalized Models



## Core-Periphery Structures and Difficulties in Decentralized Search



## Advanced Material: Analysis of Decentralized Search


