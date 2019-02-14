# Graphs


## Basic Definitions

+ Graphs: Nodes and Edges
    + __graph__: a way of specifying relationships among a collection of items
    + __nodes__: a set of objects
    + __edges__: pairs of these objects connected by links
    + __directed graph__: a set of nodes with a set of directed edges, a lik from one node to another
    + __undirected graph__: a graph not directed

+ Graphs as Models of Networks
    + mathematical models of network structures
    + Domain applied:
        + _communication network_: 13-node Arpanet
        + _social network_: nodes - people or groups of people; edges - social interaction
        + _information networks_: nodes - information resources such as Web pages or documents; edges - logical connections such as hyperlinks, citations, or cross-reference


## Paths and Connectivity

+ Paths
    + a sequence of nodes with the property that each consecutive pair in the sequence is connected by an edge

+ Cycles
    + a "ring" structure
    + a path with at least three edges, in which the first and last nodes are the same, but otherwise all nodes are distinct
    + Cycles in communication and transportation networks present to allow for redundancy -- they provide for alternate routing that go the "other way" around the cycle.

+ Connectivity
    + connected graph: a pth between every pair of nodes
    + no prori reason to expect graphs in other setting to be connected
    + collabration graph at a biological research center: nodes -- researchers; edges -- researchers apprar jointly on a co-authored publication

+ Components
    + connected component of a graph: a subset of the nodes such that
        1. every node in the subset has a path to every other
        2. the subset is not part of some larger set with the property that every node can reach every other
    + Dividing a graph into its components is only a first, global way of describing its structure.
    + Within a given component, there may be richer internal structure that is important to one's interpretation of the network.
    + One way to formalize the role of the prominent central node is to observe that the largest connected component would break apart into distinct components if this node is removed.

+ GTiant Components
    + connectivity is a fairly property, in that the behavior of a single of a single node (or small set of noes) can negate it.
    + giant component: a connected component that contains a significant fraction of all the nodes
    + Whena network contains a giant component, it almost always contains only one.
    + When there is a giant component, it is thus generally uniquew, distinguishable as a component that dwarfs all others.
    + In some of the rare cases when two giant components have co-existed for a long time in a real network.


## Distance and Breadth-First Search

+ The length of a path: 
    + the number of steps it contains from beginning to end
    + the number of edges in the sequence that comprises it
    + the distance between two nodes in a graph = the length of the shortest path between them

+ Breadth-First Search
    + searching tyhe graph outward from a starting node, reaching the closest nodes first
    + providing a method of determining 
    + serving as a useful conceptual framework to organize the structure of a graph, arranging the nodes based on their distances from a fixed starting point

+ The Small-World Phenomenon
    + the idea that the world looks "small" when you think of how short a path of friends it takes to get from you to almost anyone else.
    + six degrees of separation
    + social networks tend to have very short paths between essentially arbitrary pairs of people

+ Instant Messaging, Paul Erdos, and Kevin Bacon
    + Jure Leskovec and Eric Horvitz: 240 million active user accounts om MS Instant Messenger
    + a graph: node = user; edge = two user if they are engaged in a two-way conversation at any point during a month-long observation period
    + the graph forms a giant component containing almost all of the nodes
    + the distances within this giant component were very small
    + estimated average distance = 6.6
    + estimated median = 7
    + The graph was so large that performing breadth-first search from every single node would have taken an astronomical amount of time.
    + Limitation: only track people who are technologically-endowed enough to have access instant messaging, and rather than basing the graph on who is truly friends with whom, it can only observe who talks to whom during an observation period

+ Scientist Collaboration Networks
    + Very short paths in the collaboration networks within professional communities
    + a mathematician's Erdos number = the distance from him or her to Erdos in the graph
    + most mathematicians Erdos numbers $< 4/5$
    + Extending the collaboration graph to include co-authorship across all the sciences
    + most scientist in other fields w/ Erdos numbers: Albert Einstein = 2, Enrico Fermi = 3, Noam Chomsky = = Linus Pauling = 4, Francis Crick = 5, James Watson = 6

+ Collaboration Networks of Actors and Actresses
    + nodes = performers; edges = two performers if they've appeared together in a movie
    + Performer's Bacon number = his or her distance in the graph to Kevin Bacon
    + Over all performers in the IMDB: average Backon number ~ 2.9; rare $> 5$


## Network Datasets: An Overview

+ Several distinct reasons for studying a particular network dataset
    1. actual domain it comes from , so that fine-grained details of the data itself are potentially as interesting as the broaderpicture
    2. as proxy for a related network that may be impossible to measure
    3. network properties that appear to be common access many different domains and so finding a similar effect in unrelated settings can suggest that it has a certain universal nature

+ The result of the IM graph analysis fit into the broader framework of small-world phenomena that span many domains.

+ Collaboration Graphs
    + who works with whom in a specific setting
    + Examples: co-authorships among scientists; co-appearance in movies; highly-placed people on the board directors of the same Fortune 500 company
    + On-line world: Wikipedia collaboration graph; World-of-Warcraft collaboration graph
    + Learn about the specific domain it comes from
    + the sociological context of scientific resarch, a broader community of people is interested in scientific co-authorship network precisely, pre-digested snapshots of a rich form of social interaction tha unfolds over a long period of time

+ Who-talks-to-Whoom Graphs
    + MS IM graph: a snapshot of a large community engaged in several billion conversations over the course of a month
    + Examples
        + e-mail logs within a company or a university
        + records of phone cells -> structure of call graphs
    + "face-to-face" graphs
        + mobile phones w/ short-range wireless technology to detect other similar devices nearby
        + record physical proximity
        + node = a person carrying one of the mobile devices
        + edge = joining two people if they were detected to be in close physical proximity over a given observation period
    + Issue: strong expecation of privacy
    + "who-transacts-with-whom" structure of a market or financial community

+ Information Linkage Graphs
    + Snapshots of the Web are central examples of network datasets: nodes = web pages; edges = links from one page to another
    + Scale of full Web: well-defined subsets of the Web to simplify
    + E.g., linkages among bloggers; pages on Wikipedia; pages on social-networking sites such as Facebook or MySpace; discussions and product reviews on shopping sites
    + Information linkage graphs significantly updates the Web
        + citation analysis - the network structure of citations among scientific papers or patents, a way of tracking of the evolution of the science
        + Citation networks: popular research datasets

+ Technological Networks
    + The Web is built on a lot of sophisticated technology, it would be mistake to think of it primarily as a technological network
    + projection onto a technological backdrop of ideas, information, and social economic structure created by humans
    + convergence of social and technological networks: much interesting network data comes from the more overly technological end of the spectrum -- with nodes representing physical devices and edges representing physical connections between them
    + Internet - two-level view of network
        + lowest level: nodes = individual routers and computers; edge = two devices have a physical connection to each other
        + higher level (autonomous systems): nodes = grouped into what are essentially little "nation-states", AS, controlled by different Internet service-providers $\longrightarrow$ who-transacts-with-whom graph on AS, known as AS graph

+ Networks in the Natural World
    + graph structure: biology, natural sciences, and network research has devoted particular attention to several different types of biological networks
    + Examples
        + food webs: who-eats-whom relationships
        + structure of neural connections within an organism's brain
        + cell's metabolism




