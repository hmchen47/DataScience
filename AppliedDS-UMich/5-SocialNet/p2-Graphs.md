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



## Network Datasets: An Overview




