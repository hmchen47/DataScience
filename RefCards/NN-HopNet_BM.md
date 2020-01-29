# Hopfield Networks & Boltzmann Machine

## Overview for Hopfield Networks

+ [Hopfield Networks](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + energy-based model: properties derive from a global energy function
  + combposed of binary threshold units w/ recurrent connections between them
  + recurrent network of non-linear units
    + generally very hard to analyze
    + behave in many different ways
      + settle to a stable state
      + oscillate
      + follow chaotic trajectories that cannot be predicted far into the future unless knowing the starting state w/ infinite precision
  + John Hopfield's proposal
    + existing a global energy function w/ __symmetric__ connections
    + each binary "configuration" of the whole network w/ an energy
    + binary threshold decision rule causing the network to settle to a minimum of the energy function


## Energy Function

+ [The energy function](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + global energy: the sum of many local contributions
  + the main contributions: the form of the product of one connection weight w/ the binary of two neurons

    \[ E = - \sum_i s_i \cdot b_i - \sum_{i < j} s_i s_j \cdot w_{ij} \]

    + energy is bad $\implies$ low energy is good $\implies$ minus sign (-) for the equation
    + $s_i$: binary variable w/ values of $1$ or $0$ or in another kind of Hopfield net w/ values of $1$ or $-1$
    + $w_{ij}$: weight for the symmetric connection strength btw two neurons
    + $s_i b_i$: bias term involves the state of individual units
    + $s_i s_j$: the activities of the two connected neurons

  + simple _quadratic_ energy function makes it possible for each unit to compute locally how it's state affects the global energy:

    \[ \text{Energy gap} = \Delta E_i = E(s_i = 0) - E(s_i = 1) = b_i + \sum_j s_j \cdot w_{ij} \]

+ [Settling to an energy minimum](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + finding the minimum energy
    + start from a random state
    + sequential update: update units _one at a time_ in random order
  + commputing the goodness
    + all pairs of units w/ on and add in the weight between them
    + minimum energy


## Memories

+ [Sequential decisions](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + if units make __simultaneous__ decisions the energy could go up
  + simultaneous parallel updating $\implies$ getting oscillations (always w/ period 2)
  + the updates occur in parallel but w/ random timing $\implies$ the oscillations usually destroyed

+ [Neat way to compute sequential decisions](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + Hopfield proposal
    + memories could be energy minima of a neural net w/ symmetric weights
    + binary threshold decision rule used to "clean up" incomplete or corrupted memories
  + Principles of Literary Criticism
    + idea of memories as energy minima
    + memories are like a large crystal that can sit on different phases
  + energy minima
    + represent memories w/ a content-addressable memory
    + access an item by just knowing part of its content
    + biological property: robust against hardware damage
    + psychological point of view: like reconstructing a dinosaur from a few bones

+ [Storing memories in a Hopefield net](../ML/MLNN-Hinton/11-Hopfield.md#111-hopfield-nets)
  + with activities of $1$ and $-1$
    + stored as a binary state vector by incrementing the weight btw any two units by the product of their activities
    + treating biases as weights from a permanently on unit
    + very simple rule: not error-driven
    + both its strength and weakness
      + not error correction rule
      + able to be online but not a very efficient way to store things

    \[ \Delta w_{ij} = s_i \cdot s_j \]

  + with state of $0$ and $1$, the rule is slightly more complicated

    \[ \Delta w_{ij} = 4 (s_i - \frac{1}{2}) (s_j - \frac{1}{2}) \]


