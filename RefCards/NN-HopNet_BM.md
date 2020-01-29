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

