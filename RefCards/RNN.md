# Recurrent Neural Networks

## Overview

+ [Getting targets when modeling sequences](../ML/MLNN-Hinton/07-RNN.md#lecture-notes)
  + when applying machine learning to sequences, often turn an input sequence into an output sequence that lives in a different domain
  + when no separate target sequence, get a teaching signal by trying to predict the next term in the input sequence
  + predicting the next terms in a sequence blurs the distinction between supervised and unsupervised learning

+ [Memoryless models for sequences](../ML/MLNN-Hinton/07-RNN.md#lecture-notes)
  + autoregressive models: predict the next term in a sequence from a fixed number of previous terms using "delay taps"
  + feed-forward neural nets

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m07-01.png" style="margin: 0.1em;" alt="Autoregressive models" title="Autoregressive models" height=80>
      <img src="../ML/MLNN-Hinton/img/m07-02.png" style="margin: 0.1em;" alt="Feed-forward neural nets" title="Feed-forward neural nets" height=80>
    </a>
  </div>

+ [Beyond memoryless models](../ML/MLNN-Hinton/07-RNN.md#lecture-notes)
  + generative model w/ hidden state that has its own internal dynamics
  + inference: only tractable for two types of hidden state model

+ [Linear dynamical systems](../ML/MLNN-Hinton/07-RNN.md#lecture-notes) (engineers perspective)
  + generative models: producing the observations using a linear model w/ Gaussian noise
  + to predict the next output: computed using "Kalman filtering"

+ [Hidden Markov Models](../ML/MLNN-Hinton/07-RNN.md#lecture-notes) (computer scientists perspective)
  + a discrete one-of-N hidden state
  + to predict the next output
  + limitation
    + considering what happens when a hidden Markov model generates data
    + considering the first half of an utterance contains about the second half
    + all aspects combined could be 100 bits of information that the first half of an utterance needs to convey to the second half. $2^{100}$ is big!

+ [Recurrent neural networks](../ML/MLNN-Hinton/07-RNN.md#lecture-notes)
  + efficient way to remember the information
  + very powerful
  + Properties
    + distributed hidden state: to store a lot of information about the past efficiently
    + non-linear dynamics: to update their hidden state in complicated ways
  + with enough neurons and time RNNs able to compute anything that can be computed by your computer
  + recurrent neural networks are deterministic
  + Behavior
    + oscillation
    + settle to point attractors
    + chaostic
  + extreme requirements for computational power

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture7/lec7.pdf" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m07-05.png" style="margin: 0.1em;" alt="Recurrent neural networks" title="Recurrent neural networks" width=150>
    </a>
  </div>



