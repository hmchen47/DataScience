# Hopfield Nets and Boltzmann Machines

## 11.1 Hopfield Nets

### Lecture Notes

+ Hopfield Networks
  + combposed of binary threshold units w/ recurrent connections between them
  + recurrent network of non-linear units
    + generally very hard to analyze
    + behave in many different ways
      + settle to a stable state
      + oscillate
      + follow chaotic trajectories that cannot be predicted far into the future
  + John Hopfield's proposal
    + J. J. Hopfield, "[Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/content/pnas/79/8/2554.full.pdf)", Proceedings of the National Academy of Sciences of the USA, vol. 79 no. 8 pp. 2554–2558, April 1982
    + existing a global energy function w/ symmetric connections
    + each binary "configuration" of the whole network w/ an energy
    + binary threshold decision rule causing the network to settle to a minimum of the energy function

+ The energy function
  + global energy: the sum of many contributions
  + each contributions on one connection weight and the binary states of two neurons

    \[ E = - \sum_i s_i \cdot b_i - \sum_{i < j} s_i s_j \cdot w_{ij} \]

  + simple quadratic energy function makes it possible for each unit to compute locally how it's state affects the global energy:

    \[ \text{Energy gap} = \Delta E_i = E(s_i = 0) - E(s_i = 1) = b_i + \sum_j s_j \cdot w_{ij} \]

+ Settling to an energy minimum
  + finding the minimum energy
    + start from a random state
    + then update units one at a time in random order
    + update each unit to whichever of its two states gives the lowest global energy
    + i.e., use binary threshold units
  + two triangles in the net
    + the three units mostly support each other
    + each triangle mostly hates the other triangle
  + weight difference btw two triangles
    + left triangle w/ weight 2 while the right triangle w/ weight 3
    + turning on the units in the triangle on the right gives the deepest minimum

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
      <img src="img/m11-01.png" style="margin: 0.1em;" alt="Settle energy minimum 1" title="Settle energy minimum 1" height=150>
      <img src="img/m11-02.png" style="margin: 0.1em;" alt="Settle energy minimum 2" title="Settle energy minimum 2" height=150>
      <img src="img/m11-03.png" style="margin: 0.1em;" alt="Settle energy minimum 3" title="Settle energy minimum 3" height=150>
      <img src="img/m11-04.png" style="margin: 0.1em;" alt="Settle energy minimum 4" title="Settle energy minimum 4" height=150>
    </a>
  </div>

+ Sequential decisions
  + if units make simultaneous decisions the energy could go up
  + simultaneous parallel updating $\implies$ getting oscillations
    + they always have a period of 2
  + the updates occur in parallel but w/ random timing $\implies$ the oscillations usually destroyed
  + example:
    + at the next parallel step, both units will turn on
    + w/ high energy, both turn off again

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
        <img src="img/m11-05.png" style="margin: 0.1em;" alt="Settle energy minimum 1" title="Settle energy minimum 1" height=150>
      </a>
    </div>

+ Neat way to compute sequential decisions
  + Hopfield proposal
    + memories could be energy minima of a neural net
    + binary threshold decision rule used to "clean up" incomplete or corrupted memories
  + Principles of Literary Criticism
    + I. A. Richards (1924) proposal [wikipedia](https://en.wikipedia.org/wiki/I._A._Richards)
    + idea of memories as energy minima
  + energy minima
    + represent memories giving a content-addressable memory
    + an item able to accessed by just knowing part of its content
    + robust against hardware damage
    + like reconstructing a dinosaur from a few bones

+ Storing memories in a Hopefield net
  + with activities of $1$ and $-1$
    + stored as a binary state vector by incrementing the weight btw any two units by the product of their activities
    + treating biases as weights from a permanently on unit
    + very simple rule: not error-driven
    + both its strength and its weakness

    \[ \Delta_{ij} = s_i \cdot s_j \]
  + with state of $0$ and $1$, the rule is slightly more complicated

    \[ \Delta w_{ij} = 4 (s_i - \frac{1}{2}) (s_j - \frac{1}{2}) \]


### Lecture Video

<video src="https://youtu.be/DS6k0PhBjpI?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 11.2 Dealing with spurious minima in hopfield nets

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 11.3 Hopfields Nets with hidden units

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 11.4 Using stochastic units to improve search

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## 11.5 How a boltzmann machine models data

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>

