# Hopfield Nets and Boltzmann Machines

## 11.1 Hopfield Nets

### Lecture Notes

+ Hopfield Networks
  + energy-based model: properties deriva from a global energy function
  + combposed of binary threshold units w/ recurrent connections between them
  + recurrent network of non-linear units
    + generally very hard to analyze
    + behave in many different ways
      + settle to a stable state
      + oscillate
      + follow chaotic trajectories that cannot be predicted far into the future unless knowing the srating state w/ infinite precision
  + John Hopfield's proposal
    + J. J. Hopfield, "[Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/content/pnas/79/8/2554.full.pdf)", Proceedings of the National Academy of Sciences of the USA, vol. 79 no. 8 pp. 2554–2558, April 1982
    + existing a global energy function w/ __symmetric__ connections
    + each binary "configuration" of the whole network w/ an energy
      + binary configuration: an assignment of binary values to each neuron in the network
      + each neuron w/ a particular binary value in a configuration
    + binary threshold decision rule causing the network to settle to a minimum of the energy function

+ The energy function
  + global energy: the sum of many local contributions
  + the main contributions: the form of the product of one connection weight w/ the binary of two neurons

    \[ E = - \sum_i s_i \cdot b_i - \sum_{i < j} s_i s_j \cdot w_{ij} \]

    + energy is bad $\implies$ low energy is good $\imples$ minus sign (-) for the equation
    + $w_{ij}$: weight for the symmetric connection strength btw two neurons
    + $s_i s_j$: the activities of the two connected neurons
    + $s_i$: binary variable w/ values of $1$ or $0$ or in another kind of Hopfield net w/ values of $1$ or $-1$
    $s_i b_i$: bias term involves the state of individual units

  + simple _quadratic_ energy function makes it possible for each unit to compute locally how it's state affects the global energy:

    \[ \text{Energy gap} = \Delta E_i = E(s_i = 0) - E(s_i = 1) = b_i + \sum_j s_j \cdot w_{ij} \]

    + the energy gap for unit $i$ = difference of in the global energy of the whole configuration depending on whether or not $i$ is on
    + $E(s_i = 0) - E(s_i = 1)$: the difference btw the energy when $i$ is off and the energy when $i$ is on; computed by the binary threshold decision rule

+ Settling to an energy minimum
  + finding the minimum energy
    + start from a random state
    + then update units _one at a time_ in random order $\implies$ sequential update
    + update each unit to whichever of its two states gives the lowest global energy independent of what state it was previously
    + i.e., use binary threshold units
  + commputing the goodness: all pairs of units w/ on and add in the weight between them
  + example:
    + start w/ random global state (carefully selected random state for demo) $\to$ goodness as $-E = 3$
    + 1st fig: randomly pick a unit (top right cornere one)
      + input: $1 \time -4 + 0 \times 3 + 0 \times 3 = -4$
      + unit state: $-4 < 0 \to 0 \implies$ state off
    + 2nd fig: randomly pick a unit except for computed one (bottom left corner one)
      + input: $1 \times 3 + 0 \time (-1) = 3$
      + unit state: $3 > 0 \to 1 \implies$ state on
    + 3rd fig: randomly pick a unit except for computed ones (botton middle one)
      + input: $1 \times 2 + 1 \times (-1) + 0 \time 3 + 0 \time (-1) = 1$
      + unit state: $1 > 0 \to 0 \implies$ turn on
      + goodness: $-E = 3 + 2 -1 = 4$
    + probing any of the units: no state changed $\implies$ settle to minimum
    + 4th fig: the deepest energy minimum
      + 3rd fig is one of the two minima of the net
      + 4th fig shows the deepest energy minimum
      + when the other triangle of units supports each orther is on
      + goodness: $-E = 3 + 3 -1 = 5$+ two triangles in the net (4th fig)
      + two triangles:
        the three units mostly support each other although a little bit disagreement at the bottom
        + each triangle mostly hates the other triangle via connection on top
      + the triangle on the left differs from one on the right by having a wight of two where the other one has a weight of 3 $\implies$ the triangle on the right will provide the deepest minimum

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
      <img src="img/m11-01.png" style="margin: 0.1em;" alt="Settle energy minimum 1" title="Settle energy minimum 1" height=150>
      <img src="img/m11-02.png" style="margin: 0.1em;" alt="Settle energy minimum 2" title="Settle energy minimum 2" height=150>
      <img src="img/m11-03.png" style="margin: 0.1em;" alt="Settle energy minimum 3" title="Settle energy minimum 3" height=150>
      <img src="img/m11-04.png" style="margin: 0.1em;" alt="Settle energy minimum 4" title="Settle energy minimum 4" height=150>
    </a>
  </div>

+ Sequential decisions
  + if units make __simultaneous__ decisions the energy could go up
  + simultaneous parallel updating $\implies$ getting oscillations
    + they always have a period of 2
  + the updates occur in parallel but w/ random timing $\implies$ the oscillations usually destroyed
    + not waiting for one update to communicate the state to everybody before considering another update
    + but waiting for random length of time between doing updates of a given unit
    + the random timing will often destory these bi-phasic oscillations
    + the updates have to be sequential, not as bad as it seems from a biological perspective (?)
  + example:
    + a network w/ units have biases at +5 and weight between them of -100
    + start w/ both states are off
    + at the next parallel step, both units will turn on $\impliedby$ assume to improve via the bias term but worse due to -100
    + w/ high energy, both turn off again

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pptx" ismap target="_blank">
        <img src="img/m11-05.png" style="margin: 0.1em;" alt="Settle energy minimum 1" title="Settle energy minimum 1" width=250>
      </a>
    </div>

+ Neat way to compute sequential decisions
  + Hopfield proposal
    + memories could be energy minima of a neural net w/ symmetric weights
    + binary threshold decision rule used to "clean up" incomplete or corrupted memories
      + clean up partial memory into full memory
      + fill out out the memory caused by corrupted memory due to wrong computation or undecided
  + Principles of Literary Criticism
    + I. A. Richards (1924) proposal [wikipedia](https://en.wikipedia.org/wiki/I._A._Richards)
    + idea of memories as energy minima
    + memories are like a large crystal that can sit on different phases
  + energy minima
    + represent memories w/ a content-addressable memory
    + access an item by just knowing part of its content
      + set the states of some neurons in the net
      + randonly put the other neurons in random states
      + apply the binary threshold rule w/ a bit of luck
      + fill up that memory to be some stored item you know about
    + biological property: robust against hardware damage
    + psychological point of view: like reconstructing a dinosaur from a few bones
      + knowing something about how the bones are meant to fit together
      + the weight in the net provides infomation about how states of neurons fit together
      + given the states of a few neurons $\immplies$ fill out the whole state to recover a whole memory

+ Storing memories in a Hopefield net
  + with activities of $1$ and $-1$
    + stored as a binary state vector by incrementing the weight btw any two units by the product of their activities
    + treating biases as weights from a permanently on unit
    + very simple rule: not error-driven
      + just go through data once and done
      + a genuine ionline rule
      + not predicting what the right answer is and then making small adjustments
    + both its strength and its weakness
      + not error correction rule
      + able to be online but not a very efficient way to store things

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

