# Multiclasses Machine Learning

## Learning to predict the next word

### Lecture Notes

+ A simple example of relational information

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture4/lec4.pptx" ismap target="_blank">
      <img src="img/m04-01.png" style="margin: 0.1em;" alt="Example of family trees" title="Example of family trees" width=450>
    </a>
  </div>

+ Another way to express the same information
  + Make a set of propositions using the 12 relationships
    + son, daughter, nephew, niece, father, mother, uncle, aunt
    + brother, sister, husband, wife
  + Examples:
    + (colin has-father james)
    + (colin has-mother victoria)
    + (james has-wife victoria) <span style="color: blue;">this follow from th two above</span>
    + (charlotte has-brother colin)
    + (victoria bas-brother arthur)
    + (charlotte has-uncle arthur) <span style="color: blue;">this follow from th two above</span>

+ A relational learning task
  + Figuring out the regularities from given family trees
    + express with symbolic rules
    + e.g., (x has-mother y) & (y has-husband z) $\Rightarrow$ (x has-father z)
  + Finding the symbolic rules: difficult search through a very large discrete space of possibilities
  + Q: neural network able to capture the same knowledge of rules through a continuous space of weights

+ The structure of the neural net and example

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture4/lec4.pptx" ismap target="_blank">
      <img src="img/m04-02.png" style="margin: 0.1em;" alt="The structure of neural network to search symbolic rules" title="The structure of neural network to search symbolic rules" height=200>
      <img src="img/m04-03.png" style="margin: 0.1em;" alt="The example to search symbolic rules" title="The example to search symbolic rules" height=200>
    </a>
  </div>

+ What the network learns
  + Six hidden units in the bottleneck
    + connected to the input representation of person 1
    + learn to represent features of people
    + useful for predicting the answer
    + e.g., nationality, generation, branch of the family tree
  + Features and central layer
    + features: only useful if the other bottlenecks use similar representations
    + central layer: learn how features predict other features
    + e.g., <br/>
      input personal of generation 3 <span style="color: red;">and</span><br/>
      relationship requires answer to be one generation up<br/><span style="color: red;">implies</span><br/>
      Output person is of generation 2

+ Another way to see that it works
  + 12 relationships form 4 of the triples
    + train the network on all
    + sweep through the training set many times
    + adjust the weight slightly each time
  + Validate on the 4 held-out cases
    + about 3/4 correct
    + good for a 24-way choice
    + able to train on a much smaller fraction of a big datasets

+ A large-scale example
  + Suppose a database with millions of relational facts of the for (A R B)
    + train a net to discover vector representations of the terms
    + predict 3rd terms from the first two terms
    + using the trained net to find very unlikely triples
    + unlikely triples: potential errors in the database


### Lecture Video

<video src="https://youtu.be/ReUrmqStBd4?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## A brief diversion into cognitive science

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Another diversion_The softmax output function

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Neuro-probabilistic language models

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## ways to deal with large number of possible outputs

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>

