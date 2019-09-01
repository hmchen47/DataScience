# Convolution Neural Networks (CNN)

## Why object recognition is difficult

### Lecture Notes

+ Issues about object recognition
  + <span style="color: darkblue; font-weight: bold;">Segmentation</span>
    + real scenes cluttered with other objects
      + stereo cues: human due to two eyes but not static images
    + difficult to identify pieces as parts of the same object
    + parts of a object hidden behind other objects
  + <span style="color: darkblue; font-weight: bold;">Lighting</span>
    + intensities of pixels determined by the nature of the object
    + intensities of pixels determined by the lighting as well
    + e.g., black surface in bright light w/ much more intense pixels than whit surface in very gloomy light
    + object recognition: convert many intensities of the pixels into a class label
    + intensities varying for all sorts of reasons nothing related to the identity of the object
  + <span style="color: darkblue; font-weight: bold;">Deformation</span>
    + deforming objects in various non-affine ways
    + e.g., hand-written 2 w/ a large loop or just a cusp
  + <span style="color: darkblue; font-weight: bold;">Affordances</span>
    + object classes defined by how they are used
    + e.g., chairs designed for sitting on w/ a wide variety of physical shapes, including armchairs, modern chairs mad with steel frames and wood backs
  + <span style="color: darkblue; font-weight: bold;">Viewpoint</span>
    + 3-D object w/ variety of viewpoints
    + viewpoint changes $\implies$ changes in images
    + not coped with by standard learning methods
    + information hops btw input dimensions (i.e. pixels)
      + envision the input dimensions correspond to pixels
      + eyes not moving to follow object movement but images with different pixels
    + fix the issue in systematic way

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture5/lec5.pptx" ismap target="_blank">
      <img src="img/m05-01.png" style="margin: 0.1em;" alt="Digit with different viewpoints" title="Digit with different viewpoints" width=150>
    </a>
  </div>

+ dimension-hopping phenomenon
  + info jumping from one input dimension to another
  + e.g., age and weight of a patient are inputs
    + some coders change the info dimension
    + swap the age and weight fields
  + viewpoint carried


### Lecture Video

<video src="https://youtu.be/Qx3i7VWYwhI?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Ways to achieve viewpoint invariance

### Lecture Notes

+ Viewpoint invariance
  + Human good at viewpoint invariant
  + one of the main difficulties in making computers perceive
  + still no accepted solutions
  + Approaches
    + using redundant invariant features
    + putting a box around the object w/ normalized pixels
    + convolutional neural networks by replicating features with pooling
    + using a hierarchy of parts that have explicit poses relative to the camera

+ Invariant feature approach
  + extract a large, redundant set of invariant features under transformations
    + e.g., pair of roughly parallel lines w/ a red dot btw them
    + what baby herring gulls used to know where to peck for food
    + if paint that feature on a piece of wood, the baby herring gulls will peck at the appropriate place on the piece of wood.
  + with enough invariant features, only one way to assemble them into an object
    + no need to represent the relationships btw features directly
    + relationship captured by other features
    + Psychologist Wayne: Strings of letters
      + Shimon Ullman envisioned it
      + sort of acute point requiring a big bag of features
      + with overlapping and redundant features, one feature will tell how two other features are related
  + avoid forming features from parts of different objects
    + human recognition: having a whole bunch of features composed of parts of different objects
    + very misleading for recognition

+ Judicious normalization approach
  + putting a box around the object
  + using the box as a coordinate frame for a set of normalized pixels
  + solving the dimension-hopping problem
    + the box on the rigid shape avoid the effect of changes in viewpoint
    + correctly choosing the box results in the same normalized pixels for the same part of an object
    + box not required to be rectangular
    + any box providing invariant to many degrees of freedom: <span style="color: red;">translation, rotation, scale, shear, stretch, ...</span>
  + difficulties on choosing the box
    + segmentation errors
    + occlusion: not just shrinking a box around things
    + unusual orientations: a 'd' with an extra stroke w/ the loop D as upright one of those characters
  + chicken-egr problem
    + to get the box right requires to recognize the shape
    + to recognize the shape requires to get the box right
  + Human recognizes the letter before doing mental rotation to decide if it's a mirror image
    + letter R w/ a vertical stroke at the back and a loop facing forwards at the top
    + recognize the letter R perfect well before any mental rotation
    + then recognized its upside down to know how to rotate it
    + mental rotation for dealing with judgement like handedness, e.g., correct R or mirror, not used for dealing with upside down

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="url" ismap target="_blank">
      <img src="img/m05-02.png" style="margin: 0.1em;" alt="Recognizing letter after rotation mentally" title="Recognizing letter after rotation mentally" width=100>
    </a>
  </div>

+ Brute force normalization approach
  + Using well-segmented, upright images to fit the correct box for training
  + Designing period - try all possible boxes in a range of positions and scales
    + widely used for computer vision
    + particularly to detect upright things like faces and house numbers in unsegmented images
    + much more efficient if the recognizer can cope with some variation in position and scale
    + then able to use a coarse grid when trying all possible boxes


### Lecture Video

<video src="https://youtu.be/SxD-YVxIygc?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Convolutional neural networks for hand-written digit recognition

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Convolutional neural networks for object recognition

### Lecture Notes





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>

