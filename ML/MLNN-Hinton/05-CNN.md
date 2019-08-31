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
      <img src="img/m05-01.png" style="margin: 0.1em;" alt="text" title="caption" width=200>
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





### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width=180>
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

