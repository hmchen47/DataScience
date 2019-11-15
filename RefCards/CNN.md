# Convolutional Neural Network

## Object Recognition and Classification

+ [Issues about object recognition](../ML/MLNN-Hinton/05-CNN.md#lecture-notes)
  + Segmentation: real scenes cluttered with other objects
  + Lighting: intensities of pixels determined by the nature of the object
  + Deformation: deforming objects in various non-affine ways
  + Affordances: the quality or property of an object that defines its possible uses or makes clear how it can or should be used
  + Viewpoint: 3-D object w/ variety of viewpoints

+ [Dimension-hopping phenomenon](../ML/MLNN-Hinton/05-CNN.md#lecture-notes):
  + info jumping from one input dimension to another
  + e.g., age and weight of a patient are inputs


## Solutions for Viewpoint Invariance

+ [Viewpoint invariance](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-1)
  + one of the main difficulties in making computers perceive
  + still no accepted solutions
  + Approaches
    + redundant invariant features
    + a box around the object w/ normalized pixels
    + convolutional neural networks by replicating features with pooling
    + hierarchy of parts that have explicit poses relative to the camera

+ [Redundant invariant feature approach](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-1)
  + extract a large, redundant set of invariant features under transformations
  + with enough invariant features, only one way to assemble them into an object
  + avoid forming features from parts of different objects

+ [Judicious normalization approach](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-1)
  + putting a box around the object
  + using the box as a coordinate frame for a set of normalized pixels
  + solving the dimension-hopping problem
    + correctly choosing the box results in the same normalized pixels for the same part of an object
    + any box providing invariant to many degrees of freedom: translation, rotation, scale, shear, stretch, ...
  + Issues
    + segmentation errors
    + occlusion
    + unusual orientations
  + chicken-egg problem: getting the box right $\leftrightarrow$ recognizing the shape
  + Human recognizes the letter before doing mental rotation to decide if it's a mirror image

+ [Brute force normalization approach](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-1)
  + Using well-segmented, upright images to fit the correct box for training
  + Designing period - try all possible boxes in a range of positions and scales


## Replicated Feature Approach

+ [The replicated feature approach](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + using many different copies of the same feature detector w/ different positions
  + using several different feature types, each with its own map of replicated detectors
  + Example

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture5/lec5.pptx" ismap target="_blank">
        <img src="../ML/MLNN-Hinton/img/m05-03.png" style="margin: 0.1em;" alt="Illustriation for replicated features" title="Illustriation for replicated features" width=150>
      </a>
    </div>

+ [Backpropagation with weight constraints](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + modify the backpropagation algorithm to incorporated linear constraints btw the weights for replicated features
  + compute the gradients as usual
  + modify the gradients to satisfy the constraints
  + once the weights satisfying the linear constrains, they continue satisfying the linear constrain after weight update

+ [Replicated feature](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + not translation invariant
  + Equivariant activities
    + replicated features not make the neural activities invariant to translation
    + activities equivalent
  + Invariant knowledge
    + a feature useful in some locations during training $\to$ the feature available in all locations during testing
    + knowing how to detect a feature in one place $\to$ knowing how to detect same feature in another place

+ [Pooling for invariant activities](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + to achieve some invariance in the activities, pull the outputs of replicated feature detectors
  + get a small amount of translational invariance at each level by averaging four neighboring replicated detectors to give a single output to the next level
  + slightly better to take the maximum of the four neighboring feature detectors than averaging them
  + Problem: lost information about the precise positions of things after several levels of pooling
  + impossible to use the precise spatial relationships btw high-level parts for recognition


## Application - Hand-written Recognition

### Le Net (Yann LeCun & collaborators)

+ [Objective](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + a really good recognizer for hand-written digits
  + using backpropagation in feed-forward net

+ [Architecture](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + many hidden layers
  + many maps of replicated units in each layer
  + pooling of the outputs of nearby replicated units btw layers
  + a wide net able to cope with several characters at once even if they overlap
  + no segmented individual characters required before fed into the net
  + a clever way of training a complete system, not just a recognizer for individual characters
  + maximum margined method: way before maximum margin invented
  + input: (A)
  + feature maps (c1)
  + Subsampling/pooling

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://engmrk.com/lenet-5-a-classic-cnn-architecture/" ismap target="_blank">
      <img src="https://engmrk.com/wp-content/uploads/2018/09/LeNet_Original_Image.jpg" style="margin: 0.1em;" alt="The LeNet-5 architecture consists of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier." title="Architecture of LeNet5" width=350>
    </a>
  </div>

+ [Priors and Prejudice](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + design appropriately by applying prior knowledge about the task into the network
  + using prior knowledge to create a more training data
  + allowing optimization to discover clever ways of using the multilayer network


### Brute Force Approach for Hand-written Recognition

+ [Brute force approach](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + Designing LeNet w/ the invariant knowledge
    + local connectivity
    + weight-sharing
    + pooling
  + about 80 errors w/ origin LeNet

+ [Ciresan et. al. net (2010)](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + applying knowledge of invariance
  + creating a huge amount of carefully designed extra training data (synthetic data)
  + producing many new training examples by applying many different transformations on each training image
  + train a large deep, dumb net on a GPU w/o much overfitting
  + 3 tricks used to prevent from overfitting when generating synthetic data
  + Achieving about 35 errors


### Measurement for Hand-written Recognition

+ [McNemar test](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)
  + using the particular errors in diagrams
  + much more powerful than a test counting the numbers of errors

+ [Example](../ML/MLNN-Hinton/05-CNN.md#lecture-notes-2)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="http://www.cs.toronto.edu/~hinton/coursera/lecture5/lec5.pptx" ismap target="_blank">
      <img src="../ML/MLNN-Hinton/img/m05-08.png" style="margin: 0.1em;" alt="McNemar test for error rate comparisons" title="McNemar test for error rate comparisons" width=450>
    </a>
  </div>


## CNN for Object Recognition



