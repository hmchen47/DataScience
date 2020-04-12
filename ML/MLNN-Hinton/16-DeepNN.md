# 16. Deep Neural Networks
  
## 16.1 Learning a joint model of images and captions

### Lecture Notes

+ Modeling the joint density of images and captions
  + N. Srivastava and R. Salakhutdinov, [Multimodal Learning with Deep Boltzmann Machines](https://tinyurl.com/wzsknt8), NIPS 2012
  + goal: to build a joint density model of captions and standard computer vision feature vectors extracted from real photographs
  + issue: requiring more computation than building a joint density model of labels and digit images
  + procedure:
    + training a multilayer model of images
      + a multilayer model of the features extracted from images
      + using standard computer vision features
    + training a separated multilayer model of word-count vectors from the captions
    + adding a new top layer connected to the top layers of both individual models
      + using further joint training of the whole system
      + allowing each modality to improve the earlier layers of other modality
  + using a deep Boltzmann machine (DBM)
    + instead of using a deep belief net
    + symmetric connection btw all pairs of layers
    + further joint training of the whole DBM allows each modality to improve the earlier layers of the other modality $\to$ using a DBM
    + able to use a DBM and done generative fine-tuning w/ contrastive wake-sleep
    + fine-tuning algorithm probably working better for DBM
  + effect of pre-training on the hidden layers of the DBM
    + standard pre-training on a composite model w/ RBM $\to$ not DBM
    + combining a stack of RBM $\to$ a deep belief network

+ Combining 3 RBMs to make a DBM
  + combining a stack of pre-trained RBMs $\to$ DBM
  + network architecture (see diagram)
    + the top and bottom RBMs pre-trained w/ the weights in one direction twice as big as in the other direction
      + named scale symmetric
      + the bottom RBM trained w/ bottom-up weights being twice as big as the top-down weights
      + the top RBM trained the bottom-up weights w/ half of the top-down weights
    + the middle layers: geometric model averaging
      + trained w/ symmetric weights
    + combining these RBMs to a composite model
      + RBMs in the middle simply halved its weights
      + bottom RBM weights:  halved the up-going weights = the down-poing weights
      + top RBM weights: halved the down-going weights = the top-going weights
  + rough explanation for scale symmetric
    + separated RBMs
      + $h_1$ w/ two different ways to infer the states of the units in $h_1$ in the stack of RBM on the left
      + inferring the states of $h_1$ either bottom-up from $v$ or top-down from $h-2$
      + combining middle and bottom RBMs by taking average of these two weights of inferring $h_1$
      + taking geometric average by halving the weights
      + using half of the bottom-up model, %2W_1/2$
      + same applied to middle and top RBMs
    + combined model
      + given states of $h_2$ or $v$ $\implies$ states of $h_1$
      + halving weights to prevent the double counts
      + the state of $H_2$ depending on $v$
      + using the bottom-up input from the 1st RBM and the top-dowm input from 2nd RBM $\to$ counting evidence twice
      + inferring $h_1$ from $v$ and inferring it from $h_2$, depending on $v$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/u3whuvf" ismap target="_blank">
      <img src="img/m16-01.png" style="margin: 0.1em;" alt="Combining RBMs to a DBM" title="Combining RBMs to a DBM" width=250>
    </a>
  </div>


### Lecture Video

<a href="https://tinyurl.com/sno2msv" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 16.2 Hierarchical coordinate frames

### Lecture Notes

+ Approaches to recognizing objects
  + deep convolutional neural network
  + parts based approach
  + existing features used for computer vision to
    + extract from images
    + make histograms
    + use a lot of hand enginerring

+ Disadvantages of convolutional neural networks
  + pooling the activities of a bunch of replicated feature detectors $\to$ losing the position of the feature detector
  + pooling
    + losing the precise spatial relationship btw higher-level parts such as a nose and a mouth
    + precise spatial relationships required for identity recognition
    + partial solution: overlapping the pools
      + feature occurring in several pools
      + retaining more information about its position
  + only translations to replicated features detectors
    + unable to extrapolate their understanding of geometric relationships to radically new viewpoints, such as different orientations or scales
    + human good at extrapolation
    + seeing new shape $\to$ recognizing it from a different viewpoint
  + approach w/ convolutional neural networks
    + training viewpoints on transformed data
    + huge training sets w/ orientations, scales and lighting $\to$ coping w/ the variations
    + clumsy way to dealing w/ variantions

+ The hierarchical coordinate frame approach
  + a group of neurons to represent the conjunction of
    + the shape of a feature: neurons telling features of objects, such as nose and mouth
    + the pose relative to the retina
      + the relationship btw the coordinate frame of the retina
      + the intrinsic coordinate frame of the feature
      + should embed a coordinate frame within it
  + recognizing larger features by using the consistency of the poses of their parts
    + representation of the pose of parts of objects relative to retina
    + easier to use relationships btw parts and recognize larger objects
    + using the poses of the parts as a cue for recognizing a larger shape
  + example: nose and mouth
    + left diagram:
      + nose and mouth w/ the right spatial relationship to one another
      + using mouth or nose to predict the pose of whole face $\to$ similar prediction
    + right diagram:
      + nose and mouth w/ wrong spatial relationship
      + making prediction separately about the pose of the whole face $\to$ not agreed

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/u3whuvf" ismap target="_blank">
      <img src="img/m16-02.png" style="margin: 0.1em;" alt="Relationship of nose and mouth" title="Relationship of nose and mouth" width=250>
    </a>
  </div>

+ Two layers in a hierarchy of parts
  + a higher level visual entity
    + several level visual entities agreeing on their predictions for its pose
    + inverse computer graphics
  + network model
    + larger parts recognized by consistent predictions from smaller parts
    + $T_j$: a collection of neurons $\to$ recognizing the pose of the whole face
    + $p_j$: a single logistic neuron to representing whether or not a face there
    + similar neurons to represent the mouth, $T_i$, and nose, $T_h$
    + recognizing the face by noticing that those two representations make consistent predictions
      + taking a vector of activities to represent the pose of the mouth
      + multiplying by a matrix $T_{ij}$ to represent the spatial relationship btw a mouth and a face
      + obtaining the prediction $T_i T_{ij}$ for the pose of the face
    + same procedure for nose to get $T_h T_{hj}$
    + $T_iT_{ij} \approx T_hT_{hj} \implies$ the nose and the mouth w/ righ spatial relationship
  + inverse computer graphics
    + knowing the pose of the face $\implies$ computed by using inverse of $T_{ij}$; same as nose
    + computer graphics: from poses of larger things to poses of their parts
    + computer vision: from the poses of the parts to the poses of the larger things
    + verifying the consistency btw them

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/u3whuvf" ismap target="_blank">
      <img src="img/m16-03.png" style="margin: 0.1em;" alt="Hierarchical model for different parts" title="Hierarchical model for different parts" width=400>
    </a>
  </div>

+ A crucial property of the pose vectors
  + able to get neural network to represent these pose vectors, vectors of neural activities $\to$ property
  + property: modeling spatial transformation w/ linear operations
    + easier to learn a hierarchy of visual entities
    + easier to generalize across viewpoints
  + issue: small changes in viewpoint $\to$ the pose vectors of neural activities changed
  + invariant geometric properties of a shape
    + invariant in the weights, not in the activities
      + weights representing the relationship btw a part and a whole
      + e.g., $T_{ij}$ not depending on viewpoint
      + to get the invariant properties of a shape into a weights
      + changing viewpoint  $\to$ all those pose vectors in the activities changed
      + getting neural activities invariant to viewpoint = the pooling in a convolution neural network behaved
    + equivariant activities
      + varying the pose of the object $\implies$ varying activities
      + goal: getting neural activities equivariant = the pose of the object varying the activities of the neurons vary = the percept of an object not its label but its appearance
    + changing the viewpoint $\implies$ changing the percept of an object

+ Imposing coordinate frames in order to represent shapes
  + proposed by Irvin Rock, The Logic of Perception ([Wikipedia](https://tinyurl.com/w8ktgqr))
  + evidence of visual systems to represent shapes
  + What country is this? Hint: Sarah Palin (left diagram)
  + the square and the diamond: very different percepts $\to$ different properties
    + perceiving as a tilted square $\to$ acutely  sensitive to whether the angles are right angles
    + perceiving as an upright diamond $\to$ not sensitive to the angle
      + the angles probably 5 degrees off and not noticed
      + sensitive to whether the corners on the left and on the right w/ the same height
  + representing shapes $\to$ imposing coordinate frames on them
    + square or diamond: the same thing in appearance but the percepts totally different
    + depending on what coordinate frame imposing

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/u3whuvf" ismap target="_blank">
      <img src="img/m16-05a.png" style="margin: 0.1em;" alt="Shape of a country" title="Shape of a country" height=100>
      <img src="img/m16-05b.png" style="margin: 0.1em;" alt="Shape of square and diamond" title="Shape of square and diamond" width=100>
    </a>
  </div>


### Lecture Video

<a href="https://tinyurl.com/tkbt3fj" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 16.3 Bayesian optimization of neural network hyperparameters

### Lecture Notes

+ Learning hyperparameters
  + J. Snoek, H. Larochellel, and R. Adams, [Practical Bayesian Optimization of Machine Learning Algorithms](https://tinyurl.com/rgeervt), NIPS 2012
  + setting hyperparameters
    + one of the commonest issues using neural networks
    + requiring a lot of skill to set hyperparameters
      + not much experience $\to$ easily stuck on using completely wrong value for one of the hyperparameters
      + wrong hyperparameter $\to$ nothing work
    + common hyperparameters
      + number of layers
      + number of units per layers
      + type of unit
      + weight penalty
      + learning rate
      + momentum, etc.
  + __Naive grid search__
    + making a list of alternative values for each hyperparameter
    + trying all possible combinations
    + possibly several points along each axis w/ identical other parameters $\to$ similar results
    + Q: way to improve?
  + __Sampling random combinations__
    + much better if some hyperparameters w/o effect
    + a big waste to exactly repeat the settings of the other hyper-parametres

+ Machine learning to the rescue
  + observing the results instead of random combinations of values of hyperparameters
    + predicting regions of hyperparameter space probably w/ better results
    + requirements
      + predicting how well a new combinations will
      + modeling the uncertainty of the prediction
  + huge amount of computation in evaluating one setting of the hyperparameters
    + involved training a big neural network on a huge data set
    + building a model to predict how well a setting of hyperparameter will do
    + much less work than experimenting all settings of hyperparameters
    + requiring much less computation to fit the predicted model to the results of the experiments observed than running a single experiment
    + the model predicting the result from knowing previous results w/ different settings of the hyperparameters

+ Gaussian process models
  + assumption: similar inputs giving similar outputs
    + a very weak but very sensible prior for the effects of hyperparameters
    + not much more sophisticated prior than that
    + very good using that prior in an effective way
    + prior work probably the best strategy to set hyperparameters
  + learning the appropriate scale for measuring similarity on each input dimension
    + the number of hidden units: 200 similar to 300?
    + observing how similar the results w/ scaling
  + GP model
    + doing much more than just predicting the expected outcome of a particular experiment
    + how well the neural network trained on a validation set
    + in addition to predicting a mean value for how well thet expect the neural network to do
    + predicting a Gaussian distribution of values, mean and variance
  + test cases
    + close to several consistent training cases run $\to$ the predictions fairly sharp = low variance
    + far from any training cases: the predictions w/ high variance
  + strategy: a sensible way to decide what to try
    + a model w/ a big neural network $\to$ long time to run
    + using GP model as a different kind of model to experiment
    + trying to predict for some proposed new settings of the hyperparameters $\to$ how well the neural network to do and how uncertain that prediction is
    + keep track of the best setting of hyperparameters so far
    + the last result is best $\implies$ experiment performing better or staying the same
    + picking a setting of the hyperparameters $\to$ the __expected improvement__ in our best setting is big
    + not worrying about the downside
  + example: three predictions for settings A, B and C
    + A, B and C w/ different settings of the hyperparameters but not yet been tried
    + green curves: the predictions of the Gaussian process model for how well each setting will do
    + A: mean < current best so far w/ moderate variance
    + B: mean close to the best so far w/ small variance
    + C: mean lower than setting B w/ high variance
    + taking area above red line and the moment of that area about red line $\to$ maximum of 
    + moments of C > B > A $\implies$ A the worse bet while C the best bet

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/u3whuvf" ismap target="_blank">
        <img src="img/m16-04.png" style="margin: 0.1em;" alt="Gaussina distribution with difference variance" title="Gaussina distribution with difference variance" height=200>
      </a>
    </div>

+ Evaluating Bayesian optimization
  + much better approach than finding good combinations of hyperparameters
    + required resources to run a lot of experiments
    + not the kind of task people good at
    + unable to keep in mind the results of 50 different experiments and see what they predict
  + much less prone to doing
    + a good job for the method we preferred
    + a bad job for the method we are comparing with
    + people cannot help doing this
    + people try much harder for their own methods because they know it ought to work better


### Lecture Video

<a href="https://tinyurl.com/wysqg7g" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 16.4 The fog of progress

### Lecture Notes

+ Unable to predict the long-term future
  + considering driving at night
  + the number of photons received from the tail-lights of the car in front falls off $1/d^2$
  + suppose $\exits\;$ fog, received photons
    + small distances: $\propto 1/d^2$
    + big distances: $\propto \exp(-d)$
      + fog absorbed a certain fraction of of the photons per unit distance
  + front car completely invisible at a distance 
    + at which the short-range $1/d^2$ model predicts it will very visible
    + this kill people

+ The effect of exponential progress
  + short term effect
    + things changing slowly
    + easy to predict progress
    + e.g., able to make quite good guesses about what features in the next iPhone
  + longer run $\to$ perception of future hitting wall, like fog
  + long-term future of machine learning and neural networks
    + a total mystery
    + highly probable big, deep neural networks doing amazing things in the next 5 years (2012)


### Lecture Video

<a href="https://tinyurl.com/uvkn3xj" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>

