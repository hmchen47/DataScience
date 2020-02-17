# Neural Networks for Machine Learning

## 01. Introduction to Machine Learning

+ [Why do we need machine learning](01-IntroML.md#why-do-we-need-machine-learning)
+ [What are neural networks](01-IntroML.md#what-are-neural-networks)
+ [Some simple models of neurons](01-IntroML.md#some-simple-models-of-neurons)
+ [A simple example of learning](01-IntroML.md#a-simple-example-of-learning)
+ [Three types of learning](01-IntroML.md#three-types-of-learning)

## 02. Perceptron Learning Procedure

+ [An overview of the main types of network architecture](02-Perceprtons.md#an-overview-of-the-main-types-of-network-architecture)
+ [Perceptrons](02-Perceprtons.md#perceptrons-the-first-generation-of-neural-networks)
+ [A geometrical view of perceptrons](02-Perceprtons.md#a-geometrical-view-of-perceptrons)
+ [Why the learning works](02-Perceprtons.md#why-the-learning-works)
+ [What perceptrons can not do](02-Perceprtons.md#what-perceptrons-can-not-do)


## 03. Backpropagation Learning Procedure
  
+ [Learning the weights of a linear neuron](03-Backpropagation.md#learning-the-weights-of-a-linear-neuron)
+ [The error surface for a linear neuron](03-Backpropagation.md#the-error-surface-for-a-linear-neuron)
+ [Learning the weights of a logistic output neuron](03-Backpropagation.md#learning-the-weights-of-a-logistic-output-neuron)
+ [The backpropagation algorithm](03-Backpropagation.md#the-backpropagation-algorithm)
+ [How to use the derivatives computed by the backpropagation algorithm](03-Backpropagation.md#how-to-use-the-derivatives-computed-by-the-backpropagation-algorithm)


## 04. Multiclasses Machine Learning

+ [Learning to predict the next word](04-Multiclasses.md#learning-to-predict-the-next-word)
+ [A brief diversion into cognitive science](04-Multiclasses.md#a-brief-diversion-into-cognitive-science)
+ [Another diversion_The softmax output function](04-Multiclasses.md#another-diversion-the-softmax-output-function)
+ [Neuro-probabilistic language models](04-Multiclasses.md#neuro-probabilistic-language-models)
+ [ways to deal with large number of possible outputs](04-Multiclasses.md#dealing-with-large-number-of-possible-outputs)


## 05. Convolutional Neural Networks

+ [Why object recognition is difficult](05-CNN.md#why-object-recognition-is-difficult)
+ [Ways to achieve viewpoint invariance](05-CNN.md#ways-to-achieve-viewpoint-invariance)
+ [Convolutional neural networks for hand-written digit recognition](05-CNN.md#convolutional-neural-networks-for-hand-written-digit-recognition)
+ [Convolutional neural networks for object recognition](05-CNN.md#convolutional-neural-networks-for-object-recognition)


## 06. Mini-batch Gradient Descent

+ [Overview of mini-batch gradient descent](06-MiniBatch.md#overview-of-mini-batch-gradient-descent)
+ [A bag of tricks for mini-batch descent](06-MiniBatch.md#a-bag-of-tricks-for-mini-batch-descent)
+ [The momentum method](06-MiniBatch.md#the-momentum-methodadaptive-learning-rate-for-each-connection)
+ [Adaptive learning rate for each connection](06-MiniBatch.md#)
+ [rmsprop: Normalized the gradient](06-MiniBatch.md#rmsprop-normalized-the-gradient)


## 07. Recurrent Neural Networks

+ [Modeling sequences: A brief overview](07-RNN.md#71-modeling-sequences-a-brief-overview)
+ [Training RNNs with backpropagation](07-RNN.md#72-training-rnns-with-backpropagation)
+ [A toy example of training an RNN](07-RNN.md#73-a-toy-example-of-training-an-rnn)
+ [Why it is difficult to train an RNN](07-RNN.md#74-why-it-is-difficult-to-train-an-rnn)
+ [Long term short term memory](07-RNN.md#75-long-short-term-memory)


## 08. Multicaptive Connections
  
+ [A brief overview of Hessian-free optimization](08-RNN2.md#81-a-brief-overview-of-hessian-free-optimization)
+ [Modeling character strings with multiplicative connections](08-RNN2.md#82-modeling-character-strings-with-multiplicative-connections)
+ [Learning to predict the next character using HF](08-RNN2.md#83-learning-to-predict-the-next-character-using-hf)
+ [Echo state networks](08-RNN2.md#84-echo-state-networks)


## 09. Overfitting
  
+ [Overview of ways to improve generalization](09-Overfitting.md#91-overview-of-ways-to-improve-generalization)
+ [Limiting size of the weights](09-Overfitting.md#92-limiting-size-of-the-weights)
+ [Using noise as a regularizer](09-Overfitting.md#93-using-noise-as-a-regularizer)
+ [Introduction to the Bayesian approach](09-Overfitting.md#94-introduction-to-the-bayesian-approach)
+ [The Bayesian interpretation of weight decay](09-Overfitting.md#95-the-bayesian-interpretation-of-weight-decay)
+ [MacKays quick and dirty method of fixing weight costs](09-Overfitting.md#96-mackays-quick-and-dirty-method-of-fixing-weight-costs)


## 10. Model Combination and Dropouts
  
+ [Why it helps to combine models](10-CombineDropout.md#101-why-it-helps-to-combine-models)
+ [Mixtures of experts](10-CombineDropout.md#102-mixtures-of-experts)
+ [The idea of full bayesian learning](10-CombineDropout.md#103-the-idea-of-full-bayesian-learning)
+ [Making full bayesian learning practical](10-CombineDropout.md#104-making-full-bayesian-learning-practical)
+ [Dropout an efficient way to combine neural nets](10-CombineDropout.md#105-dropout-an-efficient-way-to-combine-neural-nets)


## 11. Hopfield Nets and Boltzmann Machines

+ [Hopfield Nets](11-Hopfield.md#111-hopfield-nets)
+ [Dealing with spurious minima in Hopfield nets](11-Hopfield.md#112-dealing-with-spurious-minima-in-hopfield-nets)
+ [Hopfields Nets with hidden units](11-Hopfield.md#113-hopfields-nets-with-hidden-units)
+ [Using stochastic units to improve search](11-Hopfield.md#114-using-stochastic-units-to-improve-search)
+ [How a Boltzmann machine models data](11-Hopfield.md#115-how-a-boltzmann-machine-models-data)


## 12. Restricted Boltzmann Machine (RBMs)

+ [The Boltzmann machine learning algorithm](12-Boltzmann.md#121-the-boltzmann-machine-learning-algorithm)
+ [More efficient ways to get the statistics](12-Boltzmann.md#122-more-efficient-ways-to-get-the-statistics)
+ [Restricted Boltzmann machines](12-Boltzmann.md#123-restricted-boltzmann-machines)
+ [An example of contrastive divergence learning](12-Boltzmann.md#124-an-example-of-contrastive-divergence-learning)
+ [RBMs for collaborative filtering](12-Boltzmann.md#125-rbms-for-collaborative-filtering)


## 13. Deep Belief Nets

+ [The ups and downs of backpropagation](13-BeliefNets.md#)
+ [Belief nets](13-BeliefNets.md#)
+ [Learning Sigmoid Belief Nets](13-BeliefNets.md#)
+ [The wake-sleep algorithm](13-BeliefNets.md#)


## 14. Generative Adversarial Networks (GANs)
  
+ [Learning layers of features by stacking RBMs](14-GANs.md#)
+ [Discriminative fine-tuning for DBNs](14-GANs.md#)
+ [What happens during discriminative fine-tuning](14-GANs.md#)
+ [Modeling real-valued data with an RBM](14-GANs.md#)
+ [RBMs are infinite sigmoid belief nets](14-GANs.md#)


## 15. Hierarchical Structure with Neural Networks

+ [From principal components analysis to autoencoders](15-Hierarchy.md#)
+ [Deep Autoencoders](15-Hierarchy.md#)
+ [Deep autoencoders for document retrieval and visualization](15-Hierarchy.md#)
+ [Semantic hashing](15-Hierarchy.md#)
+ [Learning binary codes for image retrieval](15-Hierarchy.md#)
+ [Shallow autoencoders for pre-training](15-Hierarchy.md#)


## 16. Deep Neural Networks
  
+ [Learning a joint model of images and captions](16-DeepNN.md#)
+ [Hierarchical coordinate frames](16-DeepNN.md#)
+ [Bayesian optimization of neural network hyperparameters](16-DeepNN.md#)


## Related Articles

+ Matthew Stewart, [Introduction to Neural Networks](a01-IntroNN.md)
+ Matthew Stewart, [Intermediate Topics in Neural networks](a02-IntermediateNN.md)
+ Matthew Stewart, [Neural Network Optimization](a03-Optimization.md)
+ Matthew Stewart, [Simple Guide to Hyperparameter Tuning in Neural Networks](a04-Hyperparameter.md)
+ Matthew Stewart, [Neural Style Transfer and Visualization of Convolutional Networks](a05-VisualCNN.md)
+ Random Nerd, [Delta Learning Rule & Gradient Descent | Neural Networks](a06-DeltaRule.md)
+ Drew Rollins, [Delta Function](a07-DeltaFunc.md)
+ Chris McCormick, [Deep Learning Tutorial - Softmax Regression](a08-SoftmaxReg.md)
+ [Softmax Classifier](a09-SoftmaxClass.md) in CS231n Convolutional Neural Networks for Visual Recognition, Stanford University
+ Adit Deshpande, [A Beginner's Guide To Understanding Convolutional Neural Networks](a10-CNNsGuide.md)
+ Adit Deshpande, [The 9 Deep Learning Papers You Need to Know About](a11-9Papers.md)


<hr/>

+ D. Spiegelhalter, [An Overview of the Bayesian Approach](http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/Bayes/an%20overview%20of%20the%20Bayesian%20approach.pdf), In: Bayesian Approaches to Clinical Trials and Health-Care Evaluation
+ [Bayesian Inference](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)
+ S. Ghosh, [Basics of Bayesian Methods](https://www.researchgate.net/profile/Sujit_Ghosh4/publication/45283465_Basics_of_Bayesian_Methods/links/55cce51208ae1141f6b9e8e0/Basics-of-Bayesian-Methods.pdf), in "Methods in molecular biology" (Clifton, N.J.) 620:155-78, 2010
+ [Bayesian Inference](http://www.stat.cmu.edu/~larry/=sml/Bayes.pdf), chapter 12
+ A. Julien-Laferriere, [Hopfield network](http://perso.ens-lyon.fr/eric.thierry/Graphes2010/alice-julien-laferriere.pdf)
+ [Hopfield Model of Neural Network](https://shodhganga.inflibnet.ac.in/bitstream/10603/1760/6/06_chapter%202.pdf), Chapter 2,
+ R. Rojas, [The Hopfield Model](https://page.mi.fu-berlin.de/rojas/neural/chapter/K13.pdf) in Neural Networks, Springer, 1996
+ J. J. Hopfield, "[Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/content/pnas/79/8/2554.full.pdf)", Proceedings of the National Academy of Sciences of the USA, vol. 79 no. 8 pp. 2554–2558, April 1982
+ J. Hopfield, D. Feinstein and R. Palmer, [‘Unlearning’ has a stabilizing effect in collective memories](https://www.researchgate.net/profile/John_Hopfield/publication/16333131_'Unlearning'_has_a_stabilizing_effect_in_collective_memories/links/563fef2f08aec6f17ddb84cc/Unlearning-has-a-stabilizing-effect-in-collective-memories.pdf), Nature 304(5922):158-9 · July 1983
+ G. Hinton and T. Sejnowski, [Optimal perceptual inference](https://papers.cnl.salk.edu/PDFs/Optimal%20Perceptual%20Inference%201983-646.pdf), Proceedings of the IEEE conference on Computer Vision and Pattern Recognition
+ L. Saul, T. Jaakkola, M. Jordan, [Mean field theory for sigmoid belief networks](https://www.jair.org/index.php/jair/article/download/10156/24075), Journal of artificial intelligence research, 1996
+ G. Hinton and T. Sejnowski, [Learning and relearning in Boltzmann machines](https://www.researchgate.net/profile/Terrence_Sejnowski/publication/242509302_Learning_and_relearning_in_Boltzmann_machines/links/54a4b00f0cf256bf8bb327cc/Learning-and-relearning-in-Boltzmann-machines.pdf), In Rumelhart, D. E. and McClelland, J. L., editors, Parallel Distributed Processing: Explorations in the Microstructure of Cognition. Volume 1: Foundations, MIT Press, Cambridge, MA., 1986
+ G. Hinton, R. Salakhutdinov, [A Better Way to Pretrain Deep Boltzmann Machines](http://papers.nips.cc/paper/4610-a-better-way-to-pretrain-deep-boltzmann-machines.pdf), Advances in Neural Information Processing Systems 25 (NIPS 2012)
+ H. Yu, [A gentle tutorial on Restricted Boltzmann Machine and Contrastive Divergence](https://www.researchgate.net/profile/Hongyang_Yu2/publication/315382074_A_gentle_tutorial_on_Restricted_Boltzmann_Machine_and_Contrastive_Divergence/links/58cf1a654585157b6db02f5a/A-gentle-tutorial-on-Restricted-Boltzmann-Machine-and-Contrastive-Divergence.pdf), 2017
+ A. Fischer & C. Igel, [An Introduction to Restricted Boltzmann Machines](https://www.researchgate.net/profile/Asja_Fischer/publication/243463621_An_Introduction_to_Restricted_Boltzmann_Machines/links/0a85e5320cc4851d83000000/An-Introduction-to-Restricted-Boltzmann-Machines.pdf). In Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications: 17th Iberoamerican Congress, CIARP 2012, Buenos Aires, Argentina, September 3-6, 2012. Proceedings (pp.14-36
+ Wythoff, BJ, 1993. [Backpropagation neural networks. A tutorial](https://www.researchgate.net/profile/Samreen_Sid/post/rookie/attachment/59d61d8cc49f478072e97144/AS%3A271736812048384%401441798512764/download/1993+Backpropagation+neural+networks+A+tutorial.pdf), Chemometrics and Intelligent Laboratory Systems, 18: 115-155
+ A. Kurenkov, [A 'Brief' History of Neural Nets and Deep Learning](http://www.andreykurenkov.com/writing/ai/a-brief-history-of-neural-nets-and-deep-learning/), 2015
+ Judea Pearl, [Belief networks revisited](https://ftp.cs.ucla.edu/pub/stat_ser/R175.pdf), Artificial Intelligence, 1993
+ Judea Pearl and Stuart Russell, [Bayesian Networks](https://ftp.cs.ucla.edu/pub/stat_ser/r277.pdf), Technical Report, R-277, 2000
+ Judea Pearl, [A Personal Journey into Bayesian Networks](https://ftp.cs.ucla.edu/pub/stat_ser/r476.pdf), Technical Report, R-476, 2018
+ I. Ben‐Gal, [Bayesian Networks](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470061572.eqr089), Encyclopedia of Statistics in Quality and Reliability 2008
+ M. Wellman and M. Henrion, [Explaining 'explaining away'](https://pdfs.semanticscholar.org/bffd/c2699cba4893bd6a6befdc8f46f6f23f33d1.pdf?_ga=2.60626154.639362258.1581809305-1938421553.1581809305), IEEE Transactions on Pattern Analysis and Machine Intelligence, 15(3):287-292, April 1993
+ G. Hinton, P. Dayan, B. Frey, and R. Neal, [The wake-sleep algorithm for unsupervised neural networks](https://www.cs.toronto.edu/~hinton/csc2535/readings/ws.pdf), Science, Vol. 268, Issue 5214, pp. 1158-1161, 1995
+ A. Ng and M. Jordan, [On discriminative vs. generative classifiers: a comparison of logistic regression and naive bayes](http://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes.pdf), Adv. Neural Inf. Proc. Syst. 14, 841 (2002)
+ C. Bishop and J. Lasserre, [Generative or Discriminative? Getting the Best of Both Worlds](https://www.researchgate.net/profile/David_Heckerman/publication/228993892_Generative_or_Discriminative_Getting_the_Best_of_Both_Worlds/links/5547741b0cf2e2031b36b897/Generative-or-Discriminative-Getting-the-Best-of-Both-Worlds.pdf), BAYESIAN STATISTICS 8, pp. 3–24, 2007
+ R. Tibshirani, [Modeling Basics: Assessment, Selection, and Complexity](https://www.stat.cmu.edu/~ryantibs/statml/review/modelbasics.pdf), Statistical Machine Learning, Spring 2015



## References

+ [Introduction to Neural Networks and Machine Learning - CSC321 Winter 2014](http://www.cs.toronto.edu/~tijmen/csc321/)
  + [Lecture Notes](http://www.cs.toronto.edu/~tijmen/csc321/lecture_notes.shtml)
  + [Assignmenets](http://www.cs.toronto.edu/~tijmen/csc321/assignments.shtml)
  + [Optional Readings](http://www.cs.toronto.edu/~tijmen/csc321/texts.shtml)
  + [Computing](http://www.cs.toronto.edu/~tijmen/csc321/computing.shtml)
  + [Exam](http://www.cs.toronto.edu/~tijmen/csc321/tests.shtml)
+ [2012 COURSERA COURSE LECTURES: Neural Networks for Machine Learning](http://www.cs.toronto.edu/~hinton/nntut.html)
  + [Lecture Videos](http://www.cs.toronto.edu/~hinton/coursera_lectures.html)
  + [Lecture Slides](http://www.cs.toronto.edu/~hinton/coursera_slides.html)
+ [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  + [Course Notes](http://cs231n.github.io/)
  + [Course Video](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
+ Github Related Links
  + [Fuyang Liu](https://github.com/liufuyang/course-Neural-Networks-for-Machine-Learning)
  + [Chinmay Das](https://github.com/chinmaydas96/Neural-Networks-for-Machine-Learning)
  + [Chouffe](https://github.com/Chouffe/hinton-coursera)
  + [khanhnamle1994](https://github.com/khanhnamle1994/neural-nets)
+ Harry Wechsler, [Neural Networks for Perception. Computation, Learning, and Architectures](http://93.174.95.29/main/1199000/8de6d2df3a95f9c5477b6d95f3e80a06/Harry%20Wechsler%20-%20Neural%20Networks%20for%20Perception.%20Computation%2C%20Learning%2C%20and%20Architectures-Elsevier%20Inc%2C%20Academic%20Press%20%281992%29.pdf), AP, 1992



