# Welcome

## Welcome to Machine Learning!

<video src="https://d3c33hcgiwev3.cloudfront.net/qHTMkA4fEeW2rSIAC2yC6g.processed/full/360p/index.mp4?Expires=1551916800&Signature=DDe3BWU9XUBJRiOrTaPgGhivN023rTRNAPA58ih7SjGCs6tNBpT3RBiTKpXf2b8Z3Wb7kYLnhyVOmDb0P9FPKAHjuduEpBhqhmMe1wWSgvZMumzOKNit5wHiTBqw7aNIExqkNPqxStpdFeKyvocn6ARTf2QDsuIl8eP-TjJQ0DY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" controls muted width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/r3LdPY_CTUqy3T2Pwu1KVQ?expiry=1551916800000&hmac=5ZbbjiO-PeT-6-oBjT2wY6ELs0CfQ1TdCmZGH_ExMWk&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

## Machine Learning Honor Code

### Machine Learning Honor Code

We strongly encourage students to form study groups, and discuss the lecture videos (including in-video questions). We also encourage you to get together with friends to watch the videos together as a group. However, the answers that you submit for the review questions should be your own work. For the programming exercises, you are welcome to discuss them with other students, discuss specific algorithms, properties of algorithms, etc.; we ask only that you not look at any source code written by a different student, nor show your solution code to other students.


### Guidelines for Posting Code in Discussion Forums

+ Scenario 1: Code to delete

    _Learner Question/Comment: "Here is the code I have so far, but it fails the grader. Please help me fix it."_

    __Why Delete?__: The reason is that if there is a simple fix provided by a student, a quick copy and paste with a small edit will provide credit without individual effort.

    _Learner Question: A student substitutes words for the math operators, but includes the variable names (or substitutes the equivalent greek letters (θ for 'theta', etc). This student also provides a sentence-by-sentence, line by line, description of exactly what their code implements. "The first line of my script has the equation "hypothesis equals theta times X", but I get the following error message..."._

    __Why Delete?__: This should be deleted. “Spelling out” the code in English is the same as using the regular code.

+ Scenario 2: Code not to delete

    Learner Question: How do I subset a matrix to eliminate the intercept?

    Mentor Response: This probably would be okay, especially if the person posting makes an effort to not use familiar variable names, or to use a context which has nothing to do with the contexts in the assignments.

    It is clearly ok to show examples of Octave code to demonstrate a technique. Even if the technique itself is directly applicable to a programming problem at hand. As long as what is typed cannot be "cut and pasted" into the program at hand.

    E.g. how do I set column 1 of a matrix to zero? Try this in your Octave work area:

    ```matlab
    >> A = magic(3)

    >> A(:,1) = 0
    ```

    The above is always acceptable (in my understanding). Demonstrating techniques and learning the language/syntax are important Forum activities.


### Resources

### Tutorials

Ref: [Programming Exercise Tutorials](https://www.coursera.org/learn/machine-learning/discussions/m0ZdvjSrEeWddiIAC9pDDA)

This post contains links to all of the programming exercise tutorials.

After clicking on a link, you may need to scroll down to find the highlighted post.

--- Note: Additional test cases can be found ([here](https://www.coursera.org/learn/machine-learning/discussions/0SxufTSrEeWPACIACw4G5w)) ---

---------------------

#### ex1

[computeCost()](https://www.coursera.org/learn/machine-learning/discussions/t35D1xn3EeWA7CIAC5WDNQ) tutorial - also applies to computeCostMulti().

[gradientDescent()](https://www.coursera.org/learn/machine-learning/discussions/-m2ng_KQEeSUBCIAC9QURQ) - also applies to gradientDescentMulti() - includes test cases.

[featureNormalize()](https://www.coursera.org/learn/machine-learning/module/vW94N/discussions/c7VBzJ9lEeWILRIOm1V0SQ) tutorial

Note: if you use OS X and the contour plot doesn't display correctly, see the "Resources Menu" page "Tips on Octave OS X" for how to fix it.

------------------------

#### ex2

Note: If you are using MATLAB version R2015a or later, the fminunc() function has been changed in this version. The function works better, but does not give the expected result for Figure 5 in ex2.pdf, and it throws some warning messages (about a local minimum) when you run ex2_reg.m. This is normal, and you should still be able to submit your work to the grader.

Note: If your installation has trouble with the GradObj option, see this thread: [link](https://www.coursera.org/learn/machine-learning/discussions/s6tSSB9CEeWd3iIAC7VAtA)

Note: If you are using a linux-derived operating system, you may need to remove the attribute "MarkerFaceColor" from the plot() function call in plotData.m.

------------------------

[sigmoid()](https://www.coursera.org/learn/machine-learning/discussions/-v5KABxxEea_TAo4ODIo0w) tutorial

[costFunction()](https://www.coursera.org/learn/machine-learning/module/mgpv7/discussions/0DKoqvTgEeS16yIACyoj1Q) cost tutorial - also good for costFunctionReg()

[costFunction()](https://www.coursera.org/learn/machine-learning/discussions/GVdQ9vTdEeSUBCIAC9QURQ) gradient tutorial - also good for costFunctionReg()

[predict()](https://www.coursera.org/learn/machine-learning/discussions/weeks/3/threads/j2Vn07HqEeaYcRJ-aKpq1A) - tutorial for logistic regression prediction

Discussion of plotDecisionBoundary() [link](https://www.coursera.org/learn/machine-learning/module/mgpv7/discussions/HAEss7C7EeWoGg6ulZMPEw)

Enhancements to plotDecisionBoundary() - not required, just handy - [link](https://www.coursera.org/learn/machine-learning/discussions/weeks/3/threads/4XnTNev9EeebPwrlkcpjjg)

-------------

ex3

Note: a change to displayData.m for MacOS users: ([link](https://www.coursera.org/learn/machine-learning/discussions/YlOmkiWsEeWeUyIAC44Ejw/replies/0A7DZi_BEeWOkCIAC4UG7w))

Note: if your images are upside-down, use flipud() to reverse the data. This is due to a change in gnuplot()'s defaults.

<p style="text-decoration: underline;">Tips on lrCostFunction():<p>

+ When completed, this function is identical to your costFunctionReg() from ex2, but using vectorized methods. See the ex2 tutorials for the cost and gradient - they use vectorized methods.
+ ex3.pdf tells you to first implement the unregularized parts, then to implement the regularized parts. Then you test your code, and then submit it for grading.
+ Do not remove the line "grad = grad(:)" from the end of the lrCostFunction.m script template. This line guarantees that the grad value is returned as a column vector.

[oneVsAll()](https://www.coursera.org/learn/machine-learning/discussions/weeks/4/threads/sLIsSJU1EeW70BJZtLVfGQ) tutorial

[predictOneVsAll()](https://www.coursera.org/learn/machine-learning/module/mZYiz/discussions/Hfo82qxTEeWjcBKYJq1ZMQ) tutorial (updated)

[predict()](https://www.coursera.org/learn/machine-learning/module/mZYiz/discussions/miam5q2IEeWhLRIkesxXNw) tutorial (for the NN forward propagation - updated)

-------------

#### ex4

[nnCostFunction()](https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/QFnrpQckEeWv5yIAC00Eog) - forward propagation and cost w/ regularization

[nnCostFunction()](https://www.coursera.org/learn/machine-learning/discussions/a8Kce_WxEeS16yIACyoj1Q) - tutorial for backpropagation

[Tutorial on using matrix multiplication to compute the cost value 'J'](https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/AzIrrO7wEeaV3gonaJwAFA)

-------------

#### ex5

[linearRegCostFunction()](https://www.coursera.org/learn/machine-learning/discussions/UAv1DB62EeWd3iIAC7VAtA) tutorial

[polyFeatures()](https://www.coursera.org/learn/machine-learning/discussions/weeks/6/threads/YbO2RaVGEeaCbg44JUM1Vg) - tutorial

[learningCurve()](https://www.coursera.org/learn/machine-learning/module/xAUWb/discussions/Y_DZmpkgEeWNbBIwwhtGwQ) tutorial (really just a set of tips)

[validationCurve()](https://www.coursera.org/learn/machine-learning/discussions/AdGhzAX1EeWyEyIAC7PmUA/replies/7XjBAQ-MEeWUtiIAC9TNkg) tips

-------------

#### ex6

_Note: Possible error in svmPredict.m:_ See the FAQ thread in the "Discussion Forum - Week 7" area for details.

All ex6 tutorials ([link](https://www.coursera.org/learn/machine-learning/discussions/g2VB7po6EeWKNwpBrKr_Fw))

-------------

### ex7

[findClosestCentroids()](https://www.coursera.org/learn/machine-learning/module/kxH2P/discussions/ncYc-ddQEeWaURKFEvfOjQ) tutorial

[computeCentroids()](https://www.coursera.org/learn/machine-learning/discussions/weeks/8/threads/WzfDM7LjEeatew7zqUaXxg) tutorial

[Tutorials for ex7_pca functions](https://www.coursera.org/learn/machine-learning/programming/ZZkM2/k-means-clustering-and-pca/discussions/wp_NfU55EeWxHxIGetKceQ) - pca(), projectData(), recoverData()

-------------

### ex8

selectThreshold() - use the tips in the function script template, and the bulleted list on page 6 of ex8.pdf, to compute each of the tp, fp, and fn values. Sample code for "fp" is given in the text box on the bottom of ex8.pdf - page 6.

Note: error in ex8_cofi.m (click this [link](https://www.coursera.org/learn/machine-learning/discussions/YD0v9TL_EeWj5iIACwIAYw))

Tip for estimateGaussian(): Compute the mean using "mean()". You can compute sigma2 using the equation in ex8.pdf, or you can use "var()" if you set the OPT parameter so it normalizes over the entire sample size.

[cofiCostFunc()](https://www.coursera.org/learn/machine-learning/module/HjnB4/discussions/92NKXCLBEeWM2iIAC0KUpw) tutorial

-------------



