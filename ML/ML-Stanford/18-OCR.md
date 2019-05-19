# Application Example: Photo OCR

## Photo OCR

### Problem Description and Pipeline

#### Lecture Notes

+ The Photo OCR (Optical Character Recognition) problem
  1. Given picture, detect location of text in the picture
  2. Read text at the location

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.coursera.org/learn/machine-learning/lecture/iDBMm/problem-description-and-pipeline">
      <img src="images/m18-01.png" style="margin: 0.1em;" alt="Text detection and recognition" title="Text detection and recognition" width="400">
      <img src="images/m18-02.png" style="margin: 0.1em;" alt="Text OCR" title="Text OCR" width="250">
    </a></div>
  </div>

+ Photo OCR Pipeline
  1. Text detection
  2. Character segmentation: Splitting “ADD” for example
  3. Character classification: First character “A”, second “D”, and so on

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.ritchieng.com/machine-learning-photo-ocr/#problem-description-and-pipeline">
      <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w11_application_example_ocr/photoocr.png" style="margin: 0.1em;" alt="Text OCR pipeline" title="Text OCR pipeline" width="400">
    </a></div>
  </div>

  + IVQ: When someone refers to a “machine learning pipeline,” he or she is referring to:

    1. A PhotoOCR system.
    2. A character recognition system.
    3. A system with many stages / components, several of which may use machine learning.
    4. An application in plumbing. (Haha.)

    Ans: 3

+ When you design a machine learning algorithm, one of the most important steps is defining the pipeline
  + A sequence of steps or components for the algorithms
  + Each step/module can be worked on by different groups to split the workload


#### Lecture Video

<video src="https://d18ky98rnyall9.cloudfront.net/19.1-ApplicationExamplePhotoOCR-ProblemDescriptionAndPipeline.465d8770b22b11e4bb7e93e7536260ed/full/360p/index.mp4?Expires=1558310400&Signature=L6tHa85rIFuUOUljU8jM19U8WuRlJbhPwmNvHWWLQMxsrozIRA22aIgC2KwFH-zrJs6BBGrxYuRxNOgm0aCwYHQR4OnZCl9kJw0XCv~uynF3WZODwjMVCcxPvne2mbew63vZlKfhynMvR4bkkFYrtIH89WFN707qPCi4z4Wp4O4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/2bdOXjhCSW23Tl44QvltBQ?expiry=1558310400000&hmac=os66HSqQ1uxSorn5N4f3sVjiaquoslSf2S44AOO9nGY&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Sliding Windows

#### Lecture Notes



#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Getting Lots of Data and Artificial Data

#### Lecture Notes



#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Ceiling Analysis: What Part of the Pipeline to Work on Next

#### Lecture Notes



#### Lecture Video


<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Review

### Lecture Slides

These are the [lecture slides](https://d18ky98rnyall9.cloudfront.net/_cff4fea7eaf5ad373734488ae70dc3dd_Lecture18.pdf?Expires=1558310400&Signature=guhnF5heyev0~-d61PqcpGW-~w0MLG3hiGPT3QCcCwZigqQyoqshqITbD16T79m0253jn9VZBQ1ZJeIfZ1eyggquUnm0E9LhXFjA-5Ke~d-GFEHwyqA1Mlzb9w4QcqBphaalSKY7SZL1gb69o9f2irL-sATgpNVx0SwlTKqHbws_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A) from this unit. It would be helpful to review them prior to taking the quiz.


### Quiz: Application: Photo OCR





