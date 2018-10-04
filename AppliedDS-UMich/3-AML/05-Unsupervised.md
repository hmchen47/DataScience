# Unsupervised Machine Learning

## Unsupervised Learning Notebook

+ [Web notebook Launching Page](https://www.coursera.org/learn/python-machine-learning/notebook/KDjxZ/unsupervised-learning-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/user/elkljxyoytcwjbmkgctrtg/notebooks/UnsupervisedLearning.ipynb)
+ [Local notebook](notebooks/UnsupervisedLearning.ipynb)
+ [Local python code](notebooks/UnsupervisedLearning.py)

## Introduction

+ Introduction: Unsupervised Learning
    + Unsupervised learning involves tasks that operate on datasets __without__ labeled responses or target values.
    + Instead, the goal is to capture interesting structure or information.
    + Applications of unsupervised learning:
        + Visualize structure of a complex dataset.
        + Density estimation to predict probabilities of events.
        + Compress and summarize the data.
        + Extract features for supervised learning.
        + Discover important clusters or outliers.

+ Web Clustering Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-01.png" alt="Here's an example of an unsupervised method called clustering. Suppose you're in charge of running a website that allows people to buy products from your company. And the site gets thousands of visits per day. As people access the site by clicking links to products or typing in search terms, their interactions are logged by the web server that creates a large log file. It might be useful for your business to understand who's using your site by grouping users according to their shopping behavior. For example, there might be a group of more expert users who use more advanced features to find something very specific. While another group of non-expert users might just enjoy browsing a broader set of items. By clustering users into groups, you might gain some insight into who your typical customers are and what site features different types of users find important. You could use insights from clustering users to improve the site's features for different groups or to recommend products to specific groups that would be more likely to buy them. " title= "Web Clustering Example" height="150">
    </a>

+ Two major types of unsupervised learning methods
    + Transformations
        + Processes that extract or compute information
    + Clustering
        + Find groups in the data
        + Assign every point in the dataset to one of the groups


+ Transformations: Density Estimation
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-02.png" alt="Okay, let's look at some important unsupervised learning methods that transform the input data in useful ways. One method called density estimation is used when you have a set of measurements scattered throughout an area. And you want to create what you can think of as a smooth version over the whole area that gives a general estimate for how likely it would be to observe a particular measurement in some area of that space. " title= "Transformations: Density Estimation" height="150">
    </a>

+ Density Estimation Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-03.png" alt="For example, in a medical application related to diagnosing diabetes, density estimation with one variable might be used to estimate the distribution of a specific test score. The plasma glucose concentration number from a blood test for people who have a particular form of diabetes. With this density estimate we can estimate the probability that anyone with that medical condition has a particular glucose score. Even if that specific score wasn't seen in the original dataset. " title= "Density Estimation Example: diabetic" height="150">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/fig5-04.png" alt="We could then compare this to the range of glucose levels for people who do not have that condition, which is shown by the red line here. Often, density estimates are then used in further machine learning stages as part of providing features for classification or regression. The more technical way to say this is that density estimation calculates a continuous probability density over the feature space, given a set of discrete samples in that feature space.  " title= "Density Estimation Example: non-diabetic" height="150">
    </a>

+ Kernel Density Example
    <a href="http://www.digital-geography.com/csv-heatmap-leaflet/"> <br/>
        <img src="images/fig5-05.png" alt="With this density estimate, we can estimate how likely any given combination of features is to occur. In Scikit-Learn, you can use the kernel density class in the sklearn.neighbors module to perform one widely used form of density estimation called kernel density estimation. Kernel density's especially popular for use in creating heat maps with geospatial data like this one. " title= "Kernel Density Example" height="150">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/8O0xDUFoEee3MRIl4lCYSA.processed/full/360p/index.mp4?Expires=1538784000&Signature=IouwnCBJvtm7tds2irYlEwq~KkFXqQ8RyAiZiwhB~UPJu9F~Cu75IxYtdIV3QiIwUETdpBVLa8bkEDU6AjkKb~5hVHjDunNzTzB7yBVporyuQTt2uiBLXLrEiO2V658rb1Qm0-HocicG9AXIFXl9EcvYU6ucv-3sevC5uWoTcKw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Introduction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Dimensionality Reduction and Manifold Learning

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Dimensionality Reduction and Manifold Learning" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Clustering

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Clustering" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## How to Use t-SNE Effectively

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="How to Use t-SNE Effectively" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## How Machines Make Sense of Big Data: an Introduction to Clustering Algorithms

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="How Machines Make Sense of Big Data: an Introduction to Clustering Algorithms" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

