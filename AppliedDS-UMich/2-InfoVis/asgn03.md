# Practice Assignment

## Practice Assignment: Understanding Distributions Through Sampling

+ [Launch Web Page](https://www.coursera.org/learn/python-plotting/notebook/W3OXw/practice-assignment-understanding-distributions-through-sampling)
+ [Web Notebook](https://hub.coursera-notebooks.org/user/pkfpwscjcemdtitwkaxuvv/notebooks/UnderstandingDistributionsThroughSampling.ipynb)
+ [Local Notebook](./notebooks/UnderstandingDistributionsThroughSampling.ipynb)

## Practice Peer-graded Assignment: Practice Assignment: Understanding Distributions Through Sampling

### Instruction

This assignment is optional, and I encourage you to share your solutions with me and your peers in the discussion forums!

To complete this assignment, follow the instructions in the Jupyter notebook.

### Practice Assignment: Understanding Distributions Through Sampling

This assignment is optional, and I encourage you to share your solutions with me and your peers in the discussion forums!

To complete this assignment, create a code cell that:

+ Creates a number of subplots using the pyplot subplots or matplotlib gridspec functionality.
+ Creates an animation, pulling between 100 and 1000 samples from each of the random variables (x1, x2, x3, x4) for each plot and plotting this as we did in the lecture on animation.
+ Bonus: Go above and beyond and "wow" your classmates (and me!) by looking into matplotlib widgets and adding a widget which allows for parameterization of the distributions behind the sampling animations.

Tips:

+ Before you start, think about the different ways you can create this visualization to be as interesting and effective as possible.
+ Take a look at the histograms below to get an idea of what the random variables look like, as well as their positioning with respect to one another. This is just a guide, so be creative in how you lay things out!
+ Try to keep the length of your animation reasonable (roughly between 10 and 30 seconds).

```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib notebook

# generate 4 random variables from the random, gamma, exponential, and uniform distributions
x1 = np.random.normal(-2.5, 1, 10000)
x2 = np.random.gamma(2, 1.5, 10000)
x3 = np.random.exponential(2, 10000)+7
x4 = np.random.uniform(14,20, 10000)

# plot the histograms
plt.figure(figsize=(9,3))
plt.hist(x1, normed=True, bins=20, alpha=0.5)
plt.hist(x2, normed=True, bins=20, alpha=0.5)
plt.hist(x3, normed=True, bins=20, alpha=0.5)
plt.hist(x4, normed=True, bins=20, alpha=0.5);
plt.axis([-7,21,0,0.6])

plt.text(x1.mean()-1.5, 0.5, 'x1\nNormal')
plt.text(x2.mean()-1.5, 0.5, 'x2\nGamma')
plt.text(x3.mean()-1.5, 0.5, 'x3\nExponential')
plt.text(x4.mean()-1.5, 0.5, 'x4\nUniform')
```

## Assignment 3

## Building a Custom Visualization

+ [Web Launch Page](https://www.coursera.org/learn/python-plotting/notebook/OMUVV/building-a-custom-visualization)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=x_a6Zl0sSvS2umZdLBr0bA&next=%2Fnotebooks%2FAssignment3.ipynb)
+ [Local Notebook](./notebooks/Assignment04.ipynb)

## ReadingAssignment Reading

Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). [Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions](https://drive.google.com/file/d/0B7Tj31nhk4BAeFJ1Y1lwQmpMQVk/view. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM

The authors have provided a related video which may be of further value in explaining the research.

## Assignment 3 - Building a Custom Visualization

In this assignment you must choose one of the options presented below and submit a visual as well as your source code for peer grading. The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work. The options differ in challenge level, but there are no grades associated with the challenge level you chose. However, your peers will be asked to ensure you at least met a minimum quality for a given technique in order to pass. Implement the technique fully (or exceed it!) and you should be able to earn full grades for the assignment.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). [Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM. ([video](https://www.youtube.com/watch?v=BI7GAs-va-Q))


In this [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) the authors describe the challenges users face when trying to make judgements about probabilistic data generated through samples. As an example, they look at a bar chart of four years of data (replicated below in Figure 1). Each year has a y-axis value, which is derived from a sample of a larger dataset. For instance, the first value might be the number votes in a given district or riding for 1992, with the average being around 33,000. On top of this is plotted the 95% confidence interval for the mean (see the boxplot lectures for more information, and the yerr parameter of barcharts).

<br>
<img src="readonly/Assignment3Fig1.png" alt="Figure 1" style="width: 400px;"/>
<h4 style="text-align: center;" markdown="1">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 1 from (Ferreira et al, 2014).</h4>

<br>

A challenge that users face is that, for a given y-axis value (e.g. 42,000), it is difficult to know which x-axis values are most likely to be representative, because the confidence levels overlap and their distributions are different (the lengths of the confidence interval bars are unequal). One of the solutions the authors propose for this problem (Figure 2c) is to allow users to indicate the y-axis value of interest (e.g. 42,000) and then draw a horizontal line and color bars based on this value. So bars might be colored red if they are definitely above this value (given the confidence interval), blue if they are definitely below this value, or white if they contain this value.


<br>
<img src="readonly/Assignment3Fig2c.png" alt="Figure 1" style="width: 400px;"/>
<h4 style="text-align: center;" markdown="1">  Figure 2c from (Ferreira et al. 2014). Note that the colorbar legend at the bottom as well as the arrows are not required in the assignment descriptions below.</h4>

<br>
<br>

**Easiest option:** Implement the bar coloring as described above - a color scale with only three colors, (e.g. blue, white, and red). Assume the user provides the y axis value of interest as a parameter or variable.


**Harder option:** Implement the bar coloring as described in the paper, where the color of the bar is actually based on the amount of data covered (e.g. a gradient ranging from dark blue for the distribution being certainly below this y-axis, to white if the value is certainly contained, to dark red if the value is certainly not contained as the distribution is above the axis).

**Even Harder option:** Add interactivity to the above, which allows the user to click on the y axis to set the value of interest. The bar colors should change with respect to what value the user has selected.

**Hardest option:** Allow the user to interactively set a range of y values they are interested in, and recolor based on this (e.g. a y-axis band, see the paper for more details).

---

*Note: The data given for this assignment is not the same as the data used in the article and as a result the visualizations may look a little different.*

```python
# Use the following data for this assignment:

import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df
```

## Peer-graded Assignment: Building a Custom Visualization

### Instruction

In this assignment you must implement a visualization of sample data as described in Ferreria et al (2014). The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work.

The options differ in challenge level, but there are no grades associated with the challenge level you chose.

Download the attachment for a preview of how the assignment will be graded.

[assignment3_rubric.pdf](https://d3c33hcgiwev3.cloudfront.net/_b5171d970a52dc86e7b1da5ea8bf55ea_assignment3_rubric.pdf?Expires=1529280000&Signature=SkWFoLC0H031cfPQMQQwDxMhd-Lq5~6O2PCFwsAYOeif2pzOgd1KfbXSWMN0wQ~XV4LPvzpiPOLv75A1GaEFPYaMlMUYAjEUCfP6cEDJmN4yKVG8oe9VacFHfARYTIvMfKpKipi7GOt5QnZyHyxsaoFfB723n7fmGpI5CK6dv9k_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Review Your Peers: Building a Custom Visualization








