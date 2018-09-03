# Section 8: Classification (Lec 8.1 - Lec 8.3)

+ [Launching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/03a357f8203f4dfa8aa471e06b75affe/517a860f7bb44b76a020bddf902f0521/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%40c79297d109df494db223fbf78fec7225)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec8.ipynb)
+ [Local Notebook](./notebook/lec9.ipynb)
+ [Local Python Code](./notebooks/lec8.py)

## Lec 8.1 Introduction

### Note

+ Classification: a particular kind of machine learning technique
+ Classifier: 
    + a machine learning algorithm used to answer yes or no question
    + making prediction based on historical data given some data set
    + E.g, credit risk in bank
+ Classifier Examples:
    + online stores to decide fraudulent or legitimate
        + amount of the transaction
        + what are buying, e.g., diamonds
        + where to ship
    + Applications
        + Medical: cancerous or not - pre-judge, training
        + Online dating site: important questions, such as religion

+ Machine: learning
    + Teach computers by examples
    + Large data sets -> predict the future instances
    + No fix rules to make decisions
    + Concerns on fairness and discrimination
    + Trained algorithms come out of human system with bias

### Video

<a href="https://edx-video.net/BERD83FD2018-V002800_DTH.mp4" alt="Lec 8.1 Introduction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 8.2 Nearest Neighbor

### Note


+ Demo
    ```python
    ckd = Table.read_table('ckd.csv').relabeled('Blood Glucose Random', 'Glucose')
    # labels: Age    Blood Pressure    Specific Gravity    Albumin    Sugar    Red Blood Cells    Pus Cell    Pus Cell clumps    Bacteria    Glucose    Blood Urea    Serum Creatinine    Sodium    Potassium    Hemoglobin    Packed Cell Volume    White Blood Cell Count    Red Blood Cell Count    Hypertension    Diabetes Mellitus    Coronary Artery Disease    Appetite    Pedal Edema    Anemia    Class
    # Class 1: patient with chronic kidney disease

    ckd.group('Class')
    # Class   count
    # 0       115
    # 1       43

    ckd.scatter('White Blood Cell Count', 'Glucose', colors='Class')
    # decide which attributes to make decision -> a good reference

    ckd.scatter('Hemoglobin', 'Glucose', colors='Class')    # a better one
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V002900_DTH.mp4" alt="Lec 8.2 Nearest Neighbor" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 8.3 Examples

### Note


+ Demo
    ```python
    banknotes = Table.read_table('banknote.csv')
    # class: counterfeit (1) or legitimate (0)
    # WaveletVar   WaveletSkew    WaveletCurt    Entropy    Class
    # 3.6216       8.6661         -2.8073        -0.44699    0
    # 4.5459       8.1674         -2.4586        -1.4621     0
    # 3.866       -2.6383          1.9242         0.10645    0
    # ... (rows omitted)

    banknotes.scatter('WaveletVar', 'WaveletCurt', colors='Class')

    banknotes.scatter('WaveletSkew', 'Entropy', colors='Class')

    fig = plots.figure(figsize=(8,8))
    ax = Axes3D(fig)
    ax.scatter(banknotes.column('WaveletSkew'), 
            banknotes.column('WaveletVar'), 
            banknotes.column('WaveletCurt'), 
            c=banknotes.column('Class'),
            cmap='viridis',
            s=50);
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V003000_DTH.mp4" alt="Lec 8.3 Examples" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading

This guide assumes that you have watched the videos for Section 8.

This corresponds to textbook section:

[Chapter 17: Classification](https://www.inferentialthinking.com/chapters/17/Classification)

In section 8, we learned about a new kind of machine learning: classification. Classification is about learning how to identify the class to which an individual belongs, based on past examples. We start with examples in which we have been told what the correct class was, and we want to learn from those examples how to make good predictions in the future.

You will learn how to build your own machine learning classifiers, but first, let's make sure you understand how classifiers work.



### Practice

Given this dataset, the goal is to classify the new point (5, 7) using k-nearest neighbor classifiers.

pointsBlue

| X  |  Y |
|----|----|
| 8  |  2 |
| 7  |  2 |
| 6  |  5 |
| 7  |  2 |
| 5  |  6 |
| 9  |  4 |
... (14 rows omitted)


pointsYellow

| X  |  Y |
|----|----|
| 6  |  5 |
| 4  |  3 |
| 1  |  6 |
| 1  |  6 |
| 3  |  4 |
| 5  |  5 |
... (14 rows omitted)


Here's a scatter plot of the data. Let's try to classify point (5, 7).
<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/03a357f8203f4dfa8aa471e06b75affe/4347e12ff0de4aa7831d7342e754833b/?child=first">
    <br/><img src="https://prod-edxapp.edx-cdn.org/assets/courseware/v1/80ef7ab4aa3719512719eba500a17dc5/asset-v1:BerkeleyX+Data8.3x+2T2018+type@asset+block/colors.png" alt="Scatter plot of data from pointsBlue and pointsYellow table. Point (5, 7) is surrounded by the following 7 points (from closest to smallest): blue, blue, yellow, blue, yellow, blue, yellow." title= "Scatter plot" width="350">
</a>

Q1. What is the color of point (5, 7) if you use a 3-nearest neighbors classifer?

a. Blue
b. Yellow

Ans: a


Q2. What is the color of point (5, 7) if you use a 5-nearest neighbors classifer?

a. Blue
b. Yellow

Ans: a


Q3. What is the color of point (5, 7) if you use a 7-nearest neighbors classifer?

a. Blue
b. Yellow

Ans: a



