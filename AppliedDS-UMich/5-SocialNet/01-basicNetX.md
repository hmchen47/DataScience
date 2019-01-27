# Module 1: Why Study Networks and Basics on NetworkX

## Learning Objectives

+ Recognize and categorize real world networks.
+ Identify applications and important questions about networks that network science allows us to answer.
+ Determine what type of network is best suited to model real networked data.
+ Construct and manipulate networks of different types using different network classes and node and edge attributes in NetworkX.
+ Define bipartite graphs and describe related algorithms such as graph projections.
+ Manipulate bipartite graphs and related algorithms using NetworkX.



## Course Syllabus

### Prerequisites

In order to be successful in this course, you will need to know how to program in Python. The expectation is that you have completed [Introduction to Data Science in Python](https://www.coursera.org/learn/python-data-analysis), [Applied Plotting](https://www.coursera.org/learn/python-plotting), [Charting & Data Representation in Python](https://www.coursera.org/learn/python-plotting), and [Applied Machine Learning in Python](https://www.coursera.org/learn/python-machine-learning) so that you are familiar with the numpy and pandas Python libraries for data manipulation, matplotlib for plotting, and scikit-learn for machine learning.

### Week by week

__Module One__ introduces you to different types of networks in the real world and why we study them. You will cover the basic elements of networks such as nodes, edges, and attributes and different types of networks such as directed, undirected, weighted, signed, and bipartite. You will also learn how to represent and manipulate networked data using the NetworkX library. The assignment will give you an opportunity to use NetworkX to analyze a networked dataset of employees in a small company, their relationships, and preferences of movies to watch for an upcoming movie night.

In __Module Two__ you will learn about how to analyze the connectivity of a network based on measures of distance, reachability, and redundancy of paths between nodes. This type of analysis will allow you explore the robustness of a network when it is exposed to random or targeted attacks such as the removal of nodes and edges. In the assignment, you will practice using NetworkX to compute measures of connectivity of a network of email communication among the employees of a mid-size manufacturing company.

In __Module Three__ you will explore ways of measuring the importance or centrality of a node in a network. You will cover several different centrality measures including Degree, Closeness, and Betweenness centrality, Page Rank, and Hubs and Authorities. You will learn about the assumptions each measure makes, the algorithms we can use to compute them, and the different functions available on NetworkX to measure centrality. You will also compare the ranking of nodes by centrality produced by the different measures. In the assignment, you will practice choosing the most appropriate centrality measure on a real-world setting, where you are tasked with choosing a person from a social network who should be given a promotional voucher in order to maximize the impact of the promotion on the network.

In __Module Four__ you will explore the evolution of networks over time. You will learn about different models that generate networks with realistic features such as the Preferential Attachment Model and Small World Networks. You will also explore the link prediction problem, where you will learn useful features that can predict whether a pair of disconnected nodes will be connected in the future. In the assignment, you will be challenged to identify which model generated a given network. Additionally, you will have the opportunity to combine different concepts of the course by predicting the salary, position, and future connections of the employees of a company using their logs of email exchanges.

## Enrollment Options

Coursera has made the decision to make Specializations available by monthly subscription. This means you can choose to pay a monthly fee to access all of the courses in a specific Specialization. Coursera’s switch to monthly subscriptions comes with another change -- for those learners who choose the “Audit Only” enrollment, you will no longer be able to submit assignments for grades nor see answers for those assignments. You will still have access to all the course materials but you will not be graded on your work, nor see answers to graded assignments. For further information on the different enrollment options for Coursera courses, please visit the [Enrollment Options ](https://learner.coursera.help/hc/en-us/articles/209818613-Enrollment-options) page. If you have feedback about the enrollment options shared on the Enrollment Options page, you can share your thoughts with Coursera in this [survey](https://www.surveymonkey.com/r/65DPLHG).

## Grading and Assignments

The lectures will provide you with some guidance for completing assignments, but you will need to take initiative and look beyond assignment instructions in order to be successful. You'll need to know how to ask questions in the discussion forums of your peers, and seek out new information through web searches and [Stack Overflow](http://stackoverflow.com/questions/tagged/matplotlib). Be sure to also check out the [Additional Resources](https://www.coursera.org/learn/python-social-network-analysis/resources/iZUox). If you are not sure what kind of output is required, or think there is a need for more clarity, please head to the course discussion forums. Note that some assignments and in video quizzes may not be mobile friendly.

| Course Item | Percentage of Final Grade | Passing Grade |
|:------------|:-------------------------:|:-------------:|
| Week 1 Quiz | 5% | 80% |
| Week 1 Notebook Assignment | 18% | 80% |
| Week 2 Quiz | 5% | 80% |
| Week 2 Notebook Assignment | 18% | 80% |
| Week 3 Quiz | 5% | 80% |
| Week 3 Notebook Assignment | 18% | 80% |
| Week 4 Quiz | 5% | 80% |
| Week 4 Notebook Assignment | 26% | 80% |


### Code of Conduct

Visit [Coursera’s Code of Conduct](https://learner.coursera.help/hc/en-us/articles/208280036-Coursera-Code-of-Conduct) and to abide by guidelines there. It is important when giving feedback to your peers to be polite and to be sensitive to the diversity of cultures and backgrounds of learners in your course.

## Working Offline

While the Coursera platform has an integrated Jupyter Notebook system, you can work offline on your own computer by installing Python 3.5+ and the Jupyter software packages. For more details, consult the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs).

### Accessibility

We strive to develop fully accessible courses. Occasionally, some of our content does not fully meet our accessibility goals. Please use [this form](https://goo.gl/forms/XqKzVUMTn62yrarU2) to inform us of any accessibility issues you are experiencing in this course.

### Help!

If you're having problems, here are a couple of great places to go for help: If the problem is with the Coursera platform such as verification on assignments, in video quiz problems, or the Jupyter Notebooks, please check out the [Coursera Learner Support Forums](https://learner.coursera.help/hc/en-us/requests). If the problem deals with understanding the assignment or how to use the Jupyter Notebooks, please read our Jupyter Notebook FAQ page in the course resources If you have questions with the content of the course, or questions about programming in python or with the toolkits described, you can contact your peers and the course instructors in the discussion forums, or go to [Stack Overflow](http://stackoverflow.com/questions/tagged/matplotlib). Having trouble accessing your previously submitted assignments? If your session has ended, you can access these again by selecting the "Switch Session" option. Details for how to select this can be found in this [learner help center article](https://learner.coursera.help/hc/en-us/articles/208279776-Switch-to-a-different-session). If you still have issues accessing your materials after switching sessions, please reach out to Coursera learner support via our online chat forums in the [Learner Help Center](https://learner.coursera.help/hc/en-us).

### In-Video Questions (IVQs)

In this course, in-video questions or IVQs may appear during lectures to help you learn as well as assess your understanding of the content. IVQs are optional and do not count towards your overall course grade.

#### Types of in-video questions

Many of the lectures contain in-video questions (IVQs). These questions are presented in a variety of formats. Some will ask you to write or think about a concept from the video. Others will ask for a short answer. Still others may ask you to choose from a multiple-choice list of answers. If an IVQ is a survey or a poll, you will see a summary of responses from other learners after you respond. You can look at the question again later to see new summary data as more of your peers answer. Some IVQs also contain runnable code blocks. These IVQs allow you to practice the coding concepts during the lecture. In this course, these types of IVQs will usually be directly followed with the solution code.


## Help us learn more about you!

As part of getting to know you better, your backgrounds, your interest in this specific course and in digital education in general, we at the University of Michigan have crafted a survey that should only take a few minutes to complete. Our goal is to keep our communication with you focused on learning and staying in touch, but we believe that both this and an end-of-course survey are important to our mutual educational goals.

[Take the survey](https://umich.qualtrics.com/SE/?SID=SV_735AxMay2FSApSZ&redirect=SV_9ukuaMa1VKYV9C5&phoenix_global_user_id=%GLOBAL_USER_ID:2013-may-demographics%&phoenix_session_user_id=4f54f67ce5254893536041565df2887d6826c536&name=H.-M.%20Fred%20Chen&platform_id=coursera_phoenix&course_id=python-social-network-analysis)

The link will open in this same window for the purposes of making it accessible to screen readers and other assistive devices. You may need to manually navigate back to the course afterwards. Thank you for participating!

## Additional Resources

+ Dr Chuck Severance's Coursera Specialization, [Python for Everybody](https://www.coursera.org/specializations/python)
+ [Python Docs](https://docs.python.org/3/) (for general Python documentation)
+ [Python Classes Docs](https://docs.python.org/3.5/tutorial/classes.html)
+ [Scipy](http://scipy.org/) (for [IPython](http://ipython.org/), [Numpy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), and [Matplotlib](http://matplotlib.org/))
+ [scitkit-learn Docs](http://scikit-learn.org/stable/documentation.html)
+ [scikit-learn Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
+ [NetworkX Documentation](https://networkx.readthedocs.io/en/stable/)
+ Don't forget to check [Stack Overflow](https://stackoverflow.com/questions) and [Cross Validated](https://stats.stackexchange.com/)!


## Networks: Definition and Why We Study Them

### Lecture Note

+ Networks
    + Networks: A set of objects (nodes) with interconnections (edges).
    + Why study networks? <br/>
        <b style="color:red">Because they are everywhere!</b>

+ Social Networks
    + Friendship network in a 34-person karate club [Zachary 1977]
    + E-mail communication network among 436 HP employees [Adamic & Adar 2005]
    + Network of friendship, marital tie, and family tie among 2200 people [Christakis & Fowler 2007]
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/CsyaIMmiYfjAk89JPShRf75E4B-NzPHpwhs5MagkoO8PYEj6vJrkNW1QrcomU8MCpk1JarWNpHkWlQcFNpDE7O4wTySzAHWDavfvmhJ5jg8pmFoS89tePBlU2f0TiNVnjEUkuCVW1Q=w2400" alt="Social Networks" title="Social Networks " height="300">
    </a>


+ Transportation and Mobility Networks
    + Network of direct flights around the world [Bio.Diaspora]
    + Human mobility network based on location of dollar bills (Where’s George) [Thiemann et al. 2010]
    + Ann Arbor bus transportation network
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/bc0o7l-5Bgi9H_D6DZaQvuVimxQHHR5aM4cCLkwuKyAMrd_OyhyP9d4Sb0x32oNb4vWUYgQO6_k9HNK3IUgsQKP92coBxiAyJLUdl0O4DRZMx9VmgFXa4LK9BT1dkjHjz8SMwR1jOA=w2400" alt="Transportation and Mobility Networks " title="Transportation and Mobility Networks " height="300">
    </a>


+ Information Networks
    + Communication between left-wing and rightwing political blogs [Adamic & Glance 2005]
    + Internet Connectivity [K. C. Claffy]
    + Network of Wikipedia articles about climate change [EMAPS]
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/txcCQMUFlestauM2AvT79S-4gqOP8HDRUrKWwteQgoZ7WgLWl8BcK1-VOz2W14Ry8Y8MaO-JUEuLs6afL8wTWxtFbfOpN_ghMFMCeaWlsi1suf3lwZbSrmt8h4d3_NsMpDkusjpnVA=w2400" alt="Information Networks" title="Information Networks" height="300">
    </a>

+ Biological Networks
    + Protein-protein interactions [Jeong et al. 2001]
    + Chesapeake Bay Waterbird Food Web [Perry et al. 2005]
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/8PFVbsmguLKaR2f0rQpL2T2arcuUS1bpCE_k3Q_wTRL40GyDekCp8ug6EV9paFYDHMGZzWuuVfiJ6vGQzQhp2vc4hdqFXcNog8g-g3loFofkVfewOnDZ2Gx7rhyCPpkRzpEwpuUTkw=w2400" alt="Biological Networks" title="Biological Networks" height="300">
    </a>
    
+ And More…
    + Financial networks
    + Co-authorship networks
    + Trade networks
    + Citation networks


+ Networks Applications
    + <b style="color:darkred">Networks are everywhere, but what can we do with them?</b>
    + E-mail communication network among 436 HP employees
        <a href="https://courses.cit.cornell.edu/info2040_2010fa/"> <br/>
            <img src="https://courses.cit.cornell.edu/info2040_2010fa/adamic-hier.jpg" alt="E-mail communication network among 436 HP employees" title="E-mail communication network among 436 HP employees" height="200">
        </a>
        + Is a rumor likely to spread in this network?
        + Who are the most influential people in this organization?
    + Friendship network in a 34-person karate club
        <a href="https://anthonybonato.com/2016/04/13/the-mathematics-of-game-of-thrones/"> <br/>
            <img src="https://lh3.googleusercontent.com/OQqUIVdAO_KrEiIsfGN4mARt24rHxQzWZ9IndHfY3DEvgvYp-m7PW4BzaaKpb9Trp2w8UKvvkuW3tSN6O7pJ7L7vm9P_pBX-eLOf03QKFd9y2jVQ" alt="Zachary’s Karate club graph, from Wayne Zachary’s PhD thesis in 1972. The nodes correspond to members of an actual karate club, and the edges represent their social ties. The instructors of the club are the bold nodes 1 and 34. After a dispute between them, the instructors each formed their own club. We can see here the community formed around each of the nodes 1 and 34." title="Zachary Karate Club graph" height="200">
        </a>
        + Is this club, likely to split into two groups?
        + If so, which nodes will go to which group?
    + Network of direct flights around the world
        <a href="http://www.visualisingdata.com/2012/02/bio-diaspora-visualising-interactions-between-populations-and-travel/"> <br/>
            <img src="http://www.visualisingdata.com/blog/wp-content/uploads/2012/02/World_FlightLines_BioDiaspora-600x393.jpg" alt="I got in touch with David Kossowsky, a GIS mapper, cartographer and graphic designer, to find out more about the work of Bio.Diaspora and some of the visualisations they have been working on. This image shows a visualisation of the global airline transportation network consisting of all commercial flights worldwide. While this image in its entirety does not necessarily provide information that can be used to assess an infectious disease threat, it does provide one with a greater understanding of how interconnected the world is due to air transportation, and how easy it is for some diseases to spread across very large areas in relatively short periods of time." title="global airline transportation network " height="200">
        </a>

+ Summary
    + Many complex structures can be modeled by networks.
    + Studying the structure of a network can allows us to answer questions about complex phenomena.
    + In this course, we will explore different network techniques to study the structure of social networks.


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/Vxfw6nw_Eeeybwpoukrg-A.processed/full/360p/index.mp4?Expires=1548720000&Signature=XywFBC6yEKMYXbVv-ZOPC7wENLPvbfs9HhLvHq~K5HBABn~Dg02rqNM988XsA8GAOQD6vLnY4feGCdx84dYvK5cnS0bslaNCuBxWkHZv6DrdcyqQjVE0dl5D4PlRSc07VSOwcMSlrAa6CLYFanIN-QX3eEwuRA6Jd82uAF~frIA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Networks: Definition and Why We Study Them" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Network Definition and Vocabulary

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Node and Edge Attributes

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Bipartite Graphs

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Notice for Auditing Learners: Assignment Submission

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Notebook: Loading Graphs in NetworkX

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## TA Demonstration: Loading Graphs in NetworkX

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>




