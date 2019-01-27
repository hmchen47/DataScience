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

Enrollment Options
Coursera has made the decision to make Specializations available by monthly subscription. This means you can choose to pay a monthly fee to access all of the courses in a specific Specialization. Coursera’s switch to monthly subscriptions comes with another change -- for those learners who choose the “Audit Only” enrollment, you will no longer be able to submit assignments for grades nor see answers for those assignments. You will still have access to all the course materials but you will not be graded on your work, nor see answers to graded assignments. For further information on the different enrollment options for Coursera courses, please visit the Enrollment Options Help page. If you have feedback about the enrollment options shared on the Enrollment Options page, you can share your thoughts with Coursera in this survey.

Grading and Assignments
The lectures will provide you with some guidance for completing assignments, but you will need to take initiative and look beyond assignment instructions in order to be successful. You'll need to know how to ask questions in the discussion forums of your peers, and seek out new information through web searches and Stack Overflow. Be sure to also check out the Additional Resources. If you are not sure what kind of output is required, or think there is a need for more clarity, please head to the course discussion forums. Note that some assignments and in video quizzes may not be mobile friendly.

Course Item	Percentage of Final Grade	Passing Grade
Week 1 Quiz	5%	80%
Week 1 Notebook Assignment	18%	80%
Week 2 Quiz	5%	80%
Week 2 Notebook Assignment	18%	80%
Week 3 Quiz	5%	80%
Week 3 Notebook Assignment	18%	80%
Week 4 Quiz	5%	80%
Week 4 Notebook Assignment	26%	80%
Code of Conduct
Visit Coursera’s Code of Conduct and to abide by guidelines there. It is important when giving feedback to your peers to be polite and to be sensitive to the diversity of cultures and backgrounds of learners in your course.

Working Offline
While the Coursera platform has an integrated Jupyter Notebook system, you can work offline on your own computer by installing Python 3.5+ and the Jupyter software packages. For more details, consult the Jupyter Notebook FAQ.

Accessibility
We strive to develop fully accessible courses. Occasionally, some of our content does not fully meet our accessibility goals. Please use this form to inform us of any accessibility issues you are experiencing in this course.

Help!
If you're having problems, here are a couple of great places to go for help: If the problem is with the Coursera platform such as verification on assignments, in video quiz problems, or the Jupyter Notebooks, please check out the Coursera Learner Support Forums. If the problem deals with understanding the assignment or how to use the Jupyter Notebooks, please read our Jupyter Notebook FAQ page in the course resources If you have questions with the content of the course, or questions about programming in python or with the toolkits described, you can contact your peers and the course instructors in the discussion forums, or go to Stack Overflow. Having trouble accessing your previously submitted assignments? If your session has ended, you can access these again by selecting the "Switch Session" option. Details for how to select this can be found in this learner help center article. If you still have issues accessing your materials after switching sessions, please reach out to Coursera learner support via our online chat forums in the Learner Help Center.

In-Video Questions (IVQs)
In this course, in-video questions or IVQs may appear during lectures to help you learn as well as assess your understanding of the content. IVQs are optional and do not count towards your overall course grade.

Types of in-video questions
Many of the lectures contain in-video questions (IVQs). These questions are presented in a variety of formats. Some will ask you to write or think about a concept from the video. Others will ask for a short answer. Still others may ask you to choose from a multiple-choice list of answers. If an IVQ is a survey or a poll, you will see a summary of responses from other learners after you respond. You can look at the question again later to see new summary data as more of your peers answer. Some IVQs also contain runnable code blocks. These IVQs allow you to practice the coding concepts during the lecture. In this course, these types of IVQs will usually be directly followed with the solution code.


## Help us learn more about you!

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Networks: Definition and Why We Study Them

### Lecture Note



+ Demonstration
    ```python


    ```

### Lecture Video

<a href="url" alt="text" target="_blank">
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




