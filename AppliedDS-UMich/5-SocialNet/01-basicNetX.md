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

Coursera has made the decision to make Specializations available by monthly subscription. This means you can choose to pay a monthly fee to access all of the courses in a specific Specialization. Coursera's switch to monthly subscriptions comes with another change -- for those learners who choose the “Audit Only” enrollment, you will no longer be able to submit assignments for grades nor see answers for those assignments. You will still have access to all the course materials but you will not be graded on your work, nor see answers to graded assignments. For further information on the different enrollment options for Coursera courses, please visit the [Enrollment Options ](https://learner.coursera.help/hc/en-us/articles/209818613-Enrollment-options) page. If you have feedback about the enrollment options shared on the Enrollment Options page, you can share your thoughts with Coursera in this [survey](https://www.surveymonkey.com/r/65DPLHG).

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

Visit [Coursera's Code of Conduct](https://learner.coursera.help/hc/en-us/articles/208280036-Coursera-Code-of-Conduct) and to abide by guidelines there. It is important when giving feedback to your peers to be polite and to be sensitive to the diversity of cultures and backgrounds of learners in your course.

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
        <img src="https://lh3.googleusercontent.com/CsyaIMmiYfjAk89JPShRf75E4B-NzPHpwhs5MagkoO8PYEj6vJrkNW1QrcomU8MCpk1JarWNpHkWlQcFNpDE7O4wTySzAHWDavfvmhJ5jg8pmFoS89tePBlU2f0TiNVnjEUkuCVW1Q=w2400" alt="So the first kind of network that we are going to look at are social networks. And in social networks the nodes are people, and the connections between the nodes represent some type relationship between the people in the network. So here we see an example of 34 people who belong to a karate club, and the network represents friendship between them. In this example, node number one, which you see right here, is the instructor in the karate club and everybody else is a student in the karate club. This other example is a network of friendship, marital tie, and family tie among 2,200 people. Here the edges are colored to represent the particular type of relationship between the nodes. And then next we have an e-mail communication network. So the other two networks we saw were in relationships that were happening in the offline world, but networks can also be constructed based on relationships that happen in the online world. So here in this example, we have a network between 436 HP employees and the edges represent communication through email." title="Social Networks " height="300">
    </a>

+ Transportation and Mobility Networks
    + Network of direct flights around the world [Bio.Diaspora]
    + Human mobility network based on location of dollar bills (Where's George) [Thiemann et al. 2010]
    + Ann Arbor bus transportation network
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/bc0o7l-5Bgi9H_D6DZaQvuVimxQHHR5aM4cCLkwuKyAMrd_OyhyP9d4Sb0x32oNb4vWUYgQO6_k9HNK3IUgsQKP92coBxiAyJLUdl0O4DRZMx9VmgFXa4LK9BT1dkjHjz8SMwR1jOA=w2400" alt="We can also have transportation and mobility networks. So in this example, we see a network of directed flights between the different airports around the world. And there's human mobility network that is based on the location of dollar bills using the words George website, where people look at their dollar bills, tells the website where they are located. And then this bill can be track when other people update the location of this bill. We can see this bill has travel throughout the United States, and here is the network I gets more by tracking bills movement. And here we have a network that represents the bus transportation network of Ann Arbor." title="Transportation and Mobility Networks" height="300">
    </a>

+ Information Networks
    + Communication between left-wing and rightwing political blogs [Adamic & Glance 2005]
    + Internet Connectivity [K. C. Claffy]
    + Network of Wikipedia articles about climate change [EMAPS]
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/txcCQMUFlestauM2AvT79S-4gqOP8HDRUrKWwteQgoZ7WgLWl8BcK1-VOz2W14Ry8Y8MaO-JUEuLs6afL8wTWxtFbfOpN_ghMFMCeaWlsi1suf3lwZbSrmt8h4d3_NsMpDkusjpnVA=w2400" alt="And here the edges represent direct bus routes from one stop to the next stop. Here also information networks, so in this example, we see network where the nodes are political blogs and then edges represent connections between the blogs through URLs. So who links to, what blog links to which blog? And what we see here is that they're colored by one of their left wing or right wing. And what we see is that left-wing blogs tend to connect mainly to left-wing blogs, and right-wing blogs tend to connect mainly to right-wing blogs, but there are not a lot of connections that go from one to the other. So we call this clustering, there's a lot of clustering between the two types of blogs. Here is the network that represents the Internet connectivity, and here's a network between Wikipedia articles about climate change. The ideas also represent URL connections or direct connections between one article in the next. And here we can also see that there is some clustering happening. So the colors represent different sub-topping within climate change, and we can see that they are clustered by the different sub-toppings." title="Information Networks" height="300">
    </a>

+ Biological Networks
    + Protein-protein interactions [Jeong et al. 2001]
    + Chesapeake Bay Waterbird Food Web [Perry et al. 2005]
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/8PFVbsmguLKaR2f0rQpL2T2arcuUS1bpCE_k3Q_wTRL40GyDekCp8ug6EV9paFYDHMGZzWuuVfiJ6vGQzQhp2vc4hdqFXcNog8g-g3loFofkVfewOnDZ2Gx7rhyCPpkRzpEwpuUTkw=w2400" alt="Networks also show up in biology, so here's an example of a protein to protein interaction. So the nodes are proteins and they're connected by where they interact with each other. And here is a network that represent a food web, so what animals eat what animals." title="Biological Networks" height="300">
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
            <img src="https://courses.cit.cornell.edu/info2040_2010fa/adamic-hier.jpg" alt="So I'm going to give you some examples. Let's go back to this email communication network among the 436 HP employees. The kinds of questions that we can answer by looking at the structure of the network are things like, if there's a rumor that starts in some part of the network, is it likely to spread through the whole network? And who are the most influential people in this organization? Is the rumor that's starting on say, some node that's sort of like on the outskirts of the network more or less likely to be spread than if it were to be started by somebody who's more central to the network, like someone around this area? These are the kinds of things that we can start to analyze and understand by looking at the structures of a network." title="E-mail communication network among 436 HP employees" height="200">
        </a>
        + Is a rumor likely to spread in this network?
        + Who are the most influential people in this organization?
    + Friendship network in a 34-person karate club
        <a href="https://anthonybonato.com/2016/04/13/the-mathematics-of-game-of-thrones/"> <br/>
            <img src="https://lh3.googleusercontent.com/OQqUIVdAO_KrEiIsfGN4mARt24rHxQzWZ9IndHfY3DEvgvYp-m7PW4BzaaKpb9Trp2w8UKvvkuW3tSN6O7pJ7L7vm9P_pBX-eLOf03QKFd9y2jVQ" alt="Zachary's Karate club graph, from Wayne Zachary's PhD thesis in 1972. The nodes correspond to members of an actual karate club, and the edges represent their social ties. The instructors of the club are the bold nodes 1 and 34. After a dispute between them, the instructors each formed their own club. We can see here the community formed around each of the nodes 1 and 34. -- the friendship network between the karate club members, we can answer things like, is this club likely to split into two different clubs? And actually the story is that, this is exactly what happens. If you could try to guess which member is going to join which club after it splits, you can look at the structure of this network and make a pretty educated guess. And it turns out that, as you can probably see from just looking at this network, the division between the club happens around here where the nodes on the left side of this line go to one club, and then the ones on the right go to the other club." title="Zachary Karate Club graph" height="200">
        </a>
        + Is this club, likely to split into two groups?
        + If so, which nodes will go to which group?
    + Network of direct flights around the world
        <a href="http://www.visualisingdata.com/2012/02/bio-diaspora-visualising-interactions-between-populations-and-travel/"> <br/>
            <img src="http://www.visualisingdata.com/blog/wp-content/uploads/2012/02/World_FlightLines_BioDiaspora-600x393.jpg" alt="I got in touch with David Kossowsky, a GIS mapper, cartographer and graphic designer, to find out more about the work of Bio.Diaspora and some of the visualisations they have been working on. This image shows a visualisation of the global airline transportation network consisting of all commercial flights worldwide. While this image in its entirety does not necessarily provide information that can be used to assess an infectious disease threat, it does provide one with a greater understanding of how interconnected the world is due to air transportation, and how easy it is for some diseases to spread across very large areas in relatively short periods of time. -- If we look at a network of transportation network of direct flights around the world, can we answer the questions like, if there's an epidemic or a virus spreading in the world, are there airports that we have to pay more attention to than others? Or if there are certain parts of the world that are harder to reach through air transportation, what are key connections we can make to make those areas easier to reach? " title="global airline transportation network " height="200">
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

+ Network Definition and Vocabulary
    + Network (or Graph): A representation of connections among a set of items.
        + Items are called __nodes__ (or _vertices_)
        + Connections are called __edges__ (or _link_ or _ties_)
        ```python
        import networkx as nx

        G=nx.Graph()
        G.add_edge('A','B')
        G.add_edge('B','C')
        ```
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> 
            <img src="https://lh3.googleusercontent.com/xYuECJWNQyxTK-2Ie9VtKrfBOEPKUq9CDIaO_DQ1s9z6li4OnpC3oIIZkZNuFccAz-CKiUHUajxFHvM6pmE8OAL3bN8bkMGJ0NNuJ_QpQ7J04H3bQBokhvTMaWb0DOvQnu8r8Ma8XA=w2400" alt="So here's an example of a set of things which we call nodes, just circles that have labels A through G. So we call these things nodes or vertices. And then there are connections between them that can represent various different things. And we call these connections edges, or sometimes we call them links or ties as well. And we're going to use NetworkX in Python in order to work with some of the networks that we look at. So the first thing you need to know is how to create a network in NetworkX. So the first thing we're going to do is import networkx as nx. And then we're going to use the class Graph in order to represent this network that we see here. So here I'm making G an instance of one of those graphs. And then what I can do is I can add edges. So for example here, I'm adding the edge A, B, which is this edge right here. And then I would add the next edge, which would be the edge B, C. And I could continue adding all the other edges." title="Network Definition and Vocabulary" height="250">
        </a>

+ Example
    + Network of friendship, marital tie, and family tie among 2200 people
        + __Nodes__: People
        + __Edges__: Friendship, marital, or family ties
        + (Mostly) __Symmetric relationships__
        <a href="https://www.nejm.org/doi/full/10.1056/NEJMsa066082"> <br/>
            <img src="https://www.nejm.org/na101/home/literatum/publisher/mms/journals/content/nejm/2007/nejm_2007.357.issue-4/nejmsa066082/production/images/img_small/nejmsa066082_f1.jpeg" alt="Each circle (node) represents one person in the data set. There are 2200 persons in this subcomponent of the social network. Circles with red borders denote women, and circles with blue borders denote men. The size of each circle is proportional to the person's body-mass index. The interior color of the circles indicates the person's obesity status: yellow denotes an obese person (body-mass index, ≥30) and green denotes a nonobese person. The colors of the ties between the nodes indicate the relationship between them: purple denotes a friendship or marital tie and orange denotes a familial tie. -- This is a network between people and the edges here represent friendship, marital ties and family ties among 2,200 people. When you look at this network one thing that you can see is that these edges are mostly symmetric relationships. And by that I mean if A is a friend of B, then B is also a friend of A. Or at least most of the time, that's the case. And so this network has symmetric relationships." title="Largest Connected Subcomponent of the Social Network in the Framingham Heart Study in the Year 2000." height="250">
        </a>
    + Chesapeake Bay Water bird Food Web
        + __Nodes__: Birds
        + __Edges__: What eats what
        + __Asymmetric relationships__
        <a href="https://www.usgs.gov/media/images/chesapeake-bay-waterbird-food-web-illustration-circular-1316"> <br/>
            <img src="https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/full_width/public/thumbnails/image/Circular%201316_fig14.1.jpg" alt="But that doesn't always have to be the case. So if you look at this network, which represents what animals eat other animals, these relationships are very asymmetric. For example, it's very different if you have an edge pointing from the fish to the eagle. That says that the eagle eats the fish, rather than the other way around. So the direction of the edge in this network has a very important meaning to what the edge is trying to represent. So this suggests that we need at least two different types of networks. Some that are undirected, meaning the edges don't have any direction, or that the direction of the edge is not really important, and we call these undirected edges." title="Chesapeake Bay Waterbird Food Web." height="250">
        </a>

+ Edge Direction
    + __Undirected network__: edges have no direction
        ```python
        G=nx.Graph()
        G.add_edge('A','B')
        G.add_edge('B','C')
        ```
    + __Directed network__: edges have direction
        ```python
        G=nx.DiGraph()
        G.add_edge('B', 'A')
        G.add_edge('B','C')
        ```

+ Weighted Networks
    + Not all relationships are equal.
    + Some edges carry higher weight than others.
    + Eg: Number of times coworkers had lunch together in one year
    + __Weighted network__: a network where are assigned a (typically numerical) weight.
        ```python
        G=nx.Graph()
        G.add_edge('A','B', weight = 6)
        G.add_edge('B','C', weight = 13)
        ```

+ Signed Networks
    + Some networks can carry information about friendship and antagonism based on conflict or disagreement.
    + Eg: In Epinions and Slashdot people can declare friends and foes.
    + __Signed network__: a network where edges are assigned positive or negative sign.
        ```python
        G=nx.Graph()
        G.add_edge('A','B', sign= '+')
        G.add_edge('B','C', sign= '-')
        ```

+ Other Edge Attributes
    + Edges can carry many other labels or attributes
    ```python
    G=nx.Graph()
    G.add_edge('A','B', relation= 'friend')
    G.add_edge('B','C', relation= 'coworker')
    G.add_edge('D','E', relation= 'family')
    G.add_edge('E','I', relation= 'neighbor')
    ```

+ Mutigraphs
    + A pair of nodes can have different types of relationships simultaneously
    + Multigraph: A network where multiple edges can connect the same nodes (parallel edges).
        ```python
        G=nx.MultiGraph()
        G.add_edge('A','B', relation= 'friend')
        G.add_edge('A','B', relation= 'neighbor')
        G.add_edge('G','F', relation= 'family')
        G.add_edge('G','F', relation= 'coworker')
        ```

+ __Lecture Quiz__: We would like to construct a graph on NetworkX, where the nodes represent employees of a company and the edges represent the number of times an employee sent an email to another employee. What would be the best way to represent this network?
    ```
    a. Directed graph
    b. Undirected graph
    c. Weighted, directed graph
    d. Weighted, undirected graph
    e. Weighted multigraph

    Ans: c 
    Since we want to capture who sent the email and who received it, we need a directed graph. Since we also want to capture the number of times an employee emailed another, we want the edges to have weights, hence we want to use a weighted, directed graph.
    ```

+ Summary
    <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
        <img src="https://lh3.googleusercontent.com/S4gf8u9EVJAx4htkOzGvfuW-TFxm4ph-5p4yzXnn7YzGR0dLDPQTjd15td5tDVeiIct2W-dGABDf-HR4iUyWSxbKC4uGvPou4Xii_Vz0Erj1MQt9SycOttuP_1-YInQ8fUeKagkbRw=w2400" alt="Undirected network: Nodes have symmetric relationships; Directed network: Nodes have assymmetric relationships; Weighted network: A network where edges are assigned a weight; Signed network: A network where edges are assigned positive or negative sign; Edges can carry many other labels or attributes. Attribute can be weight, sign, relation,…; Multigraph: a network where multiple edges can connect the same nodes " title="Basic Concepts" height="300">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/KVAnNZTLEeeOmgqEJWRlfA.processed/full/360p/index.mp4?Expires=1548720000&Signature=J5Sr9blGwEfWBbkNgYEYoPokhrpQE2ulvGYTSffirjefoYloIc~IwWXmqMBABZWzUI1bQ8qSMrgznQSnjTcaDm5jHguKzG3NKzNZWO31G~jP1X~4UU3euxOfiCCL7Ma~2OjUKC1BWQwEHf9SztL67eyTmlAsKTs-Q94sbFvA0c0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Network Definition and Vocabulary" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Node and Edge Attributes

### Lecture Note

+ Edge Attributes in NetworkX
    + Number of times coworkers had lunch together in one 
    + Undirected network
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hTKct/node-and-edge-attributes"> <br/>
        <img src="images/m1-01.png" alt="So here is an example of a network where the color of the edges represent the relationship between the nodes. And then there's a number on the edges that represents how many times they had lunch together. So when we were using NetworkX to construct some of these graphs, we would simply add attributes to the edges when we added them. So here we added the weight of the edge and also their relation, and the same thing for all the other edges. And so the first thing is, if we just use the function edges, this will give us a list of all the edges of the network. So far I've only added two so this will give us the two edges that I've added. Now if you wanted to get a little more data on these edges, then you would use the same function edges, but now you would say data equals true. And now these will list all the edges with the attributes that they have. So, for each edge, you would get the two nodes A, B, as well as a dictionary for the different attributes that, that edge has. In this case, relation and weight, and same thing for the other edges. Now, let's say you only wanted the information about the edges for a particular attribute, then you can say data equals relation, for example. The particular attribute you're interested in. And now you will get triplets, that will have the two nodes and then the value of the attribute relations. So in this case, A, B will have family. And B, C will have friend.  by using edge rather than edges, and then saying which edge you have by showing the two nodes, the two endpoints of the edge. And then this would return a dictionary that will have each one of the attributes of that edge.  You can also just specify which particular attribute you're interested in. So for example, if you wanted to know what the weight of the edge B, C is, then you would do it this way, and then it will tell you it's 13, as shown right here. Now notice that because this graph is an undirected graph, then the order in which we place the end points of the edge does not matter, right? So if you ask for the weight of the edge B, C, the answer is 13. But if you were to ask for the weight of the edge C, B, you would get the same thing because this is an undirected graph and the order doesn't matter." title="Edge Attributes in NetworkX" height="250">
    </a>
        ```python
        G=nx.Graph()
        G.add_edge('A','B', weight= 6, relation = 'family')
        G.add_edge('B','C', weight= 13, relation = 'friend')

        G.edges() #list of all edges
        # [('A', 'B'), ('C', 'B')]

        G.edges(data= True) #list of all edges with attributes
        # [('A', 'B', {'relation': 'family', 'weight': 6}), ('C', 'B', {'relation': 'friend', 'weight': 13})]

        G.edges(data= 'relation') #list of all edges with attribute ‘relation'
        # [('A', 'B', 'family'), ('C', 'B', 'friend')]

        # Accessing attributes of a specific edge:
        G.edge['A']['B'] # dictionary of attributes of edge (A, B)
        # {'relation': 'family', 'weight': 6}
        G.edge['B']['C']['weight']
        # 13
        G.edge['C']['B']['weight'] # undirected graph, order does not matter
        # 13
        ```
    + Directed, weighted network:
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hTKct/node-and-edge-attributes"> <br/>
        <img src="images/m1-02.png" alt="What if you have a directed case? So now this is the network, the same kind of thing but now you have direction on the edges. Well, then we would use the directed graph class and then we would add the edges in the exact same way we did before. And now, we would access the attributes of the edges in the same way. So here you would be asking for the weight of the edge C, B which is 13. But if you were to switch the order and now you're asking for the weight of the edge B, C, then you would get an error because this edge doesn't exist. So because you're using the directed graph class, the order matters and it also matters in the way that you access the attributes of the edge." title="Edge Attributes in NetworkX" height="250">
    </a>
        ```python
        G=nx.DiGraph()
        G.add_edge('A','B', weight= 6, relation = 'family')
        G.add_edge('C', 'B', weight= 13, relation = 'friend')
        
        # Accessing edge attributes:
        G.edge['C']['B']['weight']
        # 13

        G.edge['B']['C']['weight'] # directed graph, order matters
        # KeyError: 'C'
        ```
    + MultiGraph:
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hTKct/node-and-edge-attributes"> <br/>
        <img src="images/m1-03.png" alt="MultiGraph" title="Edge Attributes in NetworkX" height="250">
    </a>
        ```python
        G=nx.MultiGraph()
        G.add_edge('A','B', weight= 6, relation = 'family')
        G.add_edge('A','B', weight= 18, relation = 'friend')
        G.add_edge('C','B', weight= 13, relation = 'friend')
        
        # Accessing edge attributes:
        G.edge['A']['B'] # One dictionary of attributes per (A,B) edge
        # {0: {'relation': 'family', 'weight': 6}, 1: {'relation': 'friend', 'weight': 18}}
        
        G.edge['A']['B'][0]['weight'] # undirected graph, order does not matter
        # 6
        ```
    + Directed MultiGraph:
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hTKct/node-and-edge-attributes"> <br/>
        <img src="images/m1-04.png" alt="generalize this to MultiGraphs, right? So as we saw before, the way you would represent a network like this is by using the MultiGraph class. And then you would add the edges and you would add the edges for the same pair of nodes multiple times. And sometimes you would give them different weights or different attributes in general. So here, we're adding the edge A, B twice with different weights and different relations. And same thing for C,B. So how do we access the attributes for these? Well, if you ask for the attributes of the edge A,B, then what happens is you will get a dictionary of the attributes of A, B. But you would get one dictionary for each one of the edges. So remember the edge A, B has two different edges with different attributes. So here you get the first edge, which is label zero, and is the first one you entered, will have a dictionary that has relation, family, and then weight six. And then another dictionary with the relationship friend that has weight 18, and that's the second answer that you added. And in this case, because this is an undirected graph, the order in which you do things doesn't matter. So in this example, I'm showing you how to get the weight for the edge A, B, and I'm specifying that I want the first edge. So I add this zero here, and then the answer is six. If I wanted the weight on the second edge, then this zero will be one. And then you would get the other weight which is 18. And because this is undirected, the ordered doesn't matter. It's waited and it's also directed. For this kind of graph, we'll actually use a class we haven't seen before which is the MultiDiGraph, which stands for multi directive graph. And of course, now the edges have direction and there can be multiple edges between any two pairs of nodes. And so we'll add the edges in the same way we did before. But now we have to be careful to add them in the right direction, right? So we write A, B rather than B, A because the edge that we're adding has a particular direction. The same thing for the other ones. And when you would access the attributes of these in just the exact same way. But now the order matters, so if I ask for the edge, the way for the first edge A, B, then I would get six. But if I were to ask for the weight of the first edge B, A, I would get an error because that edge doesn't exist." title="Edge Attributes in NetworkX" height="250">
    </a>
        ```python
        G=nx.MultiDiGraph()
        G.add_edge('A','B', weight= 6, relation = 'family')
        G.add_edge('A','B', weight= 18, relation = 'friend')
        G.add_edge('C','B', weight= 13, relation = 'friend')

        # Accessing edge attributes:
        G.edge['A']['B'][0]['weight']
        # 6
        
        G.edge['B']['A'][0]['weight'] # directed graph, order matters
        # KeyError: 'A'
        ```
    + Lecture Quiz: What would be the output of the following code?
        ```python
        import networkx as nx

        G=nx.MultiDiGraph()

        G.add_edge('John', 'Ana', weight= 3, relation = 'siblings')
        G.add_edge('Ana', 'David', weight= 4, relation = 'cousins')
        G.add_edge('Ana', 'Bob', weight= 1, relation = 'friends')
        G.add_edge('Ana', 'Bob', weight= 1, relation = 'neighbors')

        print( G.edge['Bob']['Ana'][1]['relation'] )
        ```
        ```
        a. 'friends'
        b. KeyError: 'Ana'
        c. Correct
        d. 'neighbors'

        Ans: b
        G is a directed graph and while the edge (‘Ana’, ‘Bob’’) is in the network, the edge (‘Bob’, ‘Ana’) is not. Hence, the output will be an error.
        ```
    + Undirected Multigraph
    <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/hTKct/node-and-edge-attributes"> <br/>
        <img src="images/m1-04.png" alt="Okay, so we've talked about how to add attributes and how to access attributes of edges. But we could also imagine having attributes stored on the nodes. So let's go back to this example of how many times coworkers had lunch together in a particular company, and what kind of relationship they have. You could imagine that the nodes can also have a particular attribute, so in this case, imagine they're colored by their role in the company. So some are managers, some are traders and some are analysts. Then we would want a way of capturing this, also, when we construct network in NetworkX, and what we can do is the following. So first, we're constructing the graph in the usual way that we construct it. And then we would add the node attributes. So to add the node attributes, what we'll do is we'll use the function add_node, even though this node A is already added because we added the edge A, B, we would add it again. And now we'll give it an attribute role, and we'll say that the role of node A is trader. And node B is also trader, and C is a manager. And then to access those attributes, we'll do it in the following way. So first of all, if we just wanted to have the node, list of all the nodes, we can use the function nodes, and this will give us the three nodes that we've added. But if we wanted the attributes on the nodes, then just like we did for edges, we would say, data equals true. And it will give us a list of all the nodes. And along with each node, a dictionary with the attributes and the values for that node. And if we want to just for all, or a particular attribute for a particular node, then we would use node instead of nodes. And then specify which node we're at one, and which attribute we want. And they would say that node A is a manager." title="Edge Attributes in NetworkX" height="250">
    </a>
        ```python
        G=nx.Graph()
        G.add_edge('A','B', weight= 6, relation = 'family')
        G.add_edge('B','C', weight= 13, relation = 'friend')

        # Adding node attributes:
        G.add_node('A', role = 'trader')
        G.add_node('B', role = 'trader')
        G.add_node('C', role = 'manager')

        # Accessing node attributes:
        G.nodes() # list of all nodes
        # ['A', 'C', 'B']

        G.nodes(data= True) #list of all nodes with attributes
        # [('A', {'role': 'trader'}), ('C', {'role': 'manager'}) , ('B', {'role': 'trader'})]

        G.node['A']['role']
        # 'manager'
        ```

+ Summary
    ```python
    # Adding node and edge attributes:
    G=nx.Graph()
    G.add_edge('A','B', weight= 6, relation = 'family')
    G.add_node('A', role = 'trader')

    # Accessing node attributes:
    G.nodes(data= True) #list of all nodes with attributes
    G.node['A']['role'] #role of node A

    # Accessing Edge attributes:
    G.edges(data= True) #list of all edges with attributes
    G.edges(data= ‘relation’) #list of all edges with attribute ‘relation’
    G.edge['A']['B']['weight'] # weight of edge (A,B)
    ```

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/DKgHw5TLEeeClxLmJhEfgA.processed/full/360p/index.mp4?Expires=1548720000&Signature=TuR6UldQsef~Lo5-Tw0C~-AA4wElpS-pgCtQsTP7MChjRHJZ9lDjTMS3L2l8j5dtrr1aBX72Hf3ASP4Wrz3z8gi5TfJp1IzNDuv7ITDhCQhaz0WucGEy4ffQWeCFYVfz98k3nztw9R7omZJ2uSAgXy3YYlQJA3L0egVTn7t8AYA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Node and Edge Attributes" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Bipartite Graphs

### Lecture Note

+ Bipartite Graphs
    + __Bipartite Graph__: a graph whose nodes can be split into two sets L and R and every edge connects an node in L with a node in R.
        ```python
        from networkx.algorithms import bipartite

        B = nx.Graph() # No separate class for bipartite graphs
        B.add_nodes_from(['A’,'B','C','D', 'E'], bipartite=0) #label one set of nodes 0
        B.add_nodes_from([1,2,3,4], bipartite=1) # label other set of nodes 1
        B.add_edges_from([('A',1), ('B',1), ('C',1), ('C',3), ('D',2), ('E',3), ('E', 4)])

        # Checking if a graph is bipartite:
        bipartite.is_bipartite(B) # Check if B is bipartite
        # True

        B.add_edge('A', 'B')
        bipartite.is_bipartite(B) # False

        B.remove_edge('A', 'B')

        # Checking if a set of nodes is a bipartition of a graph:
        X = set([1,2,3,4])
        bipartite.is_bipartite_node_set(B,X) # True

        X = set(['A', 'B', 'C', 'D', 'E'])
        bipartite.is_bipartite_node_set(B,X) # True

        X = set([1,2,3,4, ‘A’])
        bipartite.is_bipartite_node_set(B,X) # False

        # Getting each set of nodes of a bipartite graph:
        bipartite.sets(B) # ({'A', 'B', 'C', 'D', 'E'}, {1, 2, 3, 4})

        B.add_edge('A', 'B')
        bipartite.sets(B) # NetworkXError: Graph is not bipartite.

        B.remove_edge('A', 'B')
        ```
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/">
            <img src="https://lh3.googleusercontent.com/ndSW8er1j0jDOT5WQx4sGja9qJL6fHhT5LhgpGI-Sj4Z9VnufF3X1v8U-dK-vZHDdUCgVgm0i-CdLG88r-e3z2aXO1lVeHR7vN1tggCe2JY4A0ekAw7ij0dJglpmUNjw_hGB1ECYYA=w2400" alt="Bipartite Graph is a graph whose nodes can be split into two sets L and R, and every edge connects an node in L with a node in R." title="Bipartite Graphs" height="250">
        </a>
        <a href="https://www.coursera.org/learn/python-social-network-analysis/lecture/tWwx2/bipartite-graphs">
            <img src="images/m1-06.png" alt="text" title="Bipartite" height="250">
        </a>

+ Projected Graphs
    + __L-Bipartite graph projection__: Network of nodes in group L, where a pair of nodes is connected if they have a common neighbor in R in the bipartite graph.
    + Similar definition for R-Bipartite graph projection
        ```python
        B = nx.Graph()
        B.add_edges_from([('A',1), ('B',1),('C',1),('D',1),('H',1), ('B', 2), ('C', 2), ('D',2),('E', 2), ('G', 2), ('E', 3), ('F', 3), ('H', 3), ('J', 3), ('E', 4), ('I', 4), ('J', 4) ])

        X = set(['A','B','C','D', 'E', 'F','G', 'H', 'I','J'])
        P = bipartite.projected_graph(B, X)
        nx.draw_networkx(P)
        ```
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/">
            <img src="https://lh3.googleusercontent.com/M3QnE6zndPcgzKDIbl3gFvibIyEkFOVnHeXQktPRWt2UEO-egFrHWaYIkH5X0vgvgt6b8KS2_vTRe3q2uNvM9pbynQX7KiT6oW33ju2-SuyS0Lg8yODzxGkxE08qnUw0ZoU2-rP_9A=w2400" alt="L-Bipartite graph projection" title="Bipartite Graphs" height="250">
            <img src="https://lh3.googleusercontent.com/PT-yuFMYz6NLVYckDtWW62DFN9x8tK57WDXWEAGGiwlR45C2IMldW48ZG_FoGRfmpPsfjyph_mxWa6Xdgl9BodJA9tkaYeEgQZ9lpE0fwXV8qU_bWC86ISkEcUTNq0CHjLR_AHw9hA=w2400" alt="L-Bipartite graph projection" title="Bipartite Graphs" height="250">
        </a>
    + __L-Bipartite weighted graph projection__: An LBipartite graph projection with weights on the edges that are proportional to the number of common neighbors between the nodes.
        <a href="https://harangdev.github.io/applied-data-science-with-python/applied-social-network-analysis-in-python/1/"> <br/>
            <img src="images/m1-07.png" alt="xxx" title="L-Bipartite weighted graph projection" height="250">
        </a>
        ```python
        X = set([1,2,3,4])
        P = bipartite.weighted_projected_graph(B, X)
        ```
    
+ Summary
    + No separate class for bipartite graphs in NetworkX
    + Use Graph(), DiGraph(), MultiGraph(), etc.
    + Use from networkx.algorithms import bipartite for bipartite related algorithms (Many algorithms only work on Graph()).
    ```python
    nx.bipartite.is_bipartite(B) # Check if B is bipartite
    bipartite.is_bipartite_node_set(B,X) # Check if node set X is a bipartition
    bipartite.sets(B) # Get each set of nodes of bipartite graph B
    bipartite.projected_graph(B, X) # Get the bipartite projection of node set X
    bipartite.weighted_projected_graph(B, X) # Get the weighted bipartite projection of node set X
    ```


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/A_CWc5TLEeeOmgqEJWRlfA.processed/full/360p/index.mp4?Expires=1548720000&Signature=f2Lgob2QaiIgICLpK9IsCAGnC2OmHzY5p8TvmM1jVPw3C6tEj69jt9e4DjPvZjMysV846o~H5xi7dsv~tiCPSkZL4MwKgfJdK8NJ1KpbXqnD8pyqo4088ioES3SbdtqmidE6fnMB-BnAbzPor0Mitln4fOKAyLO8zz6lf2fJkfE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Bipartite Graphs" target="_blank">
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




