# Statistical Analysis in Python and Project

## Week 4 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

[Web Notebook](https://hub.coursera-notebooks.org/user/qceqpnyfwlofzjpttttssh/notebooks/Week%204.ipynb)

[Local Notebook](./Week04.ipynb)

## Introduction

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://d3c33hcgiwev3.cloudfront.net/DsljYJeREeaK1Q4gRyvE8A.processed/full/540p/index.mp4?Expires=1526169600&Signature=KnqgJkjJOXRSOdOctE~QugCueoZnhm1qGRVeOACvVRUUa~Ym6jZVsBOK0TUB43h-yWY465E0zsf89rLFheESIpJ2NwqtFP4RcfA9dWeE7uCwgKlpPiini7JWb6SSispEFw9Jom5CX1g8QON4N~fKRJuULLm1y4nUFg3w4wZcOpg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## Distributions

+ Distributions
    + Distribution: Set of all possible random variables
    + Example:
        + Flipping Coins for heads and tails
            + a binomial distribution (two possible outcomes)
            + discrete (categories of heads and tails, no real numbers)
            + evenly weighted (heads are just as likely as tails)
        + Tornado events in Ann Arbor
            + a binomial distribution
            + Discrete
            + evenly weighted (tornadoes are rare events)

+ `binomial` method:
    + Syntax: `binomial(n, p, size=None)`
    + Draw samples from a binomial distribution.
    + `n`: int or array_like of ints  
        Parameter of the distribution, $n \geq 0$. Floats are also accepted,  but they will be truncated to integers.
    + `p`: float or array_like of floats  
        Parameter of the distribution, $0 \leq p \leq 1$.
    + `size`: int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., $(m, n, k)$, then $m * n * k$ samples are drawn.  If size is `None` (default),a single value is returned if $n$ and $p$ are both scalars. Otherwise, `np.broadcast(n, p).size` samples are drawn.
    + Simulate equal coin flips with each $n$ flips to get heads/tails and execute $size$ times

+ Demo 
    ```Python
    np.random.binomial(1, 0.5)          # record count of an event w/ equal probability

    np.random.binomial(1000, 0.5)/1000  # record counts of 1000 events and get the weight

    chance_of_tornado = 0.01/100        # prob of tornado
    np.random.binomial(100000, chance_of_tornado)   # 

    # sampling distribution
    chance_of_tornado = 0.01
    tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)

    two_days_in_a_row = 0
    for j in range(1,len(tornado_events)-1):
        if tornado_events[j]==1 and tornado_events[j-1]==1:
            two_days_in_a_row+=1
    ```
+ Quiz:
    + Suppose we want to simulate the probability of flipping a fair coin 20 times, and getting a number greater than or equal to 15. Use `np.random.binomial(n, p, size)` to do 10000 simulations of flipping a fair coin 20 times, then see what proportion of the simulations are 15 or greater.
    + Answer:
        ```python
        x = np.random.binomial(20, .5, 10000)
        print((x>=15).mean())
        ```

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## More Distributions

+ Uniform Distribution (Continuous)
    + x-axis: Values of observation
    + y-axis: Probability Observation Occurs

    ![Uniform Dist](http://slideplayer.com/4899942/16/images/2/The+Uniform+Distribution.jpg)
+ Normal (Gaussian) Distribution
    + x-axis: Value of Observation
    + y-axis: Probability Observation Occurs
    + standard deviation: measure of variability

    ![Gaussian Dist](https://cdn-images-1.medium.com/max/1600/0*RfxIdjcLvDPJjD5b.PNG)
+ Chi Squared (χ2) Distribution
    + left-skewed
    + Degrees of freedom (df)

    ![Chi-Squared Dist](https://saylordotorg.github.io/text_introductory-statistics/section_15/5a0c7bbacb4242555e8a85c9767c03ee.jpg)
+ Bimodal distributions

    ![Bimodal Dist](https://cdn-images-1.medium.com/max/1250/1*cxeqxH1_zb68td7toVvFQQ.png)
+ Demo 
    ```Python
    print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

    np.random.uniform(0, 1)

    np.random.normal(0.75)
    ```

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://d3c33hcgiwev3.cloudfront.net/WxZYpJmFEea9VA4aKWH1oA.processed/full/540p/index.mp4?Expires=1526169600&Signature=aLy3e24RkaPyLLMK91JglIa9xEmj0mdjRpklT0OFarUQOGk6p2Xd5MUabz6RBAaNsHvLt74tXfVArvwD2oOU2iHVqbo5GeWTZEkmC4z4usDPm8cSEUXrU07yoGy9GkDbTeFGycJdpABIo6hRBC~KhW-cICZwstVmMOS9cdg3ZOQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## Hypothesis Testing in Python

+ Demo 
    ```Python
    
    ```

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## End of Theory

+ Demo 
    ```Python
    
    ```

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## Discussion Prompt: Science Isn't Broken: p-hacking activity



