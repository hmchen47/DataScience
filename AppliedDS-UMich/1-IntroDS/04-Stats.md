# Statistical Analysis in Python and Project

## Week 4 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

[Web Notebook](https://hub.coursera-notebooks.org/user/qceqpnyfwlofzjpttttssh/notebooks/Week%204.ipynb)

[Local Notebook](./Week04.ipynb)

[Local Python](./Week04.py)

## Introduction

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video.jpg =120x)](https://d3c33hcgiwev3.cloudfront.net/DsljYJeREeaK1Q4gRyvE8A.processed/full/540p/index.mp4?Expires=1526169600&Signature=KnqgJkjJOXRSOdOctE~QugCueoZnhm1qGRVeOACvVRUUa~Ym6jZVsBOK0TUB43h-yWY465E0zsf89rLFheESIpJ2NwqtFP4RcfA9dWeE7uCwgKlpPiini7JWb6SSispEFw9Jom5CX1g8QON4N~fKRJuULLm1y4nUFg3w4wZcOpg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


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
    + Parameters: 
        + `n`: int or array_like of ints; Parameter of the distribution, $n \geq 0$. Floats are also accepted, but they will be truncated to integers.
        + `p`: float or array_like of floats; Parameter of the distribution, $0 \leq p \leq 1$.
        + `size`: int or tuple of ints; Output shape.  If the given shape is, e.g., $(m, n, k)$, then $m * n * k$ samples are drawn.  If size is `None` (default), a single value is returned if $n$ and $p$ are both scalars. Otherwise, `np.broadcast(n, p).size` samples are drawn.
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

    print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))            
    ```
+ Quiz:
    + Suppose we want to simulate the probability of flipping a fair coin 20 times, and getting a number greater than or equal to 15. Use `np.random.binomial(n, p, size)` to do 10000 simulations of flipping a fair coin 20 times, then see what proportion of the simulations are 15 or greater.
    + Answer:
        ```python
        x = np.random.binomial(20, .5, 10000)
        print((x>=15).mean())
        ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video.jpg =120x)](https://d3c33hcgiwev3.cloudfront.net/Flm7AZeREeaGLgqdPwfqeA.processed/full/540p/index.mp4?Expires=1526169600&Signature=UKGzeWaVEGuWdDL9Rjy4HMgE-Jrj0UGIMwUt5SalP0gdhgFeSmCIsJULPRmxqq9vjLlpDhydEs9CT0S2oh57~-p9nRxSc3sZgehUx3lykm-vrgtRyGhAlJw~noz7z7fJ-VTx9JucsCONwnDI7xVOAHcOhBTRqAYm78rkSqZTcDQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## More Distributions

+ Termonlogies:
    + Expected value: $E[x] = \sum{n * p}$
    + Standard Deviation (std): a measure of variability
    + __Skew__: a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean  
        For univariate data $Y_1, Y_2, ..., Y_N$, the formula for __skewness__ is:  
        $$g_1 = \sum_{i=1}^{N} \frac{(Y_i - \bar{Y})^3 / N}{s^3}$$  
        where $\bar{Y}$ is the mean, $s$ is the standard deviation, and $N$ is the number of data points. Note that in computing the skewness, the $s$ is computed with $N$ in the denominator rather than $N - 1$.
    + __Kurtosis__: shape of the tail of the distribution  
        For univariate data $Y_1, Y_2, ..., Y_N$, the formula for __kurtosis__ is:  
        $$kurtosis = \sum_{i=1}^{N} \frac{(Y_i - \bar{Y})^4 / N}{s^4}$$
        where $\bar{Y}$ is the mean, $s$ is the standard deviation, and $N$ is the number of data points. Note that in computing the kurtosis, the standard deviation is computed using $N$ in the denominator rather than $N - 1$.
    + __Modality__: Any value of a data variable or random variable at which the frequency curve or probability curve reaches a peak is called a mode

+ Uniform Distribution (Continuous)
    + x-axis: Values of observation
    + y-axis: Probability Observation Occurs

    ![Uniform Dist](http://slideplayer.com/4899942/16/images/2/The+Uniform+Distribution.jpg)

+ Normal (Gaussian) Distribution
    + x-axis: Value of Observation
    + y-axis: Probability Observation Occurs
    + standard deviation: measure of variability  
        $$\sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \overline{x})^2}$$

    ![Gaussian Dist](https://cdn-images-1.medium.com/max/1600/0*RfxIdjcLvDPJjD5b.PNG)

+ Chi Squared (χ2) Distribution
    + left-skewed
    + Degrees of freedom (df)

    ![Chi-Squared Dist](https://saylordotorg.github.io/text_introductory-statistics/section_15/5a0c7bbacb4242555e8a85c9767c03ee.jpg)

+ Bimodal distributions

    ![Bimodal Dist](https://cdn-images-1.medium.com/max/1250/1*cxeqxH1_zb68td7toVvFQQ.png)

+ Ref: Think Stats, Allen B. Downey
    + Probability and Statistics for Programmers
    + Available for free under CC license at: http://greenteapress.com/thinkstats2/index.html

+ `uniform` method:
    + Syntax: `uniform(low=0.0, high=1.0, size=None)`
    + Draw samples from a uniform distribution, $[low, high)$
    + Parameters: 
        + `low`: float or array_like of floats; Lower boundary of the output interval.  All values generated will be greater than or equal to low.  The default value is 0.
        + `high`: float or array_like of floats; Upper boundary of the output interval.  All values generated will be less than high.  The default value is 1.0.
        + `size`: int or tuple of ints; Output shape.  If the given shape is, e.g., $(m, n, k)$, then $m * n * k$ samples are drawn.  If size is `None` (default), a single value is returned if `low` and `high` are both scalars. Otherwise, `np.broadcast(low, high).size` samples are drawn.

+ `normal` method
    + Syntax: `normal(loc=0.0, scale=1.0, size=None)`
    + Draw random samples from a normal (Gaussian) distribution
    + Parameters: 
        + `loc`: float or array_like of floats; Mean ("centre") of the distribution.
        + `scale`: float or array_like of floats; Standard deviation (spread or "width") of the distribution.
        + `size`: int or tuple of ints; Output shape.  If the given shape is, e.g., $(m, n, k)$, then $m * n * k$ samples are drawn.  If size is `None` (default), a single value is returned if `loc` and `scale` are both scalars. Otherwise, `np.broadcast(loc, scale).size` samples are drawn.

+ `chisquare` method
    + Syntax: `chisquare(df, size=None)`
    + Draw samples from a chi-square distribution
    + Parametres: 
        + `df`: float or array_like of floats; Number of degrees of freedom, should be $> 0$.
        + `size`: int or tuple of ints; Output shape.  
            + If the given shape is, e.g., $(m, n, k)$, then $m * n * k$ samples are drawn.
            + If size is `None` (default), a single value is returned if `df` is a scalar.
            + Otherwise, `np.array(df).size` samples are drawn.

+ `std` method:
    + Syntax: `std(a, axis=None, out=None, ddof=0)`
    + ompute the standard deviation along the specified axis
    + Parametres: 
        + `a`: array_like; Calculate the standard deviation of these values.
        + `axis`: None or int or tuple of ints; Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of the flattened array.
        + `out` : ndarray; Alternative output array in which to place the result. It must have the same shape as the expected output but the type (of the calculated values) will be cast if necessary.
        + `dof`: int;  Means Delta Degrees of Freedom.  The divisor used in calculations is $N - ddof$, where $N$ represents the number of elements.

+ `stats.skew` method
    + Syntax: `skew(a, axis=0, bias=True, nan_policy='propagate')`
    + For normally distributed data, the skewness should be about $0$. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function `skewtest` can be used to determine if the skewness value is close enough to 0, statistically speaking.
    + Parameters:
        + `a`: ndarray; data for which the skew is calculated
        + `axis`: int or None; Axis along which the kurtosis is calculated. Default is 0. If None, compute over the whole array `a`.
        + `bias`: bool; If False, then the calculations are corrected for statistical bias.
        + `nan_policy`: {'propagate', 'raise', 'omit'}; Defines how to handle when input contains nan. 'propagate' returns nan, 'raise' throws an error, 'omit' performs the calculations ignoring nan values.

+ `stats.kurtosis` method:
    + Syntax: `kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate')`
    + Kurtosis is the fourth central moment divided by the square of the variance. If Fisher's definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.
    Parameters:
        + `a`: ; data for which the kurtosis is calculated
        + `axis`: int or None; Axis along which the kurtosis is calculated. Default is 0. If None, compute over the whole array `a`.
        + `fisher`: bool; 
            + `True`: Fisher's definition is used (normal ==> 0.0). 
            + `False`: Pearson's definition is used (normal ==> 3.0).
        + `bias`: bool; If False, then the calculations are corrected for statistical bias.
        + `nan_policy`: {'propagate', 'raise', 'omit'}; Defines how to handle when input contains nan. 'propagate' returns nan, 'raise' throws an error, 'omit' performs the calculations ignoring nan values.


+ Demo 
    ```Python
    np.random.uniform(0, 1)

    np.random.normal(0.75)

    distribution = np.random.normal(0.75,size=1000)     # normal distribution 

    np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution)) # standard deviation

    np.std(distribution)        # standard deviation

    # module for statistics
    import scipy.stats as stats
    stats.kurtosis(distribution)    

    stats.skew(distribution)

    chi_squared_df2 = np.random.chisquare(2, size=10000)
    stats.skew(chi_squared_df2)

    chi_squared_df5 = np.random.chisquare(5, size=10000)
    stats.skew(chi_squared_df5)

    %matplotlib inline
    import matplotlib
    import matplotlib.pyplot as plt

    output = plt.hist([chi_squared_df2,chi_squared_df5], bins=50, histtype='step', 
                    label=['2 degrees of freedom','5 degrees of freedom'])
    plt.legend(loc='upper right')
    ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video.jpg =120x)](https://d3c33hcgiwev3.cloudfront.net/WxZYpJmFEea9VA4aKWH1oA.processed/full/540p/index.mp4?Expires=1526169600&Signature=aLy3e24RkaPyLLMK91JglIa9xEmj0mdjRpklT0OFarUQOGk6p2Xd5MUabz6RBAaNsHvLt74tXfVArvwD2oOU2iHVqbo5GeWTZEkmC4z4usDPm8cSEUXrU07yoGy9GkDbTeFGycJdpABIo6hRBC~KhW-cICZwstVmMOS9cdg3ZOQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## Hypothesis Testing in Python

+ Hypothesis Testing
    + Hypothesis: A statement we can test
        + Alternative hypothesis: Our idea, e.g. there is a difference between groups
        + Null hypothesis: The alternative of our idea, e.g. there is no difference between groups
    + Critical Value alpha ($α$)
        + The threshold as to how much chance you are willing to accept
    + Typical values in social sciences are $0.1$, $0.05$, or $0.01$

+ P-hacking, or Dredging
    + At a confidence level of $0.05$, we expect to find one positive result 1 time out of 20 tests
    + Remedies:
        + Bonferroni correction: multiple tests, e.g. 0.05 w/ 3 tests -> 0.017
        + Hold-out sets: generalizable of the test - data divided into two sets and analyze each of them; havily used in ML as _cross fold validation_
        + Investigation pre-registration: outline expectations and why, then run test to backup the expectations 

+ `ttest_ind` method
    + Syntax: `ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')`
    + Calculates the T-test for the means of two independent samples of scores
    + Parameters: 
        + `a`, `b`: array_like; The arrays must have the same shape, except in the dimension corresponding to `axis` (the first, by default).
        + `axis`: int or None; Axis along which to compute test. If None, compute over the whole arrays, `a`, and `b`.
        + `equal_var`: bool; If True (default), perform a standard independent 2 sample test that assumes equal population variances. If False, perform Welch's t-test, which does not assume equal population variance.
        + `nan_policy`: {'propagate', 'raise', 'omit'}; Defines how to handle when input contains nan. 'propagate' returns nan, 'raise' throws an error, 'omit' performs the calculations ignoring nan values. Default is 'propagate'.

+ Demo 
    ```Python
    df = pd.read_csv('grades.csv')
    len(df)                         # number of rows

    early = df[df['assignment1_submission'] <= '2015-12-31']
    late = df[df['assignment1_submission'] > '2015-12-31']
    early.mean()
    late.mean()

    from scipy import stats
 
    stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade'])
    # Ttest_indResult(statistic=1.400549944897566, pvalue=0.16148283016060577)
    stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade'])
    # Ttest_indResult(statistic=1.3239868220912567, pvalue=0.18563824610067967)
    stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade'])
    # Ttest_indResult(statistic=1.7116160037010733, pvalue=0.087101516341556676)
    ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video.jpg =120x)](https://d3c33hcgiwev3.cloudfront.net/cP0Jy5mFEeazEQqJcB_rAg.processed/full/540p/index.mp4?Expires=1526169600&Signature=ERCUyHaMeGn1OVylK68SzuWz7gJRQEtScTwOhR7Lx9ichn4VWj4ljuMOw~n5aYL18eX5LLlaz6b6sGwbFpOt~LIOF4CUfT-LC2aP4jFfD1cRFXhdfY0LnsZIWseXwjbO2Y7vaYCEYKq97elKV~aIlrByQFw8e5Es0gk7nY8pFj0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## End of Theory

This course uses a third-party tool, End of Theory, to enhance your learning experience. The tool will reference basic information like your name, email, and Coursera ID.

[Open tool](https://nzh13lxjj0.execute-api.us-east-1.amazonaws.com/prod/response/4/new/)

In 2008 Chris Anderson wrote a provocative article in Wired magazine questioning whether science has moved into a new way of knowing about the world, where theory was no longer central to understanding and decisions, but data and algorithms alone could provide insights.

Read the [article](https://www.wired.com/2008/06/pb-theory/)

Do you think we have come to the end of theory? If so, what does this mean for scientific research and discover in theory-heavy fields such as the social sciences? If not, what does theory provide which data and algorithms alone cannot?

### My Own Opinion: Disagree

This is a dangerous proclaim.  The help from computational power and advances of machine learning algorithms make the era happened.  We can only analyze the correlation of the data without knowing underneath behavior.  However, a model simplifies the real world behavior by scarifying many far away factors.  It will provide people a less expensive and quicker way to find the solutions that do not need to know everything detailed.  

It’s like black and while box systems.  One is just to know the result while the other one need to know the how the system behavior.  Once we know the detail, then we can control and manipulate it.  Certainly, not every time we need to know all the details but the coarse results.  Then the approach for the Petabyte Ago kicks in to play the main role.  Just like models, there is no universal model to fit everything though.  Though a great model can explain many things, it might be too expensive too build and explore.  

IMHO, the traditional approach, hypothesize, model, and test, will coexist with the approach for the Petabyte Age, correlation.  One can help us to know the underneath world better while the other one tells us the expecting results without details. Both of them have their own value to help us knowing the real world.

### Review 1


### Review 2


### Review 3

## Discussion Prompt: Science Isn't Broken: p-hacking activity


