# Lab 5: Bootstrap

+ [Launch Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/95ee24be1f714d51bb48d73712c71aba/95f9b7a0beef469b9d2a36685acbac99/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%4036c8c64aef664dbcb437251449ac5b58)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/59d217c894d11dbd21d2d37ef6ae9675/notebooks/materials-x18/materials/x18/lab/2/lab05/lab05.ipynb)
+ [Local Notebook](./notebooks/lab05.ipynb)

Lab 5 will be due at the end of the course (June 27th 11:59 UTC), but we strongly suggest that you finish by Sunday, June 24. Start Lab 5 by selecting the "Launch Lab 5" button below. Please follow the instructions in the Jupyter notebook to complete Lab 5.

Lab grades may take up to a couple hours to update in your edX progress page.

If you cannot complete the Jupyter notebook version of Lab 5 for accessibility reasons, please send an email to wiltonwu@berkeley.edu with your accessibility needs. We will be providing you with an accessible alternative in order to complete the lab.

Note: IE / Edge browsers are currently unsupported for Jupyter notebook labs. We highly recommend you use Google Chrome. 

Answer:

Q1.1: 
1. N: population parameter
2. n: statistic, the computed number from random sample that's estimate of N

Q1.2 
```python
def plot_serial_numbers(numbers):
    numbers.hist(bins=200)
    
    # Assuming the lines above produce a histogram, this next
    # line may make your histograms look nicer.  Feel free to
    # delete it if you want.
    plt.ylim(0, .25)

plot_serial_numbers(observations)
```

Q1.3
1. We cannot tell the N immediately, but the samples indicates the most likely number of N is no less than 130.
2. The bars in the diagram represent the pobability of the samples located in the given range.
3. Due to the bins assigned as 200 and the samples only 17 without repeat.  Therefore, all the bins are only one occurence inside.  

Q1.4:
```python
def mean_based_estimator(nums):
    return np.mean(nums)

mean_based_estimate = mean_based_estimator(observations)
mean_based_estimate
```

Q1.5:
```python
max_estimate = np.max(observations)[0][0]
max_estimate
```

Q1.6:
1. The result of max_estimate is never greater than N.
2. The result (twice) of mean_based_estimate could be less or greater than N or equal to N.  It highly depends on the samples.
3. We cannot make any statement that the result (twice) of mean_based_estimate is at least x away from N.  The variance of distance from (N/2) depends on the samples.


