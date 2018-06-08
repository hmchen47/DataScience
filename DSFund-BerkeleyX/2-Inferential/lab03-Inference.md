# Lab 3: Inference, Part 1

Lab 3 will be due at the end of the course (June 27th 11:59 UTC), but we strongly suggest that you finish by Sunday, June 10. Start Lab 3 by selecting the "Launch Lab 3" button below. Please follow the instructions in the Jupyter notebook to complete Lab 3.


Lab grades may take up to a couple hours to update in your edX progress page.

If you cannot complete the Jupyter notebook version of Lab 3 for accessibility reasons, please send an email to wiltonwu@berkeley.edu Opens in new window with your accessibility needs. We will be providing you with an accessible alternative in order to complete the lab.

Note: IE / Edge browsers are currently unsupported for Jupyter notebook labs. We highly recommend you use Google Chrome. 

[Web Notebook](https://hub.data8x.berkeley.edu/user/59d217c894d11dbd21d2d37ef6ae9675/git-pull?repo=git://reposync/materials-x18&subPath=materials/x18/lab/2/lab03/lab03.ipynb)

[Local Notebook](./labs/lab03.ipynb)

## Discussion Forum

### Question 1.1.

What additional information will we need before we can check for that association? Assign extra_info to a Python list (i.e. [#] or [#, #, ...]) containing the number(s) for all of the additional facts below that we require in order to check for association.

1) What year(s) the death penalty was introduced in each state (if any).

2) Day to day data about when murders occurred.

3) What year(s) the death penalty was abolished in each state (if any).

4) Rates of other crimes in each state.

+ [Instructor Hit](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/0eeac50995794429b04ca715f4effd91/a3362f0326cd40a4b2c7284f9618db3f/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%4027d6ac90b9f64406adc4fff2adb1d29e)

>Although data on crimes other than murder are included, in the *crimes_rate.csv* file the information on whether a state had capital punishment is not. Also missing is daily data. Reviewing the Table object cited casts doubt on whether the attribution of the data to the paper cited to the paper by Dezhbakhsh, Rubin, and Shepherd is accurate ("We use a panel data set that covers 3,054 counties for the 1977-1996 period", at p. 16). Only one reference to 1960 is made, and that is in a citation, and the paper itself was published in October 2003, three month before the stated end-point of the data series. Similarly, the only reference to daily data was in a citation to another paper. The full version of the paper at https://goo.gl/WCRSzP Table 2, death penalty status, give information only as of December 31, 2000; providing no information for the remaining years except the number of executions between 1977 and 2000 with the total nationwide number of executions and the total number of death penalty states.
>
> Although the binary variable by state and year for death penalty and the "day-to-day" (time series or just average daily?) might be needed in Part 2 of the lab, none of the following questions appear to call for their use.
> 
>  we can calculate an average daily rate, we have no way of producing a time series showing for each year and day by state whether a murder was committed. As with the existence or not of the death penalty, that information can be researched, but given the possible multiplicity of data sources and coverage, it's hard to see how the grader will be able to assess the answer.
>
> Please either (1) provide the missing data, (2) clarify whether a time series or daily average is requested, or (3) eliminate from the grader any test for the existence of the penalty or day-specific data.

+ wiltonwu

> Your list should contain your choices for what additional information will we need before we can check for that association. For example, if you believe that we need options 1 and 3 from the list above, then you should assign extra_info = [1, 3]


