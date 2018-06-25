# Assignment 2 - Plotting Weather Patterns

## Plotting Weather Patterns

A detailed description is found in the Jupyter notebook. This assignment attempts to use your current geographical location to personalize the data.

[Launch Web Page](https://www.coursera.org/learn/python-plotting/notebook/84M6u/plotting-weather-patterns)

[Web Assignment Notebook](https://hub.coursera-notebooks.org/user/pkfpwscjcemdtitwkaxuvv/notebooks/Assignment2.ipynb#)

[Local Assignment Notebook](./notebooks/Assignment02.ipynb)

## Peer-graded Assignment: Plotting Weather Patterns

### Instructions

For this assignment, you will work with real world CSV weather data. You will manipulate the data to display the minimum and maximum temperature for a range of dates and demonstrate that you know how to create a line graph using matplotlib. Additionally, you will demonstrate procedure of composite charts, by overlaying a scatter plot of record breaking data for a given year.

Note: If you want more anonymity, only include the country of where the data is from rather than the specific region or city.

Download the attachment for a preview of how the assignment will be graded.


[assignment2_rubric.pdf](https://d3c33hcgiwev3.cloudfront.net/_5491561792e942ea08a955f4a18961e8_assignment2_rubric.pdf?Expires=1528502400&Signature=cHpOlRjhIDO5-ycTZFcyDJXl2owulS3SU61PbuOXqbzkJ7rKny4Vurg4mbiI07M-biJDLSbaQ3PjqkVjWfVJL4H13NvtMr~fYVw8HnZ41qlPQ~L8aJB~ukI1pItM8XGaNkkdVZpVsPL5hicWrTXA9lfhPeivcEvjIBe3V9aDstQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

#### Requirements

Each row in the assignment datafile corresponds to a single observation.

The following variables are provided to you:

* **id** : station identification code
* **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
* **element** : indicator of element type
    * TMAX : Maximum temperature (tenths of degrees C)
    * TMIN : Minimum temperature (tenths of degrees C)
* **value** : data value for element (tenths of degrees C)

For this assignment, you must:

1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.


#### Review criteria

Each component of the assignment will be graded by a peer-reviewer, using a detailed rubric.

+ Did the learner upload an image of their record highs and lows plot? Note: Not everyone received the same data, so your plot may look different from your peer’s assignment.
    + 0 pts: No, the learner did not upload an image of their record highs and lows plot.
    + 1 pt: Yes, the learner uploaded an image of their record highs and lows plot.
+ Does the plot include an accurate title?
    + 0 pts: No, the plot does not include an accurate title.
    + 1 pt: Yes, the plot does include an accurate title.
+ Does the line graph display record highs and lows for 2005-2014?
    + 0 pts: No, the line graph does not display record highs and lows for 2005-2014.
    + 3 pts: Yes, the line graph does display record highs and lows for 2005-2014.
+ Is the area between the two line graphs shaded?
    + 0 pts: No, the area between the two lines is not shaded.
    + 3 pts: Yes, the area between the two lines is shaded.
+ Does the overlaid scatter plot indicate days in 2015 that broke a record high or low for 2005-2014?
    + 0 pts: No, the overlaid scatter plot does not indicate days in 2015 that broke a record high or low for 2005-2014.
    + 3 pts: Yes, the overlaid scatter plot indicates days in 2015 that broke a record high or low for 2005-2014.
+ Is there a legend or sufficient labelling for the line graph and scatter plot?
    + 0 pts: No, there is not a legend or sufficient labelling for the line graph and scatter plot.
    + 2 pts: Yes, there is a legend or sufficient labelling for the line graph and scatter plot.
+ Describe how your visual leverages the guidelines of effective visual design outlined in module one of this course.
+ The visual leverages the guidelines of effective visual design outlined module one of this course, specifically, Cairo's principles of beauty, truthfulness, functionality, and insightfulness.
    + 0 pts: Disagree. The visual does not leverage any of the guidelines given for effective visual design (e.g., beauty, truthfulness, functionality, and insightfulness).
    + 1 pt: Neutral. The visual leverages only one or two guidelines given for effective visual design (e.g., beauty, truthfulness, functionality, and insightfulness).
        + 2 pts: Agree. The visual effectively leverages the guidelines given for effective visual design (e.g., beauty, truthfulness, functionality, and insightfulness).
+ Based on your response to the previous question, provide comments about one aspect of the plot that was especially effective and provide suggestions about how one aspect of the plot that could be improved.
+ If you want to look at the learner's code, we recommend that you open it through the Jupyter notebook system on the Coursera platform.


### My submission

+ Upload an image of your record highs and lows plot. Ensure that your plot includes the following elements:
    + an accurate title
    + correctly labelled axes
    + line graph displaying record highs and lows for 2005-2014
    + shaded area between the two lines
    + overlaid scatter plot indicates days in 2015 that broke a record high or low for 2005-2014
    + a legend or sufficient labelling for the line graph and scatter plot

    ![Diagram](./notebooks/temperature.png)

+ Describe how your visual leverages the guidelines of effective visual design outlined in module one of this course.

    Truthful: 
        1. The diagram represents all the temperature fo all stations.  
        2. The diagram is based on the maximum and minimum temperature for all station.  
        3. The TMAX and TMIN temperature of all station by each day are within the shaded area.

    Functionality: 
        1. The diagram covers all the measured temperatures of all stations with 10 year period.
        2. It represents the temperature range of the 10 year period.
        3. Temperatures broke the past 10 years are pointed out.

    Beautiful:
        1. The diagram illustrates the temperature range of each day for general public.
        2. It is easily interpreted by the public general.
        3. Legend of lines and dots are provide for audience to interpret the diagram.
        4. Extreme temperatures are using balder colors while the range is shaded with lighter color.  
        5. The extreme temperatures for 2015 are illustrated with vivid color dots for attraction.

    Insightful: 
        1. The rang of the 10 year temperature provides the general trends of maximum and minimum temperatures.
        2. The trend of temperature variation with each month can be interpreted from the diagram easily.
        3. The extreme temperature of 2015 indicates that the beginning of the year is colder that before while the end of the year is hotter than before.

    Enlightening: 
        The main purpose of the diagram provides an evidence that year 2015 is with more dramatic climate change than the past 10 years.

 + Please upload your source code.

    [Notebook](./notebooks/Assignment02.ipynb)

## Discussion Forum Links for Assignment

+ [Assignment 2 Basic Example](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/7Rh0RAQ1Eee-uQ7R_UJsXg)

    To help you with assignment 2, here is a basic example of the visualization you need to create.

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/p_2Q3gQ1Eee1_w6aJKt_EA_0605a3e0b4e950fe061e1dab526a26ea_WhiteboardExample.jpg?expiry=1530057600000&hmac=xy_2DpWNmNl_-aI-omYr1mx4kgkDlCPh-DWqSIjAqCc" alt="Basic Example" width="450">

    The number of data points in each of your min/max results should be 365

    For the data having 2005-2014, you need to group the data by the day-month parts of the Date regardless of the year part then find the max and min, temperatures in each group.

    The result should look more or less similar the plot in the original post

    You may want to review [Group By](../1-IntroDS/03-AdvPandas.md#group-by) and [Date Functionality](../1-IntroDS/03-AdvPandas.md#date-functionality) lectures in week3 of course 1

+ [Unable to find the .csv file](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/5Zuz5k57EeeEgg7HiAhd5A)

    the assignment require the use of the files provided for you.

    All the data files are in a folder named data, I suggest browsing to the data file and downloading it that way.

    here is a step by step of what you need to do

    1. go to the assignment page, i.e https://www.coursera.org/learn/python-plotting/notebook/84M6u/plotting-weather-patterns, click on open notebook

    2. when in the notebook page, you'll see the name of the file you want to download in the beginning of the second paragraph as shown below

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/DPVOaqmaEeehSg7dYzt83A_fa36ab1002bbb77b7bb5f62b3b3ef159_Screen-Shot-2017-10-05-at-07.54.10.png?expiry=1530057600000&hmac=zPJ4moQvorhZt2bqTTXo4nP0vnTCj86mIAKvaL7gu_A" alt="Notebook Page" width="600">


    3. From the notebook main menu click File-> open

    4. now you'll see the home folder and the folder data. click on data then C2A2_data then BinnedCsvs_d400

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/3umEeqmbEeeWchJSuVvCkA_e204a0d270f063634936a2bc97c2b406_getcsvfile.png?expiry=1530057600000&hmac=O1kFGUkKIGg6JfCddHrkfcfoCSpQDZoJQ1iBUV8hlac" alt="Open Fil Screens" width="600">

    5. The files list will take sometime but once loaded, the files are sorted in alphabetical order

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/CH7PoamcEeehSg7dYzt83A_085312bc384ef4cb78964bdecb41bce7_Screen-Shot-2017-10-05-at-08.03.22.png?expiry=1530057600000&hmac=9FzQEUz0YEBdVZB0EPL4E-HLQF0GWHRZrCoJJ2ZwtOI" alt="CVS File List" width="600">

    6. browse down to the file name then click to open it. once the file is open you can click File->Download to download it


    I hope this helps and Good luck

+ [The Assignment is empty](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/TdDXsI2iEeeeMgqIrQjmng)

    You will need to delete this notebook by clicking on File-> Open,

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/JaTvYY4LEee5BApqFAWhtA_d1718241280e11c4e49d156f35be9fbc_Screen-Shot-2017-08-31-at-06.11.18.png?expiry=1530057600000&hmac=Ny8ZhyhlXM7q2yCgJjQoV6ZnklEZSlAw4cEWoHhB04M" alt="Assignment Page" width="600">

    then select the file name "Assignment 2.ipynb", click delete, then click on the control panel,

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/D_P_-o4MEeeTRApMRfIQUA_d1a40a5e23a687afd454bcc8a64b9088_remove-notebook.png?expiry=1530057600000&hmac=C3yoOneFGCTSqIMUeRKggAONV5Wo5Mw64yBMHul2bvw" alt="Deletion Instruction" width="600">

    then in the control panel, click stop My server, start server

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/LZfjSY4MEee7zxJeG4wOTA_5da2a9a7679634a8bad0bc87fa2eb66f_control-panel.png?expiry=1530057600000&hmac=0BZcusv4UEtLCqYs5Q0cdYNldKTS1XyjAtatq30uKqg" alt="Control Panel Instruction" width="600">

    You may be asked to re-login, do so and access the assignment the usual way, if the assignment does not appear, click on the control panel again and stop/start server

    in the end a fresh copy of the notebook "Assignment 2.ipynb" will be automatically generated

+ [Plot not showing in notebook!](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/mx5ALTUxEee59xK-kx4BfA)

    The default backend doesn't support interactivity, If you don't change the backend to an interactive one (%matplotlib notebook) the graph will not show unless you use plt.show()

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/qMSrjjVLEeeeXw6DN6nFWg_801103f0934e29769bf656c738db0834_Screen-Shot-2017-05-10-at-07.40.55.png?expiry=1528761600000&hmac=UzWIEseiu8QA5EF_b3Cw3uq0OgG5trsfSAY8--4kzY4" alt="text" width="400">


## Review Your Peers: Plotting Weather Patterns

### Rewview 1

I have made sure my visual is truthful by not cutting off any part of the axes that corresponds to the data points. I have used opposing colors for my scatter plot to distinguish them from the line plot to keep the visual functional. I also have a legend defining the lines and points in the visual which is insightful. I have avoided using grid lines and only used month names on the x axis (versus all days) to keep the visual beautiful and functional at the same time.

[Notebook](./notebooks/Assignment2-r1.ipynb)


### Review 2

Describe how your visual leverages the guidelines of effective visual design outlined in module one of this course.

I just tried to not overload the visual with details to make it simpler but clear and still looking nice.

[Notebook](./notebooks/Assignment2-r2.ipynb)


### Review 3

My plot is truthful to the data and have no intention of misleading the users with hiding information, or flood user with too much data to obscure reality, nor distorting the data using visual forms. I used the different colors to catch users' attention of the broken records, as well as 10 years ranges. The plot contains clear labels and ticks for easy understanding.

[Notebook](./notebooks/Assignment2-r3.ipynb)



### Review 4

I have tried to use the concepts descrived by Tufte. He proposes the data-ink ratio, where he argues that all ink that is not used to present data should be removed. The data-ink ratio can be calculated by dividing the ink used for displaying data (data-ink) by the total ink used in the graphic, with the goal of having the ratio as close to 1 as possible. Thus, the greatest amount of unnecessary information was eliminated, such as the top and right spines, among others.

Also, take into account the mechanisms for misleading that Cairo outlines in his book chapter: hiding relevant data, displaying too much data and obscuring reality, and distorting data through graphic forms.


[Notebook](./notebooks/Assignment2-r4.ipynb)




