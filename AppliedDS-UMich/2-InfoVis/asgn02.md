# Assignment 2 - Plotting Weather Patterns

## Plotting Weather Patterns

A detailed description is found in the Jupyter notebook. This assignment attempts to use your current geographical location to personalize the data.

[Launch Page](https://www.coursera.org/learn/python-plotting/notebook/84M6u/plotting-weather-patterns)

[Web Notebook](https://hub.coursera-notebooks.org/user/pkfpwscjcemdtitwkaxuvv/notebooks/Assignment2.ipynb#)

[Local Notebook](./notebooks/Assignment02.ipynb)

## Peer-graded Assignment: Plotting Weather Patterns

### Instructions

For this assignment, you will work with real world CSV weather data. You will manipulate the data to display the minimum and maximum temperature for a range of dates and demonstrate that you know how to create a line graph using matplotlib. Additionally, you will demonstrate procedure of composite charts, by overlaying a scatter plot of record breaking data for a given year.

Note: If you want more anonymity, only include the country of where the data is from rather than the specific region or city.

Download the attachment for a preview of how the assignment will be graded.


[assignment2_rubric.pdf](https://d3c33hcgiwev3.cloudfront.net/_5491561792e942ea08a955f4a18961e8_assignment2_rubric.pdf?Expires=1528502400&Signature=cHpOlRjhIDO5-ycTZFcyDJXl2owulS3SU61PbuOXqbzkJ7rKny4Vurg4mbiI07M-biJDLSbaQ3PjqkVjWfVJL4H13NvtMr~fYVw8HnZ41qlPQ~L8aJB~ukI1pItM8XGaNkkdVZpVsPL5hicWrTXA9lfhPeivcEvjIBe3V9aDstQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

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

Upload an image of your record highs and lows plot. Ensure that your plot includes the following elements:

+ an accurate title
+ correctly labelled axes
+ line graph displaying record highs and lows for 2005-2014
+ shaded area between the two lines
+ overlaid scatter plot indicates days in 2015 that broke a record high or low for 2005-2014
+ a legend or sufficient labelling for the line graph and scatter plot


## Useful Links for Assignment

+ [Assignment 2 Basic Example](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/7Rh0RAQ1Eee-uQ7R_UJsXg)

    To help you with assignment 2, here is a basic example of the visualization you need to create.

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/p_2Q3gQ1Eee1_w6aJKt_EA_0605a3e0b4e950fe061e1dab526a26ea_WhiteboardExample.jpg?expiry=1528761600000&hmac=OMJh4Z2lI0GLgk0NJRgcM_4pxKWoA_NKLyIYhcbQVFc" alt="text" width="450">

    The number of data points in each of your min/max results should be 365

    For the data having 2005-2014, you need to group the data by the day-month parts of the Date regardless of the year part then find the max and min, temperatures in each group.

    The result should look more or less similar the plot in the original post

    You may want to review Group By and Date Functionality lectures in week3 of course 1

+ [Unable to find the .csv file](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/5Zuz5k57EeeEgg7HiAhd5A)

    the assignment require the use of the files provided for you.

    All the data files are in a folder named data, I suggest browsing to the data file and downloading it that way.

    here is a step by step of what you need to do

    1. go to the assignment page, i.e https://www.coursera.org/learn/python-plotting/notebook/84M6u/plotting-weather-patterns, click on open notebook

    2. when in the notebook page, you'll see the name of the file you want to download in the beginning of the second paragraph as shown below

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/DPVOaqmaEeehSg7dYzt83A_fa36ab1002bbb77b7bb5f62b3b3ef159_Screen-Shot-2017-10-05-at-07.54.10.png?expiry=1528761600000&hmac=Xob_pPkIuwF2m3F9qvpv9a0F0OsHr5vxbDX-ioaB1pg" alt="Notebook Page" width="600">


    3. From the notebook main menu click File-> open

    4. now you'll see the home folder and the folder data. click on data then C2A2_data then BinnedCsvs_d400

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/3umEeqmbEeeWchJSuVvCkA_e204a0d270f063634936a2bc97c2b406_getcsvfile.png?expiry=1528761600000&hmac=ywUwDsBXSNEjpQgbMCIM3AjkvYhmJj_ybtXPM6MMgw4" alt="Open Fil Screens" width="600">

    5. The files list will take sometime but once loaded, the files are sorted in alphabetical order

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/CH7PoamcEeehSg7dYzt83A_085312bc384ef4cb78964bdecb41bce7_Screen-Shot-2017-10-05-at-08.03.22.png?expiry=1528761600000&hmac=uZ9WK_rtMqR6DEd-NwowvIMhROREBC7S1lJwaApu9uo" alt="CVS File List" width="600">

    6. browse down to the file name then click to open it. once the file is open you can click File->Download to download it


    I hope this helps and Good luck

+ [The Assignment is empty](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/TdDXsI2iEeeeMgqIrQjmng)

    You will need to delete this notebook by clicking on File-> Open,

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/JaTvYY4LEee5BApqFAWhtA_d1718241280e11c4e49d156f35be9fbc_Screen-Shot-2017-08-31-at-06.11.18.png?expiry=1528761600000&hmac=TAGgnPiom_tNEdu5E5a13RJBKof8wW6E9NJdRsB0TX4" alt="Assignment Page" width="600">

    then select the file name "Assignment 2.ipynb", click delete, then click on the control panel,

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/JaTvYY4LEee5BApqFAWhtA_d1718241280e11c4e49d156f35be9fbc_Screen-Shot-2017-08-31-at-06.11.18.png?expiry=1528761600000&hmac=TAGgnPiom_tNEdu5E5a13RJBKof8wW6E9NJdRsB0TX4" alt="Deletion Instruction" width="600">

    then in the control panel, click stop My server, start server

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/D_P_-o4MEeeTRApMRfIQUA_d1a40a5e23a687afd454bcc8a64b9088_remove-notebook.png?expiry=1528761600000&hmac=hCfKaU-psixuvO4BHBDzxzFNv-t7ZMnfGFjeFtDsELk" alt="Control Panel Instruction" width="600">

    You may be asked to re-login, do so and access the assignment the usual way, if the assignment does not appear, click on the control panel again and stop/start server

    in the end a fresh copy of the notebook "Assignment 2.ipynb" will be automatically generated

+ [Plot not showing in notebook!](https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/mx5ALTUxEee59xK-kx4BfA)

    The default backend doesn't support interactivity, If you don't change the backend to an interactive one (%matplotlib notebook) the graph will not show unless you use plt.show()

    <br/><img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/qMSrjjVLEeeeXw6DN6nFWg_801103f0934e29769bf656c738db0834_Screen-Shot-2017-05-10-at-07.40.55.png?expiry=1528761600000&hmac=UzWIEseiu8QA5EF_b3Cw3uq0OgG5trsfSAY8--4kzY4" alt="text" width="400">


## Review Your Peers: Plotting Weather Patterns





