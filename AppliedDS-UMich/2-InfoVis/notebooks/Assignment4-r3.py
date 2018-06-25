
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **economic activity or measures** (see below) for the region of **Ann Arbor, Michigan, United States**, or **United States** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Ann Arbor, Michigan, United States** to Ann Arbor, USA. In that case at least one source file must be about **Ann Arbor, Michigan, United States**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Ann Arbor, Michigan, United States** and **economic activity or measures**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **economic activity or measures**?  For this category you might look at the inputs or outputs to the given economy, or major changes in the economy compared to other regions.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

import seaborn as sns

get_ipython().magic('matplotlib notebook')
plt.style.use('seaborn-colorblind')

df = pd.read_csv('data/C2A2_data/BinnedCsvs_d25/9bc594d0d6bf5fec16beb2afb02a3b859b7d804548c77d614b2a6b9b.csv')

df2 = pd.read_csv('https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv')

# df2

df = df.sort(['ID', 'Date'])

df2 = df2.set_index('Date')
df2 = df2[df2['Source'] == 'GCAG']
df2 = df2.reset_index()

df2['Year'] = df2['Date'].apply(lambda x: x[:4]) 
df2['Month'] = df2['Date'].apply(lambda x: x[-2:])

df2 = df2[df2['Year'] >= '2000']

# Use to store the avgrage of ocean tmp
avg_ocean_tmp = []

for i in range(1,10):
    tmp = df2[df2['Month'] == '0'+str(i)].mean()
    avg_ocean_tmp.append(tmp)

for i in range(10,13):
    tmp = df2[df2['Month'] == str(i)].mean()
    avg_ocean_tmp.append(tmp)

# plt.figure()
# plt.plot(avg_ocean_tmp)
# plt.show()


# df2.head()

# Pre-process the dataframe1
df['Year'] = df['Date'].apply(lambda x: x[:4])
df['Month-Day'] = df['Date'].apply(lambda x: x[5:])
df = df[df['Month-Day'] != '02-29']

# df['Month'] = df['Date'].apply(lambda x: x[5:7])

df_min = df[(df['Element'] == 'TMIN')]
df_max = df[(df['Element'] == 'TMAX')]

df_temp_min = df[(df['Element'] == 'TMIN')]
df_temp_max = df[(df['Element'] == 'TMAX')]

temp_min1 = df_temp_min.groupby('Month-Day')['Data_Value'].agg({'temp_min_mean_Beijing': np.mean})
temp_max1 = df_temp_max.groupby('Month-Day')['Data_Value'].agg({'temp_max_mean_Beijing': np.mean})


# Reset Index
temp_min1 = temp_min1.reset_index()
temp_max1 = temp_max1.reset_index()


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(temp_min1['temp_min_mean_Beijing']/10, 'y', alpha = 0.75, label = 'Average Low Temperature in Beijing')
ax1.plot(temp_max1['temp_max_mean_Beijing']/10, 'r', alpha = 0.5, label = 'Average High Temperature in Beijing')


a = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
b = [i+15 for i in a]

# data_m=np.array([1,2,3,4])   #(Means of your data)
# data_df=np.array([5,6,7,8])   #(Degree-of-freedoms of your data)
# data_sd=np.array([11,12,12,14])   #(Standard Deviations of your data)
# plt.errorbar([0,1,2,3], data_m, yerr=ss.t.ppf(0.95, data_df)*data_sd)


ax2.plot(b, avg_ocean_tmp, 'g', label = 'Global Ocean Surface Temperature')
# ax2.errorbar(list(range(1,13)), avg_ocean_tmp, yerr=ss.t.ppf(0.95, [1]*12)*[1]*12)

ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature in Beijing (℃)')
ax2.set_ylabel('Global Ocean Surface Temperature (℃)')
plt.title('Temperature in Beijing against \nGlobal Ocean Surface Temperature in each month')

ax1.legend()
ax2.legend()

Month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(b, Month_name)

for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)

plt.show()


# In[ ]:



