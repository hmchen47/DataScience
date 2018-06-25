%matplotlib notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import requests
from pandas.io.json import json_normalize



data1 = json.loads(requests.get('https://query.data.world/s/txibp3fq7i4msjfrgucwm6t4ebgppg').text)
df1 = json_normalize(data1["results"])
df1 = df1.set_index("year")


data2 = json.loads(requests.get('https://query.data.world/s/vqdrsdr7a5tqlsdipojrp2qvdurkaq').text)
df2 = json_normalize(data2["results"])
df2 = df2.set_index("year")

plt.figure()

                               
df1 = df1[df1["hazard_sub_type"] == "Flood"]
displaced = df1.groupby(df1.index)['new_displacements'].sum().reset_index()
plt.ylabel('Displaced Persons', alpha=0.8)
plt.title('Displaced Persons by floods vs. Total Displaced Persons in the last decade in USA')
# add a legend with legend entries (because we didn't have labels when we plotted the data series)


plt.plot(displaced["year"], displaced["new_displacements"] ,"-o", label="Persons Displaced by floods")
plt.plot(df2.index, df2["disaster_new_displacements"], "-o", label="Total Persons Displaced in the whole year")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.legend(frameon=False, framealpha=0.5)

my_xticks = np.array(df1.index)
frequency = 9
plt.xticks(alpha=0.8)

plt.gca().fill_between(df2.index , 
                   displaced["new_displacements"], df2["disaster_new_displacements"], 
                   facecolor='blue', 
                   alpha=0.25)
