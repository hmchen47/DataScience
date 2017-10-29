import numpy as np
import matplotlib.pyplot as plt

y_val = [.25, .33, .25, .125, .4]

xlabels = ['under 20', '20-24', '25-29', '30-34', '35 and over']

width = .25

ypos = np.arange(len(xlabel))
plt.subplot(211)
plt.xlabel('Age')
plt.ylabel('Rate')
plt.xticks(ypos, xlabels)
plt.title('First Child at Ages')
plt.bar(ypos, y_val, align='center', alpha=0.5)


plt.subplot(212)
plt.pie(
    y_val,
    labels=xlabels,
    autopct='%1.1f%%'
    )

plt.show()

