from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm

df = pd.DataFrame([np.random.normal(33500,150000,3650), 
                   np.random.normal(41000,90000,3650), 
                   np.random.normal(41000,120000,3650), 
                   np.random.normal(48000,55000,3650)], 
                  index=[1992,1993,1994,1995])

avg_year = df.mean(axis = 1)
yerr = df.std(axis=1) / np.sqrt(df.shape[1])

plt.figure()
plt.show()
first_plot = plt.bar(range(df.shape[0]), avg_year, yerr = yerr, color = 'grey')
fig = plt.gcf()

threshold=42000
plt.axhline(y = threshold, color = 'grey')

cm1 = mcol.LinearSegmentedColormap.from_list("CustomPalette",["blue", "white", "red"])
cpick = cm.ScalarMappable(cmap=cm1)
cpick.set_array([])

percentages = []
for bar, yerr_ in zip(first_plot, yerr):
    low = bar.get_height() - yerr_
    high = bar.get_height() + yerr_
    percentage = (high-threshold)/(high-low)
    if percentage>1: percentage = 1
    if percentage<0: percentage=0
    percentages.append(percentage)

bars = plt.bar(range(df.shape[0]), avg_year, yerr = yerr, color = cpick.to_rgba(percentages))
plt.colorbar(cpick, orientation='vertical')
plt.xticks(range(df.shape[0]), df.index)