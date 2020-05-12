# Python Visualization

## MatPlotLib

### [Official Pyplot API](https://matplotlib.org/api/pyplot_summary.html)

### Environment and Module

```python
%matplotlib notebook                                    # provides an interactive environment in Jupyter and IPuthon

import matplotlib as mpl                                # load module in CLI
import matplotlib.pyplot as plt                         # load pyplot module
import matplotlib.gridspec as gridspec
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
```

### Classes

| Method | Description | Link |
|--------|-------------|------|
| `mpl.axes.Axes` | contain most of the figure elements: Axis, Tick, Line2D, Text, Polygon, etc., and sets the coordinate system. | [Axes][030] |
| `plt.gca().xaxis` & `plt.gca().yaxis` | xaxis = class XAxis(Axis), yaxis = class YAxis(Axis) | [Line Plots][027] |
| `gridspec.GridSpec` | specifies the geometry of the grid that a subplot will be placed | [Histograms][038] |

### Official Docs

+ [The Matplotlib API][032]
  + [axis and tick API][030]
  + [PyPlot API][031]
  + [Colors in Matplotlib][033]
  + [Figure][034]
  + [Subplot Parameters][035]
  + [text][036]


[TOC](#table-of-contents)

### Methods

| Method | Description | Link |
|--------|-------------|------|
| `mpl.get_backend()` | Return the name of the current backend | [Basic Plotting][025] |
| `plt.plot(*args, **kwargs)` | Plot lines and/or markers to the Plot lines and/or markers to the `~matplotlib.axes.Axes` class; <br/> __`kwargs`__: agg_filter, alpha, animated, antialiased, axes, clip_box, clip_on, clip_path, color/c, contains, dash_capstyle, dash_joinstyle, dashes, drawstyle, figure, fillstyle, gid, label, linestyle, linewidth, marker, markeredgecolor, markeredgewidth, markerfacecolor, markerfacwidthmarkersize, markevery, path_effects, picker, pickradius, rasterized, sketch_params, snap, solid_capstyle, solid_joinstyle, transform, url, visible, xdata, ydata, zorder | [Basic Plotting][025], [Line Plots][027] |
| `mpl.figure.Figure(*args)` | The Figure instance supports callbacks through a _callbacks_ attribute which is a `matplotlib.cbook.CallbackRegistry` class instance; `args`: figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None  | [Basic Plotting][025] |
| `mpl.backends.backend_agg. FigureCanvasAgg(figure)` | The canvas the figure renders into | [Basic Plotting][025] |
| `fig.add_subplot(*args, **kwargs)` | Add a subplot; <br/> __`kwargs`__: adjustable, agg_filter, alpha, anchor, animated, aspect, autoscale_on, autoscalex_on, autoscaley_on, axes, axes_locator, axisbelow, clip_box, clip_on, clip_path, color_cycle, contains, facecolor, fc, figure, frame_on, gid, label, navigate, navigate_mode, path_effects, picker, position, rasterization_zorder, rasterized, sketch_params, snap, title, transform, url, visible, xbound, xlabel, xlim, xmargin, xscale, xticklabels, xticks, ybound, ylabel, ylim, ymargin, yscale, yticklabels, yticks, zorder | [Basic Plotting][025] |
| `subplots(nrows=1, ncols=1, *args, **fig_kw)` | Create a figure and a set of subplots <br/> `*args`: `sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None` <br/> Returns: <br/> + `fig` : `matplotlib.figure.Figure` object <br/> + `ax` (Axes object or array of Axes objects): ax can be either a single `matplotlib.axes.Axes` object or an array of Axes objects if more than one subplot was created. | [Subplots][037] |
| `plt.figure(*args, **kwargs)` | Creates a new figure; `args`: figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None | [Basic Plotting][025]; [Subplots][037] |
| `plt.gca(**kwargs)` | Docstring: Get the current `~matplotlib.axes.Axes` instance on the current figure matching the given keyword `args`, or create one.  | [Basic Plotting][025] |
| `plt.gca().axis(*v, **kwargs)` <br/> `plt.gca().axes(*v, **kwargs)` | Get the current `~matplotlib.axes.Axes` instance on the current figure matching the given keyword `args`, or create one. <br/> __`kwargs`__: adjustable, agg_filter, alpha, anchor, animated, aspect, autoscale_on, autoscalex_on, autoscaley_on, axes, axes_locator, axisbelow, clip_box, clip_on, clip_path, color_cycle, contains, facecolor, fc, figure, frame_on, gid, label, navigate, navigate_mode, path_effects, picker, position, rasterization_zorder, rasterized, sketch_params, snap, title, transform, url, visible, xbound, xlabel, xlim, xmargin, xscale, xticklabels, xticks, ybound, ylabel, ylim, ymargin, yscale, yticklabels, yticks, zorder | [Basic Plotting][025] |
| `plt.gca().get_children()` | return a list of child artists | [Basic Plotting][025] |
| `plt.scatter(x, y, *args, **kwargs)` | Make a Scatterplots of `x` vs `y`; `args`: s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None | [Scatterplots][026], [Histograms][038] |
| `plt.xlabel(s, *args, **kwargs)` | Set the `x` axis label of the current axis | [Scatterplots][026] |
| `plt.ylabel(s, *args, **kwargs)` | Set the `y` axis label of the current axis| [Scatterplots][026] |
| `plt.fill_between(x, y1, y2=0, **kwargs)` | Make filled polygons between two curves; <br/> `kwargs`: where=None, interpolate=False, step=None, *, data=None | [Line Plots][027] |
| `plt.bar(left, height, **kwargs)` <br/> `plt.barh(left, height, **kwargs)` | Make a bar plot with rectangles bounded by: `left`, `left` + `width`, `bottom`, `bottom` + `height` (left, right, bottom and top edges) <br/> `kwargs`: width=0.8, bottom=None, hold=None, data=None,  | [Bar Charts][028] |
| `plt.tick_params(axis='both', **kwargs)` | Change the appearance of ticks and tick labels | [Dejunkify][029] |
| `plt.gcf()` | Get a reference to the current figure. | [Subplots][037] |
| `plt.hist(x, *args, **kwargs)` | Plot a histogram <br/> `*args`: `bins=None, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None` | [Histograms][038] |
| `set_title(label, fontdict=None, loc='center', **kwargs)` | Set a title for the axes of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `set_xlim(left=None, right=None, emit=True, auto=False, **kw)` | Set the data limits for the x-axis of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `set_ylim(bottom=None, top=None, emit=True, auto=False, **kw)` | Set the data limits for the y-axis of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `invert_axis()` | Invert the x-axis of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `plt.boxplot(x, *args)` | Make a box and whisker plot <br/> `args`: `notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_xticks=True, autorange=False, zorder=None, hold=None, data=None` | [Box Plots][039] |
| `inset_axes(parent_axes, width, height, *args)`| Create an inset axes with a given width and height of `mpl_toolkits.axes_grid1.inset_locator`.<br/> `args`: loc=1, bbox_to_anchor=None, bbox_transform=None, axes_class=None, axes_kwargs=None, borderpad=0.5 | [Box Plots][039] |
| `margins(*args, **kw)` | Set or retrieve autoscaling margins | [Box Plots][039] |
| `tick_right()`, `tick_left()` | use ticks only on right/left of `matplotlib.axis.YAxis` | [Box Plots][039] |
| `tick_top()`, `tick_bottom()` | use ticks only on top/bottom of `matplotlib.axis.xAxis`  | [Box Plots][039] |
| `plt.hist2d(x, y, *args, **kwargs)` | Make a 2D histogram plot <br/> `*args`: bins=10, range=None, normed=False, weights=None, cmin=None, cmax=None, hold=None, data=None | [Heatmaps][040] |
| `plt.colorbar(mappable=None, cax=None, ax=None, **kw)` | Add a colorbar to a plot | [Heatmaps][040] |
| `plt.cla()` | Clear the current axes | [Animation][041] |
| `annotate(s, xy, *args, **kwargs)` | Annotate the point `xy` with text `s`<br/> `args`: xytext=None, xycoords=None, textcoords =None, arrowprops=None, annotation_clip=None | [Animation][041] |
| `animation.FuncAnimation(fig, func, *args)` | Makes an animation by repeatedly calling a function `func` <br/> `args`: frames=None, init_func=None, fargs=None, save_count=0, interval=200, repeat_delay=None, repeat=True, blit=False | [Animation][041] |
| `mpl.connect(s, func)` | Connect event with string `s` to `func`.  The signature of `func` is `def func(event)` where event is a `matplotlib.backend_bases.Event` instance | [Interactivity][042] |
| `plterrorbar(x, y, *args, **kwargs)` | Plot an errorbar graph. Plot x versus y with error deltas in `yerr` and `xerr`. Vertical errorbars are plotted if `yerr` is not None. Horizontal errorbars are plotted if `xerr` is not None.<br/> `*args`: yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, hold=None, data=None | [Agmt 3][043] |
| `plt.colormaps()` | Matplotlib provides a number of colormaps, and others can be added using `~matplotlib.cm.register_cmap`.  This function documents the built-in colormaps, and will also return a list of all registered colormaps if called. | [Agmt 3][043] |
| `plt.imshow(X, *args, **kwargs)` | Display an image on the axes <br/> `*args`: cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, hold=None, data=None | [Agmt 3][043] |
| `cm.to_rgba(x, alpha=None, bytes=False, norm=True)` | Return a normalized rgba array corresponding to *x* | [Agmt 3][043] |
| `plt.twinx(ax=None)` | Make a second axes that shares the *x*-axis.  The new axes will overlay *ax* (or the current axes if *ax* is *None*).  The ticks for *ax2* will be  placed on the right, and the *ax2* instance is returned. | [Agmt 3][043] |
| `get_legend_handles_labels( legend_handler_map=None)` | Return handles and labels for legend | [Agmt 3][043] |
| `ax.set_zorder(level)` | Set the zorder for the artist.  Artists with lower zorder values are drawn first. | [Agmt 3][043] |
| `ax.set_visible(b)` | Set the artist's visiblity. | [Agmt 3][043] |
| `fig.tight_layout(renderer=None, pad=1.08, h_pad=None, w_pad=None, rect=None)` | Adjust subplot parameters to give specified padding. | [Agmt 3][043] |
| `plt.show(*args, **kw)` | Display a figure. | [Agmt 3][043] |
| `plt,style.user(style)` | Use matplotlib style settings from a style specification. | [Plotting w/ Pandas][044] |
| `ax.set_aspect(aspect, adjustable=None, anchor=None)` | set aspect |   [Plotting w/ Pandas][044] |
| `cm.get_cmap(name=None, lut=None)` | Get a colormap instance, defaulting to rc values if *name* is None. | [Exame Data][050] |
| `plt.savefig(fname, *kwargs*)` | Save the current figure.  | [Exame Data][050] |
| `savefig(fname, *kwargs*)` | Save the current figure.  | [Exame Data][050] |
| `ListedColormap(colors, name='from_list', N=None)` | Colormap object generated from a list of colors | [Datasets][051] |



[TOC](#table-of-contents)

### Line style or marker

<table style="margin: 0 auto; border: 1px solid black; border-collapse: collapse; width: 60vw;">
  <thead>
  <tr style="border-bottom: double black;">
    <th style="width: 10vw; font-size: 1.2em; border-right: double back; background-color: #4CAF50;"> Character </th>
    <th style="width: 20vw; text-align: left; font-size: 1.2em; background-color: #4CAF50; border-right: double white;"> Description </th>
    <th style="width: 10vw; font-size: 1.2em; border-right: double back; background-color: #4CAF50;"> Character </th>
    <th style="width: 20vw; text-align: left; font-size: 1.2em; background-color: #4CAF50;"> Description </th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '-' </td>
    <td style="padding: 0.3em;">  solid line style </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '1' </td>
    <td style="padding: 0.3em;"> tri_down marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '--' </td>
    <td style="padding: 0.3em;"> dashed line style </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '2' </td>
    <td style="padding: 0.3em;"> tri_up marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '-.' </td>
    <td style="padding: 0.3em;"> dash-dot line style </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '3' </td>
    <td style="padding: 0.3em;"> tri_left marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> ':' </td>
    <td style="padding: 0.3em;"> dotted line style </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '4' </td>
    <td style="padding: 0.3em;"> tri_right marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '.' </td>
    <td style="padding: 0.3em;"> point marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '*' </td>
    <td style="padding: 0.3em;"> star marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> ',' </td>
    <td style="padding: 0.3em;"> pixel marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '+' </td>
    <td style="padding: 0.3em;"> plus marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'o' </td>
    <td style="padding: 0.3em;"> circle marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '_' </td>
    <td style="padding: 0.3em;"> hline marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'v' </td>
    <td style="padding: 0.3em;"> triangle_down marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '^' </td>
    <td style="padding: 0.3em;"> triangle_up marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '<' </td>
    <td style="padding: 0.3em;"> triangle_left marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '>' </td>
    <td style="padding: 0.3em;"> triangle_right marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 's' </td>
    <td style="padding: 0.3em;"> square marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'p' </td>
    <td style="padding: 0.3em;"> pentagon marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'h' </td>
    <td style="padding: 0.3em;"> hexagon1 marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'H' </td>
    <td style="padding: 0.3em;"> hexagon2 marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'x' </td>
    <td style="padding: 0.3em;"> x marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> '|' </td>
    <td style="padding: 0.3em;"> vline marker </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'D' </td>
    <td style="padding: 0.3em;"> diamond marker </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'd' </td>
    <td style="padding: 0.3em;"> thin_diamond marker </td>
  </tr>
  </tbody>
</table>  


### Color abbreviations


<table style="margin: 0 auto; border: 1px solid black; border-collapse: collapse; width: 40vw;">
  <thead>
  <tr style="border-bottom: double black;">
    <th style="width: 10vw; font-size: 1.2em; border-right: double back; background-color: #4CAF50;"> Character </th>
    <th style="width: 10vw; text-align: left; font-size: 1.2em; background-color: #4CAF50; border-right: double white;"> Color </th>
    <th style="width: 10vw; font-size: 1.2em; border-right: double back; background-color: #4CAF50;"> Character </th>
    <th style="width: 10vw; text-align: left; font-size: 1.2em; background-color: #4CAF50;"> Color </th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'b' </td>
    <td style="padding: 0.3em;"> blue  </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'm' </td>
    <td style="padding: 0.3em;"> magenta </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'g' </td>
    <td style="padding: 0.3em;"> green </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'y' </td>
    <td style="padding: 0.3em;"> yellow </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'r' </td>
    <td style="padding: 0.3em;"> red </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'k' </td>
    <td style="padding: 0.3em;"> black </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'c' </td>
    <td style="padding: 0.3em;"> cyan </td>
    <td style="padding: 0.3em; font-weight: bold; text-align: center;"> 'w' </td>
    <td style="padding: 0.3em;"> white </td>
  </tr>
  </tr>
  </tbody>
</table>  


### Examples - Line Plots

```python
plt.plot(x, y)        # plot x and y using default line style and color
plt.plot(x, y, 'bo')  # plot x and y using blue circle markers
plt.plot(y)           # plot y using x as index array 0..N-1
plt.plot(y, 'r+')     # ditto, but with red plusses

plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
plt.plot([1,2,3], [1,4,9], 'rs',  label='line 2')
plt.axis([0, 4, 0, 10])
plt.legend()
```

[TOC](#table-of-contents)

# Seaborn

+ Seaborn is a library for making attractive and informative statistical graphics in Python.
+ [Official Site][046]
+ [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

```python
import seaborn as sns
```

### [seaborn API][045]

| API | Description | Link |
|-----|-------------|------|
| __Axis grids__ | | |
| `FacetGrid(data[, row, col, hue, col_wrap, ...])` | Subplot grid for plotting conditional relationships. | [Link](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid) |
| `factorplot([x, y, hue, data, row, col, ...])` | Draw a categorical plot onto a FacetGrid. | [Link](https://seaborn.pydata.org/generated/seaborn.factorplot.html#seaborn.factorplot) |
| `lmplot(x, y, data[, hue, col, row, palette, ...])` | Plot data and regression model fits across a FacetGrid. | [Link](https://seaborn.pydata.org/generated/seaborn.lmplot.html#seaborn.lmplot) |
| `PairGrid(data[, hue, hue_order, palette, ...])` | Subplot grid for plotting pairwise relationships in a dataset. | [Link](https://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid) |
| `pairplot(data[, hue, hue_order, palette, ...])` | Plot pairwise relationships in a dataset. | [Link](https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot), [Seaborn][048] |
| `JointGrid(x, y[, data, size, ratio, space, ...])` | Grid for drawing a bivariate plot with marginal univariate plots. | [Link](https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid) |
| `jointplot(x, y[, data, kind, stat_func, ...])` | Draw a plot of two variables with bivariate and univariate graphs. | [Link](https://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot), [Seaborn][048] |
| __Categorical plots__ | | |
| `stripplot([x, y, hue, data, order, ...])` | Draw a scatterplot where one variable is categorical. | [Link](https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot) |
| `swarmplot([x, y, hue, data, order, ...])` | Draw a categorical scatterplot with non-overlapping points. | [Link](https://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot), [Seaborn][048] |
| `boxplot([x, y, hue, data, order, hue_order, ...])` | Draw a box plot to show distributions with respect to categories. | [Link](https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot) |
| `violinplot([x, y, hue, data, order, ...])` | Draw a combination of boxplot and kernel density estimate. | [Link](https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot), [Seaborn][048] |
| `lvplot([x, y, hue, data, order, hue_order, ...])` | Draw a letter value plot to show distributions of large datasets. | [Link](https://seaborn.pydata.org/generated/seaborn.lvplot.html#seaborn.lvplot) |
| `pointplot([x, y, hue, data, order, ...])` | Show point estimates and confidence intervals using scatter plot glyphs. | [Link](https://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot) |
| `barplot([x, y, hue, data, order, hue_order, ...])` | Show point estimates and confidence intervals as rectangular bars. | [Link](https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot) |
| `countplot([x, y, hue, data, order, ...])` | Show the counts of observations in each categorical bin using bars. | [Link](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) |
| __Distribution plots__ | | |
| `distplot(a[, bins, hist, kde, rug, fit, ...])` | Flexibly plot a univariate distribution of observations. | [Link](https://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot), [Seaborn][048] |
| `kdeplot(data[, data2, shade, vertical, ...])` | Fit and plot a univariate or bivariate kernel density estimate. | [Link](https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot), [Seaborn][048] |
| `rugplot(a[, height, axis, ax])` | Plot datapoints in an array as sticks on an axis. | [Link](https://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot) |
| __Regression plots__ | | |
| `regplot(x, y[, data, x_estimator, x_bins, ...])` | Plot data and a linear regression model fit. | [Link](https://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot) |
| `residplot(x, y[, data, lowess, x_partial, ...])` | Plot the residuals of a linear regression. | [Link](https://seaborn.pydata.org/generated/seaborn.residplot.html#seaborn.residplot) |
| __Matrix plots__ | | |
| `heatmap(data[, vmin, vmax, cmap, center, ...])` | Plot rectangular data as a color-encoded matrix. | [Link](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap) |
| `clustermap(data[, pivot_kws, method, ...])` | Plot a matrix dataset as a hierarchically-clustered heatmap. | [Link](https://seaborn.pydata.org/generated/seaborn.clustermap.html#seaborn.clustermap) |
| __Timeseries plots__ | | |
| `tsplot(data[, time, unit, condition, value, ...])` | Plot one or more timeseries with flexible representation of uncertainty. | [Link](https://seaborn.pydata.org/generated/seaborn.tsplot.html#seaborn.tsplot) |
| __Miscellaneous plots__ | | |
| `palplot(pal[, size])` | Plot the values in a color palette as a horizontal array. | [Link](https://seaborn.pydata.org/generated/seaborn.palplot.html#seaborn.palplot) |
| __Style frontend__ | | |
| `set([context, style, palette, font, ...])` | Set aesthetic parameters in one step. | [Link](https://seaborn.pydata.org/generated/seaborn.set.html#seaborn.set) |
| `axes_style([style, rc])` | Return a parameter dict for the aesthetic style of the plots. | [Link](https://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style) |
| `set_style([style, rc])` | Set the aesthetic style of the plots. | [Link](https://seaborn.pydata.org/generated/seaborn.set_style.html#seaborn.set_style) |
| `plotting_context([context, font_scale, rc])` | Return a parameter dict to scale elements of the figure. | [Link](https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context), [Seaborn][048] |
| `set_context([context, font_scale, rc])` | Set the plotting context parameters. | [Link](https://seaborn.pydata.org/generated/seaborn.set_context.html#seaborn.set_context) |
| `set_color_codes([palette])` | Change how matplotlib color shorthands are interpreted. | [Link](https://seaborn.pydata.org/generated/seaborn.set_color_codes.html#seaborn.set_color_codes) |
| `reset_defaults()` | Restore all RC params to default settings. | [Link](https://seaborn.pydata.org/generated/seaborn.reset_defaults.html#seaborn.reset_defaults) |
| `reset_orig()` | Restore all RC params to original settings (respects custom rc). | [Link](https://seaborn.pydata.org/generated/seaborn.reset_orig.html#seaborn.reset_orig) |
| __Color palettes__ | | |
| `set_palette(palette[, n_colors, desat, ...])` | Set the matplotlib color cycle using a seaborn palette. | [Link](https://seaborn.pydata.org/generated/seaborn.set_palette.html#seaborn.set_palette) |
| `color_palette([palette, n_colors, desat])` | Return a list of colors defining a color palette. | [Link](https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette) |
| `husl_palette([n_colors, h, s, l])` | Get a set of evenly spaced colors in HUSL hue space. | [Link](https://seaborn.pydata.org/generated/seaborn.husl_palette.html#seaborn.husl_palette) |
| `hls_palette([n_colors, h, l, s])` | Get a set of evenly spaced colors in HLS hue space. | [Link](https://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette) |
| `cubehelix_palette([n_colors, start, rot, ...])` | Make a sequential palette from the cubehelix system. | [Link](https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html#seaborn.cubehelix_palette) |
| `dark_palette(color[, n_colors, reverse, ...])` | Make a sequential palette that blends from dark to color. | [Link](https://seaborn.pydata.org/generated/seaborn.dark_palette.html#seaborn.dark_palette) |
| `light_palette(color[, n_colors, reverse, ...])` | Make a sequential palette that blends from light to color. | [Link](https://seaborn.pydata.org/generated/seaborn.light_palette.html#seaborn.light_palette) |
| `diverging_palette(h_neg, h_pos[, s, l, sep, ...])` | Make a diverging palette between two HUSL colors. | [Link](https://seaborn.pydata.org/generated/seaborn.diverging_palette.html#seaborn.diverging_palette) |
| `blend_palette(colors[, n_colors, as_cmap, input])` | Make a palette that blends between a list of colors. | [Link](https://seaborn.pydata.org/generated/seaborn.blend_palette.html#seaborn.blend_palette) |
| `xkcd_palette(colors)` | Make a palette with color names from the xkcd color survey. | [Link](https://seaborn.pydata.org/generated/seaborn.xkcd_palette.html#seaborn.xkcd_palette) |
| `crayon_palette(colors)` | Make a palette with color names from Crayola crayons. | [Link](https://seaborn.pydata.org/generated/seaborn.crayon_palette.html#seaborn.crayon_palette) |
| `mpl_palette(name[, n_colors])` | Return discrete colors from a matplotlib palette. | [Link](https://seaborn.pydata.org/generated/seaborn.mpl_palette.html#seaborn.mpl_palette) |
| __Palette widgets__ | | |
| `choose_colorbrewer_palette(data_type[, as_cmap])` | Select a palette from the ColorBrewer set. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_colorbrewer_palette.html#seaborn.choose_colorbrewer_palette) |
| `choose_cubehelix_palette([as_cmap])` | Launch an interactive widget to create a sequential cubehelix palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_cubehelix_palette.html#seaborn.choose_cubehelix_palette) |
| `choose_light_palette([input, as_cmap])` | Launch an interactive widget to create a light sequential palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_light_palette.html#seaborn.choose_light_palette) |
| `choose_dark_palette([input, as_cmap])` | Launch an interactive widget to create a dark sequential palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_dark_palette.html#seaborn.choose_dark_palette) |
| `choose_diverging_palette([as_cmap])` | Launch an interactive widget to choose a diverging color palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_diverging_palette.html#seaborn.choose_diverging_palette) |
| __Utility functions__ | | |
| `despine([fig, ax, top, right, left, bottom, ...])` | Remove the top and right spines from plot(s). | [Link](https://seaborn.pydata.org/generated/seaborn.despine.h) |
| `desaturate(color, prop)` | Decrease the saturation channel of a color by some percent. | [Link](https://seaborn.pydata.org/generated/seaborn.desaturate.html#seaborn.desaturate) |
| `saturate(color)` | Return a fully saturated color with the same hue. | [Link](https://seaborn.pydata.org/generated/seaborn.saturate.html#seaborn.saturate) |
| `set_hls_values(color[, h, l, s])` | Independently manipulate the h, l, or s channels of a color. | [Link](https://seaborn.pydata.org/generated/seaborn.set_hls_values.html#seaborn.set_hls_values) |


----------------------

[025]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#basic-plotting-with-matplotlib
[026]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#scatter-plot
[027]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#line-plots
[028]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#bar-charts
[029]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#dejunkifying-a-plot
[030]: https://matplotlib.org/api/axes_api.html
[031]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html
[032]: https://matplotlib.org/api/index.html
[033]: https://matplotlib.org/api/pyplot_summary.html#colors-in-matplotlib
[034]: https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html
[035]: https://matplotlib.org/api/_as_gen/matplotlib.figure.SubplotParams.html
[036]: https://matplotlib.org/api/text_api.html
[037]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#subplots
[038]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#histograms
[039]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#box-plots
[040]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#heatmaps
[041]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#animations
[042]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#interactivity
[043]: ../AppliedDS-UMich/2-InfoVis/Assigment03.md#related-methods-used
[044]: ../AppliedDS-UMich/2-InfoVis/04-AppliedVis.md#plotting-with-pandas
[045]: https://seaborn.pydata.org/api.html#api-ref
[046]: https://seaborn.pydata.org/
[047]: https://seaborn.pydata.org/tutorial.html
[048]: ../AppliedDS-UMich/2-InfoVis/04-AppliedVis.md#seaborn
[049]: ../AppliedDS-UMich/4-TextMining/01-Working.md#handling-text-in-python
[050]: ../AppliedDS-UMich/4-TextMining/01-Working.md#regular-expressions
[051]: ../AppliedDS-UMich/4-TextMining/01-Working.md#demonstration-regex-with-pandas-and-named-groups
