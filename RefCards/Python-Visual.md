# Python Visualization

## MatPlotLib

### [Official Pyplot API](https://matplotlib.org/api/pyplot_summary.html)

### Environment and Module

```python
%matplotlib notebook       # provides an interactive environment in Jupyter and IPuthon

import matplotlib as mpl                # load module in CLI
import matplotlib.pyplot as plt         # load pyplot module
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
    <th style="width: 10vw; font-size: 1.3em; border-right: double back; background-color: #4CAF50; color: darkblue; font-size: 1.4em;"> Character </th>
    <th style="width: 20vw; text-align: left; font-size: 1.3em; background-color: #4CAF50; color: darkblue; font-size: 1.4em; border-right: double white;"> Description </th>
    <th style="width: 10vw; font-size: 1.3em; border-right: double back; background-color: #4CAF50; color: darkblue; font-size: 1.4em;"> Character </th>
    <th style="width: 20vw; text-align: left; font-size: 1.3em; background-color: #4CAF50; color: darkblue; font-size: 1.4em;"> Description </th>
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
    <th style="width: 10vw; font-size: 1.3em; border-right: double back; background-color: #4CAF50; color: darkblue; font-size: 1.4em;"> Character </th>
    <th style="width: 10vw; text-align: left; font-size: 1.3em; background-color: #4CAF50; color: darkblue; font-size: 1.4em; border-right: double white;"> Color </th>
    <th style="width: 10vw; font-size: 1.3em; border-right: double back; background-color: #4CAF50; color: darkblue; font-size: 1.4em;"> Character </th>
    <th style="width: 10vw; text-align: left; font-size: 1.3em; background-color: #4CAF50; color: darkblue; font-size: 1.4em;"> Color </th>
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


<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 55vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://seaborn.pydata.org/api.html">Seaborn API reference</a></caption>
  <thead>
  <tr style="font-size: 1.3em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">Method</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Description</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#relational-plots"> Relational plots </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot" title="seaborn.relplot"><code class="xref py py-obj docutils literal notranslate">relplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;size,&nbsp;style,&nbsp;data,&nbsp;…])</p></td>
  <td><p>Figure-level interface for drawing relational plots onto a FacetGrid.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot" title="seaborn.scatterplot"><code class="xref py py-obj docutils literal notranslate">scatterplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;style,&nbsp;size,&nbsp;…])</p></td>
  <td><p>Draw a scatter plot with possibility of several semantic groupings.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot" title="seaborn.lineplot"><code class="xref py py-obj docutils literal notranslate">lineplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;size,&nbsp;style,&nbsp;data,&nbsp;…])</p></td>
  <td><p>Draw a line plot with possibility of several semantic groupings.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#distribution-plots"> Distribution plots </a></td> 
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot" title="seaborn.displot"><code class="xref py py-obj docutils literal notranslate">displot</code></a>([data,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;row,&nbsp;col,&nbsp;…])</p></td>
  <td><p>Figure-level interface for drawing distribution plots onto a FacetGrid.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot" title="seaborn.histplot"><code class="xref py py-obj docutils literal notranslate">histplot</code></a>([data,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;weights,&nbsp;stat,&nbsp;…])</p></td>
  <td><p>Plot univariate or bivariate histograms to show distributions of datasets.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot" title="seaborn.kdeplot"><code class="xref py py-obj docutils literal notranslate">kdeplot</code></a>([x,&nbsp;y,&nbsp;shade,&nbsp;vertical,&nbsp;kernel,&nbsp;bw,&nbsp;…])</p></td>
  <td><p>Plot univariate or bivariate distributions using kernel density estimation.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.ecdfplot.html#seaborn.ecdfplot" title="seaborn.ecdfplot"><code class="xref py py-obj docutils literal notranslate">ecdfplot</code></a>([data,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;weights,&nbsp;stat,&nbsp;…])</p></td>
  <td><p>Plot empirical cumulative distribution functions.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot" title="seaborn.rugplot"><code class="xref py py-obj docutils literal notranslate">rugplot</code></a>([x,&nbsp;height,&nbsp;axis,&nbsp;ax,&nbsp;data,&nbsp;y,&nbsp;hue,&nbsp;…])</p></td>
  <td><p>Plot marginal distributions by drawing ticks along the x and y axes.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot" title="seaborn.distplot"><code class="xref py py-obj docutils literal notranslate">distplot</code></a>([a,&nbsp;bins,&nbsp;hist,&nbsp;kde,&nbsp;rug,&nbsp;fit,&nbsp;…])</p></td>
  <td><p>DEPRECATED: Flexibly plot a univariate distribution of observations.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#categorical-plots"> Categorical plots </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot" title="seaborn.catplot"><code class="xref py py-obj docutils literal notranslate">catplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;row,&nbsp;col,&nbsp;…])</p></td>
  <td><p>Figure-level interface for drawing categorical plots onto a <a href="https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid" title="seaborn.FacetGrid"><code class="xref py py-class docutils literal notranslate">FacetGrid</code></a>.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot" title="seaborn.stripplot"><code class="xref py py-obj docutils literal notranslate">stripplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Draw a scatterplot where one variable is categorical.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot" title="seaborn.swarmplot"><code class="xref py py-obj docutils literal notranslate">swarmplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Draw a categorical scatterplot with non-overlapping points.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot" title="seaborn.boxplot"><code class="xref py py-obj docutils literal notranslate">boxplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Draw a box plot to show distributions with respect to categories.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot" title="seaborn.violinplot"><code class="xref py py-obj docutils literal notranslate">violinplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Draw a combination of boxplot and kernel density estimate.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot" title="seaborn.boxenplot"><code class="xref py py-obj docutils literal notranslate">boxenplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Draw an enhanced box plot for larger datasets.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot" title="seaborn.pointplot"><code class="xref py py-obj docutils literal notranslate">pointplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Show point estimates and confidence intervals using scatter plot glyphs.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot" title="seaborn.barplot"><code class="xref py py-obj docutils literal notranslate">barplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Show point estimates and confidence intervals as rectangular bars.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot" title="seaborn.countplot"><code class="xref py py-obj docutils literal notranslate">countplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;hue,&nbsp;data,&nbsp;order,&nbsp;…])</p></td>
  <td><p>Show the counts of observations in each categorical bin using bars.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#regression-plots"> Regression plots </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.lmplot.html#seaborn.lmplot" title="seaborn.lmplot"><code class="xref py py-obj docutils literal notranslate">lmplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;data,&nbsp;hue,&nbsp;col,&nbsp;row,&nbsp;…])</p></td>
  <td><p>Plot data and regression model fits across a FacetGrid.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot" title="seaborn.regplot"><code class="xref py py-obj docutils literal notranslate">regplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;data,&nbsp;x_estimator,&nbsp;…])</p></td>
  <td><p>Plot data and a linear regression model fit.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.residplot.html#seaborn.residplot" title="seaborn.residplot"><code class="xref py py-obj docutils literal notranslate">residplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;data,&nbsp;lowess,&nbsp;…])</p></td>
  <td><p>Plot the residuals of a linear regression.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#matrix-plots"> Matrix plots </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap" title="seaborn.heatmap"><code class="xref py py-obj docutils literal notranslate">heatmap</code></a>(data,&nbsp;*[,&nbsp;vmin,&nbsp;vmax,&nbsp;cmap,&nbsp;center,&nbsp;…])</p></td>
  <td><p>Plot rectangular data as a color-encoded matrix.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.clustermap.html#seaborn.clustermap" title="seaborn.clustermap"><code class="xref py py-obj docutils literal notranslate">clustermap</code></a>(data,&nbsp;*[,&nbsp;pivot_kws,&nbsp;method,&nbsp;…])</p></td>
  <td><p>Plot a matrix dataset as a hierarchically-clustered heatmap.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#facet-grids"> Multi-plot grids: Facet grids </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid" title="seaborn.FacetGrid"><code class="xref py py-obj docutils literal notranslate">FacetGrid</code></a>(data,&nbsp;*[,&nbsp;row,&nbsp;col,&nbsp;hue,&nbsp;…])</p></td>
  <td><p>Multi-plot grid for plotting conditional relationships.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.FacetGrid.map.html#seaborn.FacetGrid.map" title="seaborn.FacetGrid.map"><code class="xref py py-obj docutils literal notranslate">FacetGrid.map</code></a>(self,&nbsp;func,&nbsp;*args,&nbsp;**kwargs)</p></td>
  <td><p>Apply a plotting function to each facet’s subset of the data.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.FacetGrid.map_dataframe.html#seaborn.FacetGrid.map_dataframe" title="seaborn.FacetGrid.map_dataframe"><code class="xref py py-obj docutils literal notranslate">FacetGrid.map_dataframe</code></a>(self,&nbsp;func,&nbsp;*args,&nbsp;…)</p></td>
  <td><p>Like <code class="docutils literal notranslate">.map</code> but passes args as strings and inserts data in kwargs.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#pair-grids"> Multi-plot grids: Pair grids </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot" title="seaborn.pairplot"><code class="xref py py-obj docutils literal notranslate">pairplot</code></a>(data,&nbsp;*[,&nbsp;hue,&nbsp;hue_order,&nbsp;palette,&nbsp;…])</p></td>
  <td><p>Plot pairwise relationships in a dataset.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid" title="seaborn.PairGrid"><code class="xref py py-obj docutils literal notranslate">PairGrid</code></a>(data,&nbsp;*[,&nbsp;hue,&nbsp;hue_order,&nbsp;palette,&nbsp;…])</p></td>
  <td><p>Subplot grid for plotting pairwise relationships in a dataset.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.PairGrid.map.html#seaborn.PairGrid.map" title="seaborn.PairGrid.map"><code class="xref py py-obj docutils literal notranslate">PairGrid.map</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Plot with the same function in every subplot.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.PairGrid.map_diag.html#seaborn.PairGrid.map_diag" title="seaborn.PairGrid.map_diag"><code class="xref py py-obj docutils literal notranslate">PairGrid.map_diag</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Plot with a univariate function on each diagonal subplot.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.PairGrid.map_offdiag.html#seaborn.PairGrid.map_offdiag" title="seaborn.PairGrid.map_offdiag"><code class="xref py py-obj docutils literal notranslate">PairGrid.map_offdiag</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Plot with a bivariate function on the off-diagonal subplots.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.PairGrid.map_lower.html#seaborn.PairGrid.map_lower" title="seaborn.PairGrid.map_lower"><code class="xref py py-obj docutils literal notranslate">PairGrid.map_lower</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Plot with a bivariate function on the lower diagonal subplots.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.PairGrid.map_upper.html#seaborn.PairGrid.map_upper" title="seaborn.PairGrid.map_upper"><code class="xref py py-obj docutils literal notranslate">PairGrid.map_upper</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Plot with a bivariate function on the upper diagonal subplots.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#joint-grids"> Multi-plot grids: Joint grids </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot" title="seaborn.jointplot"><code class="xref py py-obj docutils literal notranslate">jointplot</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;data,&nbsp;kind,&nbsp;color,&nbsp;…])</p></td>
  <td><p>Draw a plot of two variables with bivariate and univariate graphs.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid" title="seaborn.JointGrid"><code class="xref py py-obj docutils literal notranslate">JointGrid</code></a>(*[,&nbsp;x,&nbsp;y,&nbsp;data,&nbsp;height,&nbsp;ratio,&nbsp;…])</p></td>
  <td><p>Grid for drawing a bivariate plot with marginal univariate plots.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.JointGrid.plot.html#seaborn.JointGrid.plot" title="seaborn.JointGrid.plot"><code class="xref py py-obj docutils literal notranslate">JointGrid.plot</code></a>(self,&nbsp;joint_func,&nbsp;…)</p></td>
  <td><p>Draw the plot by passing functions for joint and marginal axes.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.JointGrid.plot_joint.html#seaborn.JointGrid.plot_joint" title="seaborn.JointGrid.plot_joint"><code class="xref py py-obj docutils literal notranslate">JointGrid.plot_joint</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Draw a bivariate plot on the joint axes of the grid.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.JointGrid.plot_marginals.html#seaborn.JointGrid.plot_marginals" title="seaborn.JointGrid.plot_marginals"><code class="xref py py-obj docutils literal notranslate">JointGrid.plot_marginals</code></a>(self,&nbsp;func,&nbsp;**kwargs)</p></td>
  <td><p>Draw univariate plots on each marginal axes.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#themes"> Themes </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme" title="seaborn.set_theme"><code class="xref py py-obj docutils literal notranslate">set_theme</code></a>([context,&nbsp;style,&nbsp;palette,&nbsp;font,&nbsp;…])</p></td>
  <td><p>Set multiple theme parameters in one step.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style" title="seaborn.axes_style"><code class="xref py py-obj docutils literal notranslate">axes_style</code></a>([style,&nbsp;rc])</p></td>
  <td><p>Return a parameter dict for the aesthetic style of the plots.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set_style.html#seaborn.set_style" title="seaborn.set_style"><code class="xref py py-obj docutils literal notranslate">set_style</code></a>([style,&nbsp;rc])</p></td>
  <td><p>Set the aesthetic style of the plots.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context" title="seaborn.plotting_context"><code class="xref py py-obj docutils literal notranslate">plotting_context</code></a>([context,&nbsp;font_scale,&nbsp;rc])</p></td>
  <td><p>Return a parameter dict to scale elements of the figure.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set_context.html#seaborn.set_context" title="seaborn.set_context"><code class="xref py py-obj docutils literal notranslate">set_context</code></a>([context,&nbsp;font_scale,&nbsp;rc])</p></td>
  <td><p>Set the plotting context parameters.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set_color_codes.html#seaborn.set_color_codes" title="seaborn.set_color_codes"><code class="xref py py-obj docutils literal notranslate">set_color_codes</code></a>([palette])</p></td>
  <td><p>Change how matplotlib color shorthands are interpreted.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.reset_defaults.html#seaborn.reset_defaults" title="seaborn.reset_defaults"><code class="xref py py-obj docutils literal notranslate">reset_defaults</code></a>()</p></td>
  <td><p>Restore all RC params to default settings.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.reset_orig.html#seaborn.reset_orig" title="seaborn.reset_orig"><code class="xref py py-obj docutils literal notranslate">reset_orig</code></a>()</p></td>
  <td><p>Restore all RC params to original settings (respects custom rc).</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set.html#seaborn.set" title="seaborn.set"><code class="xref py py-obj docutils literal notranslate">set</code></a>(*args,&nbsp;**kwargs)</p></td>
  <td><p>Alias for <a href="https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme" title="seaborn.set_theme"><code class="xref py py-func docutils literal notranslate">set_theme()</code></a>, which is the preferred interface.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#color-palettes"> Color palettes </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set_palette.html#seaborn.set_palette" title="seaborn.set_palette"><code class="xref py py-obj docutils literal notranslate">set_palette</code></a>(palette[,&nbsp;n_colors,&nbsp;desat,&nbsp;…])</p></td>
  <td><p>Set the matplotlib color cycle using a seaborn palette.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette" title="seaborn.color_palette"><code class="xref py py-obj docutils literal notranslate">color_palette</code></a>([palette,&nbsp;n_colors,&nbsp;desat,&nbsp;…])</p></td>
  <td><p>Return a list of colors or continuous colormap defining a palette.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.husl_palette.html#seaborn.husl_palette" title="seaborn.husl_palette"><code class="xref py py-obj docutils literal notranslate">husl_palette</code></a>([n_colors,&nbsp;h,&nbsp;s,&nbsp;l,&nbsp;as_cmap])</p></td>
  <td><p>Get a set of evenly spaced colors in HUSL hue space.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette" title="seaborn.hls_palette"><code class="xref py py-obj docutils literal notranslate">hls_palette</code></a>([n_colors,&nbsp;h,&nbsp;l,&nbsp;s,&nbsp;as_cmap])</p></td>
  <td><p>Get a set of evenly spaced colors in HLS hue space.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html#seaborn.cubehelix_palette" title="seaborn.cubehelix_palette"><code class="xref py py-obj docutils literal notranslate">cubehelix_palette</code></a>([n_colors,&nbsp;start,&nbsp;rot,&nbsp;…])</p></td>
  <td><p>Make a sequential palette from the cubehelix system.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.dark_palette.html#seaborn.dark_palette" title="seaborn.dark_palette"><code class="xref py py-obj docutils literal notranslate">dark_palette</code></a>(color[,&nbsp;n_colors,&nbsp;reverse,&nbsp;…])</p></td>
  <td><p>Make a sequential palette that blends from dark to <code class="docutils literal notranslate">color</code>.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.light_palette.html#seaborn.light_palette" title="seaborn.light_palette"><code class="xref py py-obj docutils literal notranslate">light_palette</code></a>(color[,&nbsp;n_colors,&nbsp;reverse,&nbsp;…])</p></td>
  <td><p>Make a sequential palette that blends from light to <code class="docutils literal notranslate">color</code>.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.diverging_palette.html#seaborn.diverging_palette" title="seaborn.diverging_palette"><code class="xref py py-obj docutils literal notranslate">diverging_palette</code></a>(h_neg,&nbsp;h_pos[,&nbsp;s,&nbsp;l,&nbsp;sep,&nbsp;…])</p></td>
  <td><p>Make a diverging palette between two HUSL colors.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.blend_palette.html#seaborn.blend_palette" title="seaborn.blend_palette"><code class="xref py py-obj docutils literal notranslate">blend_palette</code></a>(colors[,&nbsp;n_colors,&nbsp;as_cmap,&nbsp;input])</p></td>
  <td><p>Make a palette that blends between a list of colors.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.xkcd_palette.html#seaborn.xkcd_palette" title="seaborn.xkcd_palette"><code class="xref py py-obj docutils literal notranslate">xkcd_palette</code></a>(colors)</p></td>
  <td><p>Make a palette with color names from the xkcd color survey.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.crayon_palette.html#seaborn.crayon_palette" title="seaborn.crayon_palette"><code class="xref py py-obj docutils literal notranslate">crayon_palette</code></a>(colors)</p></td>
  <td><p>Make a palette with color names from Crayola crayons.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.mpl_palette.html#seaborn.mpl_palette" title="seaborn.mpl_palette"><code class="xref py py-obj docutils literal notranslate">mpl_palette</code></a>(name[,&nbsp;n_colors,&nbsp;as_cmap])</p></td>
  <td><p>Return discrete colors from a matplotlib palette.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#palette-widgets"> Palette widgets </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.choose_colorbrewer_palette.html#seaborn.choose_colorbrewer_palette" title="seaborn.choose_colorbrewer_palette"><code class="xref py py-obj docutils literal notranslate">choose_colorbrewer_palette</code></a>(data_type[,&nbsp;as_cmap])</p></td>
  <td><p>Select a palette from the ColorBrewer set.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.choose_cubehelix_palette.html#seaborn.choose_cubehelix_palette" title="seaborn.choose_cubehelix_palette"><code class="xref py py-obj docutils literal notranslate">choose_cubehelix_palette</code></a>([as_cmap])</p></td>
  <td><p>Launch an interactive widget to create a sequential cubehelix palette.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.choose_light_palette.html#seaborn.choose_light_palette" title="seaborn.choose_light_palette"><code class="xref py py-obj docutils literal notranslate">choose_light_palette</code></a>([input,&nbsp;as_cmap])</p></td>
  <td><p>Launch an interactive widget to create a light sequential palette.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.choose_dark_palette.html#seaborn.choose_dark_palette" title="seaborn.choose_dark_palette"><code class="xref py py-obj docutils literal notranslate">choose_dark_palette</code></a>([input,&nbsp;as_cmap])</p></td>
  <td><p>Launch an interactive widget to create a dark sequential palette.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.choose_diverging_palette.html#seaborn.choose_diverging_palette" title="seaborn.choose_diverging_palette"><code class="xref py py-obj docutils literal notranslate">choose_diverging_palette</code></a>([as_cmap])</p></td>
  <td><p>Launch an interactive widget to choose a diverging color palette.</p></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.3em; background-color: lightgrey; color: gray;"><a href="https://seaborn.pydata.org/api.html#utility-functions"> Utility functions </a></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.load_dataset.html#seaborn.load_dataset" title="seaborn.load_dataset"><code class="xref py py-obj docutils literal notranslate">load_dataset</code></a>(name[,&nbsp;cache,&nbsp;data_home])</p></td>
  <td><p>Load an example dataset from the online repository (requires internet).</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.get_dataset_names.html#seaborn.get_dataset_names" title="seaborn.get_dataset_names"><code class="xref py py-obj docutils literal notranslate">get_dataset_names</code></a>()</p></td>
  <td><p>Report available example datasets, useful for reporting issues.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.get_data_home.html#seaborn.get_data_home" title="seaborn.get_data_home"><code class="xref py py-obj docutils literal notranslate">get_data_home</code></a>([data_home])</p></td>
  <td><p>Return a path to the cache directory for example datasets.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.despine.html#seaborn.despine" title="seaborn.despine"><code class="xref py py-obj docutils literal notranslate">despine</code></a>([fig,&nbsp;ax,&nbsp;top,&nbsp;right,&nbsp;left,&nbsp;bottom,&nbsp;…])</p></td>
  <td><p>Remove the top and right spines from plot(s).</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.desaturate.html#seaborn.desaturate" title="seaborn.desaturate"><code class="xref py py-obj docutils literal notranslate">desaturate</code></a>(color,&nbsp;prop)</p></td>
  <td><p>Decrease the saturation channel of a color by some percent.</p></td>
  </tr>
  <tr class="row-even"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.saturate.html#seaborn.saturate" title="seaborn.saturate"><code class="xref py py-obj docutils literal notranslate">saturate</code></a>(color)</p></td>
  <td><p>Return a fully saturated color with the same hue.</p></td>
  </tr>
  <tr class="row-odd"><td><p><a href="https://seaborn.pydata.org/generated/seaborn.set_hls_values.html#seaborn.set_hls_values" title="seaborn.set_hls_values"><code class="xref py py-obj docutils literal notranslate">set_hls_values</code></a>(color[,&nbsp;h,&nbsp;l,&nbsp;s])</p></td>
  <td><p>Independently manipulate the h, l, or s channels of a color.</p></td>
  </tr>
  </tbody>
</table><br/>


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
