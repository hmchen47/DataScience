# Charting Fundamentals

## Module 3 Jupyter Notebook

+ [Web Launcher](https://www.coursera.org/learn/python-plotting/notebook/nXjSZ/module-3-jupyter-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=0-lHeSXARWGpR3klwAVhEA&next=%2Fnotebooks%2FWeek3.ipynb)
+[Local Notebook](./labs/Week03.ipynb)

## Subplots

+ `plt.figure` method
    + Signature: `figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, **kwargs)`
    + Docstring: Creates a new figure.
    + Args:
        + `num` (integer or string): 
            + None: a new figure will be created, and the figure number will be incremented. The figure objects holds this number in a `number` attribute.
            + provided and existing id:  make it active, and returns a reference to it. 
            + provided w/o id: create it and returns it.
            + string: the window title will be set to this figure's `num`.
        + `figsize` (tuple of integers): width, height in inches. defaults to rc `figure.figsize`.
        + `dpi` (integer): resolution of the figure; defaults to rc `figure.dpi`.
        + `facecolor` : the background color. defaults to rc `figure.facecolor`
        + `edgecolor` : the border color. defaults to rc `figure.edgecolor`
    + Return: The Figure instance returned will also be passed to new_figure_manager in the backends, which allows to hook custom Figure classes into the `pylab` interface. Additional `kwargs` will be passed to the figure init function.

+ `plt.subplot` method:
    + Signature: `subplot(*args, **kwargs)`
    + Docstring: Return a subplot axes positioned by the given grid definition.
    + Keyword arguments:
        + `facecolor`: The background color of the subplot, which can be any valid color specifier.  see `matplotlib.colors`
        + `polar`: A boolean flag indicating whether the subplot plot should be a polar projection.  Defaults to *False*.
        + `projection`: A string giving the name of a custom projection to be used for the subplot. This projection must have been previously registered. See `matplotlib.projections`.
    + Typical call signature: `subplot(nrows, ncols, plot_number)`
        + `nrows`, `ncols`: used to notionally split the figure into $nrows * ncols$ sub-axes, and *plot_number* is used to identify the particular subplot that this function is to create within the notional grid. 
        + `plot_number`: starts at 1, increments across rows first and has a maximum of $nrows * ncols$.
    + `nrows`, `ncols` and `plot_number` $< 10 $: convenience notation exists

+ `plt.subplots` method
    + Signature: `subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)`
    + Docstring: Create a figure and a set of subplots
    + Args:
        + `nrows`, `ncols` (int): Number of rows/columns of the subplot grid.
        + `sharex`, `sharey` (bool or {'none', 'all', 'row', 'col'}): Controls sharing of properties among x (`sharex`) or y (`sharey`) axes:
            + True or 'all': x- or y-axis will be shared among all subplots.
            + False or 'none': each subplot x- or y-axis will be independent.
            + 'row': each subplot row will share an x- or y-axis.
            + 'col': each subplot column will share an x- or y-axis.
        + `squeeze` (bool): 
            + True - extra dimensions are squeezed out from the returned Axes object:
                + `nrows`=`ncols`=1: the resulting single Axes object is returned as a scalar.
                + $Nx1$ or $1xN$ subplots: the returned object is a 1D numpy object array of Axes objects are returned as numpy 1D arrays.
                + $NxM$ subplots with $N>1$ and $M>1$: returned as a 2D arrays.
            + False - no squeezing at all is done: the returned Axes object is always a 2D array containing Axes instances, even if it ends up being 1x1.
        + `subplot_kw` (dict): Dict with keywords passed to the `~matplotlib.figure.Figure.add_subplot` call used to create each subplot.
        + `gridspec_kw` (dict): Dict with keywords passed to the `~matplotlib.gridspec.GridSpec` class constructor used to create the grid the subplots are placed on.
        + `**fig_kw` : All additional keyword arguments are passed to the `figure` call.
    + Returns:
        + `fig` : `matplotlib.figure.Figure` object
        + `ax` (Axes object or array of Axes objects): ax can be either a single :class:`matplotlib.axes.Axes` object or an array of Axes objects if more than one subplot was created.  The dimensions of the resulting array can be controlled with the `squeeze` keyword.

+ `plt.gcf` method
    + Signature: `plt.gcf()`
    + Docstring: Get a reference to the current figure.
    
+ Demo
    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    # subplot with 1 row, 2 columns, and current axis is 1st subplot axes
    plt.subplot(1, 2, 1)

    linear_data = np.array([1,2,3,4,5,6,7,8])
    plt.plot(linear_data, '-o')

    exponential_data = linear_data**2 
    # subplot with 1 row, 2 columns, and current axis is 2nd subplot axes
    plt.subplot(1, 2, 2)
    plt.plot(exponential_data, '-o')

    # plot exponential data on 1st subplot axes
    plt.subplot(1, 2, 1)
    plt.plot(exponential_data, '-x')

    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(linear_data, '-o')
    # pass sharey=ax1 to ensure the two subplots share the same y axis
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    plt.plot(exponential_data, '-x')

    plt.figure()
    # the right hand side is equivalent shorthand syntax
    plt.subplot(1,2,1) == plt.subplot(121)

    # create a 3x3 grid of subplots
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
    # plot the linear_data on the 5th subplot axes 
    ax5.plot(linear_data, '-')

    # set inside tick labels to visible
    for ax in plt.gcf().get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_visible(True)

    # necessary on some systems to update the plot
    plt.gcf().canvas.draw()
    ```


<a href="https://d3c33hcgiwev3.cloudfront.net/XD9_O_0eEeahPwr4NTFsvg.processed/full/360p/index.mp4?Expires=1529193600&Signature=F9mwszxkvX5LWTNgjh44kDlaR3lLq-PtIMK5Gt2HJq~jvLJ1yJpK93hPwYrWTGPDVRZ9wMbG5STWEtT2ozkzne1x~VZ4c6zvxkBKQeNwwSD0fzJ3WpljuQQX7P6Jis4HXdbmzr3CYvMPVws~CL8RbhMaNfJXGGG2EMEoKU87kxg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Subplots
" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Histograms

+ `np.random.normal` method:
    + Signature: `normal(loc=0.0, scale=1.0, size=None)`
    + Docstring: Draw random samples from a normal (Gaussian) distribution.
    + Parameters:
        + `loc` (float or array_like of floats): Mean ("centre") of the distribution.
        + `scale` (float or array_like of floats): Standard deviation (spread or "width") of the distribution.
        + `size` (int or tuple of ints): Output shape.  
            + shape w/ `(m, n, k)`: $m * n * k$ samples are drawn.  
            + `None`: 
                + `loc` and `scale` both scalars: a single value is returned 
                + otherwise: `np.broadcast(loc, scale).size` samples are drawn.
    + Returns: <br/>
        out (ndarray or scalar): Drawn samples from the parameterized normal distribution.

+ `np.random.gamma` method
    + Signature: `np.random.gamma(shape, scale=1.0, size=None)`
    + Docstring: Draw samples from a Gamma distribution. <br/>
        Samples are drawn from a Gamma distribution with specified parameters, `shape` (sometimes designated "k") and `scale` (sometimes designated "theta"), where both parameters are > 0.
    + Parameters
        + `shape` (float or array_like of floats): The shape of the gamma distribution. Should be greater than zero.
        + `scale` (float or array_like of floats): The scale of the gamma distribution. Should be greater than zero.
        + `size` (int or tuple of ints): Output shape.  
            + If the given shape is, e.g., `(m, n, k)`, then `m * n * k` samples are drawn.  
            + If size is `None` (default), a single value is returned if `shape` and `scale` are both scalars.
            + Otherwise, `np.broadcast(shape, scale).size` samples are drawn.
    + Returns: 
        + `out` (ndarray or scalar): Drawn samples from the parameterized gamma distribution.

    

+ `plt.hist` method:
    + Signature: `plt.hist(x, bins=None, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None, **kwargs)`
    + Docstring: Plot a histogram
    + Usage: 
        + Compute and draw the histogram of *x*. The return value is a tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*, [*patches0*, *patches1*,...]) if the input contains multiple data.
        + Multiple data can be provided via *x* as a list of datasets of potentially different length ([*x0*, *x1*, ...]), or as a 2-D ndarray in which each column is a dataset.  Note that the ndarray form is transposed relative to the list form.
    + Parameters
        + `x` ((n,) array or sequence of (n,) arrays): Input values, this takes either a single array or a sequency of arrays which are not required to be of the same length
        + `bins` (integer or array_like or 'auto'): 
            + `bins + 1` bin edges, consistently with `numpy.histogram`
            +  `bins` a sequence: Unequally spaced bins
            + `auto`: Numpy 1.11
            + Default: the rcParam `hist.bins`
        + `range` (tuple or None): 
            + The lower and upper range of the bins.
            + Lower and upper outliers are ignored.
            + Default: `range` = (x.min(), x.max()).
            + Range has no effect if `bins` is a sequence.
            + If `bins` is a sequence or `range` is specified, autoscaling is based on the specified bin range instead of the range of x.
        + `normed` (boolean): 
            + `True`: the first element of the return tuple will be the counts normalized to form a probability density, i.e., $n/(len(x)*bin)$, i.e., the integral of the histogram will sum to 1. If *stacked* is also *True*, the sum of the histograms is normalized to 1.
        + `weights` ((n, ) array_like or None): An array of weights, of the same shape as `x`.  Each value in `x` only contributes its associated weight towards the bin count (instead of 1).  If `normed` is True, the weights are normalized,   so that the integral of the density over the range remains 1.
        + `cumulative` (boolean): 
            + `True`: a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values. The last bin gives the total number of datapoints.  
            + `normed` & `True`: the histogram is normalized such that the last bin equals 1.
            + evaluate to less than $0$ (e.g., $-1$), the direction of accumulation is reversed.
        + `bottom` (array_like, scalar, or None): Location of the bottom baseline of each bin.  
            + `scalar`: the base line for each bin is shifted by the same amount.
            + `array`: each bin is shifted independently and the length of bottom must match the number of bins.  If None, defaults to 0.
        + `histtype` ({'bar', 'barstacked', 'step',  'stepfilled'}): The type of histogram to draw.
            + 'bar': a traditional bar-type histogram.  If multiple data are given the bars are aranged side by side.
            + 'barstacked': a bar-type histogram where multiple data are stacked on top of each other.
            + 'step': generate a lineplot that is by default unfilled.
            + 'stepfilled': generatesa lineplot that is by default filled.
        + `align` ({'left', 'mid', 'right'}): Controls how the histogram is plotted.
            + 'left': bars are centered on the left bin edges.
            + 'mid': bars are centered between the bin edges.
            + 'right': bars are centered on the right bin edges.
        + `orientation` ({'horizontal', 'vertical'}): If 'horizontal', `~matplotlib.pyplot.barh` will be used for bar-type histograms and the *bottom* kwarg will be the left edges.
        + `rwidth` (scalar or None): 
            + The relative width of the bars as a fraction of the bin width.  
            + `None`: automatically compute the width.
            + Ignored if `histtype` is 'step' or 'stepfilled'.
        + `log` (boolean): 
            + `True`: the histogram axis will be set to a log scale. 
            + `True` and `x` 1D array: empty bins will be filtered out and only the non-empty (`n`, `bins`, `patches`) will be returned.
        + `color` (color or array_like of colors or None): Color spec or sequence of color specs, one per dataset.  Default (`None`) uses the standard line color sequence.
        + `label` (string or None): 
            + String, or sequence of strings to match multiple datasets.  
            + Bar charts yield multiple patches per dataset, but only the first gets     the label, so that the legend command will work as expected.
        + `stacked` (boolean): 
            + `True`: multiple data are stacked on top of each other 
            + `False`: multiple data are aranged side by side if `histtype` is 'bar' or on top of each other if histtype is 'step'
        + `kwargs` : `~matplotlib.patches.Patch` properties
    + Return: 
        + `n` (array or list of arrays):  The values of the histogram bins. See **normed** and **weights** for a description of the possible semantics. If input **x** is an array, then this is an array of length **nbins**. If input is a sequence arrays `[data1, data2,..]`, then this is a list of arrays with the values of the histograms for each of the arrays in the same order.
        + `bins` (array): The edges of the bins. Length $nbins + 1$ ($nbins$ left edges and right edge of last bin).  Always a single array even when multiple data sets are passed in.
        + `patches` (list or list of lists): Silent list of individual patches used to create the histogram or list of such list if multiple input datasets.

+ `plt.scatter` method:
    + Signature: `plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)`
    + Docstring: Make a scatter plot of `x` vs `y`
    + Parameters
        + `x`, `y` (array_like, shape (n, )): Input data
        + `s` (scalar or array_like, shape (n, )): size in $points^2$.  Default is $rcParams['lines.markersize']^2$.
        + `c` (color, sequence, or sequence of color): default: 'b'
            + single color format string, or a sequence of color specifications of length `N`, or a sequence of `N` numbers to be mapped to colors using the `cmap` and `norm` specified via `kwargs` (see below). 
            + Should not be a single numeric RGB or RGBA sequence because that is indistinguishable from an array of values to be colormapped.  
            + A 2-D array in which the rows are RGB or RGBA, however, including the case of a single row to specify the same color for all points.
        + `marker` (`~matplotlib.markers.MarkerStyle`): default: 'o'
            + See `~matplotlib.markers` for more information on the different styles of markers scatter supports. 
            + Either an instance of the class or the text shorthand for a particular marker.
        + `cmap` (`~matplotlib.colors.Colormap`): 
            + A `~matplotlib.colors.Colormap` instance or registered name.
            + Only used if `c` is an array of floats. 
            + None: defaults to rc `image.cmap`.
        + `norm` (`~matplotlib.colors.Normalize`):
            + A `~matplotlib.colors.Normalize` instance is used to scale luminance data to $0$, $1$. 
            + `norm` only used if `c` is an array of floats
            + `None`: use the default `normalize`.
        + `vmin`, `vmax` (scalar):
            + used in conjunction with `norm` to normalize luminance data
            + either `None`: the min and max of the color array used
            + Note if a `norm` instance passed, settings for `vmin` and `vmax` ignored.
        + `alpha` (scalar): The alpha blending value, between $0$ (transparent) and $1$ (opaque)
        + `linewidths` (scalar or array_like): defaults to (`lines.linewidth`,).
        + `verts` (sequence of (x, y)): 
            + `marker` = None: these vertices used to construct the marker.  
            + The center of the marker is located at (0,0) in normalized units.  
            + The overall marker is rescaled by `s`.
        + `edgecolors` (color or sequence of color): 
            + None: defaults to 'face'
            + 'face': the edge color will always be the same as the face color.
            + 'none': the patch boundary will not be drawn.
            + For non-filled markers, `kwarg` is ignored and forced to 'face' internally.
        + `kwargs`: `~matplotlib.collections.Collection` properties
    + Return: paths: `~matplotlib.collections.PathCollection`

+ `gspec.GridSpec` class
    + A class that specifies the geometry of the grid that a subplot will be placed. The location of grid is determined by similar way as the SubplotParams.
    + Constructor: `_init__(self, nrows, ncols, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None)`  <br/>
        The number of rows and number of columns of the grid need to be set. Optionally, the subplot layout parameters (e.g., left, right, etc.) can be tuned.
    + Methods: 
        + `get_subplot_params(fig=None)`: return a dictionary of subplot layout parameters. The default parameters are from rcParams unless a figure attribute is set.
        + `locally_modified_subplot_params()`: 
        + `tight_layout(fig, renderer=None, pad=1.08, h_pad=None, w_pad=None, rect=None)`: Adjust subplot parameters to give specified padding.
            + Parameters:
                + `pad` (float): padding between the figure edge and the edges of subplots, as a fraction of the font-size.
                + `h_pad`, `w_pad` (float): padding (height/width) between edges of adjacent subplots. Defaults to `pad_inches`.
                + `rect` : if rect is given, it is interpreted as a rectangle (left, bottom, right, top) in the normalized figure coordinate that the whole subplots area (including labels) will fit into. Default is (0, 0, 1, 1).
        + `update(**kwargs)`: Update the current values.  If any kwarg is None, default to the current value, if set, otherwise to rc.
    + Methods inherited from `GridSpecBase`:
        + `get_geometry()`: get the geometry of the grid, e.g., 2,3
        + `get_grid_positions(fig)`: return lists of bottom and top position of rows, left and right positions of columns.
        + `get_height_ratios()`
        + `get_width_ratios()`
        + `new_subplotspec(loc, rowspan=1, colspan=1)`: create and return a SuplotSpec instance.
        + `set_height_ratios(height_ratios)`
        + `set_width_ratios(width_ratios)`

+ `set_title` method of `matplotlib.axes._subplots.AxesSubplot`
    + Signature: `set_title(label, fontdict=None, loc='center', **kwargs)`
    + Docstring: Set a title for the axes
    + Parametres:
        + `label` (str): Text to use for the title
        + `fontdict` (dict): A dictionary controlling the appearance of the title text, the default `fontdict` is <br/>
            ```python
            {'fontsize': rcParams['axes.titlesize'],
             'fontweight' : rcParams['axes.titleweight'],
             'verticalalignment': 'baseline',
             'horizontalalignment': loc}
            ```
        + `loc` ({'center', 'left', 'right'}, str): Which title to set, defaults to 'center'
        + `kwargs` : text properties
    + Returns: 
        + `text`: `~matplotlib.text.Text` class; The matplotlib text instance representing the title

+ `set_xlim` meyjod of `matplotlib.axes._subplots.AxesSubplot`
    + Signature: `set_xlim(left=None, right=None, emit=True, auto=False, **kw)`
    + Docstring: Set the data limits for the x-axis
    + Parameters
        + `left` (scalar): The left xlim (default: None, which leaves the left limit unchanged).
        + `right` (scalar): The right xlim (default: None, which leaves the right limit unchanged).
        + `emit` (bool): Whether to notify observers of limit change (default: True).
        + `auto` (bool or None): Whether to turn on autoscaling of the x-axis. 
            + True: on
            + False: off (default action)
            + None: unchanged.
        + `xlimits` (tuple): The left and right xlims may be passed as the tuple (`left`, `right`) as the first positional argument (or as the `left` keyword argument).
    + Returns
        + `xlimits` (tuple): Returns the new x-axis limits as (`left`, `right`).

+ `set_ylim` method of `matplotlib.axes._subplots.AxesSubplot`
    + Signature: `set_ylim(bottom=None, top=None, emit=True, auto=False, **kw)`
    + Docstring: Set the data limits for the y-axis
    + Parameters
        + `bottom` (scalar): The bottom ylim (default: None, which leaves the bottom limit unchanged).
        + `right` (scalar): The top ylim (default: None, which leaves the top limit unchanged).
        + `emit` (bool): Whether to notify observers of limit change (default: True).
        + `auto` (bool or None): Whether to turn on autoscaling of the y-axis. 
            + True: on
            + False: off (default action)
            + None: unchanged.
        + `ylimits` (tuple): The bottom and top ylims may be passed as the tuple (`bottom`, `top`) as the first positional argument (or as the `bottom` keyword argument).
    + Returns
        + `ylimits` (tuple): Returns the new x-axis limits as (`bottom`, `top`).

+ `invert_axis` method of `matplotlib.axes._subplots.AxesSubplot`
    + Signature: `invert_xaxis()`
    + Docstring: Invert the x-axis.


+ Demo
    ```python
    # create 2x2 grid of axis subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
    axs = [ax1,ax2,ax3,ax4]

    # draw n = 10, 100, 1000, and 10000 samples from the normal distribution and plot corresponding histograms
    for n in range(0,len(axs)):
        sample_size = 10**(n+1)
        sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
        axs[n].hist(sample)
        axs[n].set_title('n={}'.format(sample_size))

    # repeat with number of bins set to 100
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
    axs = [ax1,ax2,ax3,ax4]

    for n in range(0,len(axs)):
        sample_size = 10**(n+1)
        sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
        axs[n].hist(sample, bins=100)
        axs[n].set_title('n={}'.format(sample_size))

    plt.figure()
    Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
    X = np.random.random(size=10000)
    plt.scatter(X,Y)

    # use gridspec to partition the figure into subplots
    import matplotlib.gridspec as gridspec

    plt.figure()
    gspec = gridspec.GridSpec(3, 3)

    top_histogram = plt.subplot(gspec[0, 1:])
    side_histogram = plt.subplot(gspec[1:, 0])
    lower_right = plt.subplot(gspec[1:, 1:])

    Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
    X = np.random.random(size=10000)
    lower_right.scatter(X, Y)
    top_histogram.hist(X, bins=100)
    s = side_histogram.hist(Y, bins=100, orientation='horizontal')

    # clear the histograms and plot normed histograms
    top_histogram.clear()
    top_histogram.hist(X, bins=100, normed=True)
    side_histogram.clear()
    side_histogram.hist(Y, bins=100, orientation='horizontal', normed=True)
    # flip the side histogram's x axis
    side_histogram.invert_xaxis()

    # change axes limits
    for ax in [top_histogram, lower_right]:
        ax.set_xlim(0, 1)
    for ax in [side_histogram, lower_right]:
        ax.set_ylim(-5, 5)

    %%HTML
    <img src='http://educationxpress.mit.edu/sites/default/files/journal/WP1-Fig13.jpg' />
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/A8NyMP0gEearIRLZY_MkaA.processed/full/360p/index.mp4?Expires=1529280000&Signature=I2A9c7A~PNcINWpfZH5qeOvuWydgefz8mxOKFDVOqBmTsxod8XParEeRbGy25b1AT3TZbselw09AMeaVkV1O0EHfDEwCMrE7hmLjbSKHVOXwdSo4iWuZEVjg9mn46ujqeIN2bqE7jHNyKOJHO3A4puPijLfCa6HaJ3AtXlEw45s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Histograms" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Selecting the Number of Bins in a Histogram: A Decision Theoretic (Optional)

He, K., & Meeden, G. (1997). [Selecting the Number of Bins in a Histogram: A Decision Theoretic Approach](http://users.stat.umn.edu/~gmeeden/papers/hist.pdf). Journal of Statistical Planning and inference, 61(1), 49-59.

## Box Plots

+ `pd.describe` method of `pandas.core.frame.DataFrame`
    + Signature: `describe(percentiles=None, include=None, exclude=None)`
    + Docstring: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding `NaN` values.
    + Parameters
        + `percentiles` (list-like of numbers): The percentiles to include in the output. All should fall between 0 and 1. The default is `[.25, .5, .75]`, which returns the 25th, 50th, and 75th percentiles.
        + `include` ('all', list-like of dtypes or None): A white list of data types to include in the result. Ignored  for `Series`. Here are the options:
            + 'all': All columns of the input will be included in the output.
            + A list-like of dtypes : Limits the results to the provided data types. To limit the result to numeric types submit `numpy.number`. To limit it instead to categorical objects submit the `numpy.object` data type. Strings can also be used in the style of `select_dtypes` (e.g. `df.describe(include=['O'])`)
            + None (default): The result will include all numeric columns.
        + `exclude` (list-like of dtypes or None): A black list of data types to omit from the result. Ignored for `Series`. Here are the options:
            + A list-like of dtypes: Excludes the provided data types from the result. To select numeric types submit `numpy.number`. To select categorical objects submit the data type `numpy.object`. Strings can also be used in the style of `select_dtypes` (e.g. `df.describe(include=['O'])`)
        + None (default): The result will exclude nothing.
    + Returns
        + `summary`:  Series/DataFrame of summary statistics

+ `plt.boxplot` method
    + Signature: `boxplot(x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_xticks=True, autorange=False, zorder=None, hold=None, data=None)`
    + Docstring: Make a box and whisker plot. <br/>
        Make a box and whisker plot for each column of `x` or each vector in sequence `x`.  The box extends from the lower to upper quartile values of the data, with a line at the median. The whiskers extend from the box to show the range of the data.  Flier points are those past the end of the whiskers.
    + Parameters
        + `x` (Array or a sequence of vectors): The input data.
        + `notch` (bool): 
            + `True`: notched box plot produced
            + `False`: a rectangular boxplot is produced. The notches represent the confidence interval (CI) around the median. 
        + `sym` (str): 
            + The default symbol for flier points. 
            + Enter an empty string ('') if no fliers
            + `None`: the fliers default to 'b+' 
            + flierprops `kwarg`: more control
        + `vert` (bool): 
            + `True`: makes the boxes vertical
            + `False`: everything is drawn horizontally.
        + `whis` (float, sequence, or string) (default = 1.5):  determine the reach of the whiskers to the beyond the first and third quartiles. 
        + `bootstrap` (int): Specifies whether to bootstrap the confidence intervals around the median for notched boxplots.
        + `usermedians` (array-like):  An array or sequence whose first dimension (or length) is compatible with `x`.
        + `conf_intervals` (array-like): Array or sequence whose first dimension (or length) is compatible with `x` and whose second dimension is 2. 
        + `positions` (array-like): Sets the positions of the boxes. Defaults to
        `range(1, N+1)` where N is the number of boxes to be drawn.
        + `widths` (scalar or array-like): Sets the width of each box either with a scalar or a sequence. The default is 0.5, or $0.15*(distance between
        extreme positions)$, if that is smaller.
        + `patch_artist` (bool): If `False` produces boxes with the Line2D artist. Otherwise, boxes and drawn with Patch artists.
        + `labels` (sequence): Labels for each dataset.
        + `manage_xticks` (bool): If the function should adjust the xlim and xtick locations.
        + `autorange` (bool): When `True` and the data are distributed such that the  25th and  75th percentiles are equal, `whis` is set to `'range'` such that the whisker ends are at the minimum and maximum of the data.
        + `meanline` (bool): If `True` (and ``showmeans`` is `True`), will try to render the mean as a line spanning the full width of the box according to `meanprops`.
        + `zorder` (scalar): Sets the zorder of the boxplot.
        + `showcaps` (bool): Show the caps on the ends of whiskers.
        + `showbox` (bool): Show the central box.
        + `showfliers` (bool): Show the outliers beyond the caps.
        + `showmeans` (bool): Show the arithmetic means.
        + `capprops` (dict):  Specifies the style of the caps.
        + `boxprops` (dict):  Specifies the style of the box.
        + `whiskerprops` (dict):  Specifies the style of the whiskers.
        + `flierprops` (dict):  Specifies the style of the fliers.
        + `medianprops` (dict):  Specifies the style of the median.
        + `meanprops` (dict):  Specifies the style of the mean.
    + Returns
        + `result` (dict):  A dictionary mapping each component of the boxplot to a list of the :class:`matplotlib.lines.Line2D` instances created. That dictionary has the following keys (assuming vertical boxplots):
            + `boxes`: the main body of the boxplot showing the quartiles and the median's confidence intervals if enabled.
            + `medians`: horizontal lines at the median of each box.
            + `whiskers`: the vertical lines extending to the most extreme, non-outlier data points.
            + `caps`: the horizontal lines at the ends of the whiskers.
            + `fliers`: points representing data that extend beyond the whiskers (fliers).
            + `means`: points or lines representing the means.

+ `plt.clear` method
    + Signature: `plt.clf()`
    + Docstring: Clear the current figure.

+ `inset_axes` method of `mpl_toolkits.axes_grid1.inset_locator`
    + Signature: `inset_axes(parent_axes, width, height, loc=1, bbox_to_anchor=None, bbox_transform=None, axes_class=None, axes_kwargs=None, borderpad=0.5)`
    + Docstring: Create an inset axes with a given width and height. <br/>
        Both sizes used can be specified either in inches or percentage of the parent axes.
    + Parameters
        + `parent_axes` (`matplotlib.axes.Axes`): Axes to place the inset axes.
        + `width`, `height` (float or str): Size of the inset axes to create.
        + `loc` (int or string): Location to place the inset axes. The valid locations are

            | Location | Num | | Location | Num |
            |----------|-----|-|----------|-----|
            | 'upper right'  | 1| | 'upper left'   | 2|
            | 'lower left'   | 3| | 'lower right'  | 4|
            | 'right'        | 5| | 'center left'  | 6|
            | 'center right' | 7| | 'lower center' | 8|
            | 'upper center' | 9| | 'center'       | 10|
        + `bbox_to_anchor` (tuple or `matplotlib.transforms.BboxBase`): Bbox that the inset axes will be anchored. Can be a tuple of [left, bottom, width, height], or a tuple of [left, bottom].
        + `bbox_transform` (`matplotlib.transforms.Transform`): Transformation for the bbox. if None, `parent_axes.transAxes` is used.
        + `axes_class` (`matplotlib.axes.Axes` type): If specified, the inset axes created with be created with this class's constructor.
        + `axes_kwargs` (dict): Keyworded arguments to pass to the constructor of the inset axes.<br/>
            Valid arguments include:
            + `adjustable`: [ 'box' | 'datalim' | 'box-forced'] 
            + `agg_filter`: unknown
            + `alpha`: float (0.0 transparent through 1.0 opaque) 
            + `anchor`: unknown
            + `animated`: [True | False] 
            + `aspect`: unknown
            + `autoscale_on`: unknown
            + `autoscalex_on`: unknown
            + `autoscaley_on`: unknown
            + `axes`: an `~matplotlib.axes.Axes` instance 
            + `axes_locator`: unknown
            + `axisbelow`: [ *True* | *False* | 'line' ] 
            + `clip_box`: a `matplotlib.transforms.Bbox` instance 
            + `clip_on`: [True | False] 
            + `clip_path`: [ (`~matplotlib.path.Path`, `~matplotlib.transforms.Transform`) | `~matplotlib.patches.Patch` | None ] 
            + `color_cycle`: unknown
            + `contains`: a callable function 
            + `facecolor`: unknown
            + `fc`: unknown
            + `figure`: unknown
            + `frame_on`: [ *True* | *False* ] 
            + `gid`: an id string 
            + `label`: string or anything printable with '%s' conversion. 
            + `navigate`: [ *True* | *False* ] 
            + `navigate_mode`: unknown
            + `path_effects`: unknown
            + `picker`: [None|float|boolean|callable] 
            + `position`: unknown
            + `rasterization_zorder`: unknown
            + `rasterized`: [True | False | None] 
            + `sketch_params`: unknown
            + `snap`: unknown
            + `title`: unknown
            + `transform`: `~matplotlib.transforms.Transform` instance 
            + `url`: a url string 
            + `visible`: [True | False] 
            + `xbound`: unknown
            + `xlabel`: unknown
            + `xlim`: unknown
            + `xmargin`: unknown
            + `xscale`: ['linear' | 'log' | 'logit' | 'symlog']
            + `xticklabels`: sequence of strings
            + `xticks`: sequence of floats 
            + `ybound`: unknown
            + `ylabel`: unknown
            + `ylim`: unknown
            + `ymargin`: unknown
            + `yscale`: ['linear' | 'log' | 'logit' | 'symlog']
            + `yticklabels`: sequence of strings
            + `yticks`: sequence of floats
            + `zorder`: any number 
        + `borderpad` (float): Padding between inset axes and the bbox_to_anchor. Defaults to 0.5.
    + Returns
        + `inset_axes` (`axes_class`): Inset axes object created.

+ `margins` method of `mpl_toolkits.axes_grid1.parasite_axes.AxesHostAxes`
    + Signature: `margins(*args, **kw)`
    + Docstring: Set or retrieve autoscaling margins
    + Returns:  `xmargin`, `ymargin`

+ `tick_right`, `tick_left` method of `matplotlib.axis.YAxis`
    + Signature: `tick_right()`, `tick_left()`
    + Docstring: use ticks only on right/left

+ `tick_top`, `tick_bottom` method of `matplotlib.axis.xAxis`
    + Signature: `tick_top()`, `tick_bottom()`
    + Docstring: use ticks only on top/bottom

+ Demo 
    ```python
    import pandas as pd

    normal_sample = np.random.normal(loc=0.0, scale=1.0, size=10000)
    random_sample = np.random.random(size=10000)
    gamma_sample = np.random.gamma(2, size=10000)

    df = pd.DataFrame({'normal': normal_sample, 
                    'random': random_sample, 
                    'gamma': gamma_sample})

    df.describe()

    plt.figure()
    # create a boxplot of the normal data, assign the output to a variable to supress output
    _ = plt.boxplot(df['normal'], whis='range')

    # clear the current figure
    plt.clf()
    # plot boxplots for all three of df's columns
    _ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')

    plt.figure()
    _ = plt.hist(df['gamma'], bins=100)

    import mpl_toolkits.axes_grid1.inset_locator as mpl_il

    plt.figure()
    plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')
    # overlay axis on top of another 
    ax2 = mpl_il.inset_axes(plt.gca(), width='60%', height='40%', loc=2)
    ax2.hist(df['gamma'], bins=100)
    ax2.margins(x=0.5)

    # switch the y axis ticks for ax2 to the right side
    ax2.yaxis.tick_right()

    # if `whis` argument isn't passed, boxplot defaults to showing 1.5*interquartile (IQR) whiskers with outliers
    plt.figure()
    _ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ] )
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/JHejcf0gEearIRLZY_MkaA.processed/full/360p/index.mp4?Expires=1529280000&Signature=ItKBIP6v95~SjLUrUeA~NFO834sqPSamyKahtBSNAAINMVRrzmNNQHvccQIm1kTRv5YvMp0uIcmvxz~ig9EWJetIKroBgYcO0elWmXQk4QMYzqx9OApgRsbStnBTsUtlQrCj5bud0gyHXf0OVoJJuJQIxnZjkF2wW-Ns8mEdSU0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Box Plots" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Heatmaps

+ `plt.hist2d` method
    + Signature: `plt.hist2d(x, y, bins=10, range=None, normed=False, weights=None, cmin=None, cmax=None, hold=None, data=None, **kwargs)`
    + Docstring: Make a 2D histogram plot.
    + Parameters
        + `x`, `y` (array_like, shape (n, )): Input values
        + `bins` ([None | int | [int, int] | array_like | [array, array]]): 
            + `int`: the number of bins for the two dimensions ($nx=ny=bins$).
            + `[int, int]`: the number of bins in each dimension ($nx, ny = bins$).
            + `array_like`: the bin edges for the two dimensions ($x_edges=y_edges=bins$).
            + `[array, array]`: the bin edges in each dimension ($x_edges, y_edges = bins$).
            + default: 10
        + `range` (array_like shape(2, 2)): 
            + The leftmost and rightmost edges of the bins along each dimension (if not specified explicitly in the bins parameters): `[[xmin, xmax], [ymin, ymax]]`. 
            + All values outside of this range will be considered outliers and not tallied in the histogram.
        + `normed` (boolean): Normalize histogram.
        + `weights` (array_like, shape (n, )): An array of values w_i weighing each sample (x_i, y_i).
        + `cmin` (scalar): All bins that has count less than cmin will not be displayed and these count values in the return value count histogram will also be set to nan upon return
        + `cmax` (scalar): All bins that has count more than cmax will not be displayed (set to none before passing to imshow) and these count values in the return value count histogram will also be set to nan upon return
        + `cmap`: {Colormap, string}; A `matplotlib.colors.Colormap` instance.  If not set, use rc settings.
        + `norm`: Normalize; A :class:`matplotlib.colors.Normalize` instance is used to
        scale luminance data to `[0, 1]`. If not set, defaults to `Normalize()`.
        + `vmin`/`vmax` {None, scalar}: Arguments passed to the `Normalize` instance.
        + `alpha` ($0 <= scalar <= 1$ or `None`): The alpha blending value.
    + Returns: `(counts, xedges, yedges, Image)` 

+ `plt.colorbar` method
    + Signature: `plt.colorbar(mappable=None, cax=None, ax=None, **kw)`
    + Docstring: Add a colorbar to a plot.
    + Parametres:
        + `mappable`: the `~matplotlib.image.Image`,`~matplotlib.contour.ContourSet`, etc. to which the colorbar applies; this argument is mandatory for the `~matplotlib.figure.Figure.colorbar` method but optional for the `~matplotlib.pyplot.colorbar` function, which sets the default to the current image.
        + `cax`: axes object into which the colorbar will be drawn
        + `ax`: parent axes object(s) from which space for a new colorbar axes will be stolen. If a list of axes is given they will all be resized to make room for the colorbar axes.
        + `use_gridspec`: 
            + `cax` = None: a new `cax` is created as an instance of Axes. 
            + `ax` is an instance of Subplot and `use_gridspec`= True: `cax` is created as an instance of Subplot using the grid_spec module.
    + Axes Properties: 
        + `orientation`: vertical or horizontal
        + `fraction`: 0.15; fraction of original axes to use for colorbar
        + `pad`: 0.05 if vertical, 0.15 if horizontal; fraction of original axes between colorbar and new image axes
        + `shrink`: 1.0; fraction by which to shrink the colorbar
        + `aspect`: 20; ratio of long to short dimensions
        + `anchor`: (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal; the anchor point of the colorbar axes
        + `panchor`: (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal; the anchor point of the colorbar parent axes. If False, the parent axes' anchor will be unchanged
    + colorbar Properties:
        + `extend`      [ 'neither' | 'both' | 'min' | 'max']: 
        + `extendfrac`  [ None | 'auto' | length | lengths ]
        + `extendrect`  [ False | True ]
        + `spacing`     [ 'uniform' | 'proportional' ]
        + `ticks`       [ None | list of ticks | Locator object ]
        + `format`      [ None | format string | Formatter object ]
        + `drawedges`   [ False | True ] 


+ Demo
    ```python
    plt.figure()

    Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
    X = np.random.random(size=10000)
    _ = plt.hist2d(X, Y, bins=25)

    plt.figure()
    _ = plt.hist2d(X, Y, bins=100)

    # add a colorbar legend
    plt.colorbar()
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/Opdlk_0gEeaI9Q7Pym09lA.processed/full/360p/index.mp4?Expires=1529280000&Signature=laoJZ4o8OB6YrDuNQ5OML60vOlkdJylh8KuBUoBeCIoGkSczIIX204JoS93vLfE1OHNYx~eiCa51UmBpi3AXl~rZBoabBUNTLk9VUvAOI~zEmdr4G-lIb7iy5cvBVv8we55DQmh8ftpmnpvWhijHUSPlKfqS61igb3GDEHNcXfU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Heatmaps" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Animation

+ `plt.cla` method:
    + Signature: `plt.cla()`
    + Doccstring: Clear the current axes.

+ `plt.annotate` method
    + Signature: `plt.annotate(s, xy, xytext=None, xycoords=None, textcoords =None, arrowprops=None, annotation_clip=None, **kwargs)`
    + Docstring: Annotate the point `xy` with text `s`.
    + Parameters:
        + `s` (str): The text of the annotation
        + `xy` (iterable): Length 2 sequence specifying the *(x,y)* point to annotate
        + `xytext` (iterable): Length 2 sequence specifying the *(x,y)* to place the text at.  If None, defaults to ``xy``.
        + `xycoords` (str, Artist, Transform, callable or tuple): 
            + `str`: 
                + 'figure points': points from the lower left of the figure
                + 'figure pixels': pixels from the lower left of the figure
                + 'figure fraction': fraction of figure from lower left
                + 'axes points': points from lower left corner of axes
                + 'axes pixels': pixels from lower left corner of axes
                + 'axes fraction': fraction of axes from lower left
                + 'data': use the coordinate system of the object being annotated (default)
                + 'polar': `(theta,r)` if not native 'data' coordinates
            + If a `~matplotlib.artist.Artist` object is passed in the units are fraction if it's bounding box.
            + If a `~matplotlib.transforms.Transform` object is passed in use that to transform ``xy`` to screen coordinates
            + If a callable it must take a `~matplotlib.backend_bases.RendererBase` object as input and return a `~matplotlib.transforms.Transform` or `~matplotlib.transforms.Bbox` object
            + If a `tuple` must be length 2 tuple of str, `Artist`, `Transform` or callable objects.  The first transform is used for the *x* coordinate and the second for *y*.
        + `textcoords` (str, `Artist`, `Transform`, callable or tuple): The coordinate system that `xytext` is given, which may be different than the coordinate system used for `xy`.
            + 'offset points': offset (in points) from the *xy* value
            + 'offset pixels': offset (in pixels) from the *xy* value
        + `arrowprops` (dict): If not None, properties used to draw a `~matplotlib.patches.FancyArrowPatch` arrow between `xy` and `xytext`.
            + width: the width of the arrow in points
            + headwidth: the width of the base of the arrow head in points
            + headlength: the length of the arrow head in points
            + shrink: fraction of total length to 'shrink' from both ends
            + ?: any key to :class:`matplotlib.patches.FancyArrowPatch`
            + `'arrowstyle'`:
                + `'-'`: None
                + `'->'`: head_length=0.4,head_width=0.2
                + `'-['`: widthB=1.0,lengthB=0.2,angleB=None
                + `'|-|'`: widthA=1.0,widthB=1.0
                + `'-|>'`: head_length=0.4,head_width=0.2
                + `'<-'`: head_length=0.4,head_width=0.2
                + `'<->'`: head_length=0.4,head_width=0.2
                + `'<|-'`: head_length=0.4,head_width=0.2
                + `'<|-|>'`: head_length=0.4,head_width=0.2
                + `'fancy'`: head_length=0.4,head_width=0.4,tail_width=0.4
                + `'simple'`: head_length=0.5,head_width=0.5,tail_width=0.2
                + `'wedge'`: tail_width=0.3,shrink_factor=0.5
            + keys for `~matplotlib.patches.FancyArrowPatch` are:
                + arrowstyle: the arrow style
                + connectionstyle: the connection style
                + relpos: default is (0.5, 0.5)
                + patchA: default is bounding box of the text
                + patchB: default is None
                + shrinkA: default is 2 points
                + shrinkB: default is 2 points
                + mutation_scale: default is text size (in points)
                + mutation_aspect: default is 1.
                + ?: any key for :class:`matplotlib.patches.PathPatch`
        + `annotation_clip` (bool): Controls the visibility of the annotation when it goes outside the axes area.
            + `True`: the annotation will only be drawn when the ``xy`` is inside the axes. 
            + `False`:  the annotation will always be drawn regardless of its position.
            + `None` (default): behave as `True` only if `xycoords` is "data".
    + Returns: Annotation

+ `FuncAnimation(TimedAnimation)` Class
    + Signature: `animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, save_count=0, interval=200, repeat_delay=None, repeat=True, blit=False)`
    + Docstring: Makes an animation by repeatedly calling a function `func`
    + Parameters
        + `fig` (matplotlib.figure.Figure): The figure object that is used to get draw, resize, and any other needed events.
        + `func` (callable):  The function to call at each frame.  The first argument will be the next value in `frames`.   Any additional positional arguments can be supplied via the `fargs` parameter.
        + `frames` (iterable, int, generator function, or None): Source of data to pass `func` and each frame of the animation
            + If an iterable, then simply use the values provided.  If the iterable has a length, it will override the `save_count` kwarg.
            + If an integer, equivalent to passing `range(frames)`
            + If a generator function, then must have the signature  `def gen_function() -> obj:`
            + In all of these cases, the values in `frames` is simply passed through to the user-supplied `func` and thus can be of any type.
            + `None`: equivalent to passing `itertools.count`.
        + `init_func` (callable):  A function used to draw a clear frame. 
            + If not given, the results of drawing from the first item in the frames sequence will be used. This function will be called once before the first frame.
            + If blit=True, ``init_func`` must return an iterable of artists to be re-drawn.
        + `fargs` (tuple or None): Additional arguments to pass to each call to `func`
        + `save_count` (int): The number of values from `frames` to cache.
        + `interval` (number): Delay between frames in milliseconds.
        + `repeat_delay` (number):  If the animation in repeated, adds a delay in milliseconds before repeating the animation.
        + `repeat` (bool): Controls whether the animation should repeat when the sequence of frames is completed.
        + `blit` (bool): Controls whether blitting is used to optimize drawing.  

+ Demo
    ```python
    import matplotlib.animation as animation

    n = 100
    x = np.random.randn(n)

    # create the function that will do the plotting, where curr is the current frame
    def update(curr):
        # check if animation is at the last frame, and if so, stop the animation a
        if curr == n: 
            a.event_source.stop()
        plt.cla()
        bins = np.arange(-4, 4, 0.5)
        plt.hist(x[:curr], bins=bins)
        plt.axis([-4,4,0,30])
        plt.gca().set_title('Sampling the Normal Distribution')
        plt.gca().set_ylabel('Frequency')
        plt.gca().set_xlabel('Value')
        plt.annotate('n = {}'.format(curr), [3,27])

    fig = plt.figure()
    a = animation.FuncAnimation(fig, update, interval=100)
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/d1tGBf0gEearIRLZY_MkaA.processed/full/360p/index.mp4?Expires=1529280000&Signature=T1Bh9qH5yrxq19J-Nit1QzuPimVwS2szgy26XWmtqV0I55Urt7fsZVwIisr77~pEvpwHK344YRkZuuL1nHII1vQkYQZjf2zXHfaL4uza2rMtyZuPgozJU5fXbqpcubuE4PuVfMnTNCIooinTKkYLSoDGNh35gxGDt7CX6WppkdQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Animation" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Interactivity


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>
