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


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Selecting the Number of Bins in a Histogram: A Decision Theoretic (Optional)


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Box Plots


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Heatmaps


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Animation


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Interactivity


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>
