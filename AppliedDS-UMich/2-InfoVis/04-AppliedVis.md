# Module 4: Applied Visualization

## Module 4 Jupyter Notebook

+ [Launch Web Page](https://www.coursera.org/learn/python-plotting/notebook/bXNWg/module-4-jupyter-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=99Bc14VzRpSQXNeFc8aUoA&next=%2Fnotebooks%2FWeek4.ipynb)
+ [Local Notebook](./notebooks/Week04.ipynb)

+ Environment Code
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    %matplotlib notebook
    ```

## Plotting with Pandas

+ `plt.style.availabe`: 
    + list of pre-defined styles
    + [Style shhets reference](https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html)

+ `plt.style.use` method of matplotlib.style.core
    + Signature: `plt,style.user(style)`
    + Docstring: Use matplotlib style settings from a style specification.
    + Parameters
        + `style` (str, dict, or list): A style specification. Valid options are:
            + str: The name of a style or a path/URL to a style file. For a list of available style names, see `style.available`.
            + dict: Dictionary with valid key/value pairs for `matplotlib.rcParams`.
            + list: A list of style specifiers (str or dict) applied from first to last in the list.
    
+ `ax.set_aspect` method of `matplotlib.axes._subplots.AxesSubplot`
    + Signature: `ax.set_aspect(aspect, adjustable=None, anchor=None)`
    + Docstring: set aspect
    + Parameters:
        + *aspect*
            + 'auto': automatic; fill position rectangle with data
            + 'normal': same as 'auto'; deprecated
            + 'equal':same scaling from data to plot units for x and y
            + num: a circle will be stretched such that the height is num times the width. aspect=1 is the same as `aspect='equal'`.
        + *adjustable*
            + 'box': change physical size of axes
            +  'datalim': change xlim or ylim
            + 'box-forced': same as 'box', but axes can be shared
        + *anchor*
            + 'C': centered
            + 'SW': lower left corner
            + 'S': middle of bottom edge
            + 'SE': lower right corner

+ `df.plot.box` method of `pandas.plotting._core.FramePlotMethods`
    + Signature: `df.plot.box(by=None, **kwds)`
    + Docstring: Boxplot
    + Parameters
        + `by` (string or sequence): Column in the DataFrame to group by.
        + `**kwds`: Keyword arguments to pass on to `pandas.DataFrame.plot`.
    + Returns: `axes`: matplotlib.AxesSubplot or np.array of them
    
+ `df.plot.hist` method of `pandas.plotting._core.FramePlotMethods`
    + Signature: `df.plot.hist(by=None, bins=10, **kwds)`
    + Docstring: Histogram
    + Parameters
        + `by` (string or sequence): Column in the DataFrame to group by.
        + `bins` (integer): Number of histogram bins to be used
        + `**kwds`: Keyword arguments to pass on to :py:meth:`pandas.DataFrame.plot`.
    + Returns: `axes`: matplotlib.AxesSubplot or np.array of them

+ + `df.plot.kde` method of `pandas.plotting._core.FramePlotMethods`
    + Signature: `df.plot.kde(**kwds)`
    + Docstring: Kernel Density Estimate plot
    + Parameters
        + `**kwds`: Keyword arguments to pass on to :py:meth:`pandas.DataFrame.plot`.
    + Returns: `axes`: matplotlib.AxesSubplot or np.array of them

+ `pd.plotting.scatter_matrix` method 
    + Signature: `pd.plotting.scatter_matrix(frame, alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds)`
    + Docstring: Draw a matrix of scatter plots.
    + Parameters:
        + `frame`: DataFrame
        + `alpha` (float): amount of transparency applied
        + `figsize` (float,float): a tuple (width, height) in inches
        + `ax` (Matplotlib axis object): 
        + `grid` (bool): setting this to True will show the grid
        + `diagonal` ({‘hist’, ‘kde’}): pick between ‘kde’ and ‘hist’ for either Kernel Density Estimation or Histogram plot in the diagonal
        + `marker` (str): Matplotlib marker type, default ‘.’
        + `hist_kwds`: other plotting keyword arguments to be passed to hist function
        + `density_kwds`: other plotting keyword arguments to be passed to kernel density estimate plot
        + `range_padding` (float): relative extension of axis range in x and y with respect to $(x_max - x_min)$ or $(y_max - y_min)$, default $0.05$
        + `kwds`` (other plotting keyword arguments to be passed to scatter function

+ `pd.tools.plotting.parallel_coordinates`
    + Signature: `pd.tools.plotting.parallel_coordinates(data. col)`
    + Docstring: Parallel coordinates is a plotting technique for plotting multivariate data, see the Wikipedia entry for an introduction. Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.
    <br/><img src="https://pandas.pydata.org/pandas-docs/stable/_images/parallel_coordinates.png" alt="tParallel Coordinatesext" width="450">

+ Demo
    ```python
    # see the pre-defined styles provided.
    plt.style.available
    # ['bmh',
    #  'classic',
    #  'dark_background',
    #  'fivethirtyeight',
    #  'ggplot',
    #  'grayscale',
    #  'seaborn-bright',
    #  'seaborn-colorblind',
    #  'seaborn-dark-palette',
    #  'seaborn-dark',
    #  'seaborn-darkgrid',
    #  'seaborn-deep',
    #  'seaborn-muted',
    #  'seaborn-notebook',
    #  'seaborn-paper',
    #  'seaborn-pastel',
    #  'seaborn-poster',
    #  'seaborn-talk',
    #  'seaborn-ticks',
    #  'seaborn-white',
    #  'seaborn-whitegrid',
    #  'seaborn',
    #  '_classic_test']

    # use the 'seaborn-colorblind' style
    plt.style.use('seaborn-colorblind')

    # ### DataFrame.plot

    np.random.seed(123)

    df = pd.DataFrame({'A': np.random.randn(365).cumsum(0), 
                    'B': np.random.randn(365).cumsum(0) + 20,
                    'C': np.random.randn(365).cumsum(0) - 20}, 
                    index=pd.date_range('1/1/2017', periods=365))
    df.head()
    #                   A            B            C
    # 2017-01-01    -1.085631    20.059291    -20.230904
    # 2017-01-02    -0.088285    21.803332    -16.659325

    df.plot(); # add a semi-colon to the end of the plotting call to suppress unwanted output

    # We can select which plot we want to use by passing it into the 'kind' parameter.

    df.plot('A','B', kind = 'scatter');

    # You can also choose the plot kind by using the `DataFrame.plot.kind` methods instead of providing the `kind` keyword argument.
    # 
    # `kind` :
    # - `'line'`` (line plot (default)
    # - `'bar'`` (vertical bar plot
    # - `'barh'`` (horizontal bar plot
    # - `'hist'`` (histogram
    # - `'box'`` (boxplot
    # - `'kde'`` (Kernel Density Estimation plot
    # - `'density'`` (same as 'kde'
    # - `'area'`` (area plot
    # - `'pie'`` (pie plot
    # - `'scatter'`` (scatter plot
    # - `'hexbin'`` (hexbin plot

    # create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'
    df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')

    ax = df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')
    ax.set_aspect('equal')

    df.plot.box();
    df.plot.hist(alpha=0.7);
    # [Kernel density estimation plots](https://en.wikipedia.org/wiki/Kernel_density_estimation) are useful for deriving a smooth continuous function from a given sample.
    df.plot.kde();

    # ### pandas.tools.plotting
    # [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)

    iris = pd.read_csv('iris.csv')
    iris.head()

    pd.tools.plotting.scatter_matrix(iris);

    plt.figure()
    pd.tools.plotting.parallel_coordinates(iris, 'Name');
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/R2quYemDEea5-BKh39zqpA.processed/full/360p/index.mp4?Expires=1529798400&Signature=HayJDR06425xrB0zPr6ImPe5YeXYeykZztEkgILIoossIO4QA9w1E-wddRw8MwbAoiNTCSowJcNrPKr1Y2tyBLxXk48Zaf2~90NDoJ8HuqhaKLZbbZYUOW99d8kQ18iVW8SEpC1VqQvUS9hLIqjtJTOTwDhB38DScg~lO8jhyQA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Plotting w/ Pandas" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Seaborn

+ seaborn: statistical data visualization
    + [Official Website](https://seaborn.pydata.org/)
    + Python visualization library based on matplotlib
    + Some of the features that seaborn offers:
        + Several [built-in themes](https://seaborn.pydata.org/tutorial/aesthetics.html#aesthetics-tutorial) for styling matplotlib graphics
        + Tools for choosing [color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial) to make beautiful plots that reveal patterns in your data
        + Functions for visualizing [univariate](https://seaborn.pydata.org/examples/distplot_options.html#distplot-options) and [bivariate](https://seaborn.pydata.org/examples/joint_kde.html#joint-kde) distributions or for [comparing](https://seaborn.pydata.org/examples/grouped_violinplots.html#grouped-violinplots) them between subsets of data
        + Tools that fit and visualize [linear regression](https://seaborn.pydata.org/examples/anscombes_quartet.html#anscombes-quartet) models for different kinds of [independent](https://seaborn.pydata.org/examples/pointplot_anova.html#pointplot-anova) and [dependent](https://seaborn.pydata.org/examples/logistic_regression.html#logistic-regression) variables
        + Functions that visualize [matrices of data](https://seaborn.pydata.org/examples/heatmap_annotation.html#heatmap-annotation) and use clustering algorithms to [discover structure](https://seaborn.pydata.org/examples/structured_heatmap.html#structured-heatmap) in those matrices
        + A function to plot [statistical timeseries](https://seaborn.pydata.org/examples/timeseries_from_dataframe.html#timeseries-from-dataframe) data with flexible estimation and representation of uncertainty around the estimate
        + High-level abstractions for structuring [grids of plots](https://seaborn.pydata.org/examples/faceted_histogram.html#faceted-histogram) that let you easily build [complex](https://seaborn.pydata.org/examples/many_facets.html#many-facets) visualizations
    + Seaborn aims to make visualization a central part of exploring and understanding data.
    + [Seborn API](https://seaborn.pydata.org/api.html)
    + [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

+ `np.concatenate` method of `numpy.core.multiarray`
    + Signature: `concatenate((a1, a2, ...), axis=0)`
    + Docstring: Join a sequence of arrays along an existing axis.
    + Parameters
        + `a1, a2, ...` (sequence of array_like): The arrays must have the same shape, except in the dimension corresponding to `axis` (the first, by default).
        + `axis` (int): The axis along which the arrays will be joined.
    + Returns: res (ndarray): The concatenated array.

+ `sns.set_style` method
    + Signature: `sns.set_style(style=None, rc=None)`
    + Docstring: Set the aesthetic style of the plots.  This affects things like the color of the axes, whether a grid is enabled by default, and other aesthetic elements.
    + Parameters:
        + `style` (dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}): A dictionary of parameters or the name of a pre-configured set.
        + `rc` (dict): Parameter mappings to override the values in the preset seaborn style dictionaries. This only updates parameters that are considered part of the style definition.

+ `sns.kdeplot` method:
    + Signature: `sns.kdeplot(data, data2=None, shade=False, vertical=False, kernel='gau', bw='scott', gridsize=100, cut=3, clip=None, legend=True, cumulative=False, shade_lowest=True, cbar=False, cbar_ax=None, cbar_kws=None, ax=None, **kwargs)`
    + Docstring: Fit and plot a univariate or bivariate kernel density estimate.
    + Parameters:
        + `data` (1d array-like): Input data.
        + `data2` (1d array-like): Second input data. If present, a bivariate KDE will be estimated.
        + `shade` (bool): If True, shade in the area under the KDE curve (or draw with filled contours when data is bivariate).
        + `vertical` (bool): If True, density is on x-axis.
        + `kernel` ({‘gau’ | ‘cos’ | ‘biw’ | ‘epa’ | ‘tri’ | ‘triw’ }): Code for shape of kernel to fit with. Bivariate KDE can only use gaussian kernel.
        + `bw` ({‘scott’ | ‘silverman’ | scalar | pair of scalars }): Name of reference method to determine kernel size, scalar factor, or scalar for each dimension of the bivariate plot.
        + `gridsize` (int): Number of discrete points in the evaluation grid.
        + `cut` (scalar): Draw the estimate to cut * bw from the extreme data points.
        + `clip` (pair of scalars, or pair of pair of scalars): Lower and upper bounds for datapoints used to fit KDE. Can provide a pair of (low, high) bounds for bivariate plots.
        + `legend` (bool): If True, add a legend or label the axes when possible.
        + `cumulative` (bool): If True, draw the cumulative distribution estimated by the kde.
        + `shade_lowest` (bool): If True, shade the lowest contour of a bivariate KDE plot. Not relevant when drawing a univariate plot or when shade=False. Setting this to False can be useful when you want multiple densities on the same Axes.
        + `cbar` (bool): If True and drawing a bivariate KDE plot, add a colorbar.
        + `cbar_ax` (matplotlib axes): Existing axes to draw the colorbar onto, otherwise space is taken from the main axes.
        + `cbar_kws` (dict): Keyword arguments for fig.colorbar().
        + `ax` (matplotlib axes): Axes to plot on, otherwise uses current axes.
        + `kwargs` (key, value pairings): Other keyword arguments are passed to `plt.plot()` or `plt.contour{f}` depending on whether a univariate or bivariate plot is being drawn.
    + Returns: `ax` (matplotlib `Axes`): `Axes` with plot.
    + Example <br/><img src="https://seaborn.pydata.org/_images/seaborn-kdeplot-3.png" alt="https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot" width="600">

+ `sns.distplot` method:
    + Signature: `sns.distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)`
    + Docstring: Flexibly plot a univariate distribution of observations.  This function combines the matplotlib hist function (with automatic calculation of a good default bin size) with the seaborn kdeplot() and rugplot() functions. It can also fit scipy.stats distributions and plot the estimated PDF over the data.
    + Parameters:
        + `a` (Series, 1d-array, or list.): Observed data. If this is a Series object with a name attribute, the name will be used to label the data axis.
        + `bins` (argument for matplotlib hist(), or None): Specification of hist bins, or None to use Freedman-Diaconis rule.
        + `hist` (bool): Whether to plot a (normed) histogram.
        + `kde` (bool): Whether to plot a gaussian kernel density estimate.
        + `rug` (bool): Whether to draw a rugplot on the support axis.
        + `fit` (random variable object): An object with fit method, returning a tuple that can be passed to a pdf method a positional arguments following an grid of values to evaluate the pdf on.
        + `{hist, kde, rug, fit}_kws` (dictionaries): Keyword arguments for underlying plotting functions.
        + `color` (matplotlib color): Color to plot everything but the fitted curve in.
        + `vertical` (bool): If True, oberved values are on y-axis.
        + `norm_hist` (bool): If True, the histogram height shows a density rather than a count. This is implied if a KDE or fitted density is plotted.
        + `axlabel` (string, False, or None): Name for the support axis label. If None, will try to get it from a.namel if False, do not set a label.
        + `label` (string): Legend label for the relevent component of the plot
        + `ax` (matplotlib axis): if provided, plot on this axis
    + Returns: `ax` (matplotlib `Axes`): Returns the `Axes` object with the plot for further tweaking
    + Example <br/><img src="https://seaborn.pydata.org/_images/seaborn-distplot-6.png" alt="https://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot" width="600">

+ `sns.joinplot` method:
    + Signature: `sns.jointplot(x, y, data=None, kind='scatter', stat_func=<function pearsonr>, color=None, size=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)`
    + Docstring: Draw a plot of two variables with bivariate and univariate graphs.  This function provides a convenient interface to the `JointGrid` class, with several canned plot kinds. This is intended to be a fairly lightweight wrapper; if you need more flexibility, you should use `JointGrid` directly.
    + Parameters: 
        + `x`, `y` (strings or vectors): Data or names of variables in data.
        + `data` (DataFrame): DataFrame when x and y are variable names.
        + `kind` ({ “scatter” | “reg” | “resid” | “kde” | “hex” }): Kind of plot to draw.
        + `stat_func` (callable or None): Function used to calculate a statistic about the relationship and annotate the plot. Should map x and y either to a single value or to a (value, p) tuple. Set to None if you don’t want to annotate the plot.
        + `color` (matplotlib color): Color used for the plot elements.
        + `size` (numeric): Size of the figure (it will be square).
        + `ratio` (numeric): Ratio of joint axes size to marginal axes height.
        + `space` (numeric): Space between the joint and marginal axes
        + `dropna` (bool): If True, remove observations that are missing from x and y.
        + `{x, y}lim` (two-tuples): Axis limits to set before plotting.
        + `{joint, marginal, annot}_kws` (dicts): Additional keyword arguments for the plot components.
        + `kwargs` (key, value pairings): Additional keyword arguments are passed to the function used to draw the plot on the joint Axes, superseding items in the joint_kws dictionary.
    + Returns: `grid` (`JointGrid`): `JointGrid` object with the plot on it. 
    + Example <br/><img src="https://seaborn.pydata.org/_images/seaborn-jointplot-2.png" alt="https://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot" width="600">

+ `sns.pairplot` method:
    + Signature: `sns.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='hist', markers=None, size=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None)`
    + Docstring: Plot pairwise relationships in a dataset.  By default, this function will create a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. The diagonal Axes are treated differently, drawing a plot to show the univariate distribution of the data for the variable in that column.
    + Parameters:
        + `data` (DataFrame): Tidy (long-form) dataframe where each column is a variable and each row is an observation.
        + `hue` (string (variable name)): Variable in data to map plot aspects to different colors.
        + `hue_order` (list of strings
        + `Order for the levels of the hue variable in the palette
        + `palette` (dict or seaborn color palette
        + `Set of colors for mapping the hue variable. If a dict, keys should be values in the hue variable.
        + `vars` (list of variable names): Variables within data to use, otherwise use every column with a numeric datatype.
        + `{x, y}_vars` (lists of variable names): Variables within data to use separately for the rows and columns of the figure; i.e. to make a non-square plot.
        + `kind` ({‘scatter’, ‘reg’}): Kind of plot for the non-identity relationships.
        + `diag_kind` ({‘hist’, ‘kde’}): Kind of plot for the diagonal subplots.
        + `markers` (single matplotlib marker code or list): Either the marker to use for all datapoints or a list of markers with a length the same as the number of levels in the hue variable so that differently colored points will also have different scatterplot markers.
        + `size` (scalar): Height (in inches) of each facet.
        + `aspect` (scalar): Aspect * size gives the width (in inches) of each facet.
        + `dropna` (boolean): Drop missing values from the data before plotting.
        + `{plot, diag, grid}_kws` (dicts): Dictionaries of keyword arguments.
    + Returns: `grid` (`PairGrid`): Returns the underlying `PairGrid` instance for further tweaking.
    + Example: <br/><img src="https://seaborn.pydata.org/_images/seaborn-pairplot-2.png" alt="https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot" width="600">

+ `sns.swarmplot` method:
    + Signature: `sns.swarmplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs)`
    + Docstring: Draw a categorical scatterplot with non-overlapping points.  This function is similar to stripplot(), but the points are adjusted (only along the categorical axis) so that they don’t overlap. This gives a better representation of the distribution of values, although it does not scale as well to large numbers of observations (both in terms of the ability to show all the points and in terms of the computation needed to arrange them). This style of plot is often called a “beeswarm”.
    + Parameters: 
        + `x`, `y`, `hue` (names of variables in data or vector data): Inputs for plotting long-form data. See examples for interpretation.
        + `data` (DataFrame, array, or list of arrays): Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise it is expected to be long-form.
        + `order`, `hue_order` (lists of strings): Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
        + `split` (bool): When using hue nesting, setting this to True will separate the strips for different hue levels along the categorical axis. Otherwise, the points for each level will be plotted in one swarm.
        + `orient` (“v” | “h”): Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables, but can be used to specify when the “categorical” variable is a numeric or when plotting wide-form data.
        + `color` (matplotlib color): Color for all of the elements, or seed for a gradient palette.
        + `palette` (palette name, list, or dict): Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
        + `size` (float): Diameter of the markers, in points. (Although plt.scatter is used to draw the points, the size argument here takes a “normal” markersize and not size^2 like plt.scatter.
        + `edgecolor` (matplotlib color, “gray” is special-cased): Color of the lines around each point. If you pass "gray", the brightness is determined by the color palette used for the body of the points.
        + `linewidth` (float): Width of the gray lines that frame the plot elements.
        + `ax` (matplotlib Axes): Axes object to draw the plot onto, otherwise uses the current Axes.
    + Returns: ax` (matplotlib `Axes`): Returns the `Axes` object with the boxplot drawn onto it.
    + Example <br/><img src="https://seaborn.pydata.org/_images/seaborn-swarmplot-1.png" alt="tehttps://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplotxt" width="450">

+ `sns.violinplot` method:
    + Signature: `sns.violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs)`
    + Docstring: Draw a combination of boxplot and kernel density estimate.  A violin plot plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.
    + Parameters:
        + `x`, `y`, `hue` (names of variables in data or vector data): Inputs for plotting long-form data. See examples for interpretation.
        + `data` (DataFrame, array, or list of arrays): Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise it is expected to be long-form.
        + `order`, `hue_order` (lists of strings): Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
        + `bw` ({‘scott’, ‘silverman’, float}): Either the name of a reference rule or the scale factor to use when computing the kernel bandwidth. The actual kernel size will be determined by multiplying the scale factor by the standard deviation of the data within each bin.
        + `cut` (float): Distance, in units of bandwidth size, to extend the density past the extreme datapoints. Set to 0 to limit the violin range within the range of the observed data (i.e., to have the same effect as trim=True in ggplot.
        + `scale` ({“area”, “count”, “width”}): The method used to scale the width of each violin. If area, each violin will have the same area. If count, the width of the violins will be scaled by the number of observations in that bin. If width, each violin will have the same width.
        + `scale_hue` (bool): When nesting violins using a hue variable, this parameter determines whether the scaling is computed within each level of the major grouping variable (scale_hue=True) or across all the violins on the plot (scale_hue=False).
        + `gridsize` (int): Number of points in the discrete grid used to compute the kernel density estimate.
        + `width` (float): Width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable.
        + `inner` ({“box”, “quartile”, “point”, “stick”, None}): Representation of the datapoints in the violin interior. If box, draw a miniature boxplot. If quartiles, draw the quartiles of the distribution. If point or stick, show each underlying datapoint. Using None will draw unadorned violins.
        + `split` (bool): When using hue nesting with a variable that takes two levels, setting split to True will draw half of a violin for each level. This can make it easier to directly compare the distributions.
        + `dodge` (bool): When hue nesting is used, whether elements should be shifted along the categorical axis.
        + `orient` (“v” | “h”): Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables, but can be used to specify when the “categorical” variable is a numeric or when plotting wide-form data.
        + `linewidth` (float): Width of the gray lines that frame the plot elements.
        + `color` (matplotlib color): Color for all of the elements, or seed for a gradient palette.
        + `palette` (palette name, list, or dict): Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
        + `saturation` (float): Proportion of the original saturation to draw colors at. Large patches often look better with slightly desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.
        + `ax` (matplotlib Axes): Axes object to draw the plot onto, otherwise uses the current Axes.
    + Returns: `ax` (matplotlib Axes): Returns the Axes object with the boxplot drawn onto it.
    + Example <br/><img src="https://seaborn.pydata.org/_images/seaborn-violinplot-1.png" alt="https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot" width="600">


+ Demo
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    %matplotlib notebook

    np.random.seed(1234)
    v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
    v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')

    plt.figure()
    plt.hist(v1, alpha=0.7, bins=np.arange(-50,150,5), label='v1');
    plt.hist(v2, alpha=0.7, bins=np.arange(-50,150,5), label='v2');
    plt.legend();

    # plot a kernel density estimation over a stacked barchart
    plt.figure()
    plt.hist([v1, v2], histtype='barstacked', normed=True);
    v3 = np.concatenate((v1,v2))
    sns.kdeplot(v3);

    plt.figure()
    # we can pass keyword arguments for each individual component of the plot
    sns.distplot(v3, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'});

    sns.jointplot(v1, v2, alpha=0.4);

    grid = sns.jointplot(v1, v2, alpha=0.4);
    grid.ax_joint.set_aspect('equal')

    sns.jointplot(v1, v2, kind='hex');

    # set the seaborn style for all the following plots
    sns.set_style('white')

    sns.jointplot(v1, v2, kind='kde', space=0);

    iris = pd.read_csv('iris.csv')
    iris.head()

    sns.pairplot(iris, hue='Name', diag_kind='kde', size=2);

    plt.figure(figsize=(8,6))
    plt.subplot(121)
    sns.swarmplot('Name', 'PetalLength', data=iris);
    plt.subplot(122)
    sns.violinplot('Name', 'PetalLength', data=iris);
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/Col2oemCEea0PA6g5Mr3FA.processed/full/360p/index.mp4?Expires=1529798400&Signature=H8YAOdNQ7C3Tn~t6pryO9JPVAyG0V-5P9t9LOZ0MJvBiPjOP5WGxRsxzZBG69uImZ1gJoh21IRFbbu3iJk0-ySGdAswxWIfeHB2dIhq-BisdCLPRw5~e~ooAJyao2ek1CJ6To~HJFbVMzcrocyZtDM9cA7WF8KXVgVe1mLCdhbk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Seaborn" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Spurious Correlations

If you look long enough you'll find correlations all over! A spurious correlation is a correlation that happens from chance alone, and these correlations are one of the reasons scientists use statistics with confidence levels (p-values or confidence intervals) and/or limit their searching to theoretically grounded effects (e.g. pre-registration of hypotheses).

The [spurious correlations](http://www.tylervigen.com/spurious-correlations) website lists a number of correlations which exist but seem unlikely to be causally related to one another. Do you see any in this list which you think are not spurious, and may have a causal relationship? Discuss in the forums, and provide links as appropriate.

