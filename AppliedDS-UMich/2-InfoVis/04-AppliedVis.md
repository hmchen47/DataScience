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
        + `kwds` : other plotting keyword arguments to be passed to scatter function

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
    # - `'line'` : line plot (default)
    # - `'bar'` : vertical bar plot
    # - `'barh'` : horizontal bar plot
    # - `'hist'` : histogram
    # - `'box'` : boxplot
    # - `'kde'` : Kernel Density Estimation plot
    # - `'density'` : same as 'kde'
    # - `'area'` : area plot
    # - `'pie'` : pie plot
    # - `'scatter'` : scatter plot
    # - `'hexbin'` : hexbin plot

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


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Seaborn

+ Demo
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Spurious Correlations

+ Demo
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

