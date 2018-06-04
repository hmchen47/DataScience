# Basic Charting

## Module 2 Jupyter Notebook


## Introduction

<a href="https://d3c33hcgiwev3.cloudfront.net/4zWLUP0bEearIRLZY_MkaA.processed/full/360p/index.mp4?Expires=1528243200&Signature=EraViNehUyefRIEkis6KAgHH74uxhl8sP1N2aZGUK1zfj8spaoVfnRyfyc5xOiuU~bQsCoHdm06Qmnr~T29Kfs6lQZQFe0zM~5fuYlqF0jHeP0Wxno~UdC7HYQbTzMh4Z9On0~7qYpTOCo5uz6yXWciVCTlY6rh2bpHMCfmXwv4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Introduction" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Matplotlib Architecture

+ Matplotlib Architecture
    + Backend Layer
        + Deals with the rendering of plots to screen or files
        + In Jupyter notebooks we use the inline backend
    + Artist Layer
        + Contains containers such as Figure, Subplot, and Axes
        + Contains primitives, such as a Line2D and Rectangle, and collections, such as a PathCollection
    + Scripting Layer
        + Simplifies access to the Artist and Backend layers
        + `pyplot` used in this 
+ Building Visualization
    + Procedural methods for building a visualzation, eg. pyplot
    + Declarative methods for vizualizing data, e.g HTML, D3.js, SVG

    [<br/><img src="https://matplotlib.org/_images/inheritance-3c2c45b4bd2f47461e9113da50594813ad9f98d0.png" alt="artist module" width="600"><br/>](http://matplotlib.org/api/artist_api.html) Artisitc Module

+ Demo
    ```python
    %matplotlib notebook # enable matplotlib in IPython Jyputer notebook

    import matplotlib as mpl
    mpl.get_backend()   # 'TkAgg' for CLI Python3, 'nbAgg' for iPython

    ```
<a href="https://d3c33hcgiwev3.cloudfront.net/K0bxmP0cEeaeKwpzIn5n7A.processed/full/360p/index.mp4?Expires=1528243200&Signature=D6cbxN4rLY99Vn80YNlOh33YTF1MsqIOvULT3aTza75V0rUZ51y6-1uC4oAJL-R-WiwZ-yc70I2JzV8kf2hhgVB5OLppIX~3U45-bQn402pKfT26fE2vpw2MTuwfUTmWQPY0IVLJ7yTidVaussi8-te49qDOI0pBn98eFOVo5KM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Matplotlib Architecture" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Matplotlib

Hunter, J., & Droettboom, M. (2012). [matplotlib](http://www.aosabook.org/en/matplotlib.html) in A. Brown (Ed.), The Architecture of Open Source Applications, Volume II: Structure, Scale, and a Few More Fearless Hacks (Vol. 2). [lulu.com](https://www.lulu.com/)

### Overview of matplotlib Architecture

+ Three layers 
    + From bottom to top are: backend, artist, and scripting
    + Each layer that sits above another layer knows how to talk to the layer below it, but the lower layer is not aware of the layers above it.

#### Backend Layer

+ Provide concrete implementations of the abstract interface classes
    + `FigureCanvas`: encapsulate the concept of a surface to draw onto (e.g. "the paper")
    + `Render`: provide a low-level drawing interface for putting ink onto the canvas
    + `Event`: handle user inputs such as keyboard and mouse events
    [<br/><img src="https://delftswa.gitbooks.io/desosa-2017/content/matplotlib/images-matplotlib/functional_view.png" alt="Functional View" width="450">](https://delftswa.gitbooks.io/desosa-2017/content/matplotlib/chapter.html)

+ `FigureCanvas`: `Qt` user interface toolkit as example
    + Know how to insert itself into a native `Qt` window
    + Transfer the matplotlib `Renderer` commands onto the canvas
    + Translate native Qt events into the matplotlib Event framework
    + The abstract base classes reside in `matplotlib.backend_bases` and all of the derived classes live in dedicated modules.

+ `Render`:
    + Provide a low-level drawing interface for putting ink onto the canvas
    + Originally motivated by the GDK `Drawable` interface, which implements such _primitive methods_ as `draw_point`, `draw_line`, `draw_rectangle`, `draw_image`, `draw_polygon`, and `draw_glyphs`
    + Core pixel-based renderer using the C++ template library _Anti-Grain Geometry_ or `agg`
        + A high-performance library for rendering anti-aliased 2D graphics
        + Produce attractive images
    + matplotlib provides support for inserting pixel buffers rendered by the `agg` backend into each user interface toolkit

+ `Event`:
    + Map underlying UI events like `key-press-event` or `mouse-motion-event` to the matplotlib classes `KeyEvent` or `MouseEvent`
    + Connect events to callback functions and interact with figure and data
    + e.g., how to toggle all of the lines in an Axes window when the user types `t'
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def on_press(event):
        if event.inaxes is None: return
        for line in event.inaxes.lines:
            if event.key=='t':
                visible = line.get_visible()
                line.set_visible(not visible)
        event.inaxes.figure.canvas.draw()

    fig, ax = plt.subplots(1)
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax.plot(np.random.rand(2, 20))
    plt.show()
    ```

#### Artist Layer

+ The middle layer of the matplotlib stack and where much of the heavy lifting happens
+ The object that knows how to take the `Renderer` (the paintbrush) and puts ink on the canvas
+ Artist instance: everything in a matplotlib Figure, including, the title, the lines, the tick labels, the images, and so on
+ `matplotlib.artist.Artist`: base class
    + the `transformation`: translate the artist coordinate system to the canvas coordinate system 
    + the `visibility`
    + the `clip box`: define the region the artist can paint into
    + the `label`
    + the `interface`: to handle user interaction such as "picking"; i.e., detecting when a mouse click happens over the artist

    <img src="http://www.aosabook.org/images/matplotlib/artists_figure.png" alt=" A figure" width="250">
    <img src="http://www.aosabook.org/images/matplotlib/artists_tree.png" alt="The hierarchy of artist instances used to draw" width="340">

+ Coupling the `Artist` hierarchy and the backend `draw` method, e.g.,  create SomeArtist which subclasses Artist, the essential method that SomeArtist must implement is draw, which is passed a renderer from the backend
    ```python
    class SomeArtist(Artist):
        'An example Artist that implements the draw method'

        def draw(self, renderer):
            """Call the appropriate renderer methods to paint self onto canvas"""
            if not self.get_visible():  return

            # create some objects and use renderer to draw self here
            renderer.draw_path(graphics_context, path, transform)
    ```
+ Two types of `Artist`s in the hierarchy
    + __Primitive artists__: the kinds of objects seen in a plot, including `Line2D`, `Rectangle`, `Circle`, and `Text`
    + __Composite artists__: collections of Artists such as the `Axis`, `Tick`, `Axes`, and `Figure`

+ `Axes`: the most important composite artist
    + Define most of the matplotlib API plotting methods
    + Containing most of the graphical elements that make up the background of the plot — the ticks, the axis lines, the grid, the patch of color (plot background)
    + Containing numerous helper methods that create primitive artists and adding them to the Axes instance

+ A simple Python script
    ```python
    # Import the FigureCanvas from the backend of your choice
    #  and attach the Figure artist to it.
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    fig = Figure()
    canvas = FigureCanvas(fig)

    # Import the numpy library to generate the random numbers.
    import numpy as np
    x = np.random.randn(10000)

    # Now use a figure method to create an Axes artist; the Axes artist is
    #  added automatically to the figure container fig.axes.
    # Here "111" is from the MATLAB convention: create a grid with 1 row and 1
    #  column, and use the first cell in that grid for the location of the new
    #  Axes.
    ax = fig.add_subplot(111)

    # Call the Axes method hist to generate the histogram; hist creates a
    #  sequence of Rectangle artists for each histogram bar and adds them
    #  to the Axes container.  Here "100" means create 100 bins.
    ax.hist(x, 100)

    # Decorate the figure with a title and save it.
    ax.set_title('Normal distribution with $\mu=0, \sigma=1$')
    fig.savefig('matplotlib_histogram.png')
    ```

### Scripting Layer (pyplot)

+ Using the API works very well, especially for programmers
+ Usually the appropriate programming paradigm when writing a web application server, a UI application, or perhaps a script to be shared with other developers
+ `matplotlib.pyplot` interface: provide a lighter scripting interface to simplify common tasks
+ `pyplot` script: same as the Python code above
    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.random.randn(10000)
    plt.hist(x, 100)
    plt.title(r'Normal distribution with $\mu=0, \sigma=1$')
    plt.savefig('matplotlib_histogram.png')
    plt.show()
    ```
    + `import matplotlib.pyplot as plt`: parses a local configuration file in which the user state their preference for a default backend
    + `plt.hist(x, 100)`:  the first plotting command in the script;  check internal data structures to confirm existence of a current `Figure` instance
        + True: extract the current `Axes` and direct plotting to the `Axes.hist` API call
        + False: create a `Figure` and `Axes`, set these as current, and direct the plotting to `Axes.hist`
    + `plt.title(r'Normal distribution with $\mu=0, \sigma=1$')`: call the existing Axes instance method Axes.set_title or create new instance
    + `plt.show()`: force the `Figure` to render

+ A stateful interface that handles much of the boilerplate for 
    + creating figures and axes 
    + connecting them to the backend of your choice
    + maintaining module-level internal data structures representing the current figure and axes to which to direct plotting commands

+ `matplotlib.pyplot.plot`: a somewhat stripped-down and simplified version of `pyplot`'s frequently used line plotting function

+ Example:
    ```python
    @autogen_docstring(Axes.plot)
    def plot(*args, **kwargs):
        ax = gca()

        ret = ax.plot(*args, **kwargs)
        draw_if_interactive()

        return ret
    ```
    + `@autogen_docstring(Axes.plot)`: decorator; extract the documentation string from the corresponding API method and attache a properly formatted version to the `pyplot.plot` method
    + `*args` and `**kwargs`: all the arguments and keyword arguments that are passed to the method
    + `ax = gca()`: invoke the stateful machinery to "get current Axes" (each Python interpreter can have only one "current axes"), and create the `Figure` and `Axes` if necessary.
    + `ret = ax.plot(*args, **kwargs`: forward the function call and its arguments to the appropriate `Axes` method, and store the return value to be returned later.

### Backend Refactoring

+ Old methods of drawing APIs: `draw_arc`, `draw_image`, `draw_line_collection`, `draw_line`, `draw_lines`, `draw_point`, `draw_quad_mesh`, `draw_polygon_collection`, `draw_polygon`, `draw_rectangle`, `draw_regpoly_collection`
+ Issue: more backend methods meant it took much longer to write a new backend
+ Refactored for matplotlib version 0.98, the only backend APIs
    + `draw_path`: Draws compound polygons, made up of line and Béezier segments
    + `draw_image`: Draws raster images.
    + `draw_text`: Draws text with the given font properties.
    + `get_text_width_height_descent`: Given a string of text, return its metrics.
+ Further refactoring: removing the need for the `draw_text` method by drawing text using `draw_path`
+ Optional backend API methods
    + `draw_markers`: Draws a set of markers.
    + `draw_path_collection`: Draws a collection of paths.
    + `draw_quad_mesh`: Draws a quadrilateral mesh.

### Transforms

+ Coordinate systems include
    + __data__: the original raw data values
    + __axes__: the space defined by a particular axes rectangle
    + __figure__: the space containing the entire figure
    + __display__: the physical coordinates used in the output (e.g. points in PostScript, pixels in PNG)

+ __Transformation node__: how to transform from one coordinate system to another w/ every `Artist`
    + Cnnected together in a directed graph, where each node is dependent on its parent
    + Coordinates in data space transformed all the way to coordinates in the final output file
+ Most transformations invertible
    + Transform graph sets up dependencies between transformation nodes: when a parent node's transformation changes, such as when an `Axes`'s limits are changed, any transformations related to that
    + 
+ Types of transform nodes: 
    + Affine transformations:
        + The family of transformations that preserve straight lines and ratios of distances, including rotation, translation, scale and skew
        + 2-dim affine transformations using $3 \times 3$ affine transform matrix:
            $$ \left[ \begin{array}{c} x^{\prime} \\ y^{\prime} \\ 1 \right] 
                = \left[ \begin{array}{c} 
                    s_x & \thera_x & t_x \\ \theta_y & s_y & t_y \\ 0 & 0 & 1
                  \right] 
                  \left[ \begin{array}{c} x \\ y \\ 1 \right]
            $$
        + Property: composed transformations by using matrix multiplication; i.e., to perform a series of affine transformations, the transformation matrices can first be multiplied together only once, and the resulting matrix can be used to transform coordinates
    + Non-affine transformations:
        + Used for logarithmic scaling, polar plots and geographical projections
        + Able to freely mix with affine ones in the transformation graph

        <br/><img src="http://www.aosabook.org/images/matplotlib/nonaffine_transforms.png" alt="The same data plotted with three different non-affine transformations: logarithmic, polar and Lambert" width="450">

+ Blended transformation
    + A special transformation node using one transformation for the x axis and another for the y axis
    + __Separable__ transformation: the x and y coordinates are independent, but the transformations themselves may be either affine or non-affine
    + e.g., plot logarithmic plots where either or both of the x and y axes may have a logarithmic scale
    + Allowing the available scales to be combined in arbitrary ways
    + Allowing sharing of axes: to "link" the limits of one plot to another and ensure that when one is panned or zoomed, the other is updated to match

        <br/><img src="http://www.aosabook.org/images/matplotlib/transform_tree.png" alt="An example transformation graph" width="450">

### The Polyline Pipeline

+ Earlier version issue: tangle multiple steps together
+ Refactoring for discrete steps in a "path conversion" pipeline
    + __Transformation__: 
        + The coordinates are transformed from data coordinates to figure coordinates. + Affine transformation: simple matrix multiplication
        + Arbitrary transformations: transformation functions called to transform the coordinates into figure space.
    + __Handle missing data__: 
        + The data array may have portions where the data is missing or invalid. 
        + Either stting those values to NaN, or using numpy masked arrays
        + skip over the missing data segments using MOVETO commands for Vector output formats, such as PDF, and rendering libraries, such as Agg -> not often have a concept of missing data when plotting a polyline, and tell the renderer to pick up the pen and begin drawing again at a new point.
    + __Clipping__: 
        + Points outside of the boundaries of the figure can increase the file size by including many invisible points.
        + overflow errors in the rendering of the output file w/ large or very small coordinate values can cause
        + Pipeline clips as it enters and exits the edges of the figure to prevent both of these problems
    + __Snapping__: 
        + Perfectly vertical and horizontal lines can look fuzzy due to antialiasing when their centers are not aligned to the center of a pixel
        + The snapping step determining whether the entire polyline is made up of horizontal and vertical segments (such as an axis-aligned rectangle)
        + Rounds each resulting vertex to the nearest pixel center
        + Only used for raster backends, since vector backends should continue to have exact data points
    + __Simplification__: 
        + Dense plots: many of the points on the line may not visible
        + Including these points in the plot increases file size, and may even hit limits on the number of points allowed in the file format
        + Any points that lie exactly on the line between their two neighboring points are removed
        + Decision depends on a threshold based on what would be visible at a given resolution specified by the user
        <br/><img src="http://www.aosabook.org/images/matplotlib/path_simplification.png" alt="The figure on the right is a close-up of the figure on the left. The circled vertex is automatically removed by the path simplification algorithm, since it lies exactly on the line between its neighboring vertices, and therefore is redundant." width="300">

### Math Text

+ Two ways to render math expressions.
    + `usetex`:
        + Using a full copy of TeX on the user's machine to render the math expression
        + TeX outputs the location of the characters and lines in the expression in its native DVI (device independent) format
        + matplotlib parses the DVI file and converts to its set of drawing commands that one of its output backends then renders directly onto the plot
        + Handling a great deal of obscure math syntax
        + Requirement: a full and working installation of TeX
    + `mathtext`: 
        + A direct port of the TeX math-rendering engine
        + glued onto a much simpler parser written using the `pyparsing` parsing framework
        + Based on the published copy of the TeX source code
        + Building up a tree of boxes and glue (in TeX nomenclature), then laid out by the layout engine
        + Featuresported on an as-needed basis, with an emphasis on frequently used and non-discipline-specific features firs

### Regression Testing

+ Lack of automated tests -> lack regressions in features
+ Script that generated a number of plots exercising various features of matplotlib, particularly those that were hard to get right
+ The current matplotlib testing script generates a number of plots, but instead of requiring manual intervention, those plots are automatically compared to baseline images. 
+ All of the tests are run inside of the nose testing framework, which makes it very easy to generate a report of which tests failed.
+ Testing framework computes the histogram of both images, and calculates the root-mean-square of their difference w/ a given threshold
+ The testing framework tests multiple backends for each plot: PNG, PDF and SVG
+ Vector backends: the testing framework first renders the file to a raster using an external tool (Ghostscript for PDF and Inkscape for SVG) and then uses those rasters for comparison



<br/><img src="url" alt="text" width="450">
<br/><img src="url" alt="text" width="450">
<br/><img src="url" alt="text" width="450">
<br/><img src="url" alt="text" width="450">


## Ten Simple Rules for Better Figures

Rougier et al. share their ten simple rules for drawing better figures, and use matplotlib to provide illustrative examples. As you read this paper, reflect on what you learned in the first module of the course -- principles from Tufte and Cairo -- and consider how you might realize these using matplotlib.

Rougier NP, Droettboom M, Bourne PE (2014) [Ten Simple Rules for Better Figures](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833). PLoS Comput Biol 10(9): e1003833. doi:10.1371/journal.pcbi.

+ Rules
    + Rule 1: Know Your Audience
    + Rule 2: Identify Your Message
    + Rule 3: Adapt the Figure to the Support Medium
    + Rule 4: Captions Are Not Optional
    + Rule 5: Do Not Trust the Defaults
    + Rule 6: Use Color Effectively
    + Rule 7: Do Not Mislead the Reader
    + Rule 8: Avoid “Chartjunk”
    + Rule 9: Message Trumps Beauty
    + Rule 10: Get the Right Tool

## Basic Plotting with Matplotlib

+ Deno
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Scatterplots

+ Deno
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Line Plots

+ Deno
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Bar Charts

+ Deno
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Dejunkifying a Plot

+ Deno
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

