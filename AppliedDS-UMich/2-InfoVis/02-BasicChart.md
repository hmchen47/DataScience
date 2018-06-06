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
    %matplotlib notebook # provides an interactive environment

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
    + Know how to insert it into a native `Qt` window
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

        def draw(, renderer):
            """Call the appropriate renderer methods to paint  onto canvas"""
            if not .get_visible():  return

            # create some objects and use renderer to draw  here
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



## Ten Simple Rules for Better Figures

Rougier et al. share their ten simple rules for drawing better figures, and use matplotlib to provide illustrative examples. As you read this paper, reflect on what you learned in the first module of the course -- principles from Tufte and Cairo -- and consider how you might realize these using matplotlib.

> Rougier NP, Droettboom M, Bourne PE (2014) [Ten Simple Rules for Better Figures](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833). PLoS Comput Biol 10(9): e1003833. doi:10.1371/journal.pcbi.


+ Rule 1: Know Your Audience
    + How a visual is perceived differs significantly from the intent of the conveyer
    + Identify, as early as possible in the design process, the audience and the message the visual is to convey
    + Audiences
        + Your and your direct collaborators: possibly skip a number of steps in the design process
        + Scientific journal: make sure your figure is correct and conveys all the relevant information to a broader audience
        + Students: special care about the goal is to explain a concept
        + General public: the most difficult audience of all since you need to design a simple, possibly approximated, figure that reveals only the most salient part of your research

+ Rule 2: Identify Your Message
    + Figure: express an idea or introduce some facts or a result that would be too long (or nearly impossible) to explain only with words
    + For an article or during a time-limited oral presentation
    + Clearly identify the role of the figure, i.e., what is the underlying message and how can a figure best express this 
    + Only after identifying the message will it be worth the time to develop your figure

+ Rule 3: Adapt the Figure to the Support Medium
    + A figure can be displayed on a variety of media, such as a poster, a computer monitor, a projection screen (as in an oral presentation), or a simple sheet of paper (as in a printed article).
    + Different medium implies different ways of viewing and interacting with the figure
    + Oral presentation: 
        + The figure must be kept simple and the message must be visually salient in order to grab attention
        + Video-projected figures will be seen from a distance
        + Figure elements must consequently be made thicker (lines) or bigger (points, text), colors should have strong contrast, and vertical text should be avoided, etc.
    + Journal article: 
        + Able to view the figure as long as necessary
        + Details added, along with complementary explanations in the caption
    + Computer screens: possibility to zoom and drag the figure
    + Each type of support medium requires a different figure

+ Rule 4: Captions Are Not Optional
    + A figure should be accompanied by a caption
    + Caption: explain how to read the figure and provide additional precision for what cannot be graphically represented

+ Rule 5: Do Not Trust the Defaults
    + Default settings are used to specify size, font, colors, styles, ticks, markers, etc.
    + Recognize the specific style of each software package (Matlab, Excel, Keynote, etc.) or library (LaTeX, matplotlib, gnuplot, etc.)
    + Default settings: used for virtually any type of plot, not fine-tuned for a specific type of plot
    + All plots require at least some manual tuning of the different settings
        + to better express the message
        + be it for making a precise plot more salient to a broad audience
        + to choose the best colormap for the nature of the data

+ Rule 6: Use Color Effectively
    + Color is an important dimension in human vision and is consequently equally important in the design of a scientific figure.
    + Edward Tufte: greatest ally or worst enemy if not used properly
    + Do not use the default colormap (e.g., jet or rainbow) unless there is an explicit reason to do so.
    + Main categories of colormaps:
        + __Sequential__: one variation of a unique color, used for _quantitative data_ varying from low to high.
        + __Diverging__: variation from one color to another, used to _highlight deviation_ from a median value.
        + __Qualitative__: rapid variation of colors, used mainly for _discrete or categorical data_.

+ Rule 7: Do Not Mislead the Reader
    + Incorrect perception of quantities
        + Software automatically re-scales values
        + Inadvertently misled your readers into visually believing something that does not exist in your data
    + Ensure to always use the simplest type of plots that can convey message
    + Ensure to use labels, ticks, title, and the full range of values when relevant
    + Do not hesitate to ask colleagues about their interpretation of your figures.

+ Rule 8: Avoid “Chartjunk”
    + Chartjunk: the unnecessary or confusing visual elements found in a figure 
    + Any decorations that do not tell the viewer something new must be banned - Edward Tutfe
    + Graphs should ideally “represent all the data that is needed to see and understand what's meaningful.” - Stephen Few

+ Rule 9: Message Trumps Beauty
    + Each scientific domain w/ own set of best practices
    + Knowing these standards
        + facilitate a more direct comparison between models, studies, or experiments
        + help to spot obvious errors in results
    + Online graphics: aesthetic first and content second
    + Science: message and readability of the figure is the most important aspect while beauty is only an option

+ Rule 10: Get the Right Tool
    + Matplotlib
        + A python plotting library, primarily for 2-D plotting, but with some 3-D support
        + Produce publication-quality figures in a variety of hardcopy formats and interactive environments across platforms
        + Cover virtually all scientific domains
    + R: 
        + a language and environment for statistical computing and graphics
        + Provide a wide variety of statistical (linear and nonlinear modeling, classical statistical tests, time-series analysis, classification, clustering, etc.) and graphical techniques, and is highly extensible.
    + Inkscape: 
        + A professional vector graphics editor
        + To a professional vector graphics editor
        + To improve a script-generated figure
        + To read a PDF file in order to extract figures and transform them any way you like
    + TikZ and PGF: 
        + TeX packages for creating graphics programmatically
        + To create sophisticated graphics in a rather intuitive and easy manner
    + GIMP 
        + GNU Image Manipulation Program
        + For photo retouching, image composition, and image authoring
    + ImageMagick
        + A software suite to create, edit, compose, or convert bitmap images from the command line
        + To quickly convert an image into another format
    + D3.js (Data-Driven Documents): a JavaScript library that offers an easy way to create and control interactive data-based graphical forms which run in web browsers
    + Cytoscape: a software platform for visualizing complex networks and integrating these with any type of attribute data
    + Circos: designed for visualizing genomic data but useful if you have data that describes relationships or multilayered annotations of one or more scales


## Basic Plotting with Matplotlib

+ Show matplotlib figures directly in the notebook by using the `%matplotlib notebook` and `%matplotlib inline` magic commands

+ `mpl.get_backend` method
    + Signature: `mpl.get_backend()`
    + Docstring: Return the name of the current backend.

+ `plt.plot` method
    + Signature: `plt.plot(*args, **kwargs)`
    + Docstring: Plot lines and/or markers to the Plot lines and/or markers to the `~matplotlib.axes.Axes` class.  
    + Args: 
        + `args`: variable length argument, allowing for multiple *x*, *y* pairs with an optional format string.
        + `kwargs`: used to set line properties (any property that has a `set_*` method).  You can use this to set a line label (for auto legends), linewidth, anitialising, marker face color, etc.
    + Example for `args`: each of the following is legal::
        ```python
        plot(x, y)        # plot x and y using default line style and color
        plot(x, y, 'bo')  # plot x and y using blue circle markers
        plot(y)           # plot y using x as index array 0..N-1
        plot(y, 'r+')     # ditto, but with red plusses
        ```
        + If `x` and/or `y` is 2-dimensional, then the corresponding will be plotted.
        + If used with labeled data, make sure that the color spec is not included as an element in data, as otherwise the last case `plot("v","r", data={"v":..., "r":...)` can be interpreted as the first case which would do `plot(v, r)` using the default line style and color.
        + If not used with labeled data (i.e., without a data argument), an arbitrary number of `x`, `y`, `fmt` groups can be specified, as in `a.plot(x1, y1, 'g^', x2, y2, 'g-')` 
        + Return value is a list of lines that were added.
        + By default, each line is assigned a different style specified by a 'style cycle'.  To change this behavior, you can edit the `axes.prop_cycle rcParam`.
    + Line style or marker

        | character | description | | character | description |
        |-----------|-------------|-|-----------|-------------|
        | `'-'`  |  solid line style | | `'1'`  |  tri_down marker |
        | `'--'` |  dashed line style | | `'2'`  |  tri_up marker |
        | `'-.'` |  dash-dot line style | | `'3'`  |  tri_left marker |
        | `':'`  |  dotted line style | | `'4'`  |  tri_right marker |
        | `'.'`  |  point marker | | `'*'`  |  star marker |
        | `','`  |  pixel marker | | `'+'`  |  plus marker |
        | `'o'`  |  circle marker | | `'_'`  |  hline marker |
        | `'v'`  |  triangle_down marker | | `'^'`  |  triangle_up marker |
        | `'<'`  |  triangle_left marker | | `'>'`  |  triangle_right marker |
        | `'s'`  |  square marker | | `'p'`  |  pentagon marker |
        | `'h'`  |  hexagon1 marker | | `'H'`  |  hexagon2 marker |
        | `'x'`  |  x marker | | `'|'`  |  vline marker |
        | `'D'`  |  diamond marker | | `'d'`  |  thin_diamond marker |

    + Color abbreviations

        | character |  color | | character |  color |
        |-----------|--------|-|-----------|--------|
        | 'b'       |  blue  | | 'm'       |  magenta |
        | 'g'       |  green | | 'y'       |  yellow |
        | 'r'       |  red   | | 'k'       |  black |
        | 'c'       |  cyan  | | 'w'       |  white |
    + Example for `kwargs`
        ```python
        plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
        plot([1,2,3], [1,4,9], 'rs',  label='line 2')
        axis([0, 4, 0, 10])
        legend()
        ```
        + If you make multiple lines with one plot command, the `kwargs` apply to all those lines, e.g. `plot(x1, y1, x2, y2, antialiased=False)`. Neither line will be `antialiased`.
        + You do not need to use format strings, which are just abbreviations.  All of the line properties can be controlled by keyword arguments.  For example, you can set the `color`, `marker`, `linestyle`, and `markercolor` with 
            ```python 
            plot(x, y, color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
            ```
    + The kwargs are properties of `~matplotlib.lines.Line2D` class:
        + `agg_filter`: unknown
        + `alpha`: float (0.0 transparent through 1.0 opaque)
        + `animated`: [True | False]
        + `antialiased` or aa: [True | False]
        + `axes`: an `~matplotlib.axes.Axes` class instance
        + `clip_box`: a `matplotlib.transforms.Bbox` class instance
        + `clip_on`: [True | False]
        + `clip_path`: [ (`~matplotlib.path.Path` class, `~matplotlib.transforms.Transform` class) | `~matplotlib.patches.Patch` class | None ]
        + `color` or `c`: any matplotlib color
        + ``contains`: a callable function
        + `dash_capstyle`: ['butt' | 'round' | 'projecting']
        + `dash_joinstyle`: ['miter' | 'round' | 'bevel']
        + ``dashes`: sequence of on/off ink in points 
        + `drawstyle`: ['default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post']
        + `figure`: a `matplotlib.figure.Figure` class instance
        + `fillstyle`: ['full' | 'left' | 'right' | 'bottom' | 'top' | 'none']
        + `gid`: an id string
        + `label`: string or anything printable with '%s' conversion.
        + `linestyle` or `ls`: ['solid' | 'dashed', 'dashdot', 'dotted' | (offset, on-off-dash-seq) | `'-'` | `'--'` | `'-.'` | `':'` | `'None'` | `' '` | `''`]
        + `linewidth` or `lw`: float value in points
        + `marker`: :mod:`A valid marker style <matplotlib.markers>`
        + `markeredgecolor` or `mec`: any matplotlib color
        + `markeredgewidth` or `mew`: float value in points
        + `markerfacecolor` or `mfc`: any matplotlib color
        + `markerfacecoloralt` or `mfcalt`: any matplotlib color
        + `markersize` or `ms`: float
        + `markevery`: [None | int | length-2 tuple of int | slice | list/array of int | float | length-2 tuple of float]
        + `path_effects`: unknown
        + `picker`: float distance in points or callable pick function `fn(artist, event)`
        + `pickradius`: float distance in points
        + `rasterized`: [True | False | None]
        + `sketch_params`: unknown
        + `snap`: unknown
        + `solid_capstyle`: ['butt' | 'round' |  'projecting']
        + `solid_joinstyle`: ['miter' | 'round' | 'bevel']
        + `transform`: a `matplotlib.transforms.Transform` class instance
        + `url`: a url string
        + `visible`: [True | False]
        + `xdata`: 1D array
        + `ydata`: 1D array
        + `zorder`: any number

    + kwargs `scalex` and `scaley`, if defined, are passed on to `~matplotlib.axes.Axes.autoscale_view` method to determine whether the `x` and `y` axes are autoscaled; the default is `True`.

+ `mpl.figure.Figure` method
    + Init signature: `mpl.figure.Figure(figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)`
    + Docstring: The Figure instance supports callbacks through a _callbacks_ attribute which is a `matplotlib.cbook.CallbackRegistry` class instance.  The events you can connect to are 'dpi_changed', and the callback will be called with `func(fig)` where fig is the `Figure` class instance.
    + Args:
        + `figsize`: w,h tuple in inches
        + `dpi`: Dots per inch
        + `facecolor`: The figure patch facecolor; defaults to rc `figure.facecolor`
        + `edgecolor`: The figure patch edge color; defaults to rc `figure.edgecolor`
        + `linewidth`: The figure patch edge linewidth
        + `frameon`: If `False`, suppress drawing the figure frame
        + `subplotpars`: A `SubplotParams` class instance, defaults to rc
        + `tight_layout`: 
            + False: use `subplotpars`
            + True: adjust subplot parameters using `tight_layout` with default padding. When providing a dict containing the keys `pad`, `w_pad`, `h_pad` and `rect`, the default `tight_layout` paddings will be overridden.

+ `mpl.backends.backend_agg.FigureCanvasAgg` method
    + Init signature: `mpl.backends.backend_agg.FigureCanvasAgg(figure)`
    + Docstring: The canvas the figure renders into.  Calls the `draw` and `print fig` methods, creates the renderers, etc...

+ `fig.add_subplot` method
    + Signature: `fig.add_subplot(*args, **kwargs)`
    + Docstring: Add a subplot
    + Args:
        + `kwargs`: `~matplotlib.axes.Axes` class kwargs plus `projection`, which chooses a projection type for the axes.
    + Valid values for `projection` are: ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear'].  Some of these projections support additional `kwargs`, which may be provided to `add_axes` method.
    + Examples:
        ```python
        fig.add_subplot(111)
        fig.add_subplot(1,1,1)      # equivalent but more general
        fig.add_subplot(212, facecolor='r')     # add subplot with red background
        fig.add_subplot(111, projection='polar')    # add a polar subplot
        fig.add_subplot(sub)        # add Subplot instance sub
        ```
    + Supported `kwargs`:

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
        + `axisbelow`: [ _True_ | _False_ | 'line' ] 
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

+ `plt.figure` method
    + Signature: `plt.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, **kwargs)`
    + Docstring: Creates a new figure.
    + Args: 
        + `num` (int or string): 
            + Not provided: a new figure created, and the figure number will be incremented. The figure objects holds this number in a `number` attribute.
            + Provided & id existed: make it active, and returns a reference to it or  create return it and returns it.
            + string: the window title will be set to this figure's `num`.
        + `figsize` (tuple of integers): width, height in inches, default to  `figure.figsize.`
        + `dpi` (int): resolution of the figure, default to rc `figure.dpi`.
        + `facecolor`: the background color, default to rc `figure.facecolor`
        + `edgecolor`: the border color, default to rc `figure.edgecolor`

+ `print_png` method
    + Signature: `mpl.backends.backend_agg.print_png(filename_or_obj, *args, **kwargs)`

+ `plt.gca` method
    + Signature: `plt.gca(**kwargs)`
    + Docstring: Get the current `~matplotlib.axes.Axes` instance on the current figure matching the given keyword `args`, or create one.
    + Supported `kwargs`:
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
        + `clip_path`: [ (`~matplotlib.path.Path`, `~matplotlib.transforms.Transform`) | `~matplotlib.patches.Patch` None ] 
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

+ `plt.gca().axis` method
    + Signature: `plt.gca()..axis(*v, **kwargs)`
    + Docstring: Set axis properties
    + Args: 
        + `v`: list of float or {Value}
            | Value     | Description |
            |-----------|------|
            | 'on'      | Toggle axis lines and labels on |
            | 'off'     | Toggle axis lines and labels off |
            | 'equal'   | Equal scaling by changing limits |
            | 'scaled'  | Equal scaling by changing box dimensions |
            | 'tight'   | Limits set such that all data is shown |
            | 'auto'    | Automatic scaling, fill rectangle with data |
            | 'normal'  | Same as 'auto'; deprecated |
            | 'image'   | 'scaled' with axis limits equal to data limits |
            | 'square'  | Square plot; similar to 'scaled', but initially forcing xmax-xmin = ymax-ymin |
        + `emit` (bool): Passed to set_{x,y}lim functions, if observers are notified of axis limit change
        + `xmin`, `ymin`, `xmax`, `ymax` (float): The axis limits to be set
    + Valid signature:
        ```python
        xmin, xmax, ymin, ymax = axis()
        xmin, xmax, ymin, ymax = axis(list_arg)
        xmin, xmax, ymin, ymax = axis(string_arg)
        xmin, xmax, ymin, ymax = axis(**kwargs)
        ```
+ `plt.gca().get_children` method
    + Signature: `plt.gca().get_children()`
    + Docstring: return a list of child artists

+ Demo
    ```python
    import matplotlib.pyplot as plt

    # because the default is the line style '-', 
    # nothing will be shown if we only pass in one point (3,2)
    plt.plot(3, 2)

    # we can pass in '.' to plt.plot to indicate that we want
    # the point (3,2) to be indicated with a marker '.'
    plt.plot(3, 2, '.')

    # First let's set the backend without using mpl.use() from the scripting layer
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure()                  # create a new figure
    canvas = FigureCanvasAgg(fig)   # associate fig with the backend
    ax = fig.add_subplot(111)       # add a subplot to the fig
    ax.plot(3, 2, '.')              # plot the point (3,2)

    # save the figure to test.png
    # you can see this figure in your Jupyter workspace afterwards by going to
    # https://hub.coursera-notebooks.org/
    canvas.print_png('test.png')

    # We can use html cell magic to display the image.
    %%html
    <img src='test.png' />

    plt.figure()                # create a new figure
    plt.plot(3, 2, 'o')         # plot the point (3,2) using the circle marker
    ax = plt.gca()              # get the current axes
    ax.axis([0,6,0,10])         # Set axis properties [xmin, xmax, ymin, ymax]
    
    plt.figure()                # create a new figure
    plt.plot(1.5, 1.5, 'o')     # plot the point (1.5, 1.5) using the circle marker
    plt.plot(2, 2, 'o')         # plot the point (2, 2) using the circle marker
    plt.plot(2.5, 2.5, 'o')     # plot the point (2.5, 2.5) using the circle marker
    ax = plt.gca()              # get current axes
    ax.get_children()           # get all the child objects the axes contains
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/gHvXBv0cEeaI9Q7Pym09lA.processed/full/360p/index.mp4?Expires=1528329600&Signature=kQmxs5co1w1DcbHaegKSc-lIeK~590M~0gkDX~Lxm-7ieMbtPpVAeNfi5H6~NfNrE7fq2c1MYOyvLT1L7DEmZP2dcLBUNrUnEOAJ30aFrhu0IMfEFZsuD0SD5ewb9wRoP1ZWAiJQ3S92cjhd~gvVcs7KIas6T75gD4kFmVs-EAQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Basic Plotting with Matplotlib" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Scatterplots

+ `pyplot` retrieving the current figure with the function `gcf` and then the current axis with the function `gca`.
+ `pyplot` mirrors the API of the axis objects -> call the `plot` function against the `pyplot` module.
+ Function declaration from most of the functions in matplotlib end with an open set of keyword arguments -> property control via keyword arguments

+ `plt.scatter` method
    + Signature: `plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)`
    + Docstring: Make a scatter plot of `x` vs `y`
    + Parameters
        + `x`, `y` (array_like, shape (n, )): Input data
        + `s` (scalar or array_like, shape (n, )): size in $points^2$.  Default is `rcParams['lines.markersize'] ** 2`.
        + `c` (color, sequence, or sequence of color): default: 'b'; a single color format string, or a sequence of color specifications of length `N`, or a sequence of `N` numbers to be mapped to colors using the `cmap` and `norm` specified via kwargs
        + `marker` : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
        + `cmap` : `~matplotlib.colors.Colormap`; A `~matplotlib.colors.Colormap` instance or registered name. `cmap` is only used if `c` is an array of floats. If None, defaults to rc `image.cmap`.
        + `norm` : `~matplotlib.colors.Normalize`; A `~matplotlib.colors.Normalize` instance is used to scale luminance data to 0, 1. `norm` is only used if `c` is an array of floats. If `None`, use the default :func:`normalize`.
        + `vmin`, `vmax` (scalar): `vmin` and `vmax` are used in conjunction with `norm` to normalize luminance data.  If either are `None`, the min and max of the color array is used. 
        + `alpha` (scalar):  The alpha blending value, between 0 (transparent) and 1 (opaque)
        + `linewidths` (scalar or array_like): If None, defaults to `(lines.linewidth,)`.
        + `verts` (sequence of (x, y)): If `marker` is None, these vertices will be used to construct the marker.  The center of the marker is located at $(0,0)$ in normalized units.  The overall marker is rescaled by `s`.
        + `edgecolors` (color or sequence of color): 
            + If None, defaults to 'face'
            + If 'face', the edge color will always be the same as the face color.
            + If it is 'none', the patch boundary will not be drawn.
            + For non-filled markers, the `edgecolors` kwarg is ignored and forced to 'face' internally.

+ `plt.xlabel`, `plt.ylabel` methods:
    + Signature: `plt.xlabel(s, *args, **kwargs)`, `plt.ylabel(s, *args, **kwargs)`
    + Docstring: Set the `x`, `y` axis label of the current axis
    + Default override is:
        ```json
        override = {
            'fontsize'            : 'small',
            'verticalalignment'   : 'top',
            'horizontalalignment' : 'center'
            }
        ```
+ `plt.title` method:
    + Signature: `plt.title(s, *args, **kwargs)`
    + Docstring: Set a title of the current axes.
    + Parameters
        + `label` (str): Text to use for the title
        + `fontdict` (dict): A dictionary controlling the appearance of the title text, the default `fontdict` is:
            ```json
            {'fontsize': rcParams['axes.titlesize'],
            'fontweight' : rcParams['axes.titleweight'],
            'verticalalignment': 'baseline',
            'horizontalalignment': loc}
            ```
        + `loc` ({'center', 'left', 'right'}, str): Which title to set, defaults to 'center'
        + `kwargs`: text properties

+ `plt.ledgend` method:
    + Signature: `plt.legend(*args, **kwargs)`
    + Docstring: Places a legend on the axes.
    + To make a legend for lines which already exist on the axes (via plot for instance), simply call this function with an iterable of strings, one for each legend item.
    + Parametrs: 
        + `loc` (int or string or pair of floats), default: 'upper right': The location of the legend. Possible codes are:

            | Location String |  Location Code | | Location String |  Location Code | 
            |-----------------|----------------|-|-----------------|----------------|
            | 'best'         |  0 |   | 'upper right'   |  1 | 
            | 'upper left'    |  2 |   | 'lower left'    |  3 | 
            | 'lower right'   |  4 |   | 'right'         |  5 | 
            | 'center left'   |  6 |   | 'center right'  |  7 | 
            | 'lower center'  |  8 |   | 'upper center'  |  9 | 
            | 'center'        |  10 |
        + `bbox_to_anchor` (`matplotlib.transforms.BboxBase` instance or tuple of floats): Specify any arbitrary location for the legend in `bbox_transform`    coordinates (default Axes coordinates).
        + `ncol` (int): The number of columns that the legend has. Default is 1.
        + `prop` (None or `matplotlib.font_manager.FontProperties` class or dict):     The font properties of the legend. Default-the current     `matplotlib.rcParams` data will be used.
        + `fontsize` (int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}): Controls the font size of the legend.
        + `numpoints` (None or int): The number of marker points in the legend when creating a legend entry for a line/`matplotlib.lines.Line2D` class. Default `legend.numpoints` `rcParam<matplotlib.rcParams>` data.
        + `scatterpoints` (None or int): The number of marker points in the legend when creating a legend entry for a scatter plot/`matplotlib.collections.PathCollection` class.  Default `legend.scatterpoints` `rcParam<matplotlib.rcParams>` data.
        + `scatteryoffsets` (iterable of floats): The vertical offset (relative to the font size) for the markers created for a scatter plot legend entry. $0.0$ is at the base the legend text, and 1.0 is at the top. To draw all markers at the same height, set to $[0.5]$. Default $[0.375, 0.5, 0.3125]$.
        + `markerscale` (None or int or float): The relative size of legend markers compared with the originally drawn ones. Default `legend.markerscale` in `rcParam <matplotlib.rcParams>`.
        + `markerfirst` (bool): 
            + True: legend marker is placed to the left of the legend label
            + False: legend marker is placed to the right of the legend label
        + `frameon` (None or bool): Control whether the legend should be drawn on a patch (frame).  Default `legend.frameon`
        + `fancybox` (None or bool): Control whether round edges should be enabled around the `~matplotlib.patches.FancyBboxPatch` class which makes up the legend's background.  Default `legend.fancybox`
        + `shadow` (None or bool): Control whether to draw a shadow behind the legend. Default=`None` - `legend.shadow`
        + `framealpha` (None or float): Control the alpha transparency of the legend's background. Default `legend.framealpha`
        + `facecolor` (None or "inherit" or a color spec): Control the legend's background color. Default `legend.facecolor` `inherit`= `axes.facecolor`
        + `edgecolor` (None or "inherit" or a color spec): Control the legend's background patch edge color. Default `legend.edgecolor` `inherit` = `axes.edgecolor`
        + `mode` ({"expand", None}): `expand` = the legend horizontally expanded to fill the axes area (or `bbox_to_anchor` if defines the legend's size).
        + `bbox_transform` (None or `matplotlib.transforms.Transform` class): The transform for the bounding box (`bbox_to_anchor`). Default using Axes' `~matplotlib.axes.Axes.transAxes` transform.
        + `title` (str or None): The legend's title. Default is no title.
        + `borderpad` (float or None): The fractional whitespace inside the legend border. Measured in font-size units. Default `legend.borderpad`
        + `labelspacing` (float or None): The vertical space between the legend entries. Measured in font-size units. Default `legend.labelspacing`
        + `handlelength` (float or None): The length of the legend handles. Measured in font-size units. Default `legend.handlelength`
        + `handletextpad` (float or None): The pad between the legend handle and text. Measured in font-size units. Default `legend.handletextpad`
        + `borderaxespad` (float or None): The pad between the axes and legend border. Measured in font-size units. Default `legend.borderaxespad`
        + `columnspacing` (float or None): The spacing between columns. Measured in font-size units. Default `legend.columnspacing`
        + `handler_map` (dict or None): The custom dictionary mapping instances or types to a legend handler. This `handler_map` updates the default handler map  found at `matplotlib.legend.Legend.get_legend_handler_map` function.

+ `zip` fucntion
    + Init signature: `zip(*args, **kwargs)`
    + Docstring: zip(iter1 [,iter2 [...]]) --> zip object
    + Return a zip object whose `.__next__()` method returns a tuple where the i-th element comes from the i-th iterable argument.  The `.__next__()` method continues until the shortest iterable in the argument sequence is exhausted and then it raises StopIteration.

+ Demo
    ```python
    import numpy as np

    x = np.array([1,2,3,4,5,6,7,8])
    y = x

    plt.figure()
    plt.scatter(x, y) # similar to plt.plot(x, y, '.'), but the underlying child objects in the axes are not Line2D

    x = np.array([1,2,3,4,5,6,7,8])
    y = x

    # create a list of colors for each point to have
    # ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'red']
    # plot the point with size 100 and chosen colors
    colors = ['green']*(len(x)-1)
    colors.append('red') 

    plt.figure()
    plt.scatter(x, y, s=100, c=colors)

    # ZIP Function usage
    # convert the two lists into a list of pairwise tuples
    zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
    print(list(zip_generator))
    # the above prints: [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]

    zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
    # The single star * unpacks a collection into positional arguments
    print(*zip_generator)
    # the above prints: (1, 6) (2, 7) (3, 8) (4, 9) (5, 10)

    # use zip to convert 5 tuples with 2 elements each to 2 tuples with 5 elements each
    print(list(zip((1, 6), (2, 7), (3, 8), (4, 9), (5, 10))))
    # the above prints: [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]

    # Unpack zip object
    zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
    # let's turn the data back into 2 lists
    x, y = zip(*zip_generator) # This is like calling zip((1, 6), (2, 7), (3, 8), (4, 9), (5, 10))
    print(x)
    print(y)
    # the above prints: (1, 2, 3, 4, 5) & (6, 7, 8, 9, 10)

    plt.figure()
    # plot a data series 'Tall students' in red using the first two elements of x and y
    plt.scatter(x[:2], y[:2], s=100, c='red', label='Tall students')
    # plot a second data series 'Short students' in blue using the last three elements of x and y 
    plt.scatter(x[2:], y[2:], s=100, c='blue', label='Short students')

    # add a label to the x axis
    plt.xlabel('The number of times the child kicked a ball')
    # add a label to the y axis
    plt.ylabel('The grade of the student')
    # add a title
    plt.title('Relationship between ball kicking and grades')

    # add a legend (uses the labels from plt.scatter)
    plt.legend()

    # add the legend to loc=4 (the lower right hand corner), also gets rid of the frame and adds a title
    plt.legend(loc=4, frameon=False, title='Legend')

    # get children from current axes (the legend is the second to last item in this list)
    plt.gca().get_children()

    # get the legend from the current axes
    legend = plt.gca().get_children()[-2]

    # you can use get_children to navigate through the child artists
    legend.get_children()[0].get_children()[1].get_children()[0].get_children()

    # import the artist class from matplotlib
    from matplotlib.artist import Artist

    def rec_gc(art, depth=0):
        if isinstance(art, Artist):
            # increase the depth for pretty printing
            print("  " * depth + str(art))
            for child in art.get_children():
                rec_gc(child, depth+2)

    # Call this function on the legend artist to see what the legend is made up of
    rec_gc(plt.legend())
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/xSk2r_0dEearIRLZY_MkaA.processed/full/540p/index.mp4?Expires=1528416000&Signature=dkqpm85W8V6th8JzE6LAdVPCEQRSFPl9bMmLXpSMO0mVcCnEG8eVPWnnuofeKPMPhcbMPLHEwds0u1YAI0FR6v2Ac4FDMTaRlbjZfkGOGFPrklE7UHx7YVs478dQXmoGHOhSgNiyTiFcVotfc4~J4T4PQh~xK4b~3kpBdadQCoo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Scatterplots" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Line Plots

+ Characters of Line Plots:
    + Utilize `plt.plot` fucntion
    + Only y-axis calues, no x-axis values -> using index as the x-axis values
    + Color of two series of data assigned autoamtically,, while scatterplots require manualy labeled
    + Marker style assigned automatically w/ different series of data
+ `plt.gca().fill_between` method
    + Signature: `fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)`
    + Docstring: Make filled polygons between two curves; Method of matplotlib.axes._subplots.AxesSubplot instance
    + Paramters: 
        + `x` (array): An N-length array of the x data
        + `y1` (array): An N-length array (or scalar) of the y data
        + `y2` (array): An N-length array (or scalar) of the y data
        + `where` (array): 
            + `None`: fill between everywhere.  
            + not `None`: an N-length numpy boolean array and the fill will only happen over the regions where ``where==True``.
        + `interpolate` (bool): 
            + `True`: interpolate between the two lines to find the precise point of intersection.  
            + `False`: the start and end points of the filled region will only occur on explicit values in the _x_ array.
        + `step` ({'pre', 'post', 'mid'}): If not None, fill with step logic.

+ Date Time handling
    + Standard library: 
        + `datetime` module: classes for manipulating dates and times in both simple and complex ways
        + `time` module: time access and conversion
        + `calendar` module: classes for working with dates to manage year/month/week oriented values
    + Numpy:
        + `datetime64` dtype: encode dates as 64-bit integers, and thus allowing arrays of dates to be represented very compactly
        + `timedelta` dtype
    + Pandas:
        + `Timestamp`: combining the ease-of-use of datetime and dateutil with the efficient storage and vectorized interface of `numpy.datetime64`

+ `pd.to_datetime` method: &nbsp;&nbsp; [Date Functions](../1-IntroDS/03-AdvPandas.md#date-functionality)

+ `plt.gca().xaxis` & `plt.gca().yaxis` classes:
    + `xaxis = class XAxis(Axis)` 
    + Attributes: 
        + `axes.transData` - transform data coords to display coords
        + `axes.transAxes` - transform axis coords to display coords
        + `labelpad` - number of points between the axis and its label
    + Methods: 
        + `contains(mouseevent)`: Test whether the mouse event occured in the x axis.
        + `get_data_interval()`: return the Interval instance for this axis data limits
        + `get_minpos()`
        + `get_text_heights(renderer)`: Returns the amount of space one should reserve for text above and below the axes.  Returns a tuple (above, below)
        + `get_tick_space()`: Return the estimated number of ticks that can fit on the axis.
        + `get_ticks_position()`: Return the ticks position (top, bottom, default or unknown)
        + `get_view_interval()`: return the Interval instance for this axis view limits
        + `set_data_interval(vmin, vmax, ignore=False)`: set the axis data limits
        + `set_default_intervals()`: set the default limits for the axis interval if they are not mutated
        + `set_label_position(position)`: Set the label position (top or bottom);             ACCEPTS: [ 'top' | 'bottom' ]
        + `set_ticks_position(position)`: 
            + Set the ticks position (top, bottom, both, default or none) both sets the ticks to appear on both positions, but does not change the tick labels.  
            + 'default': reset the tick positions to the default: ticks on both positions, labels at bottom.  
            + 'none': be used if you don't want any ticks. 
            + 'none' and 'both': affect only the ticks, not the labels.
            + ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]
        + `set_view_interval(vmin, vmax, ignore=False)`: for mpl internal use
        + `tick_bottom()`: use ticks only on bottom
        + `tick_top()`: use ticks only on top
    + Methods inherited from Axis:
        + `axis_date(tz=None)`: Sets up x-axis ticks and labels that treat the x data as dates. *tz* is a `tzinfo` instance or a timezone string. This timezone is used to create date labels.
        + `cla()`: clear the current axis
        + `convert_units(x)`
        + `draw(artist, renderer, *args, **kwargs)`: Draw the axis lines, grid lines, tick lines and labels
        + `get_children()`: Return a list of the child `Artist`s this `Artist` contains.
        + `get_gridlines()`: Return the grid lines as a list of Line2D instance
        + `get_label()`: Return the axis label as a Text instance
        + `get_label_position()`: Return the label position (top or bottom)
        + `get_label_text()`: Get the text of the label
        + `get_major_formatter()`: Get the formatter of the major ticker
        + `get_major_locator()`: Get the locator of the major ticker
        + `get_major_ticks(numticks=None)`: get the tick instances; grow as necessary
        + `get_majorticklabels()`: Return a list of Text instances for the major ticklabels
        + `get_majorticklines()`: Return the major tick lines as a list of Line2D instances
        + `get_majorticklocs()`: Get the major tick locations in data coordinates as a numpy array
        + `get_minor_formatter()`: Get the formatter of the minor ticker
        + `get_minor_locator()`: Get the locator of the minor ticker
        + `get_minor_ticks(numticks=None)`: get the minor tick instances; grow as necessary
        + `get_minorticklabels()`: Return a list of Text instances for the minor ticklabels
        + `get_minorticklines()`: Return the minor tick lines as a list of Line2D instances
        + `get_minorticklocs()`: Get the minor tick locations in data coordinates as a numpy array
        + `get_offset_text()`: Return the axis offsetText as a Text instance
        + `get_pickradius()`: Return the depth of the axis used by the picker
        + `get_scale()`
        + `get_smart_bounds()`: get whether the axis has smart bounds
        + `get_tick_padding()`
        + `get_ticklabel_extents(renderer)`: Get the extents of the tick labels on either side of the axes.
        + `get_ticklabels(minor=False, which=None)`: Get the x tick labels as a list of `~matplotlib.text.Text` instances.
            + `minor` (bool)`: If True return the minor ticklabels, else return the major ticklabels
            + `which` (None, {'minor', 'major', 'both'})`: Selects which ticklabels to return
        + `get_ticklines(minor=False)`: Return the tick lines as a list of Line2D instances
        + `get_ticklocs(minor=False)`: Get the tick locations in data coordinates as a numpy array
        + `get_tightbbox(renderer)`: Return a bounding box that encloses the axis. It only accounts tick labels, axis label, and offsetText.
        + `get_transform()`: Return the `~matplotlib.transforms.Transform` instance used by this artist.
        + `get_units()`: return the units for axis
        + `grid(b=None, which='major', **kwargs)`: Set the axis grid on or off; b is a boolean. Use *which* = 'major' | 'minor' | 'both' to set the grid for major or minor ticks.
            + If *b* is *None* and len(kwargs)==0, toggle the grid state.  
            + If *kwargs* are supplied, it is assumed you want the grid on and *b* will be set to True.
            + *kwargs* are used to set the line properties of the grids, e.g., xax.grid(color='r', linestyle='-', linewidth=2)
        + `have_units()`: Return *True* if units are set on the *x* or *y* axes
        + `iter_ticks()`: Iterate through all of the major and minor ticks.
        + `limit_range_for_scale(vmin, vmax)`
        + `pan(numsteps)`: Pan *numsteps* (can be positive or negative)
        + `reset_ticks()
        + `set_clip_path(clippath, transform=None)`: Set the artist's clip path, which may be:
            + a `~matplotlib.patches.Patch` (or subclass) instance
            + a `~matplotlib.path.Path` instance, in which case an optional `~matplotlib.transforms.Transform` instance may be provided, which will be applied to the path before using it for clipping.
            + *None*, to remove the clipping path
            + For efficiency, if the path happens to be an axis-aligned rectangle, this method will set the clipping box to the corresponding rectangle and set the clipping path to *None*.
            + ACCEPTS`: [ (`~matplotlib.path.Path`, `~matplotlib.transforms.Transform`) | `~matplotlib.patches.Patch` | None ]
        + `set_label_coords(x, y, transform=None)`: Set the coordinates of the label.  By default, the x coordinate of the y label is determined by the tick label bounding boxes, but this can lead to poor alignment of multiple ylabels if there are multiple axes.  Ditto for the y coodinate of the x label.
        + `set_label_text(label, fontdict=None, **kwargs)`: Sets the text value of the axis label; ACCEPTS`: A string value for the label
        + `set_major_formatter(formatter)`: Set the formatter of the major ticker; ACCEPTS`: A `~matplotlib.ticker.Formatter` instance
        + `set_major_locator(locator)`: Set the locator of the major ticker; ACCEPTS`: a `~matplotlib.ticker.Locator` instance
        + `set_minor_formatter(formatter)`: Set the formatter of the minor ticker; ACCEPTS`: A `~matplotlib.ticker.Formatter` instance
        + `set_minor_locator(locator)`: Set the locator of the minor ticker; ACCEPTS`: a `~matplotlib.ticker.Locator` instance
        + `set_pickradius(pickradius)`: Set the depth of the axis used by the picker; ACCEPTS`: a distance in points
        + `set_smart_bounds(value)`: set the axis to have smart bounds
        + `set_tick_params(which='major', reset=False, **kw)`: Set appearance parameters for ticks and ticklabels.
        + `set_ticklabels(ticklabels, *args, **kwargs)`: Set the text values of the tick labels. Return a list of Text instances.  Use *kwarg* *minor=True* to select minor ticks.
            + All other kwargs are used to update the text object properties.
            + As for get_ticklabels, label1 (left or bottom) is affected for a given tick only if its label1On attribute is True, and similarly for label2.  The list of returned
            label text objects consists of all such label1 objects followed by all such label2 objects.
            + The input *ticklabels* is assumed to match the set of tick locations, regardless of the state of label1On and label2On.
            + ACCEPTS`: sequence of strings or Text objects
        + `set_ticks(ticks, minor=False)`: Set the locations of the tick marks from sequence ticks; ACCEPTS`: sequence of floats
        + `set_units(u)`: set the units for axis; ACCEPTS`: a units tag
        + `update_units(data)`: introspect *data* for units converter and update the axis.converter instance if necessary. Return *True* if *data* is registered for unit conversion.
        + `zoom(direction)`: Zoom in/out on axis; if *direction* is >0 zoom in, else zoom out
 

+ Demo
    ```python
    import numpy as np

    linear_data = np.array([1,2,3,4,5,6,7,8])
    exponential_data = linear_data**2

    plt.figure()
    # plot the linear data and the exponential data
    plt.plot(linear_data, '-o', exponential_data, '-o')

    # plot another series with a dashed red line
    plt.plot([22,44,55], '--r')

    plt.xlabel('Some data')
    plt.ylabel('Some other data')
    plt.title('A title')
    # add a legend with legend entries (because we didn't have labels when we plotted the data series)
    plt.legend(['Baseline', 'Competition', 'Us'])

    # fill the area between the linear data and exponential data
    plt.gca().fill_between(range(len(linear_data)), 
                        linear_data, exponential_data, 
                        facecolor='blue', 
                        alpha=0.25)

    # Let's try working with dates!
    plt.figure()
    # Numpy provides a linear sampling dates
    observation_dates = np.arange('2017-01-01', '2017-01-09', dtype='datetime64[D]')
    plt.plot(observation_dates, linear_data, '-o',  observation_dates, exponential_data, '-o')

    # Let's try using pandas
    import pandas as pd

    plt.figure()
    observation_dates = np.arange('2017-01-01', '2017-01-09', dtype='datetime64[D]')
    observation_dates = map(pd.to_datetime, observation_dates) 
    # trying to plot a map will result in an error 
    #  map function return iterator and matplotlib cannot handle iterator
    plt.plot(observation_dates, linear_data, '-o',  observation_dates, exponential_data, '-o')

    plt.figure()
    observation_dates = np.arange('2017-01-01', '2017-01-09', dtype='datetime64[D]')
    observation_dates = list(map(pd.to_datetime, observation_dates)) 
    # convert the map to a list to get rid of the error - memory efficient way
    plt.plot(observation_dates, linear_data, '-o',  observation_dates, exponential_data, '-o')

    x = plt.gca().xaxis

    # rotate the tick labels for the x axis
    for item in x.get_ticklabels():
        item.set_rotation(45)

    # adjust the subplot so the text doesn't run off the image
    plt.subplots_adjust(bottom=0.25)

    ax = plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Units')
    ax.set_title('Exponential vs. Linear performance')

    # you can add mathematical expressions in any text element w/ LaTex by $
    ax.set_title("Exponential ($x^2$) vs. Linear ($x$) performance")
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/0sxDE_9aEeaREQ7irMqVoA.processed/full/360p/index.mp4?Expires=1528416000&Signature=Busiy-6aXd7nlofhxE4fUsqI9JE4k0XSeyIvdJcKA0gBxHbdcvqkSwfmb5~7LcFyxaNkUvinWx3o42wsCarZRx8ovwxlu3wx9lHrwvssZRTUX-ryvquu-pPznOmPEspOkinBI-Dc5Et38q41~EzFK2uW2N65sgSQPTc8JG3HNkM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Bar Charts" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Bar Charts

+ Demo
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Dejunkifying a Plot

+ Demo
    ```python

    ```

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

