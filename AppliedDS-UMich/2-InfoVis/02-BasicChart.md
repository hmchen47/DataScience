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

    [<br/><img src="https://delftswa.gitbooks.io/desosa-2017/content/matplotlib/images-matplotlib/functional_view.png" alt="Functional View" width="450">](https://delftswa.gitbooks.io/desosa-2017/content/matplotlib/chapter.html)
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


## Ten Simple Rules for Better Figures

Rougier et al. share their ten simple rules for drawing better figures, and use matplotlib to provide illustrative examples. As you read this paper, reflect on what you learned in the first module of the course -- principles from Tufte and Cairo -- and consider how you might realize these using matplotlib.

Rougier NP, Droettboom M, Bourne PE (2014) [Ten Simple Rules for Better Figures](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833). PLoS Comput Biol 10(9): e1003833. doi:10.1371/journal.pcbi.1003833

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

