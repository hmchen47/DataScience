# Introduction

## Lec 1.1 Course Overview

### Note

+ Course Instructors: Ani Adhikari, John DeNero, and David Wanger

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/tIQz3ldACvM)

## Lec 1.2 Why Data Science

### Notes

+ What is Data Science?

        Drawing useful conclusion from data using computation

    + Exploration

        + Identifying patterns in info
        + Uses visualization
    + Inference

        + Quantifying whether those patterns are reliable
        + Uses randomization
    + Prediction

        + Making informed guesses
        + Uses machine learning

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/BKgdDLrSC5s)

## Lec 1.3 Programming

### Notes

[Example](https://hub.data8x.berkeley.edu/user/cade73f52ccce256ebbca3384ef48d9c/notebooks/materials-x18/lec/x18/1/Example.ipynb)

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/PFgBCG_evEg)

## Lec 1.4 Demo: Little Women

### Notes

+ [Example](https://hub.data8x.berkeley.edu/user/cade73f52ccce256ebbca3384ef48d9c/notebooks/materials-x18/lec/x18/1/lec01.ipynb)

    + Cell 1:
        ```python
        from datascience import *
        import numpy as np
        %matplotlib inline
        import matplotlib.pyplot as plots
        plots.style.use('fivethirtyeight')
        import warnings
        warnings.simplefilter(action="ignore", category=FutureWarning)

        from urllib.request import urlopen 
        import re
        def read_url(url): 
            return re.sub('\\s+', ' ', urlopen(url).read().decode())
        ```
    + Cell 2:
        ```python
        little_women_url = 'https://www.inferentialthinking.com/chapters/01/3/little_women.txt'
        little_women_text = read_url(little_women_url)
        chapters = little_women_text.split('CHAPTER ')[1:]    
        ```

    + Cell 3: `Table().with_column('Text', chapters)`
    + Cell 4: `np.char.count(chapters, 'Christmas')`
    + Cell 5: `np.char.count(chapters, 'Jo')`
    + Cell 6:
        ```python
        Table().with_columns(
            'Jo', np.char.count(chapters, 'Jo'),
            'Meg', np.char.count(chapters, 'Meg'),
            'Amy', np.char.count(chapters, 'Amy'),
            'Beth', np.char.count(chapters, 'Beth'),
            'Laurie', np.char.count(chapters, 'Laurie')
        )
        ```
    

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Yt-hzH_1u_A)

## Lec 1.5 Demo: Visualizations (1)

### Notes

```python
Table().with_columns(
    'Jo', np.char.count(chapters, 'Jo'),
    'Meg', np.char.count(chapters, 'Meg'),
    'Amy', np.char.count(chapters, 'Amy'),
    'Beth', np.char.count(chapters, 'Beth'),
    'Laurie', np.char.count(chapters, 'Laurie')
).cumsum().plot()
```

+ Questions:

    + Who is tha main character of this book?
    + Who get married in the middle of the book?
    + Who developed relationship with Laurie, the boy next door?

+ Answer:
    + Jo
    + Meg - almost flat line after ch 28
    + Amy - ch 35+, Laurie's line along with Amy's line 

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/N2Sq_2HMWFw)

## Lec 1.6 Demo: Visualizations (2)

### Notes

```python
Table().with_columns([
        'Chapter Length', [len(c) for c in chapters],
        'Number of Periods', np.char.count(chapters, '.'),
    ]).scatter('Number of Periods')
```

+ Number of periods vs. Lines of the chapter: 
    + almost linear line
    + sentences of the chapter -> periods ~ lines of the chapter

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/GeZ3ugunMn4)

