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

    + __Exploration__

        + Identifying patterns in info
        + Uses visualization
    + __Inference__

        + Quantifying whether those patterns are reliable
        + Uses randomization
    + __Prediction__

        + Making informed guesses
        + Uses machine learning

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/BKgdDLrSC5s)

## Lec 1.3 Programming

### Notes

[Example](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.1x+1T2018/courseware/413fff9cb76c471fa0ccb32d7d08ace6/679974c6a9ef47be9c4f091ac35884d5/3?activate_block_id=block-v1%3ABerkeleyX%2BData8.1x%2B1T2018%2Btype%40vertical%2Bblock%4032e21bdef731460b86123fc70cab72bd#examples-external-resource)

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/PFgBCG_evEg)

## Lec 1.4 Demo: Little Women

### Notes

+ [Example](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.1x+1T2018/courseware/413fff9cb76c471fa0ccb32d7d08ace6/679974c6a9ef47be9c4f091ac35884d5/3?activate_block_id=block-v1:BerkeleyX+Data8.1x+1T2018+type@vertical+block@32e21bdef731460b86123fc70cab72bd#examples-external-resource)

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

