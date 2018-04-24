# Section 9a: Comparing Histograms (Lec 9.1 - Lec 9.2)

## Lec 9.1 Comparing Histograms

### Notes

+ Overlaid graph: for visually comparing two populatons
+ Demo
    ```python
    height = Table.read_table('galton.csv').select(1, 2, 7).relabeled(2, 'child')
    height.hist('father', unit='inch')          # histogram for father
    height.hist('child', unit='inch')           # histogram for mother
    height.hist(unit='inch', bins=np.arange(55, 80, 2)) # overlaid histograms for 3 cols
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/0ImGO0BG630){:target="_blank"}


## Lec 9.2 Comparing Histograms Discussion

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


# Section 9b: Functions (Lec 9.3 - Lec 9.6)

## Lec 9.3 Defining Functions

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Lec 9.4 Defining Functions Discussion

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Lec 9.5 Apply

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Lec 9.6 Example Prediction

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Reading and Practice for Section 9b

### Reading

### Practice



