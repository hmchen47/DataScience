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

+ Discussion Questions  
    The histogram describes a __year__ of daily temperatures  
    Answer these questions , if possible:
    + What portion of days had a high temp in the range $60-70$?
    + What portion had a low of 45 or more?
    + How many days had a difference of more than 20 degrees btw their high & low temperatures.

    ![diagram](./Diagrams/sec09-sec09-temp.png)

+ Answers:
    + $(70 - 60) \times 4.8\% = 48\%$
    + $1 - (0.1\% \times (35 - 10)) - (.7\% \times (40-35)) - (2\% \times (45 - 40)) = 1 - 14\% = 86\%$
    + N/A

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Ag2929CN3MA){:target="_blank"}


# Section 9b: Functions (Lec 9.3 - Lec 9.6)

## Lec 9.3 Defining Functions

### Notes

+ `def` Statements  
    User-defined functions give names to blocks of code
    ```python
    def spread(values):
        return max(values) - min(values)
    ```
    + Name: `spread`
    + Arguments: `values`
    + Body: `max(values) - min(values)`
    + Return expression: `return max(values) - min(values)`
+ Demo
    ```python
    def double(x):
        """ Double x """
        return 2*x

    double(3)
    double(-4)
    y = 5;  double(y/4)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/DEEsmyz3oRo){:target="_blank"}


## Lec 9.4 Defining Functions Discussion

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/4dat6zBtddM){:target="_blank"}


## Lec 9.5 Apply

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/A9lKV2QBTXs){:target="_blank"}


## Lec 9.6 Example Prediction

### Notes

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/eLtLrb_Mfnk){:target="_blank"}


## Reading and Practice for Section 9b

### Reading

### Practice



