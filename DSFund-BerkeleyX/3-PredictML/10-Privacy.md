# Section 10: Privacy (Lec 10.1 - Lec 10.5)

+ [Launching WebPage](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/5b8ee52fd5644c26995eda55b83306ce/c71400648f574acb8bf74b240efe475b/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%406547b979dd1a470ba6ebefa8a1e13a2d)
+ [Web notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec10.ipynb)
+ [Local Notebook](notebooks/lec10.ipynb)
+ [Local Python Code](notebook/lec10.py)

## Lec 10.1 Examples

### Note

+ Why is Facebook in the news?
    + Cambridge Analytica: 
        + Access private 87 mil users w/ permission
        + Involve with political campaigning
    + Start with survey about personality for academic research
    + Not only the survey questions but also personal profiles and their friends' profiles
    + Bought data from academic researchers

+ Knowing your Facebook Likes is enough to predict:
    + Black vs. white: 95% accuracy
    + Republican vs Democrat: 85% accuracy
    + Christian vs Muslim: 82% accuracy
    + Homosexual male vs heterosexual mal: 88% accuracy

+ Voter turnout:
    + Show one of two messages to 61M people in US
    + Variant #1: "you should vote!"
    + Variant #2: "you should vote!", plus thumbnails of friends who have already voted
    + Results: 340K more people vote when they saw Variant #2

+ The myth of anonymity:
    + Netflix releases dataset of how people rated movies they'd seen on Netflix
    + Researchers discovered that knowing two of your movie ratings and when you rated them enough to uniquely find you in the dataset, for 68% of people

+ Terminology: disclosure, collection, inference
    + Disclosure: home address with Amazon
    + Collection: collecting information without my awareness, such as web site visiting
    + Disclosure: least privacy implication
    + Collection: more privacy implications because3 people not aware the info is being collected
    + Inference: most sensitive privacy implication because it may be hard for people to anticipate what inferences can be drawn about them



### Video 

<a href="https://edx-video.net/BERD83FD2018-V003700_DTH.mp4" alt="Lec 10.1 Examples" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 10.2 Meaning

### Note

+ What is _privacy_?
    + "the right to be let alone" - Justice DSamuel Warren and Louis Brandeis
    + control over who can obtain or use information about me
    + Fair Information Practices: notice, consent (opt-in/opt-out), access
        + providing people notice if you are collecting information - awareness
        + consent: people should have right to have some say over whether you collect info about them or what you about them
            + opt-in: ask for explicit consent, no permission until you say yes
            + opt-out: similar to opt-in but default assume that your are ok to do something unless someone tells you they don't want you to do it
        + access: people should have right to know what information credit agencies have about them
    + contextual integrity - Helen Nissenbaum
        + people feel like their privacy has been violated if info is used in a way that surprises them, that differs from what they would expect
        + E.g. bar talk about boss or company to buddies

+ Why care about privacy?
    + reputation management
    + maintaining social boundaries
    + limits on government power
    + freedom of though, speech, politics
    + opportunity for second chances

+ Trend: architectures of persuasion
    + use the info to persuade them
    + E.g. voters
    + Concern: how far could this go?


### Video 

<a href="https://edx-video.net/BERD83FD2018-V004500_DTH.mp4" alt="Lec 10.2 Meaning" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 10.3 Case Study

### Note

+ Automated License Plate Readers
    + Cameras on policy cars to scan parking cars and log the time, place and so one into database
    + Useful info about the car stolen or other criminal events
    + Be aware of bias on the same location of police HQ

+ Demo
    ```python
    # ## Data collection

    # First, we'll gather the data.  It turns out the data is publicly available on the 
    # Oakland public records site.  I downloaded it and combined it into a single CSV 
    # file by myself before lecture.
    lprs = Table.read_table('https://inferentialthinking.com/data/all-lprs.csv.gz', 
        compression='gzip', sep=',')
    # red_VRM    red_Timestamp             Location
    # 1275226    01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)
    # 27529C     01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)
    # 1158423    01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)
    # ... (rows omitted)

    # Let's start by renaming some columns, and then take a look at it.
    lprs.relabel('red_VRM', 'Plate')
    lprs.relabel('red_Timestamp', 'Timestamp')
    # Plate      Timestamp                 Location
    # 1275226    01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)
    # 27529C     01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)
    # 1158423    01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)
    # ... (rows omitted)

    # Phew, that's a lot of data: we can see about 2.7 million license plate reads here.
    # Let's start by seeing what can be learned about someone, using this data -- assuming you 
    # know their license plate.

    # ## Searching for Individuals

    # As a warmup, we'll take a look at ex-Mayor Jean Quan's car, and where it has been seen.  
    # Her license plate number is 6FCH845.  (How did I learn that?  Turns out she was in the
    # news for getting $1000 of parking tickets, and [the news article](http://www.sfgate.com/bayarea/matier-ross/article/Jean-Quan-Oakland-s-new-mayor-gets-car-booted-3164530.php) 
    # included a picture of her car, with the license plate visible.  You'd be amazed by 
    # what's out there on the Internet...)
    lprs.where('Plate', '6FCH845')
    # Plate      Timestamp                 Location
    # 6FCH845    11/01/2012 09:04:00 AM    (37.79871, -122.276221)
    # 6FCH845    10/24/2012 11:15:00 AM    (37.799695, -122.274868)
    # 6FCH845    10/24/2012 11:01:00 AM    (37.799693, -122.274806)
    # 6FCH845    10/24/2012 10:20:00 AM    (37.799735, -122.274893)
    # 6FCH845    05/08/2014 07:30:00 PM    (37.797558, -122.26935)
    # 6FCH845    12/31/2013 10:09:00 AM    (37.807556, -122.278485)

    # OK, so her car shows up 6 times in this data set.  However, it's hard to make sense of
    # those coordinates.  I don't know about you, but I can't read GPS so well.
    # 
    # So, let's work out a way to show where her car has been seen on a map.  We'll need to
    # extract the latitude and longitude, as the data isn't quite in the format that the
    # mapping software expects: the mapping software expects the latitude to be in one column
    # and the longitude in another.  Let's write some Python code to do that, by splitting
    # the Location string into two pieces: the stuff before the comma (the latitude) and the
    # stuff after (the longitude).
    '(37.79871, -122.276221)'.split(',')     # ['(37.79871', ' -122.276221)']

    def get_latitude(s):
        before, after = s.split(',')         # Break it into two parts
        lat_string = before.replace('(', '') # Get rid of the annoying '('
        return float(lat_string)             # Convert the string to a number

    def get_longitude(s):
        before, after = s.split(',')                 # Break it into two parts
        long_string = after.replace(')', '').strip() # Get rid of the ')' and spaces
        return float(long_string)                    # Convert the string to a number

    # Let's test it to make sure it works correctly.
    get_latitude('(37.797558, -122.26935)')     # 37.797558
    get_longitude('(37.797558, -122.26935)')    # -122.26935

    # Good, now we're ready to add these as extra columns to the table.
    lprs = lprs.with_columns(
        'Latitude',  lprs.apply(get_latitude, 'Location'),
        'Longitude', lprs.apply(get_longitude, 'Location')
    )
    # Plate      Timestamp                 Location                                     Latitude   Longitude
    # 1275226    01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)    37.7983    -122.276
    # 27529C     01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)    37.7983    -122.276
    # 1158423    01/19/2011 02:06:00 AM    (37.798304999999999, -122.27574799999999)    37.7983    -122.276
    # ... (rows omitted)

    # And at last, we can draw a map with a marker everywhere that her car has been seen.
    jean_quan = lprs.where('Plate', '6FCH845').select('Latitude', 'Longitude', 'Timestamp')
    Marker.map_table(jean_quan)
    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V004800_DTH.mp4" alt="Lec 10.3 Case Study" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 10.4 Inferences

### Note


+ Demo
    ```html

    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V004600_DTH.mp4" alt="Lec 10.4 Inferences" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 10.5 Implications

### Note


+ Demo
    ```html

    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V004700_DTH.mp4" alt="Lec 10.5 Implications" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>







