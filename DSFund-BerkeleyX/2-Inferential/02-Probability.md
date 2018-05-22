# Section 2: Probability (Lec 2.1 - Lec 2.5)

+ Demo
    ```python
    from datascience import *
    import numpy as np
    ```

## Lec 2.1 Monty Hall Problem

### Notes

+ [Monty Hall Problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
    + Build a simulation to get the probability

+ Demo
    ```python
    goats = make_array('first goat', 'second goat')
    hidden_behind_doors = np.append(goats, 'car')
    hidden_behind_doors

    def other_goat(goat):
        if goat == 'first goat':
            return 'second goat'
        if goat == 'second goat':
            return 'first goat'

    other_goat("first goat")    # second goat
    other_goat("second goat")   # first goat
    other_goat("sheep")         # None

    contestant_goat = np.random.choice(hidden_behind_doors)
    contestant_goat     # randomly select from first/second goat

    def monty_hall_game():
        """[contestant's guess, revealed, remains]"""
        contestant_guess = np.random.choice(hidden_behind_doors)
        
        if contestant_guess == 'first goat':
            return ['first goat', 'second goat', 'car']
        elif contestant_guess == 'second goat':
            return ['second goat', 'first goat', 'car']
        elif contestant_guess == 'car':
            revealed = np.random.choice(goats)
            return ['car', revealed, other_goat(revealed)]

    monty_hall_game()

    trials = Table(['guess', 'revealed', 'remains'])
    trials

    trials.append(monty_hall_game())

    trials.append(monty_hall_game())
    trials.append(monty_hall_game())
    trials.append(monty_hall_game())
    trials.append(monty_hall_game())
    trials

    for i in np.arange(9995):
        trials.append(monty_hall_game())

    trials.pivot('remains', 'guess')
    trials.group('guess')
    trials.group('remains')
    ```


### Video

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/148bc397ac774e18a846abdb54ee2e1a/7cd2c9c09383493da3127ca92c1610e4/4?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%40d9dedc9b412a42009af73e0510261e0f" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.2 Probability

### Notes


### Video

<a href="URL" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.3 Multiplication Rule

### Notes


### Video

<a href="URL" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.4 Addition Rule

### Notes


### Video

<a href="URL" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.5 Probability Example

### Notes


### Video

<a href="URL" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>


## Reading and Practice for Section 2

### Readings


### Practices

