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

+ Basics
    + Lowest value: $0$
        + Chance of an event that is impossible
    + Highest value: $1$ (or $100\%$)
        + Chance of an event is certain
    + If an event has chance $70\%$, then the chance that  it doesn't happen is
        + $100\% - 70\% = 30\%$
        + $ 1 - 0.7 = 0.3$
+ Equally Likely Outcome  
    Assuming all outcomes are equally likely, the chance of an even A is  
    $$ P(A) = \frac{\text{number of outcomes that make A happen}}{\text{total number of outcomes}} $$


### Video

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/148bc397ac774e18a846abdb54ee2e1a/7cd2c9c09383493da3127ca92c1610e4/4?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%40d9dedc9b412a42009af73e0510261e0f" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.3 Multiplication Rule

### Notes

+ Fraction of a Fraction
+ Multiplication Rule  
    $$ \text{Chance that two events A and B both happen} = P(\text{A happen} \times P(\text{B happens given that A happens})) $$  
    + The answer is _less than or equal to_ each of the two chances being multiplied
    + The more conditions you have to satisfy, the less likely you are to satisfy them all

### Video

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/148bc397ac774e18a846abdb54ee2e1a/7cd2c9c09383493da3127ca92c1610e4/4?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%40d9dedc9b412a42009af73e0510261e0f" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.4 Addition Rule

### Notes

+ Addition Rule  
    If event A can happen in _exactly one_ of two ways, then  
    $$ P(A) = P(\text{first way}) + P(\text{second way}) $$

### Video

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/148bc397ac774e18a846abdb54ee2e1a/7cd2c9c09383493da3127ca92c1610e4/4?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%40d9dedc9b412a42009af73e0510261e0f" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>

## Lec 2.5 Probability Example

### Notes

+ Example: At Least One Head
    + In 3 tosses:
        + Any outcome _except_ TTT
        + $P(TTT) = (1/2)(1/2)(1/2) = 1/8$
        + $P(\text{at least one head}) = 1 - P(TTT) = 7/8 = 87.5\%$
    + In 10 tosses:
        + $1 - (1/2)^{10} $
        + $99.9\%$

### Video

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/148bc397ac774e18a846abdb54ee2e1a/7cd2c9c09383493da3127ca92c1610e4/4?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%40d9dedc9b412a42009af73e0510261e0f" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:48px;height:48px;border:0;"> 
</a>


## Reading and Practice for Section 2

### Readings


### Practices

