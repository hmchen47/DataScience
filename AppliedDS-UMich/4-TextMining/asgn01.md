# Assignment 1

## [Extra resources on Regex](https://www.coursera.org/learn/python-text-mining/discussions/weeks/1/threads/HyeJPFWWEei_yQqv26nkEA)

### YW Init

If you need more complicated expressions of regex for the assignment or just in general want to know more, below are some useful resources. Enjoy!

[Regex expression reference and tester](https://regexr.com)

[Regex cheatsheet](http://www.rexegg.com/regex-quickstart.html)

Another [regex tester](https://www.debuggex.com/) (I highly recommend this one as it has relationship graph to demonstrate the relationship of different groups and show you which group is working and which is not)

[Regex tutorial](https://www.guru99.com/python-regular-expressions-complete-tutorial.html) (if you want to know more, I strongly recommend reading it through, you will know more about the difference among search, match, and findall functions and when to use each of them)

### Jo Are By Amended

For the curious-minded, let me add a few references to how regular expressions work behind the scenes (it helped me deepen my understanding of regular expressions, and I wish someone would have told me sooner).

https://en.wikipedia.org/wiki/Regular_expression#Implementations_and_running_times

https://en.wikipedia.org/wiki/Finite-state_machine

https://en.wikipedia.org/wiki/Deterministic_finite_automaton

https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton

https://www.youtube.com/results?search_query=regular+expression+finite+automata



## [Something wrong with the grader](https://www.coursera.org/learn/python-text-mining/discussions/weeks/1/threads/AoLX8rSbEeiqnRI0WnAb-A)

### Uwe F Mayer Replay 1

First up, your code is not good, sorry to say. You are getting zero because the autograder cannot run your code. That's what the message "Unable to load student data file" means. Here are the most common causes for that:

1. Notebook imports matplotlib or has anything else related to matplotlib such as %matplotlib notebook

2. Notebook plots

3. Notebook prints (using print)

4. Notebook has a syntax error

5. Notebook imports a library not available to the grader

6. Notebook uses an option / parameter to a more recent version of a library that the autograder's older library version does not support

The autograder works by first running the entire notebook, and only afterwards calls the assigned functions to check if the expected answer is produced. If it cannot run the entire notebook you'll get the message about unable to load student data file.

The steps to fix are to make sure none of the above points happen. You may want to use Kernel>Restart and Run all to check for 4. You can upload your notebook to the online system and run it there to check for 5. The first 3 you check by simply looking at your notebook and making sure you don't do any of that (after all, where would the autograder display the output?).


### Uwe F Mayer Replay 2

Yeifer, first run the notebook on the Coursera online system. Then add a cell at the end that calls your assigned function. Run the entire notebook again via Kernel > Restart and Run All. This simulates what the autograder does, it first runs your entire notebook, and only then calls your assigned function. In that cell you added at the end, does it still produce the correct answer?

Another issue I didn't list above is that some learners change the function definition. The assignment asks to implement the function date_sorter(), however quite a few learner don't do that and instead implement date_sorter(df), which as far as the grader is concerned is something entirely different.

Does any of that help?

### Uwe F Mayer Replay 3

We don't know what you are submitting, but I am guessing you misunderstood the instructions. The correct answer should look like:

```python
0 9
1 84
2 2
3 53
4 28
5 474
6 153
7 13
8 129
9 98
10 111
...
490 152
491 235
492 464
493 253
494 231
495 427
496 141
497 186
498 161
499 413
Length: 500, dtype: int32
```

I have seen it repeatedly that learners switched index and value in their submission, or sorted by the wrong thing. Please check that your submission matched (or mostly matches) the above. You don't have to get exactly what I got to get 100%, there's a bit of a fudge factor built into the grader.

Please update this thread with what you find out.

Anurag, the first two lines in my posted answer say that the records with the earliest date in the dataset is record 9, then record with the second-earliest date in the dataset is record 84. Your post relates to the first two records of the dataset, which do not contain the two earliest dates in the dataset, and hence will not be the first two indices of the answer. It appears you misunderstood what the answer is supposed to contain: Extract all the dates, then sort the records by those dates, and return the resorted original index. As it turns out record #9 (that's the 10th record) has the earliest date, and so on.


### Uwe F Mayer Replay 4

Miroslav, start with a brand new unedited assignment notebook, and submit it. It should load, but of course you won't get any points. Then start adding your code to it, piece by piece. Submit after each piece (saving your notebook of course before submitting and always doing a Kernel>Restart and Run All to capture errors in the notebook). At some point or the other the submission will fail to load. Look at that code and chances are you will find an overlooked print or plot or import statement. Should you not have such a statement, then rewrite that code until the grader can run it.

Of course if you work offline you should do the same thing with the online system. Your libraries might differ, and some libraries now have migrated far enough away so that code that runs locally no longer runs on the online system.

Please let us know what you find.


### Uwe F Mayer Replay 5

Sahil, do you get "cannot load student data file" as part of the grader's messages? If so there's something in your notebook the autograder cannot handle, and it never even gets to your answer. As a recap, submissions must:

+ not use print
+ not use matplotlib (not even import)
+ have no syntax or other error
+ after a kernel restart be able to be run all the way through and then in function calls afterwards (added by the autograder) produce desired answers
+ not expect the autograder to run the same new library versions as learners run offline (the autograder is using the same ones as the online Coursera notebook, and they are about 1 year old)

### Uwe F Mayer Replay 6

Well, you asked for it: You are wrong :-).

There are indeed identical dates, but the scoring accounts for that. It computes a proximity score of your returned record order to the one the autograder has, and if it's close enough you'll get 100%. You might want to write your parsed dates along with the original records into a file and open that with Excel. The data is tiny (500 records) so you can easily check what you've missed. Quite often incorrectly parsed dates have the day as 1. Another common issue is to make sure you parse dates in the US fashion, that is month is first in the raw data. Finally make sure you sort correctly, I'd suggest to transform all dates into a single YYYYMMDD string and sort by that.


### Uwe F Mayer Replay 7

or good measure I resubmitted with a tie-breaking sort. Specifically if I had multiple records with the same date extracted I sorted each such group of records with identical date by the index in increasing order for one submission, and by decreasing index order in the other submission. That resulted in 92 differences. Of course most differences are simply by one position or so. Both submissions got 100% from Grader version 2017.10.16a. So as said, as long as the extracted dates are correct and the sorting is done correctly the sort order of records with identical dates does not matter.


## [Notice for Auditing Learners: Assignment Submission](https://www.coursera.org/learn/python-text-mining/discussions/weeks/1/threads/kx2S5rKrEeefpw51rAEiYg)

Please note: only verified learners can submit assignments. If you are auditing this course, you will be able to go through the quizzes or assignments, but you will not be able to submit your assignment for a grade. If you wish to have your assignments graded and receive a course certificate, we encourage you to upgrade to the Certified Learner track for this course. Coursera has provided [information about purchasing a certificate](https://learner.coursera.help/hc/en-us/articles/208280146-Pay-for-a-course-or-Specialization), and you can also get help from the [Coursera Help Center](https://learner.coursera.help/hc/en-us).





