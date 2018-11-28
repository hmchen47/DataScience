# Assignment 3 

## Discussion Forum

+ [Assignment 3 Questions 9 & 11 Autograder Issues](https://www.coursera.org/learn/python-text-mining/discussions/weeks/3/threads/liUnu-3PEeePUQ6MjeQj7A)
    + Uwe F Mayer reply 1
        In general when computing AUC scores one needs to use the true labels (y_test in our course) and the computed scores from decision_function() or predict_proba() or some such method. However, the autograder version 2017.10.16a expects the AUC to be computed for both Questions 9 and 11 using the predicted class labels, that is, one incorrectly needs to use the predict() method instead of a scoring function. This is not the right thing to do but is necessary until this autograder bug is fixed.

        One more clarification on Question 11, the instructions say that the return value of the function should be "(AUC score as a float, smallest coefs list, largest coefs list)". This makes one think that the two lists should be python lists. However the autograder expects the lists to be pandas Series, so the instructions should more correctly say: This function should return a tuple (AUC score as a float, smallest coefs series, largest coefs series). The solution to Question 11 is something like the following:

        ```python
        (0.97..., feature_name
        .     -0.869749
        ..    -0.860869
        ?     -0.676969
        <more rows>
        so    -0.411370
        :)    -0.403203
        Name: coef, dtype: float64, feature_name
        digit_count    1.212219
        ne             0.597775
        ia             0.541486
        <more rows>
        ba            0.402199
        en            0.402157
        Name: coef, dtype: float64)
        ```
    + Uwe F Mayer reply 2

        for an ROC or AUC computation you need two data series, and if it's done correctly that's the true labels and the predicted labels. As the post says, because of the grader bug, one needs to use the true labels as the first series and the computed score (instead of the predicted labels) as the second series.

+ [Assignment 3 Question 4 Clarification](https://www.coursera.org/learn/python-text-mining/discussions/weeks/3/threads/4y22x-3KEeeaYw6MXNSBHA)
    + Uwe F Mayer init

        Unfortunately Assignment 3 Question 4 is a poorly worded question leaving out two key points.

        The question "What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?" is not clear, after all the tf-idf value of each feature depend on the document, not just the feature. What the autograder expects is more precisely the following: For each feature compute first the maximum tf-idf value across all documents in X_train. What 20 features have the smallest maximum tf-idf value and what 20 features have the largest maximum tf-idf value?

        The pair of series to be returned should not have any names, nor should their index have a name. Instructions: for a series s you can remove its name and its index's name by setting them to None as in s.name=None and s.index.name=None.

        The correct answer (as of autograder 2017.10.16a) is the following [obviously I won't post the entire answer here :-)]:
        ```python
        (aaniye          0.074475
        athletic        0.074475
        chef            0.074475
        ... 16 more rows
        approaching     0.091250
        dtype: float64, 146tf150p    1.000000
        645          1.000000
        anything     1.000000
        ... 16 more rows
        blank        0.932702
        dtype: float64)
        ```

    + Azk Question

        I am so sorry, despite your really helpful advice in this thread, I still have so questions.

        After creating the TFIDF matrix, I created an array that took the maximum tfidf values for each feature (i.e the max of each column in the matrix), across all documents (the rows).

        This array, called "b", returns a list that seems to be sorted in ascending order, starting with 0.07 and up, with long decimal numbers repeated often, (as in 0.074, 0.074, 0.074, , 0.074, 0.09, 0.09, ....)

        When I create a series "s" from the array "b", and return "s", I don't get it sorted in the same way.

        So what is confusing me is -- why this difference between "b" and "s" ? And also, isn't it odd that there are so many td-idf values that are exactly the same?

    + Uwe Rply for Zak

        Zak, this is where you finally need to look at what TFIDF actually is. After you understood that you may want to think about what the TFIDF value of a word is that occurs exactly once (or exactly once in 2 documetns, ...) in the corpus.

        I just noticed that I didn't comment on your "b" vs "s" question. If you have a list in the pyton sense (meaning type(b) actual is list) and you create a series s=pd.Series(b) then it will be sorted the the same way and have a default integer range index. If not, something's wrong. Are you sure "b" is in fact a list?

    + Uwe F Mayer Reply 2

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

        ascending : bool or list of bool, default True

        Sort ascending vs. descending. Specify list for multiple sort orders. If this is a list of bools, must match the length of the by.

    + Uwe Reply 3

        On a high level, let's try to understand if your answer is correct and the grader fails to accept it, or if your answer is incorrect. Here's some check code, if you get the same output it's fairly sure your answer is correct. If so, you may want look into why the autograder fails to accept it (there's lots of discussion of that elsewhere, if you submit online you may want to restart your server, and in general, make sure your notebook does the right things).
        ```python
        # compute summary stats on characters
        from collections import defaultdict
        d0 = defaultdict(lambda: 0)
        for c in "".join(answer_four()[0].index):
            d0[c] = d0[c]+1
        d1 = defaultdict(lambda: 0)
        for c in "".join(answer_four()[1].index):
            d1[c] = d1[c]+1
        idx = sorted(list(set(d0.keys()).union(set(d1.keys()))))
        pd.DataFrame(data={'smallest':d0, 'largest':d1}, index=idx)

        # largest	smallest
        # 0	1.0	NaN
        # 1	2.0	NaN
        # 4	2.0	NaN
        # 5	2.0	NaN
        # 6	2.0	NaN
        # a	7.0	17.0
        # b	2.0	1.0
        # c	1.0	8.0
        # d	1.0	5.0
        # e	13.0	21.0
        # f	1.0	2.0
        # g	2.0	5.0
        # h	6.0	7.0
        # i	6.0	13.0
        # k	5.0	NaN
        # l	2.0	6.0
        # m	2.0	5.0
        # n	9.0	13.0
        # o	6.0	11.0
        # p	2.0	9.0
        # r	3.0	11.0
        # s	NaN	11.0
        # t	9.0	15.0
        # u	1.0	4.0
        # v	1.0	1.0
        # w	1.0	NaN
        # x	1.0	1.0
        # y	3.0	5.0
        # z	NaN	1.0

        sum(answer_four()[0]), sum(answer_four()[1])
        # (1.5230406459656354, 19.91286747680313)
        ```

    + Uwe Reply 4

        Satya, you have the correct solution. So you need to figure out why it doesn't pass the autograder. Often the reason is that learners are doing something below an assigned function in their notebook that affects what that function computes. "Something" usually is modification of a global object, such as a provided dataframe. You might want to check for this by adding a cell at the end of your notebook that calls answer_one(), another cell that calls answer_two() and so on. Then restart your kernel and run the entire notebook top to bottom (Kernel > Restart and Run All) and check if those calls produce the expected answers. That's how the autograder does it, it first runs the entire notebook, and then checks by calling one function after the other.

        If that produces the correct answers, then I don't know what to tell you. Of course I assume you are not getting the "unable to load student datafile" message, if you are getting that, check the discussion forum, there's lots on that.

        If all fails and you just can't get it sorted out you might want to hardcode your answer:
        ```python
        def answer_four():
            return (pd.Series(index=['aaniye',
        'athletic',
        'chef',
        ...
        'approaching'],
        data=[0.074475,
        0.074475,
        0.074475,
        ...
        0.091250]),
        pd.Series(index=['146tf150p',
        '645',
        'anything',
        ...
        'blank'],
        data=
        [1.000000,
        1.000000,
        1.000000,
        ...
        0.932702]))
        ```

    + Uwe reply 5

        Probably never. On the other hand, being able to handle all the NLTK tools and being able to complete this analysis is likely a reasonable way to get learners to use the tools, and to think about concepts. Furthermore, it is checkable with an automated grader. You opinion on all of that of course may differ.

    + Uwe Reply 6

        Moshfiqur, that all depends on what those other records are. Also, you need to understand what the sentence "sorted by tf-idf value and then alphabetically by feature name" means. It means you should sort by tf-idf value, and use the alphabetical sort as a tie break where needed. This is very different from sorting first by tf-idf, and then in a second step doing a complete sort alphabetically. Below is an example (note: in the example I sort increasing by both tf-idf and by feature name).
        ```python
        import pandas as pd
        df = pd.DataFrame({'tf-idf':[1,1,0.5], 'feature name': ['c', 'a', 'b']})
        print('original df')
        print(df)
        print('\nsort with tie-break')
        print(df.sort_values(['tf-idf', 'feature name']))
        print('\nsort with two consecutive sorting operations')
        print(df.sort_values('tf-idf').sort_values('feature name'))

        # original df
        #   feature name  tf-idf
        # 0            c     1.0
        # 1            a     1.0
        # 2            b     0.5
        # 
        # sort with tie-break
        #   feature name  tf-idf
        # 2            b     0.5
        # 1            a     1.0
        # 0            c     1.0
        # 
        # sort with two consecutive sorting operations
        #   feature name  tf-idf
        # 1            a     1.0
        # 2            b     0.5
        # 0            c     1.0
        ```


## Solution

