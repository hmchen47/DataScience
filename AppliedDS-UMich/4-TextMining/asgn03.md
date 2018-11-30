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

+ [Question 11 (Q11)](https://www.coursera.org/learn/python-text-mining/discussions/weeks/3/threads/5793y4PfEee6eQpzQFu5YA)

    + Jim Soiland init

        I've searched all the other threads regarding this topic and I'm just not seeing where I'm going wrong! I've got all the others right so I'm just pounding my head trying to get this guy correct, as well.

        Steps taken:

        1. Created CountVectorizer with parameters set min=5, ngram_range=(2,5) and analyzer set per instructions. Fitted on X_Train, then transformed Xtrain/test. Feature names extracted and synthetic feature names added upfront.
        2. Series objects created for each necessary xtrain/xtest function (char-length, digits, and special chars). Each series is appended to the feature matrix with course-supplied function.
        3. Logistic Regression model fitted with C=100 to augmented train data.
        4. Model coefficients and feature names zipped together and converted to dataframe. Dataframe gets sorted by coefficient value. Get the first ten feature names from the dataframe as the smallest features, then get the last ten feature names (and reverse them) as the largest features. I've also tried sorting the dataframe by the coefficients' absolute value. Still no dice.
        Does anyone see anything glaring in my approach? I'm really scratching my head here!

        Answer: (.991...86, [' .','..',...,' m'],['digit_count',...,'ar'])

    + Valdimir reply

        A correct way to add new feature names to a list: using np.append. Please note that np.append returns a copy of the original array, hence for the change to take place it should be assigned to a feature list!

        Like this: `feature_names = np.append(feature_names, ['name1','name2','name3'])`

    + Marcus Fischer Reply

        just in case you haven't solved the problem yet. To me it seems like you didn't manage to add the features to the vectorisations of X_train and X_test or you are simply not using them in your computations.

        Try something like:

        X_train_addfeatures = add_feature(X_train_vect, [len_doc, # of digits, # of non_words])

        X_test_addfeatures= add_feature(X_test_vect, [len_doc, # of digits, # of non_words])

        Then use X_train_addfeatures and X_test_addfeatures to compute the AUC_score using the setup you described above.

        Ultimately, you should be able to extract the feature names of the smallest coefficients by simply using:

        sorted_coef_index_min = clf.coef_[0].argsort()

        To reverse the descending order of this array you just have to add [::-1] to argsort().

        sorted_coef_index_max = clf.coef_[0].argsort()[::-1]

        Now you should be ready to go. However, don't forget to convert the resulting np.arrays to python list before you submit your assignment. ;-)

+ [Hints for Q11](https://www.coursera.org/learn/python-text-mining/discussions/weeks/3/threads/OlPMVljnEei6sA7UYifM3A)

    Dear all,

    I had some troubles with Q11, after reading the previous posts I could not find the bug. I was convinced I was doing something wrong with the list.

    Finally I realised that I was too fast coping from the previous answers I provided. So my errors came from not correctly applying the instruction ...

    + I was not using the Count Vectorizer but the TfidfVectorizer as in q9
    + I haven't pass in analyzer='char_wb'



## Solution

+ Q1: What percentage of the documents in spam_data are spam?
    ```python
    def answer_one():

        return spam_data['target'].sum() / spam_data.shape[0] * 100 #Your answer here
    # 13.406317300789663
    ```

+ Q2: Fit the training data `X_train` using a Count Vectorizer with default parameters.

    What is the longest token in the vocabulary?
    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    def answer_two():

        cntvec = CountVectorizer().fit(X_train)

        tokens = cntvec.get_feature_names()

        return max(tokens, key=len)#lambda x: len(x) if x.isalpha() else -1) #Your answer here
    # 'com1win150ppmx3age16subscription'
    ```

+ Q3: Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.

    Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
    ```python
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import roc_auc_score

    def answer_three():

        cntvec = CountVectorizer().fit(X_train)
        X_train_vectorized = cntvec.transform(X_train)

        mnnbclf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

        predictions = mnnbclf.predict(cntvec.transform(X_test))

        return roc_auc_score(y_test, predictions) #Your answer here
    # 0.97208121827411165
    ```

+ Q4: Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.

    What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?

    Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.

    The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first.
    ```python
    def answer_four():

        tfidfvec = TfidfVectorizer().fit(X_train)
        X_train_vectorized = tfidfvec.transform(X_train)

        feature_names = np.array(tfidfvec.get_feature_names())

        tfidf_index = X_train_vectorized.max(0).toarray()[0]

        tfidf_ser = pd.Series(tfidf_index, index=[feature_names]).sort_values()

        small_20 = tfidf_ser[:20]
        large_20 = tfidf_ser[:-21:-1]

        return (small_20, large_20) #Your answer here
    # (sympathetic     0.074475
    #  healer          0.074475
    #  aaniye          0.074475
    #  dependable      0.074475
    #  companion       0.074475
    #  listener        0.074475
    #  athletic        0.074475
    #  exterminator    0.074475
    #  psychiatrist    0.074475
    #  pest            0.074475
    #  determined      0.074475
    #  chef            0.074475
    #  courageous      0.074475
    #  stylist         0.074475
    #  psychologist    0.074475
    #  organizer       0.074475
    #  pudunga         0.074475
    #  venaam          0.074475
    #  diwali          0.091250
    #  mornings        0.091250
    #  dtype: float64,
    #  146tf150p    1.000000
    #  havent       1.000000
    #  home         1.000000
    #  okie         1.000000
    #  thanx        1.000000
    #  er           1.000000
    #  anything     1.000000
    #  lei          1.000000
    #  nite         1.000000
    #  yup          1.000000
    #  thank        1.000000
    #  ok           1.000000
    #  where        1.000000
    #  beerage      1.000000
    #  anytime      1.000000
    #  too          1.000000
    #  done         1.000000
    #  645          1.000000
    #  tick         0.980166
    #  blank        0.932702
    #  dtype: float64)
    ```

+ Q5: Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 3.

    Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1 and compute the area under the curve (AUC) score using the transformed test data.
    ```python
    def answer_five():

        tfidfvec = TfidfVectorizer(min_df=3).fit(X_train)
        X_train_vectorized = tfidfvec.transform(X_train)

        mnnbclf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

        predictions = mnnbclf.predict(tfidfvec.transform(X_test))

        return roc_auc_score(y_test, predictions) #Your answer here
    # 0.94162436548223349
    ```

+ Q6: What is the average length of documents (number of characters) for not spam and spam documents?
    ```python
    def answer_six():

        spam_doc = spam_data[spam_data['target'] == 1]
        spam_doc.loc[:, 'text_len'] = spam_doc['text'].str.len()

        non_spam_doc = spam_data[spam_data['target'] == 0]
        non_spam_doc.loc[:, 'text_len'] = non_spam_doc['text'].str.len()

        return (non_spam_doc['text_len'].sum()/non_spam_doc.shape[0],
                spam_doc['text_len'].sum()/spam_doc.shape[0]) #Your answer here
    # (71.02362694300518, 138.8661311914324)
    ```

+ Q7: Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5.

    Using this document-term matrix and an additional feature, the length of document (number of characters), fit a Support Vector Classification model with regularization C=10000. Then compute the area under the curve (AUC) score using the transformed test data.
    ```python
    from sklearn.svm import SVC

    def answer_seven():

        tfidfvec = TfidfVectorizer(min_df=5).fit(X_train)
        X_train_vectorized = tfidfvec.transform(X_train)

        X_train_aug = add_feature(X_train_vectorized, X_train.apply(len))

        svmclf = SVC(C=10000).fit(X_train_aug, y_train)

        X_test_vectorized = tfidfvec.transform(X_test)
        X_test_aug = add_feature(X_test_vectorized, X_test.apply(len))

        predictions = svmclf.predict(X_test_aug)

        return roc_auc_score(y_test, predictions)  #_vectorized #Your answer here
    # 0.95813668234215565
    ```

+ Q8: What is the average number of digits per document for not spam and spam documents?
    ```python
    def answer_eight():

        # ''.join(c for c in my_string if c.isdigit())

        spam_doc = spam_data[spam_data['target'] == 1]
        spam_doc.loc[:, 'digits'] = spam_doc['text'].apply(lambda x: len(''.join(c for c in x if c.isdigit())))

        non_spam_doc = spam_data[spam_data['target'] == 0]
        non_spam_doc.loc[:, 'digits'] = non_spam_doc['text'].apply(lambda x: len(''.join(c for c in x if c.isdigit())))

        return (non_spam_doc['digits'].sum()/non_spam_doc.shape[0],
                spam_doc['digits'].sum()/spam_doc.shape[0]) #Your answer here
    # (0.2992746113989637, 15.759036144578314)
    ```

+ Q9: Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5 and using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams).

    Using this document-term matrix and the following additional features:
    + the length of document (number of characters)
    + number of digits per document

    fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
    ```python
    from sklearn.linear_model import LogisticRegression

    def answer_nine():

        tfidfvec = TfidfVectorizer(ngram_range=(1, 3), min_df=5).fit(X_train)
        X_train_vectorized = tfidfvec.transform(X_train)

        X_train_aug = add_feature(add_feature(X_train_vectorized, X_train.apply(len)), 
                                X_train.apply(lambda x: len(''.join(c for c in x if c.isdigit()))))

        lregclf = LogisticRegression(C=100).fit(X_train_aug, y_train)

        X_test_vectorized = tfidfvec.transform(X_test)
        X_test_aug = add_feature(add_feature(X_test_vectorized, X_test.apply(len)),
                                X_test.apply(lambda x: len(''.join(c for c in x if c.isdigit()))))

        predictions = lregclf.predict(X_test_aug)

        return roc_auc_score(y_test, predictions) #Your answer here
    # 0.96533283533945646
    ```

+ Q10: What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
    ```python
    def answer_ten():

        # re.sub('[\w]+' ,'', x)

        import re

        spam_doc = spam_data[spam_data['target'] == 1]
        spam_doc.loc[:, 'nonword'] = spam_doc['text'].apply(lambda x: len(re.sub('[\w]+', '', x)))

        non_spam_doc = spam_data[spam_data['target'] == 0]
        non_spam_doc.loc[:, 'nonword'] = non_spam_doc['text'].apply(lambda x: len(re.sub('[\w]+', '', x)))

        return (non_spam_doc['nonword'].sum()/non_spam_doc.shape[0],
                spam_doc['nonword'].sum()/spam_doc.shape[0]) #Your answer here
    # (17.29181347150259, 29.041499330655956)
    ```

+ Q11:Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.

    To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

    Using this document-term matrix and the following additional features:
    + the length of document (number of characters)
    + number of digits per document
    + number of non-word characters (anything other than a letter, digit or underscore.)

    fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.

    Also find the 10 smallest and 10 largest coefficients from the model and return them along with the AUC score in a tuple.

    The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.

    The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients: ['length_of_doc', 'digit_count', 'non_word_char_count']
    ```python
    def answer_eleven():

        import re 

        tfidfvec = CountVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=5).fit(X_train)
        X_train_vectorized = tfidfvec.transform(X_train)

        X_train_aug = add_feature(add_feature(
                add_feature(X_train_vectorized, X_train.apply(len)), 
                X_train.apply(lambda x: len(''.join(c for c in x if c.isdigit())))),
                    X_train.apply(lambda x: len(re.sub('[\w]+', '', x))))

        lregclf = LogisticRegression(C=100).fit(X_train_aug, y_train)

        X_test_vectorized = tfidfvec.transform(X_test)
        X_test_aug = add_feature(add_feature(add_feature(X_test_vectorized, X_test.apply(len)),
                                X_test.apply(lambda x: len(''.join(c for c in x if c.isdigit())))),
                                X_test.apply(lambda x: len(re.sub('[\w]+', '', x))))

        predictions = lregclf.predict(X_test_aug)

        sorted_coef_index = lregclf.coef_[0].argsort()

        feature_names = np.append(tfidfvec.get_feature_names(), ['doc_len', 'digit_count', 'non_char'])

        return (roc_auc_score(y_test, predictions),
                list(feature_names[sorted_coef_index[:20]]), 
                list(feature_names[sorted_coef_index[:-21:-1]])) #Your answer here
    # (0.97885931107074342,
    #  array(['. ', '..', '? ', ' i', ' y', ' go', ':)', ' h', 'go', ' m', 'h ',
    #         'he', 'hen', ' ok', ' 6', 'ok', 'ca', 'pe', 'so', ':) '],
    #        dtype='<U11'),
    #  array(['digit_count', 'ne', 'ia', 'co', 'xt', ' ch', 'mob', ' x', 'ww',
    #         'ar', 'eply ', ' a ', 'ply ', ' dar', 'uk', 'art', 'rt', 'dar',
    #         ' ba', ' en'],
    #        dtype='<U11'))
    ```


