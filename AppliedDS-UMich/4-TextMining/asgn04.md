# Assignment 4

## Discussion Forum

### [Assignment 4 Guide](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/hyd7rkwBEeinuQpIIhlTbA)

+ Jo Are By Init

    A few things I wish I knew before I started working on this assignment.

    __Part 1__

    __doc_to_synsets__

    + Tokenize words
    + Tag tokens
    + Convert tags with predefined function convert_tag. Note: convert_tag may return None and that's ok. No token should be removed at this point.
    + Get synsets from (token, wordnet_tag) pairs using wn.synsets (including tokens with wordnet_tag None)
    + wn.synsets returns zero or more synsets. Keep only the first synset. In the case of no synsets, discard the empty list.
    + Return the list of synsets.

    __similarity_score__

    + For each synset in s1, loop over all the synsets in s2 and find the largest similarity value for each one. Use x.path_similarity(y) where x is a synset from s1 and y is a synset from s2.
    + path_similarity may return None in which case it is to be discarded (and make sure it does not count when you later divide the sum for normalization).
    + Return the sum of scores divided by the number of scores found (not including None).
    + Just to make this point clear, None should not be replaced by zero, but completely discarded.

    Here is some additional debugging info (A similar post by mentor Uwe F Mayer helped me a lot, thanks)

    ```python
    doc1 = 'Ali saw the man with the telescope.'
    doc2 = 'The man with the telescope was seen by Ali.'
    synsets1 = doc_to_synsets(doc1)
    print(synsets1)
    synsets2 = doc_to_synsets(doc2)
    print(synsets2)
    score1 = similarity_score(synsets1, synsets2)
    print(score1)
    score2 = similarity_score(synsets2, synsets1)
    print(score2)
    score3 = document_path_similarity(doc1, doc2)
    print(score3)

    [Synset('ali.n.01'), Synset('saw.v.01'), Synset('man.n.01'), Synset('telescope.n.01')]
    [Synset('man.n.01'), Synset('telescope.n.01'), Synset('be.v.01'), Synset('see.v.01'), Synset('by.r.01'), Synset('ali.n.01')]
    0.7916666666666667
    0.6619047619047619
    0.7267857142857144
    ```

    __Part 2__

    I found this part more straight forward. Make sure to read the assignment carefully.

    __ldamodel__

    Read the assignment and documentation. Try to get it right the first time, as this part takes a while to run, especially if you get it wrong.

    __lda_topics__

    Check the documentation. This should be straight forward.

    __topic_distribution__

    I failed to notice that I returned a nested list the first time around. The easy fix for me was to return the first element of the nested list.

    __topic_names__

    I spent some time trying to select topics based on similarity. After reading a bit on the forums I ended up with just manually listing the topics I thought would fit best subjectively.


+ Santosh Tamhane reply

    I am getting all of Part 1 wrong. It looks like path_similarity is giving me asymmetric values when comparing the token 'correct' from doc2 to any of the tokens from doc1 (be, function, test). This makes my similarity scores possibly incorrect?

    I have independently verified the asymmetry with the following code. Please note that this code is in no way related to the submission. Please help.

    ```python
    cor = wn.synset('correct.a.01')

    oth = wn.synset('test.v.01')

    x = wn.path_similarity(cor, oth)

    if x is None:
        print('Regular Order\nnull')
    else:
        print('Regular Order\n'+str(x))

    x = wn.path_similarity(oth, cor)

    if x is None:
        print('Reverse Order\nnull')
    else:
        print('Reverse Order\n'+str(x))
    ```

    The results are -
    ```python
    Regular Order
    null
    Reverse Order
    0.2
    ```

+ Yusuf Ertas reply

    Hi Santosh, just so that I understand this better I think your question is about the order of the synsets when evaluating path similarity. I have also found that the correct order matters for evaluating path similarity.

    That being said the definition of document_path_similarity is symmetric, so this should not make any difference. Maybe the error is somewhere else.


+ Santosh Tamhane reply

Thanks for your response, Yusuf. The logic I have used is as follows -

1. Word_tokenize
1. POS tagging
1. Convert to Wordnet tags
1. find list of Synsets (first element) for each token. I get (be, function, test) for Doc1 and (use, function, see, code, be, correct) for Doc2
1. Declare a numpy 2-d array of size len(s1) X len(s2). Nest iterations over elements of s2 inside iterations over elements of s1 using enumerate. Use enumerate indices to populate 2-d numpy array with path_similarities between the two elements. Find row-wise max across all columns, then find mean of the resultant array - using numpy functions that do not propagate nan. If compliant with code of conduct, I can post the resultant 2-day array, the array of max of each row and the mean. I am getting a mean score of 0.7333333 for s1 vs s2 and a score of 0.545xxx for the s2 vs s1 combination, giving a symmetric mean score of 0.63xxxx.
Please let me know if this is incorrect.


+ Yusuf Ertas reply

Here is what I get for the 4th step as a test:

<a href="url"> <br/>
    <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/OKYi55DxEeiHxhIKzPUaYA_45a38c59bb65010680786b23a7295bce_synset.PNG?expiry=1544572800000&hmac=DDbJihd_TuELlbp87jtSRrK28d8KCbvEjb-BTR23QfI" alt="text" title="caption" height="100">
</a>

As for your fifth step it sounds fine to me. It is the 4th step that is probably problematic though.


+ Santosh Tamhane reply

    Thanks for the pointer, Yusuf. I checked my code again. Is my dict right? - {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}. Having checked the Wordnet database online, the only way can bring in Angstrom and inch is if I add the underlined elements to my dict - {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v', 'I': 'n', 'D': 'n'}. With that, I get a test_document_path_similarity() answer of 0.554xxxxx.


+ Yusuf Ertas reply

    Hi Santosh, I am not certain whether you are using wn.synset() with the right arguments. In my answer that I use the first argument is the word gathered from the pos_tag and the second is the tag derived from the dictionary.

    The word 'a' should turn into the synset Synset('angstrom.n.01') (which is the first element of the list) without any addition to the dictionary.


+ Santosh Tamhane reply

    Hi Yusuf, after POS tagging I get the following - Since 'a' is returned with 'DT' and there is no 'D' in the original dict, I was getting a None returned from dict lookup - which is why the Synset('angstrom.n.01') was missing in my list of synsets. A similar problem happened with 'in' in the second string. The POS tagging returned with a 'IN'. That led me to believe that - perhaps - my dict was corrupted. Am I doing something wrong in POS tagging?

    ```python
    [('This', 'DT'),
    ('is', 'VBZ'),
    ('a', 'DT'),
    ('function', 'NN'),
    ('to', 'TO'),
    ('test', 'VB'),
    ('document_path_similarity', 'NN'),
    ('.', '.')]
    ```

+ Santosh Tamhane reply
    ```python
    [('Use', 'VB'),
    ('this', 'DT'),
    ('function', 'NN'),
    ('to', 'TO'),
    ('see', 'VB'),
    ('if', 'IN'),
    ('your', 'PRP$'),
    ('code', 'NN'),
    ('in', 'IN'),
    ('doc_to_synsets', 'NNS'),
    ('and', 'CC'),
    ('similarity_score', 'NN'),
    ('is', 'VBZ'),
    ('correct', 'JJ'),
    ('!', '.')]
    ```


+ Yusuf Ertas reply

    Hi Santosh, I had another look at this and I am convinced that the error is in the wn.sysnet conversion. Not sure how I could help more without posting code here. But here is a partial pseudo-code of what I did:
    ```python
    tags=#gather the pos_tags
    for tag in tags:
        tagwn = #Convert tags according to the dictionary
        output = wn.synsets(tag[0],tagwn)
        #Get the first item in the list 'output'.
    ```

    Let me know if this helps..



+ Santosh Tamhane reply

    Thanks for all the help you have given me Yusuf. I will recheck. Best, Santosh.


+ Raoul Biagioni reply

    I have found that in order to pass Part 1 I had to implement the suggestion by Santosh Tamhane and add 'I': 'n', 'D': 'n' to tag_dict.

    This is rather inconvenient considering that the instructions clearly state that the function convert_tag should NOT be modified...


+ Aditya Singh · 2 months agoreply 

    I have got the value for the synsets for the two docs as correct i am getting a value of 0.48****333. If a synset in s1 would have equivalent similarity scores with synsets in s2 for example a synset in s1 has [0.3, 0.5, 0.5 , 0.4] similarity scores with elements of s2. Then are we supposed to add 0.5 (the max) twice ? And also the division would also increase by 1?

+ Thad Wengert reply

    Jo Are notes that some convert_tag results will return None ... but should that be a lot of them ?? I notice that the tags returned by nltk.pos_tag such as DT, VBZ, NN, TO, VB are not in the tag_dict, which has one-char values like N,J,R,V. So naturally the match rate which seems to fish out the first letter, is still pretty low.

    Is this as expected??
    ```python
    Putting this sentence into convert_tag
    [('This', 'DT'),
    ('is', 'VBZ'),
    ('a', 'DT'),
    ('function', 'NN'),
    ('to', 'TO'),
    ('test', 'VB'),
    ('document_path_similarity', 'NN'),
    ('.', '.')]
    
    #gives you this output
    [None, 'v', None, 'n', None, 'v', 'n', None]
    ```

+ Miranda Lam reply

    I found choosing the first synset element may not always work. For the test sentence, 'I like cats', the 3rd element of the token 'I' is the correct one. (See output below). Does anyone know how to determine the best matching sense? Or do I create a set of all synset combination?
```python
[Synset('iodine.n.01'), Synset('one.n.01'), Synset('i.n.03')]
```


+ Miranda Lam reply

    I read the instructions again, more carefully this time. We are instructed to use the first element. Is there a way to help identify/select which sense/element to use? Wish there are more discussion about Wordnet.

+ Yusuf Ertas reply

    Just to add something to the similarity score. You need to drop similarity scores of 0, otherwise they change the mean significantly. I wish I had this to guide me when I was doing the assignment. Upvoted.


+ Jo Are By reply

    I discard similarity values of None, and that is all my code does. I don't explicitly discard values of zero, yet my code passes the grader.

    Are you converting None to zero and then discard the zeros?

    Here's my approach:

    For every element in s1 I make a list of path_similarity values with respect to every element in s2, excluding values of None. For every non-empty list I place the max value in another list.

    Lastly I return the mean of the list of max values.


+ Yusuf Ertas reply

    I have the same approach, however sometimes the max value happens to be 0, and when I add it in the list the mean changes. For reference, what value do you get for test_document_path_similarity()? I get 0.55****73.


+ Justin Mahlik reply

    I found setting max value to 0 works but you need to check if the similarity = None in your maximization function then at the end don't append max value if ==0.


+ Justin Mahlik reply

    Might be a more elegant way to handle this...


+ omar medhat moslhi reply

    I got this number 0.55****73 but I did't pass the grader


+ Yusuf Ertas reply

    @ Omar, that should be the correct answer. Have you tried restarting your kernel and submitting one more time.


+ Chion John Wong reply

    i see that you use stopwords and your output omits "with", "the"

    but it did not omit "by"

    When I use this stopwords, sw = stopwords.words('english')?

    "by" is one of the stopwords. How come your stopwords include the word "by"?

    [Synset('ali.n.01'), Synset('saw.v.01'), Synset('man.n.01'), Synset('telescope.n.01')]

    [Synset('man.n.01'), Synset('telescope.n.01'), Synset('be.v.01'), Synset('see.v.01'), Synset('by.r.01'), Synset('ali.n.01')]


+ James D Lin repy

    I needed to combine Jo Are's suggestion:

    For "path_similarity may return None in whic h case it is to be discarded (and make sure it does not count when you later divide the sum for normalization)." in Jo Are's suggestion,

    and Yusuf Erta's suggestion:

    You need to drop similarity scores of 0, otherwise they change the mean significantly.

    to get the similarity_score() correct.

    To use convert_tag(), be sure to pass correct value to it. I was confused at first. I wish there was some example code in the lecture. e.g. you should pass entity[1] as the tag value to convert_tag() as shown below:





### [Week 4 Notebook Provided Here](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/y1xnKsJ2EeiTdg5seYVqZA)

+ Uwe F. Mayer init

    There isn't a provided one. I assembled a workbook when I took the class from the material presented. Here's the code, each block is a cell.

    ```python
    import re
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import wordnet as wn

    # Use path length in wordnet to find word similarity
    # find sense of words via synonym set
    # n=noun, 01=synonym set for first meaning of the word
    deer = wn.synset('deer.n.01')
    deer

    elk = wn.synset('elk.n.01')
    deer.path_similarity(elk)

    horse = wn.synset('horse.n.01')
    deer.path_similarity(horse)

    # Use an information criteria to find word similarity
    from nltk.corpus import wordnet_ic
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    deer.lin_similarity(elk, brown_ic)

    deer.lin_similarity(horse, brown_ic)

    # Use NLTK Collocation and Association Measures
    from nltk.collocations import *
    # load some text for examples
    from nltk.book import *
    # text1 is the book "Moby Dick"
    # extract just the words without numbers and sentence marks and make them lower case
    text = [w.lower() for w in list(text1) if w.isalpha()]

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(text)
    finder.nbest(bigram_measures.pmi,10)

    # find all the bigrams with occurrence of at least 10, this modifies our "finder" object
    finder.apply_freq_filter(10)
    finder.nbest(bigram_measures.pmi,10)

    # Working with Latent Dirichlet Allocation (LDA) in Python
    # Several packages available, such as gensim and lda. Text needs to be
    # preprocessed: tokenizing, normalizing such as lower-casing, stopword
    # removal, stemming, and then transforming into a (sparse) matrix for
    # word (bigram, etc) occurences.
    # generate a set of preprocessed documents
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    from nltk.book import *

    len(stopwords.words('english'))

    stopwords.words('english')

    # extract just the stemmed words without numbers and sentence marks and make them lower case
    p_stemmer = PorterStemmer()
    sw = stopwords.words('english')
    doc1 = [p_stemmer.stem(w.lower()) for w in list(text1) if w.isalpha() and not w.lower() in sw]
    doc2 = [p_stemmer.stem(w.lower()) for w in list(text2) if w.isalpha() and not w.lower() in sw]
    doc3 = [p_stemmer.stem(w.lower()) for w in list(text3) if w.isalpha() and not w.lower() in sw]
    doc4 = [p_stemmer.stem(w.lower()) for w in list(text4) if w.isalpha() and not w.lower() in sw]
    doc5 = [p_stemmer.stem(w.lower()) for w in list(text5) if w.isalpha() and not w.lower() in sw]
    doc_set = [doc1, doc2, doc3, doc4, doc5]

    # under Windows this generates a warning
    import gensim
    from gensim import corpora, models

    dictionary = corpora.Dictionary(doc_set)
    dictionary

    # transform each document into a bag of words
    corpus = [dictionary.doc2bow((doc)) for doc in doc_set]

    # The corpus contains the 5 documents
    # each document is a list of indexed features and occurrence count (freq)
    print(type(corpus))
    print(type(corpus[0]))
    print(type(corpus[0][0]))
    print(corpus[0][::2000])

    # let's try 4 topics for our 5 documents
    # 50 passes takes quite a while, let's try less
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=10)

    print(ldamodel.print_topics(num_topics=4, num_words=10))
    ```

### [Seven attempts at the first part and still zero points](https://www.coursera.org/learn/python-text-mining/programming/2qbcK/assignment-4-submission/discussions/threads/7tqMktuKEeigcg53SNTyCg)

+ Nathan Thompson Init

    I can't figure out how to solve the first part of this assignment. I'm following, as closely as I can, the provided instructions as well as all the tips I can find on the forums. I have no idea why it's not working and I'm getting extremely frustrated.

    In doc_to_synsets, I'm:

    + converting the doc to tokens using nltk.word_tokenize
    + generating tags for the tokens using nltk.pos_tag
    + converting the tags using convert_tag
    + fetching the first synset returned by wn.synsets for each token and tag pair and ignoring it if none are returned

    In similarity_score, I'm:

    + iterating through each synset in s1 (as syn1...and likewise for s2)
    + getting syn1.path_similarity(syn2) for each syn2 in s2 (I've also tried swapping syn1 and syn2 with no luck)
    + throwing out all 'None' responses
    + grabbing the max similarity value for each synset in s1 and appending it to a list
    + returning the score as sum(max_similarity_scores) / len(max_similarity_scores)

    Still zero points. I'm completely lost. I have no idea what I'm doing wrong.

    Any assistance would be greatly appreciated.

+ Uwe F Mayer Reply

    What do you do when
        + grabbing the max similarity value for each synset in s1 and appending it to a list

    is trying to grab from an empty list?

    Here are a couple of test cases posted on the Discussion Forum elsewhere:

    + Case 1
        ```python
        doc1="I don't like green eggs and ham"
        doc2="Bacon and eggs with a dose of Iodine"
        synsets1 = doc_to_synsets(doc1)
        synsets2 = doc_to_synsets(doc2)
        print(synsets1)
        print(synsets2)
        print(similarity_score(synsets1, synsets2))
        print(similarity_score(synsets2, synsets1))
        print(document_path_similarity(doc1, doc2))
        ```

    + Case 2
        ```python
        [Synset('iodine.n.01'), Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01')]
        [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('angstrom.n.01'), Synset('dose.n.01'), Synset('iodine.n.01')]
        0.5202380952380953
        0.5116666666666667
        0.5159523809523809
        ```

    + Case 3
        ```python
        # these contain a few typos, on purpose
        # contains "love" as a noun and a verb
        doc1="I don't like green eggs and ham, i do not, no love at all."
        doc2="Bacon and eggs with a dose of Iodine, I reallly love!"
        synsets1 = doc_to_synsets(doc1)
        synsets2 = doc_to_synsets(doc2)
        print(synsets1)
        print(synsets2)
        print(similarity_score(synsets1, synsets2))
        print(similarity_score(synsets2, synsets1))
        print(document_path_similarity(doc1, doc2))
        ```

    + Case 4
        ```python
        [Synset('iodine.n.01'), Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01'), Synset('make.v.01'), Synset('not.r.01'), Synset('no.n.01'), Synset('love.n.01'), Synset('astatine.n.01'), Synset('all.a.01')]
        [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('angstrom.n.01'), Synset('dose.n.01'), Synset('iodine.n.01'), Synset('iodine.n.01'), Synset('love.v.01')]
        0.42059483726150393
        0.555952380952381
        0.48827360910694245
        ```

+ Nathan Thompson Reply

    I understand that it should matter. I should rephrase what I said: regardless of whether I toss out empty lists or substitute a zero in place of an empty list, I still don't get the correct answer.

    No, I don't get the same results as the examples above.

    If I skip missing values, as the assignment specifies, I get the following similarity scores for the test cases you posted.

    ```python
    # Case 1
    [Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01')]
    [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('dose.n.01'), Synset('iodine.n.01')]
    0.400297619048
    0.395833333333
    0.39806547619

    # case 2
    [Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01'), Synset('make.v.01'), Synset('not.r.01'), Synset('love.n.01')]
    [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('dose.n.01'), Synset('iodine.n.01'), Synset('love.v.01')]
    0.393518518519
    0.383333333333
    0.388425925926
    ```

+ Nathan Thompson reply

    OK, these helped me figure it out. 100/100 now.

    The problem was in my doc_to_synsets function. While debugging exceptions, I kept adding a check to see whether a tag converted to None, and skipped converting that tag if it did.

    Other assignments had ways to individually debug each function independent of each other, but this one didn't. That compounded my problem immensely.

    In addition, I don't understand how having the word "I" get tagged to "Iodine" is in any way valid. Aside from their first letter, they are not similar, much less equivalent.

    Thanks for your help. I was just about ready to quit.

+ Kumar Rishank 

    the code is throwing the following error, when I run the test code:

    ```python
    ValueError                                Traceback (most recent call last)
    <ipython-input-2-ab549505ab53> in <module>()
        5 print(synsets1)
        6 print(synsets2)
    ----> 7 print(similarity_score(synsets1, synsets2))
        8 print(similarity_score(synsets2, synsets1))
        9 print(document_path_similarity(doc1, doc2))

    <ipython-input-1-7fc75f48bafd> in similarity_score(s1, s2)
        56                 wordset.append((w1,w2,w1.path_similarity(w2)))
        57                 #print(wordset)
    ---> 58         match.append(max(wordset,key=lambda x:x[2]))
        59     mean= np.mean([x[2] for x in match])
        60 

    ValueError: max() arg is an empty sequence
    ```

+ Uwe F MayerMentor 

    Kumar, your code is trying to take the max of an empty sequence, and that empty sequence is wordset. So you need to add some logic to handle that case. This is where most learners run into problems, that is, you need to handle carefully the case when there are no synsets for a given token.

+ Kumar Rishank reply

    Hi Uwe,

    I tried my code by switching the sequence of doc1 and doc2 being sent to document_path_similarity

    for doc1 = 'This is a function to test document_path_similarity.'

    & doc2 = 'Use this function to see if your code in doc_to_synsets \

    and similarity_score is correct!'

    I get a mean of 0.6***

    but for doc2 = 'This is a function to test document_path_similarity.'

    & doc1 = 'Use this function to see if your code in doc_to_synsets \

    and similarity_score is correct!'

    I get the following output:

    ```python
    (Synset('use.v.01'), Synset('be.v.01'), 0.3333333333333333)
    (Synset('use.v.01'), Synset('angstrom.n.01'), 0.1)
    (Synset('use.v.01'), Synset('function.n.01'), 0.14285714285714285)
    (Synset('use.v.01'), Synset('test.v.01'), 0.2)
    (Synset('function.n.01'), Synset('angstrom.n.01'), 0.1)
    (Synset('function.n.01'), Synset('function.n.01'), 1.0)
    (Synset('see.v.01'), Synset('be.v.01'), 0.25)
    (Synset('see.v.01'), Synset('angstrom.n.01'), 0.09090909090909091)
    (Synset('see.v.01'), Synset('function.n.01'), 0.125)
    (Synset('see.v.01'), Synset('test.v.01'), 0.16666666666666666)
    (Synset('code.n.01'), Synset('angstrom.n.01'), 0.1)
    (Synset('code.n.01'), Synset('function.n.01'), 0.14285714285714285)
    (Synset('inch.n.01'), Synset('angstrom.n.01'), 0.25)
    (Synset('inch.n.01'), Synset('function.n.01'), 0.1111111111111111)
    (Synset('be.v.01'), Synset('be.v.01'), 1.0)
    (Synset('be.v.01'), Synset('angstrom.n.01'), 0.1)
    (Synset('be.v.01'), Synset('function.n.01'), 0.14285714285714285)
    (Synset('be.v.01'), Synset('test.v.01'), 0.2)

    #for mean i get this error
    ValueError: max() arg is an empty sequence
    ```

+ Uwe F MayerMentor reply

    Kumar, you need to debug your code. I suggest you enter a bunch of print statements. At some point you are likely having a double loop over the the synsets from the two documents. You need to ask yourself: what happens if one of the tokens does not have a synset (such as the token "doc_to_synsets" for example)?


### [Assignment 4 : Part - 2](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/95ul5tzJEeiXpgopafspSg)

+ Sahil Soni Init

    Please help me out in this.

    Steps followed topic_distribution():

    + Created a default CountVectorizer()
    + Transform new_doc using the vectorizer in Step 1
    + Created Sparse2Corpus from transformed vector
    + using already created LDA model get_document_topics() on step 3 corpus, it returns a TransformedCorpus type object
    + created list from TransformedCorpus object using list function
    + Return 0th index of list which only returns list of 4 tuples with topic # 0, 4, 6, 7

    Also not able to understand, topic_names. please suggest.


+ Uwe F MayerMentor · a month ago

    Sahil, the assignment provides code for the vectorization:
    ```python
    vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english',
        token_pattern='(?u)\\b\\w\\w\\w+\\b')
    ```

    You need to use vect and not create a new one. So Step 1 is incorrect in your list. Finally what's returned from Step 6 should be a list of 10 pairs
    ```python
    [(0, 0.020001831829864054),
    (1, 0.02000204822465949),
    ...
    (9, 0.3436751665320027)]
    ```

+ 
+ Allen Wang reply

    Please help. When I try to vect.fit_transform(new_doc), I get this

    ValueError: max_df corresponds to < documents than min_df

    Does this mean the number of documents in new_doc is less than the parameter specified by max_df = 20? What have I missed? Thanks

+ Uwe F MayerMentor reply

    Allen, what’s the reason behind using vect.fit_transform(new_doc)? Shouldn’t you use vect.transform(new_doc)?


### [Lessons Learned For Assignment 4](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/2240ZdDpEeiwGg4oArjr5g)

+ Chion John Wong Init

    After spending a lot of time (26 hours) on Assignment 4 , here are the lessons learned. This is essentially a compilation of the various tips from several posts.

    In the lecture video, it spent a brief demonstration on the use of wn.synset(deer.n.01) (singular synset, no s at the end). Later in the LDAModel video, it mentioned a few times to perform text preprocessing prior to use LDA.

    For Part I, the keys are as follow:

    + Except for word tokenization, do not perform any other text preprocessing. Any attempt such as lower casing, removing stopwords, stemming, lemmatizing, will affect the result sets of synsets1 and synsets2
    + Perform pos_tagging and tag conversion
    + The instruction mentioned to use WordNet wn.synsets (plural). That does not mean applying wn.synset several times. There is an actual function called wn.synsets. If you mistakenly used wn,synset, you will discover that it will not work on input like are.v.01. Then you will go down the rabbit hole of lemmatizing the words before pos_tagging, to get (be.v.01). Furthermore, wn.synset does not work with None returned type.
    + Instead, use wn.synsets(word, pos) to return a list of matches on the tokens, including tokens that has None pos tag from convert_tag. Use the first match from the list.
    + In the Similarity score calculation, with the 2 synsets, it makes a difference on how to apply the path_similarity: For each item in synset1, use path_similarity on each item in synset2. That is, items1.path_similarity(items2). There is a 0.008 difference from the test_document_path_similarity if you use items2.path_similarity(items1), which is enough to fail the autograder
    + For label_accuracy, the existing Quality column is the y_test column. The new paraphrase labels (1,0) column is the y_Predict for the accuracy_score function

    For Part II:

    + Using passes=1 and passes=25 on the LdaModel will both satisfy the autograder. passes=1 will run much much faster than passes=25
    + lda_topics will create a fitted ldamodel. This already fitted ldamodel is used in topic_distribution. Use the countvectorizer to transform the new_doc into a new_X vector. Do not use fit_transform.
    + Create a new corpus with the new X
    + For the new corpus, apply LDA model get_document_topics (This is not covered in the video), convert the object into a list, and get the first item from the list as the answer.
    + For topic_names, even though the instruction said "If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.". The autograder will fail if you make up a lot of the topic names. Just pick from the list of 12 suggested topics names, and assign them to the 10 topics. Some of the topic names can be used more than once.

    HTH,

    John Wong


+ Joey Corea Reply

    Using passes=1 and passes=25 on the LdaModel will both satisfy the autograder. passes=1 will run much much faster than passes=25

    Please note that you will get different answers from get_document_topics() on the new_doc depending on whether you use passes=1 vs passes=25


### [Question about lemmatizing in Part1](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/BEGKnow8EeehOgoZX4bTHA)

+ Taehee Jeong reply

    I tried to lemmatizing as following.

    doc_token = nltk.word_tokenize(doc)

    WNlemma = nltk.WordNetLemmatizer()

    [WNlemma.lemmatize(t) for t in doc_token]

    the output is ['Fish', 'are', 'nvqjp', u'friend', '.'].

    'are' does not converted to 'be'.

    Please help me!

+ Sophie Greene reply

    you only need to apply nltk.pos_tag and nltk.word_tokenize and synsets after converting the tag to be wordnet compatible

    here are the steps

    1. apply nltk.pos_tag and nltk.word_tokenize, the result will be a list of tuples where the first element in the tuple is the token and the second elemnt is the tag
    2. for each tuple in the list above use wn.synsets on token and convert_tag(tag), check if the result is a list with length more than 0, get the first elemet, store all the first elements in a list 
    3. return the list

    here is the above applied to only one of the tuples
    <a href="https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/BEGKnow8EeehOgoZX4bTHA"> <br/>
        <img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/tjzE5Iy-EeebeBIyUknsWA_fa7b7d12c457231347079fccef102232_Screen-Shot-2017-08-29-at-14.33.20.png?expiry=1544227200000&hmac=gRuy_-PhSiTvoYPhRdGgPU5RFbCawUbT5KdinZoo0P8" alt="text" title="diagram for reslut" height="300">
    </a>

### [topic_distribution()](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/ONzFsYNZEeet4w4UpkEocg)

+ seunghoon lee init

    for solving this problem, the method(in ldamodel) below should be used. but that method required the bow(bag of word) as a input.

    ```python
    ldamodel.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None,    per_word_topics=False) 
    ```

    in problem statement, said as below,

    + Remember to use vect.transform on the the new doc,
    + and Sparse2Corpus to convert the sparse matrix to gensim corpus.

    so my code is below and my final data type is corpus. for using get_document, the corpus should be converted to bow, but i don't know how to do it.
    ```python
    new_X = vect.transform(new_doc)
        new_corpus = gensim.matutils.Sparse2Corpus(new_X, documents_columns=False)
    ```
    or are there other method to solve this problem?

+ seunghoon lee reply

    however, the parameter name in other method below is corpus

    ```python
    bound(corpus, gamma=None, subsample_ratio=1.0)
    top_topics(corpus, num_words=20)
    ```

    https://radimrehurek.com/gensim/models/ldamodel.html

    and moreover, it is just parameter name, I don't know why my code was not work.

    ```python
    def topic_distribution():
        new_X = vect.transform(new_doc)
        new_corpus = gensim.matutils.Sparse2Corpus(new_X, documents_columns=False)
        return ldamodel.get_document_topics(new_corpus)
    ```

    the output of that code is below.

    ```python
    <gensim.interfaces.TransformedCorpus at 0x7f5fedd87cc0>
    ```

+ seunghoon lee · a year ago

    Thanks for your lot of help. I should figure out the data structure in nltk, and gensim.

    That code is running but, the output is different from that I expected.

    the method ldamodel.get_document_topics(new_corpus) should return the "topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples." as it described in documentation( https://radimrehurek.com/gensim/models/ldamodel.html ).

    but it return other thing, corpus, <gensim.interfaces.TransformedCorpus at 0x7f5fedd87cc0>.

    because of that reason, the different output of the method, I thought that the other input type(BOW) should be taken in that method.

    Again, thanks for your lot of help.

+ Isabel Camilla Hutchison reply

I get the following output, but it won't get accepted by the AG:
```python
[[(0, 0.020000000317099309),
  (1, 0.020000000316238595),
  (2, 0.020000000316343886),
  (3, 0.020000000315819035),
  (4, 0.020000000317976295),
  (5, 0.020000000317598902),
  (6, 0.020000000316114326),
  (7, 0.8199999971463352),
  (8, 0.020000000318816692),
  (9, 0.020000000317657578)]]
```

Could someone please give me a hint as to why?

My strategy is:

1. transform the new_doc using vect
2. use Sparse2Corpus to define corpus based on transformed new_doc
3. generate ldamodel with corpus, same id_map as previous task, num_topics = 10, passes = 25, random_state = 34
4. finally return list(ldamodel2.get_document_topics(corpus2))

+ Philipp Stempel reply

    @Isabel:

    Two things:

    + My understanding of the question is that you use the same lda model that you trained before, i.e. you don't create a new model just based on that one sentence.
    + You need to return a list of tuples. You're returning a list of a list of tuples. Basically, you need to add [0] to the end of your return statement.


+ abdulkader hasan reply

    I got it,

    Thanks for sharing ideas, I also found this link interesting :

    https://stackoverflow.com/questions/31742630/gensim-lda-for-text-classification


### [PART 1. I don´t know what is wrong](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/H6vxWUFjEeidIQpMV-kQOg)

+ Silvia Init

I obtain 2.46 and I know that it is not correct.

this is my code in doc_to_synsets:

... code removed by moderator ...

In similarity_score(s1, s2)

... code removed by moderator ...

if(w1.path_similarity(w2)!=None):

... code removed by moderator ...

sum_maximos = sum(maximos)

max_maximos = max(maximos)

res = sum_maximos/max_maximos

return res

What can be wrong? I have spent 3 days trying to find the error :(


+ Philipp Vogt reply

    hi, i just have a small error in my answer of part I, i did the checkup for doc_to_synsets(doc) and have the example output, if i do the checkup for similarity_score(s1, s2) i get the output: 0.7333333333333334 instead of 0.73333333333333339.

    i my function i am creating a helping list and pick the maximum over i.path_similarity(a) for i in s2 if i.path_similarity(a) is not None

    then i calculate the similiartity score as sum/len

    any ideas what could be wrong?

    Thanks

+ Philipp Vogt reply

    Hi i figured out by my own now, but it took me some hours, i had to change the i.path_similarity(a) to a.path_similiarty(i) and instead of caclulation it in with one row with the maximum function, i had to put an if-function on top, like in this example:

    https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/JZSUJYQlEeeq-Q5IiJh3XA


+ Oscar Rene Chamberlain Pravia replt

    Please, I read all the discussion but I found the following with my code testing with different examples. I always found a small different result for similarity_score. Any suggestion?

    TEST (1)

    I don't like green eggs and ham, i do not, no love at all. Bacon and eggs with a dose of Iodine, I reallly love!

    [Synset('iodine.n.01'), Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01'), Synset('make.v.01'), Synset('not.r.01'), Synset('no.n.01'), Synset('love.n.01'), Synset('astatine.n.01'), Synset('all.a.01')]

    [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('angstrom.n.01'), Synset('dose.n.01'), Synset('iodine.n.01'), Synset('iodine.n.01'), Synset('love.v.01')]

    value = 0.42059483726150393 my result = 0.398779461279

    value = 0.555952380952381 my result = 0.555952380952

    --------------

    Test (2)

    ('This is a method to check mym_function', 'Your method might be a different one testing your_function')

    synsets1: [Synset('be.v.01'), Synset('angstrom.n.01'), Synset('method.n.01'), Synset('check.v.01')]

    synsets2: [Synset('method.n.01'), Synset('might.n.01'), Synset('be.v.01'), Synset('angstrom.n.01'), Synset('different.a.01'), Synset('one.n.01'), Synset('testing.n.01')]

    value = 0.8125 my result = similarity_score(synsets1, synsets2): 0.8125

    value = 0.5470899470899472 my result = similarity_score(synsets2, synsets1): 0.520124716553

    --------------

    Test (3)

    [Synset('be.v.01'), Synset('angstrom.n.01'), Synset('function.n.01'), Synset('test.v.01')]

    [Synset('use.v.01'), Synset('function.n.01'), Synset('see.v.01'), Synset('code.n.01'), Synset('inch.n.01'), Synset('be.v.01'), Synset('correct.a.01')]

    value = 0.6125 my result = 0.6125

    value = 0.4251700680272109 my result = 0.472789115646


+ Uwe F Mayer Reply

    Oscar, sorry to say, but there's nothing like a "small" difference. Either it's right or it's not. And you have a few that are not. I'd suggest working on Test (1). Quite a few learners make a mistake when handling synsets that have no match. Specifically, the instructions say: For each synset in s1, find the synset in s2 with the largest similarity value. What does your code do if for a given synset from s1 there is no synset in s2 with a similarity value? In that case you need to ignore that synset from s1, as the instruction say: Missing values should be ignored.

    Also, your Test (2) has a typo, you have mym_function, it should be my_function, not sure if that matters.

+ Silvia reply

    I cannot pass it. I obtain: 0.5426445578231293 in "test_document_path_similarity".

    My steps are:

    In doc_to_synsets:
    - Tokenize and pos

    - I convert pos into wordnet pos using convert_tag

    - I extract synsets and I choose the [0] synset in each list, if len of word.synsets(x,z) >0

    - I obtain a list of 4 synsets for doc1 and one of 7 synsets for doc2.

    2. In similarity_score:

    - I create a list where I will store the max values

    - for each synset in s1:

    - path_similarity with each synset in s2, if is not None, we select the maximum and append it to the list

    I obtain 0.5426445578231293

    I don´t have idea what can be wrong.

+ Uwe F MayerMentor reply

    Sisyphus, glad to read you figured it out!

    For others reading this post, Sisyphus's result above were incorrect for similarity_score(synsets2,synsets1). Find my post earlier in this thread for the expected answers of that test case.

+ Miranda Lam reply

    I tried both tests that Uwe suggested and got exactly the same results but failed the autograder.

    doc1="I don't like green eggs and ham"

    doc2="Bacon and eggs with a dose of Iodine"
    ```python
    [Synset('iodine.n.01'), Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01')]
    [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('angstrom.n.01'), Synset('dose.n.01'), Synset('iodine.n.01')]
    0.5202380952380953
    0.5116666666666667
    0.5159523809523809
    ```

    (doc1, doc2) = ('This is a method to check my_function', 'Your method might be a different one testing your_function')
    ```python
    ('This is a method to check my_function', 'Your method might be a different one testing your_function')
    synsets1: [Synset('be.v.01'), Synset('angstrom.n.01'), Synset('method.n.01'), Synset('check.v.01')]
    synsets2: [Synset('method.n.01'), Synset('might.n.01'), Synset('be.v.01'), Synset('angstrom.n.01'), Synset('different.a.01'), Synset('one.n.01'), Synset('testing.n.01')]
    similarity_score(synsets1, synsets2): 0.8125
    similarity_score(synsets2, synsets1): 0.5470899470899472
    document_path_similarity(doc1, doc2) 0.6797949735449735
    ```

    I checked for empty/None returns from path_similarity; potential empty array for and potential zero count. I get the same answers as previous posts on all the test documents. Please help.


+ Uwe F Mayer reply

    Miranda, there are obviously two scenarios, either you have the wrong answer, or you have the correct answer but the grader does not accept it. Let's try to figure out which one it is. Here is some code:
    ```python
    # test the learner's solution
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    print("synsets1", synsets1) # a list with 4 elements
    print("synsets2", synsets2) # a list with 7 elements
    s1s2_score = similarity_score(synsets1, synsets2)
    s2s1_score = similarity_score(synsets2, synsets1)
    print("s1s2_score", s1s2_score) # 0.6?2?0?0?0
    print("s2s1_score", s2s1_score) # 0.4?6?3?7?6
    print("s1 s2 doc similarity score", (s1s2_score + s2s1_score) / 2) # 0.5?4?6?8?3
    ```

    If your results match the results in the comments above then likely you have the correct answer, and it appears to be something that keeps the grader from accepting it. Typical causes for failure are printing, plotting, importing matplotlib, changing the data after the question, syntax errors. Run your notebook via Kernel -> Restart and run all, and then at the end add a new cell with only test_document_path_similarity(), does it produce the same answer?

+ Miranda Lam reply

    Thank you for the quick response. I think I've narrowed down the problem. It has to do with whether covert_tag returns None. In my original code, I use 'try', which produces very close results but returns a list of 4 elements for doc1 and 7 elements for doc2 ('in' is included). When I change my code to use an if statement to exclude cases when covert_tag returns None, I get 3 elements for doc1 ('a' was dropped) and 6 elements for doc2. I think it is because wn.synsets takes None as an argument for pos. Therefore I think 'try' is better than 'if'. Below are my results.

    I have also tried removing stopwords first but that seem to mess up the pos_tag function, which makes sense. Are we supposed to get pos using pos_tag, then remove stopwords, then rematch the pos? Doing so will require iterating over the entire text multiple times.

    Thanks again for your help. Can you give me hints to what I can try next?
    ```python
    # use 'try' resulting in 7 elements in doc2synsets1 [Synset('be.v.01'), Synset('angstrom.n.01'), ...]synsets2 [Synset('use.n.01'), Synset('function.n.01'), ...)]
    s1s2_score 0.6?2?0?0
    s2s1_score 0.4?8?9?7
    s1 s2 doc similarity score 0.5?5?4?3

    # use 'if' resulting in 3 elements in doc1synsets1 [Synset('be.v.01'), Synset('function.n.01'), Synset(...)]synsets2 [Synset('use.n.01'), Synset('function.n.01'), ...]
    s1s2_score 0.7?3?3?3
    s2s1_score 0.5?0?9?6
    s1 s2 doc similarity score 0.6?7?6?4
    ```

    [This post was edited, please don't provide too many details of your solution to stay within the Honor Code.]

+ Uwe F Mayer reply

    Miranda, while I personally used an "if" method to check for None cases to excluded, for you it appears the "try" method indeed is better. Both of your solutions suffer from the same error, you have "Synset('use.n.01')", while if you look at the sentence being parsed "use" is clearly a verb and not a noun, and indeed it should be "Synset('use.v.01')". So that's where you need to look.

    And it should be 7, not 6, for doc2, I had a typo in my earlier reply (now corrected). Sorry about that.

    Note that testing for None is done with "s is None", something like "s == None" does not work. This problem also involves two tests for None in a wider sense, one is for testing if a token is None, the other is to test if a list is empty (and that is done with "s == []").

+ Miranda Lam reply

    Thank you so much. I think I found the problem. I use .lower() to normalize the original text, which results in pos_tag returning 'NN' (noun). Removing that normalization seems to work. I will edit out more contents in the future.

+ Uwe F Mayer reply

    Miranda, no harm done, what exactly is too much detail to post is a question of judgement. And yes, you shouldn’t use lower() nor should stop words be removed.

    Please post a quick note when you get it to pass, or if needed, ask more.

+ Manuel Martinez reply

    Hi Uwe, Miranda,

    I just wanted to butt into the conversation.

    I had almost the exact same issue, I was getting 36/100 score from the auto-grader. The whole of Part 1 was being scored incorrectly, even though the whole of Part 2 was being scored correctly. I tried all the debugging snippets that Uwe posted, and I was getting them all exactly right - even though the auto-grader was scoring my submission as incorrect.

    I realized that I also had used the .lower() method on the get_doc_synsets function. I tried a submission without the method, and voila, 100/100!

    Thanks, and I hope this is of help to other victims of the incredibly hermetic auto-grader.

+ Tse Ching Yan reply

    My answer is not correct since "similarity_score(synsets2, synsets1)"

    , with my similarity_score like this,

    + make an empty list
    + loop for every word in s1, set sim = 0
    + loop for every word in s2, find s1.path_similarity(s2)
    + if s1.path_similarity(s2) is not none and value > sim, then update sim
    + append sim to the list
    + then return list total / list length

    Can anyone tell me what's wrong in my code? Thanks

+ Uwe F Mayer reply

    Tse, that logic looks correct for similarity_score(s1, s2), in your post you write about similarity_score(synsets2, synsets1), not sure what that has to do with it.

    List total presumably means sum of the entries in the list, right?

    Also, what does your function return if list length is zero?

+ Uwe F Mayer reply

    Silvia, please do not post a solution to an assignment, even an incorrect one, this violates the honor code. I have edited your post correspondingly. Note that I left a few lines of your code, those are problematic. First, this is not how you test for None. Second, the similarity score is not defined as the sum divided by the maximum. Also, the way you had posted your code I could not tell how the code was supposed to be indented, and I am guessing there was an error in that as well.

    I highly recommend you insert print statements into your function for debugging so you see if your loops are actually doing what you think they should be doing.

    Good luck hunting this down, it's actually fairly close.

### [topic distribution - tips from this forum](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/Ci32uma4EeilEg6NrP0J4g)

Sergio CruzWeek 4 · 6 months ago · Edited
Hello

Topic distribution was a complicated question for me, as I did some assumptions which were wrong, and I got the clarification from different people in this forum, so I share some ideas which were relevant to me:

1) As the question said, use vect.transform and not vect.fit_transform. This is because the fit was already done for the previous question.

2) There is not need to generate a new LDA model. So just use the same one of the previous question.

3) There is a problem with the versions of gensim. The autograder works in version 2.4.1 but current Junyper notebooks in 3.4.0. You can check that via the following command: gensim.__version__

4) As some generous people already suggested before in the threads available in this forum just use the following line at the end of the function:

return list(ldamodel.get_document_topics(corpus2)) [0]

5) get_document_topics is described here: https://radimrehurek.com/gensim/models/ldamodel.html

Source of several of this ideas: https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/ONzFsYNZEeet4w4UpkEocg

Good luck !

### [Hint for topic_names](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/x1DSkZ44EeiclQpjXTIwdA)

Jun Wang

From the list of the following given topics, assign topic names to the topics you found. If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.

Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, Society & Lifestyle, Religion, Education.

This function should return a list of 10 strings.

Of course, we can manually select the topics, as suggested by Jo Are By.

With that method, I got 100%. But I still fell uncomfortable, that's not it is suppose to be.

So I tried another method, a more intuitively right one. And it works.

Here it is.

1. In lda_topics, we've already got the most significant 10 words in each topic like this:
(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')

2. We also defined 'document_path_similarity(doc1, doc2)' at the beginning.

3. Put them together. We can find the similarity between the 10 words in each topic and the 12 candidate topic names, i.e. ["Health", 'Science', 'Automobiles', 'Politics', 'Government', 'Travel', 'Computers & IT', 'Sports', 'Business', 'Society & Lifestyle', 'Religion', 'Education']. And name each topic with the candidate topic name that has the highest score of similarity.

For example, if similarity between (9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"') and "Science" is 0.6 higher than 'Health' and others. We can name the topic 9 "Science".


### [doc_to_synsets does not work on every sentence?!](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/xYatmprrEeiqGxIKTTFnwg)

+ Pieter-Jan init

    Hi,

    My doc_to_synsets function seems to work. However when testing it on different sentences, it gives some erros.

    It works on:

    doc = 'This example with words like , computer, human being, , mother and father.'

    doc2 = 'This is an extensive example with words'

    It does not work on:

    Some characters (\) --> doc3 = 'Use this function to see if your code in doc_to_synsets \ and similarity_score is correct!'
    Error seems to be given at the \, because as I move it, the same error occurs earlier in the sentence at the place of the \

    Spelling mistakes --> doc4 = 'Bacon and eggs! with a dose of Iodine, I reallly love'
    Error seems to be at reallly with 3 l's. If I spell it with 2 l's, it works just fine.
    ```python
    ('Use', 'VB')
    ('this', 'DT')
    ('function', 'NN')
    ('to', 'TO')
    ('see', 'VB')
    ('if', 'IN')
    ('your', 'PRP$')
    ('code', 'NN')
    ('in', 'IN')
    ('doc_to_synsets', 'NNS')
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    <ipython-input-32-aa7dc7b63a21> in <module>()
    ----> 1 doc_to_synsets(doc3)

    <ipython-input-31-4b08507c9127> in doc_to_synsets(doc)
        48 
        49         lemma = lemmatzr.lemmatize(token[0], pos=wn_tag)
    ---> 50         synsets.append(wn.synsets(lemma, pos=wn_tag)[0])
        51 
        52     return synsets

    IndexError: list index out of range
    ```

    Any idea how to fix this?

+ Uwe F Mayer reply

    Well, yes. The error is that the index [0] is out of range, and that happens when wn.synsets(lemma, pos=wn_tag) does not even have a single element in it. Your code needs to handle that case.

### [part 1 issue](https://www.coursera.org/learn/python-text-mining/discussions/weeks/4/threads/Jj2-sH6JEeiSqgoLbkGR2g)

+ Syeda Farheen Naz init

    I am getting below as output for
    ```python
    doc_to_synsets('Fish are nvqjp friends.')
    output: [Synset('fish.n.01'), Synset('are.n.01'), Synset('friend.n.01')]
    ```

    instead of given output which is
    ```python
    [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    ```

+ Uwe F Mayer reply

    Syeda, a more direct answer: no, that is not OK. Your output must match exactly what is given. Please look through the discussion forum, I have posted more examples for you to use to test your code.

+ YW reply

    Hi Syeda, looking at your input, I think the problem is that the code currently treats "are" as a noun instead of a verb. Also, the right output also requires the verb to be displayed in its simplest form (are - be).

    Could you tell me a little bit more on how you write your doc_to_synsets function? (Without showing any code) I may be able to assist better if I have more details.

+ Syeda Farheen Naz reply

    I have tokenized the documnet. then tagged tokens using nltk.pos_tag() and then converted these tags using convert_tag(). Then I have passed token and converted tags to wn.synsets() and I am keeping synset at index 0 and appending it to final list which is output.

+ Uwe F Mayer reply

    Syeda, some other learner ran into similar problems, and in his case it turned out that he transformed the document into lower case during tokenization. This should not be done. Neither should lemmatization be performed. Did you do anything like that?

+ Uwe F Mayer reply

    Hmm, maybe your lookup of the synsets with wordnet isn't quite right. For this example before the synsets lookup the token should indeed be ('are', 'v'), but that should look up to Synset('be.v.01'). You may want to add a bunch of print statements and check each intermediate result.

    After all that you also might want to work through the other examples in the thread "PART 1. I don´t know what is wrong".

+ Uwe F Mayer reply

    Hmm, maybe your lookup of the synsets with wordnet isn't quite right. For this example before the synsets lookup the token should indeed be ('are', 'v'), but that should look up to Synset('be.v.01'). You may want to add a bunch of print statements and check each intermediate result.

    After all that you also might want to work through the other examples in the thread "PART 1. I don´t know what is wrong".

+ Syeda Farheen Naz reply

    Yes all of them till this one:
    ```python
    (doc1, doc2) = ('This is a method to check mym_function', 'Your method might be a different one testing your_function')
    print((doc1, doc2))
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    print('synsets1:', synsets1)
    print('synsets2:', synsets2)
    print('similarity_score(synsets1, synsets2):', similarity_score(synsets1, synsets2))
    print('similarity_score(synsets2, synsets1):', similarity_score(synsets2, synsets1))
    print('document_path_similarity(doc1, doc2)', document_path_similarity(doc1, doc2))

    ('This is a method to check my_function', 'Your method might be a different one testing your_function')
    synsets1: [Synset('be.v.01'), Synset('angstrom.n.01'), Synset('method.n.01'), Synset('check.v.01')]
    synsets2: [Synset('method.n.01'), Synset('might.n.01'), Synset('be.v.01'), Synset('angstrom.n.01'), Synset('different.a.01'), Synset('one.n.01'), Synset('testing.n.01')]
    similarity_score(synsets1, synsets2): 0.8125
    similarity_score(synsets2, synsets1): 0.5470899470899472
    document_path_similarity(doc1, doc2) 0.6797949735449735
    ```

above is my output which is exactly same as yours.

+ Syeda Farheen Naz reply

    I just came across your old post:

    Raul, pos_tag should return ('I', 'PRP') for the word 'I', just as you say, then convert_tag should transform this into ('I', None), but that is not the same as discarding the word. Then you feed ('I', None) into wn.synsets and you get iodine as the first synset.

    Presumably you are dropping the terms with None returned by convert_tag, that's not how it's expected to be done.

    I am passing only tag to convert_tag(), I hope this isn't causing any issue since I am getting correct outputs to all your test codes.

+ Uwe F Mayer reply

    Syeda, convert_tag() should just be passed the tag, that's correct. The question is what your code does if convert_tag() returns None. Do you keep that record and pass it to wn.synsets() or do you discard it? Presumably you are passing it on, as you should. Other learners have had errors when wn.synsets() returns an empty list, those need to be dropped. Also you need to use the tokenizer on the documents as they are, no stemming or lower-casing or other cleanups should be done. Here's another example:
    ```python
    1="I don't like green eggs and ham, i do not, no love at all."
    doc2="Bacon and eggs with a dose of Iodine, I reallly love!"
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    print(synsets1)
    print(synsets2)
    print(similarity_score(synsets1, synsets2))
    print(similarity_score(synsets2, synsets1))
    print(document_path_similarity(doc1, doc2))

    [Synset('iodine.n.01'), Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01'), Synset('make.v.01'), Synset('not.r.01'), Synset('no.n.01'), Synset('love.n.01'), Synset('astatine.n.01'), Synset('all.a.01')]
    [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('angstrom.n.01'), Synset('dose.n.01'), Synset('iodine.n.01'), Synset('iodine.n.01'), Synset('love.v.01')]
    0.42059483726150393
    0.555952380952381
    0.48827360910694245
    ```

+ Syeda Farheen Naz reply

    This is my output to above test code.
    ```python
    [Synset('iodine.n.01'), Synset('make.v.01'), Synset('wish.v.02'), Synset('green.s.01'), Synset('egg.n.02'), Synset('ham.n.01'), Synset('make.v.01'), Synset('not.r.01'), Synset('no.n.01'), Synset('love.n.01'), Synset('astatine.n.01'), Synset('all.a.01')]
    [Synset('bacon.n.01'), Synset('egg.n.02'), Synset('angstrom.n.01'), Synset('dose.n.01'), Synset('iodine.n.01'), Synset('iodine.n.01'), Synset('love.v.01')]
    0.42059483726150393
    0.555952380952381
    0.48827360910694245
    ```

+ Uwe F Mayer reply

    Syeda, Part 1 has several parts. Do you get 0 for all of them?

    I am asking because sometimes learners mess up somewhere else in the notebook and the autograder cannot run the notebook because of that (for example, have a cell at the end with a syntax error or print statement or matplotlib code). The autograder first runs the entire notebook, and if it cannot do that, then it says "failed to load student data file". Is this maybe what's going on? Or do you get credit for some of the parts?

+ Syeda Farheen Naz reply

    After removing few function calls and an extra cell added by me, I am now getting credit for my correct parts of code. It took me a month to fix this issue. Thanks a lot.



## Solution


