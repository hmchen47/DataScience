# Module 2: Basic Natural Language Processing


## Basic Natural Language Processing

### Lecture Notes

+ What is Natural Language?
    + Language used for everyday communication by humans
        + English
        + 中⽂文
        + ру́сский язы́к
        + español
    + compared to the artificial computer languages

+ What is Natural Language Processing?
    + Any computation, manipulation of natural language
    + Natural languages evolve
        + new words get added; e.g. selfie
        + old words lose popularity; e.g. thou
        + meanings of words change; e.g. learn
        + language rules themselves may change; e.g. position of verbs in sentences!

+ NLP Tasks: A Broad Spectrum
    + Counting words, counting frequency of words
    + Finding sentence boundaries
    + Part of speech tagging
    + Parsing the sentence structure
    + Identifying semantic roles
    + Identifying entities in a sentence
    + Finding which pronoun refers to which entity
    + and much more ...

    <a href="url"> <br/>
        <img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/Xqf0hGgFEeedjgoGzm8emA.processed/full/360p/index.mp4?Expires=1543017600&Signature=RxENKsaVj2xuLVE01tK8v1X5LaEQW-gAjoElMb6pXwXxj4c-R~G5tJTIkYvvSykIo4bW5Wtd3HFpNu3YcC4904vrMvtliufY4lRKY-5803JV~sG1dZJrGCZtQXRE3VJyq18FggYDYlRQQpXSp6iWq51TVqzQvjopJVAzjOxpCDI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Basic Natural Language Processing" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Notebook: Module 2 (Python 3)

### Lecture Notes

+ [Launching web Page](https://www.coursera.org/learn/python-text-mining/notebook/NcOOH/module-2-python-3)
+ [Notebook Web page](https://hub.coursera-notebooks.org/user/dfxbyieeexzfjsmxjreyig/notebooks/Module%202%20(Python%203).ipynb)
+ [Local Notebook](notebooks/02-Module+2+Python.3.ipynb)
+ [Local Python code](notebooks/02-Module+2+Python.3.py)


## Basic NLP tasks with NLTK

### Lecture Notes

+ An Introduction to NLTK
    + NLTK: Natural Language Toolkit
    + Open source library in Python
    + Has support for most NLP tasks
    + Also provides access to numerous text corpora

+ Demo
    ```Python
    import nltk

    nltk.download()     # download the nltk collection, only the first time

    from nltk.book import *
    # showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
    # *** Introductory Examples for the NLTK Book ***
    # Loading text1, ..., text9 and sent1, ..., sent9
    # Type the name of the text or sentence to view it.
    # Type: 'texts()' or 'sents()' to list the materials.
    # text1: Moby Dick by Herman Melville 1851
    # text2: Sense and Sensibility by Jane Austen 1811
    # text3: The Book of Genesis
    # text4: Inaugural Address Corpus
    # text5: Chat Corpus
    # text6: Monty Python and the Holy Grail
    # text7: Wall Street Journal
    # text8: Personals Corpus
    # text9: The Man Who Was Thursday by G . K . Chesterton 1908

    # ### Counting vocabulary of words
    text7       # <Text: Wall Street Journal>
    sents()
    # sent1: Call me Ishmael .
    # sent2: The family of Dashwood had long been settled in Sussex .
    # sent3: In the beginning God created the heaven and the earth .
    # sent4: Fellow - Citizens of the Senate and of the House of Representatives :
    # sent5: I have a problem with people PMing me to lol JOIN
    # sent6: SCENE 1 : [ wind ] [ clop clop clop ] KING ARTHUR : Whoa there !
    # sent7: Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .
    # sent8: 25 SEXY MALE , seeks attrac older single lady , for discreet encounters .
    # sent9: THE suburb of Saffron Park lay on the sunset side of London , as red and ragged as a cloud of sunset .
    sent7
    # ['Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board',
    #  'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.']

    len(sent7)              # 18
    len(text7)              # 100676
    len(set(text7))         # 12408
    list(set(text7))[:10]   
    # ['Sebastian', 'Midland', 'sounding', '2.75', 'Sotheby', 'youngsters', 'B-1B',
    #  'pick-up', '43-year-old', 'publicized']

    # ### Frequency of words
    dist = FreqDist(text7)
    len(dist)               # 12408

    vocab1 = dist.keys()
    # vocab1[:10]
    # In Python 3 dict.keys() returns an iterable view instead of a list
    list(vocab1)[:10]
    # ['Pierre', 'Vinken', ',', '61', 'years', 'old', 'will', 'join', 'the', 'board']

    dist['four']    # 20

    freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
    # ['billion', 'company', 'president', 'because', 'market', 'million', 'shares', 'trading', 'program']

    # ### Normalization and stemming
    input1 = "List listed lists listing listings"
    words1 = input1.lower().split(' ')
    # ['list', 'listed', 'lists', 'listing', 'listings']

    porter = nltk.PorterStemmer()
    [porter.stem(t) for t in words1]
    # ['list', 'list', 'list', 'list', 'list']

    # ### Lemmatization
    udhr = nltk.corpus.udhr.words('English-Latin1')
    udhr[:20] # ['list', 'listed', 'lists', 'listing', 'listings']

    [porter.stem(t) for t in udhr[:20]] # Still Lemmatization
    # ['univers', 'declar', 'of', 'human', 'right', 'preambl', 'wherea', 'recognit', 'of', 
    # 'the', 'inher', 'digniti', 'and', 'of', 'the', 'equal', 'and', 'inalien', 'right', 'of']

    # Lemmatization: Stemming, but resulting stems are all valid words
    WNlemma = nltk.WordNetLemmatizer()
    [WNlemma.lemmatize(t) for t in udhr[:20]]
    # ['Universal', 'Declaration', 'of', 'Human', 'Rights', 'Preamble', 'Whereas', 'recognition',
    #  'of', 'the', 'inherent', 'dignity', 'and', 'of', 'the', 'equal', 'and', 'inalienable', 'right', 'of']

    # ### Tokenization
    # Recall splitting a sentence into words / tokens
    text11 = "Children shouldn't drink a sugary drink before bed."
    text11.split(' ')
    # ['Children', "shouldn't", 'drink', 'a', 'sugary', 'drink', 'before', 'bed.']

    # NLTK has an in-built tokenizer
    nltk.word_tokenize(text11)
    # ['Children', 'should', "n't", 'drink', 'a', 'sugary', 'drink', 'before', 'bed', '.']

    # Sentence Splitting
    # How would you split sentences from a long text string?
    text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
    # NLTK has an in-built sentence splitter too!
    sentences = nltk.sent_tokenize(text12)
    len(sentences)      # 4
    sentences
    # ['This is the first sentence.',
    #  'A gallon of milk in the U.S. costs $2.99.',
    #  'Is this the third sentence?',
    #  'Yes, it is!']
    ```

+ Take Home Concepts
    + NLTK is a widely used toolkit for text and natural language processing
    + NLTK gives access to many corpora and handy tools
    + Sentence splitting, tokenization, and lemmatization are important, and non-trivial, pre-processing tasks

+ `FreqDist` function
    + Init signature: `FreqDist(samples=None)`
    + Docstring: A frequency distribution for the outcomes of an experiment.  
    + Notes:
        + A frequency distribution records the number of times each outcome of an experiment has occurred.  For example, a frequency distribution could be used to record the frequency of each word type in a document.  Formally, a frequency distribution can be defined as a function mapping from each sample to the number of times that sample occurred as an outcome.
        + Frequency distributions are generally constructed by running a number of experiments, and incrementing the count for a sample every time it is an outcome of an experiment.  For example, the following code will produce a frequency distribution that encodes how often each word occurs in a text:
        ```python
        >>> from nltk.tokenize import word_tokenize
        >>> from nltk.probability import FreqDist
        >>> sent = 'This is an example sentence'
        >>> fdist = FreqDist()
        >>> for word in word_tokenize(sent):
        ...    fdist[word.lower()] += 1
        >>> fdist = FreqDist(word.lower() for word in word_tokenize(sent)) # equivalent
        ```
    + Init docstring: Construct a new frequency distribution.  If `samples` is given, then the frequency distribution will be initialized with the count of each object in `samples`; otherwise, it will be initialized to be empty. <br/>
    In particular, `FreqDist()` returns an empty frequency distribution; and `FreqDist(samples)` first creates an empty frequency distribution, and then calls `update` with the list `samples`.
    + Parameter
        + samples (Sequence): The samples to initialize the frequency distribution with.

+ `nltk.corpus` class
    + Class docstring: Lazy module class.
    + Init docstring: Create a LazyModule instance wrapping module name.
    + Docstring: NLTK corpus readers.  The modules in this package provide functions that can be used to read corpus files in a variety of formats.  These functions can be used to read both the corpus files that are distributed in the NLTK corpus package, and corpus files that are part of external corpora.
    + Available Corpora: see http://www.nltk.org/nltk_data/ for a complete list. Install corpora using nltk.download().
    + Corpus Reader Functions
        + Each corpus module defines one or more "corpus reader functions", which can be used to read documents from that corpus.  These functions take an argument, `item`, which is used to indicate which document should be read from the corpus:
            + If `item` is one of the unique identifiers listed in the corpus module's `items` variable, then the corresponding document will be loaded from the NLTK corpus package.
            + If `item` is a filename, then that file will be read.
        + Additionally, corpus reader functions can be given lists of item names; in which case, they will return a concatenation of the corresponding documents.
        + Corpus reader functions are named based on the type of information they return.  Some common examples, and their return types, are:
            + `words()`: list of str
            + `sents()`: list of (list of str)
            + `paras()`: list of (list of (list of str))
            + `tagged_words()`: list of (str,str) tuple
            + `tagged_sents()`: list of (list of (str,str))
            + `tagged_paras()`: list of (list of (list of (str,str)))
            + `chunked_sents()`: list of (Tree w/ (str,str) leaves)
            + `parsed_sents()`: list of (Tree with str leaves)
            + `parsed_paras()`: list of (list of (Tree with str leaves))
            + `xml()`: A single xml ElementTree
            + `raw()`: unprocessed corpus contents
    + Note: Lazy modules are imported into the given namespaces whenever a non-special attribute (there are some attributes like `__doc__` that class instances handle without calling `__getattr__`) is requested. The module is then registered under the given name in locals usually replacing the import wrapper instance. The import itself is done using globals as global namespace.

+ `nltk.corpus.udhr` class
    + Class docstring: Reader for corpora that consist of plaintext documents.  Paragraphs are assumed to be split using blank lines.  Sentences and words can be tokenized using the default tokenizers, or by custom tokenizers specified as parameters to the constructor. <br/> This corpus reader can be customized (e.g., to skip preface sections of specific document formats) by creating a subclass and overriding the `CorpusView` class variable.
    + Init docstring: Construct a new plaintext corpus reader for a set of documents located at the given root directory.  Example usage:
        ```python
        >>> root = '/usr/local/share/nltk_data/corpora/webtext/'
        >>> reader = PlaintextCorpusReader(root, '.*\.txt') # doctest: +SKIP
        ```
    + Parameters:
        + `root`: The root directory for this corpus.
        + `fileids`: A list or regexp specifying the fileids in this corpus.
        + `word_tokenizer`: Tokenizer for breaking sentences or paragraphs into words.
        + `sent_tokenizer`: Tokenizer for breaking paragraphs into words.
        + `para_block_reader`: The block reader used to divide the corpus into paragraph blocks.

+ `nltk.corpus.udhr.words` method
    + Signature: `nltk.corpus.udhr.words(fileids=None)`
    + Docstring: return the given file(s) as a list of words and punctuation symbols.
    + Return: list(str)

+ `nltk.PorterStemmer` class
    + Init signature: `nltk.PorterStemmer(mode='NLTK_EXTENSIONS')`
    + Docstring: A word stemmer based on the Porter stemming algorithm.
    Notes:
        + Porter, M. "An algorithm for suffix stripping." Program 14.3 (1980): 130-137.
        + See http://www.tartarus.org/~martin/PorterStemmer/ for the homepage of the algorithm.
        + Martin Porter has endorsed several modifications to the Porter algorithm since writing his original paper, and those extensions are included in the implementations on his website. Additionally, others have proposed further improvements to the algorithm, including NLTK contributors. There are thus three modes that can be selected by passing the appropriate constant to the class constructor's `mode` attribute
        + For the best stemming, you should use the default NLTK_EXTENSIONS version. However, if you need to get the same results as either the original algorithm or one of Martin Porter's hosted versions for compability with an existing implementation or dataset, you can use one of the other modes instead.
    + Versions:
        + PorterStemmer.ORIGINAL_ALGORITHM: Implementation that is faithful to the original paper. <br/> Note that Martin Porter has deprecated this version of the algorithm. Martin distributes [implementations of the Porter Stemmer](http://www.tartarus.org/~martin/PorterStemmer/) in many languages and all of these implementations include his extensions. He strongly recommends against using the original, published version of the algorithm; only use this mode if you clearly understand why you are choosing to do so.
        + PorterStemmer.MARTIN_EXTENSIONS: Implementation that only uses the modifications to the algorithm that are included in the implementations on Martin Porter's website. He has declared Porter frozen, so the behaviour of those implementations should never change.
        + PorterStemmer.NLTK_EXTENSIONS (default): Implementation that includes further improvements devised by NLTK contributors or taken from other modified implementations found on the web.

+ `nltk.PorterStemmer.stem` method
    + Signature: `nltk.PorterStemmer.stem(self, word)`
    + Docstring: Strip affixes from the token and return the stem.
    + Parameter:
        + `token`: The token that should be stemmed.
    + Return: `token` (str)

+ `nltk.WordNetLemmatizer` class
    + Init signature: `nltk.WordNetLemmatizer()`
    + Docstring: Lemmatize using WordNet's built-in morphy function.
    + Returns the input word unchanged if it cannot be found in WordNet.

+ `wnl.lemmatize` method
    + Signature: `nltk.WordNetLemmatizer.lemmatize(word, pos='n')`
    + Docstring: Lemmatize using WordNet's built-in morphy function

+ `nlkt.word_tokenize` method
    + Signature: `nltk.word_tokenize(text, language='english', preserve_line=False)`
    + Docstring: Return a tokenized copy of text, using NLTK's recommended word tokenizer (currently an improved `.TreebankWordTokenizer` along with `.PunktSentenceTokenizer` for the specified language).
    + Parameters:
        + `text` (str): text to split into words
        + `language` (str): the model name in the Punkt corpus
        + `preserve_line` (bool): An option to keep the preserve the sentence and not sentence tokenize it.

+ `nltk.sent_tokenize` method
    + Signature: `nltk.sent_tokenize(text, language='english')`
    + Docstring: Return a sentence-tokenized copy of text, using NLTK's recommended sentence tokenizer (currently `.PunktSentenceTokenizer` for the specified language).
    + Parameters
        + `text` (str): text to split into sentences
        + `language` (str): the model name in the Punkt corpus



### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/PQRh5GbEEeeSBw5DxGzUwg.processed/full/360p/index.mp4?Expires=1543017600&Signature=GHLosr6GN~iIlD9jFVpqMQPa92VG7wsmTQ0M47l48v3IIvFeoDaY1JK5kgrdUc2En6uBMaXmsdzOPJ3xyGdw4KuISbXx2X9nkNNMhe5YuyO3Uu8dDm~jruwyph04DNsk8Clh7P4yAv~cuA3OZLro19Hy846K1OjU3LaCjzs3WYA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Basic NLP tasks with NLTK" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Advanced NLP tasks with NLTK

### Lecture Notes

+ Demo
    ```Python

    ```

    <a href="url"> <br/>
        <img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Practice Quiz: Practice Quiz

### Lecture Notes

+ Demo
    ```Python

    ```

    <a href="url"> <br/>
        <img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Discussion Prompt: Finding your own prepositional phrase attachment

### Lecture Notes

+ Demo
    ```Python

    ```

    <a href="url"> <br/>
        <img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Quiz: Module 2 Quiz

### Lecture Notes

+ Demo
    ```Python

    ```

    <a href="url"> <br/>
        <img src="url" alt="text" title= "caption" height="200">
    </a>

### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



