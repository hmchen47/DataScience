# Module 4: Topic Modeling

## Semantic Text Similarity

### Lecture Notes

+ Which pair of words are most similar?
    + deer, elk
    + deer, giraffe
    + deer, horse
    + deer, mouse
    + Ans: deer, elk
    + How can we quantify such similarity?

+ Applications of Text Similarity
    + Grouping similar words into semantic concepts
    + As a building block in natural language understanding tasks
        + Textual entailment: the smaller sentence or one of the two sentences derives its meaning or entails its meaning from another piece of text.
        + Paraphrasing: a task where you rephrase or rewrite some sentence you get into another sentence that has the same meaning.

+ WordNet
    + Semantic dictionary of (mostly) English words, interlinked by semantic relations
    + Includes rich linguistic information
    + part of speech, word senses, synonyms, hypernyms/hyponyms, meronyms, distributional related forms, …
    + Machine-readable, freely available

+ Semantic Similarity Using WordNet
    + WordNet organizes information in a hierarchy
    + Many similarity measures use the hierarchy in some way
    + Verbs, nouns, adjectives all have separate hierarchies

+ Coming back to our deer example
    <a href="url"> <br/>
        <img src="images/p4-01.png" alt="text" title="Semantic similarity with Hierarchy" height="200">
    </a>

+ Path Similarity
    + Find the shortest path between the two concepts
    + Similarity measure inversely related to path distance
    + `PathSim(deer, elk) = 0.5` (1 step)
    + `PathSim(deer, giraffe) = 0.33` (2 steps)
    + `PathSim(deer, horse) = 0.14` (6 steps)

+ Lowest Common Subsumer (LCS)
    + Find the closest ancestor to both concepts
    + `LCS(deer, elk) = deer`
    + `LCS(deer, giraffe) = ruminant`
    + `LCS(deer, horse) = ungulate`

+ Lin Similarity
    + Similarity measure based on the information contained in the LCS of the two concepts
    + $LinSim(u, v) = 2 \times \log P(LCS(u,v)) / (\log P(u) + \log P(v))$
    + `P(u)` is given by the information content learnt over a large corpus.

+ How to do it in Python?
    + WordNet easily imported into Python through NLTK
        ```python
        import nltk
        from nltk.corpus import wordnet as wn
        ```
    + Find appropriate sense of the words
        ```python
        deer = wn.synset('deer.n.01')   # deer as noun w/ the first meaning
        elk = wn.synset('elk.n.01')     # elk as noun w/ the first meaning
        …
        ```
    + Find path similarity
        ```python
        deer.path_similarity(elk)       # 0.5
        deer.path_similarity(horse)     # 0.14285714285714285
        ```
    + Use an information criteria to find Lin similarity
        ```python
        from nltk.corpus import wordnet_ic

        brown_ic = wordnet_ic.ic('ic-brown.dat')

        deer.lin_similarity(elk, brown_ic)      # 0.7726998936065773
        deer.lin_similarity(horse, brown_ic)    # 0.8623778273893673
        ```


+ `wordnet` class
    + Init Signature: `wordnet(root, omw_reader)`
    + Docstring: A corpus reader used to access wordnet or its variants.
    + Methods:
        + `__init__(root, omw_reader)`: Construct a new wordnet corpus reader, with the given root directory.
        + `all_lemma_names(pos=None, lang='eng')`: Return all lemma names for all synsets for the given part of speech tag and language or languages. If pos is not specified, all synsets for all parts of speech will be used.
        + `all_synsets(pos=None)`: Iterate over all synsets with a given part of speech tag. If no pos is specified, all synsets for all parts of speech will be loaded.
        + `citation(lang='omw')`: Return the contents of citation.bib file (for omw) use lang=lang to get the citation for an individual language
        + `custom_lemmas(tab_file, lang)`: Reads a custom tab file containing mappings of lemmas in the given language to Princeton WordNet 3.0 synset offsets, allowing NLTK's WordNet functions to then be used with that language.
            + `tab_file` (str): Tab file as a file or file-like object
            + `lang` (str): ISO 639-3 code of the language of the tab file
        + `get_version(self)`
        + `ic(corpus, weight_senses_equally=False, smoothing=1.0)`: Creates an information content lookup dictionary from a corpus.
            + `corpus` (CorpusReader): The corpus from which we create an information content dictionary.
            + `weight_senses_equally` (bool): If this is True, gives all possible senses equal weight rather than dividing by the number of possible senses. (If a word has 3 synses, each sense gets 0.3333 per appearance when this is False, 1.0 when it is true.)
            + `smoothing` (float): How much do we smooth synset counts (default is 1.0)
            + Return: An information content dictionary
        + `jcn_similarity(synset, other, ic, verbose=False)`: __Jiang-Conrath Similarity__: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets. The relationship is given by the equation $1 / (IC(s1) + IC(s2) - 2 * IC(lcs))$.
            + `other` (Synset): The `Synset` that this `Synset` is being compared to.
            + `ic` (dict): an information content object (as returned by `nltk.corpus.wordnet_ic.ic()`).
            + Return: A float score denoting the similarity of the two `Synset` objects.
        + `langs(self)`: return a list of languages supported by Multilingual Wordnet
        + `lch_similarity(synset, other, verbose=False, simulate_root=True)`: __Leacock Chodorow Similarity__: Return a score denoting how similar two word senses are, based on the shortest path that connects the senses (as above) and the maximum depth of the taxonomy in which the senses occur. The relationship is given as $-\log(p/2d)$ where $p$ is the shortest path length and d is the taxonomy depth.
            + `other` (Synset): The `Synset` that this `Synset` is being compared to.
            + `simulate_root` (bool): The various verb taxonomies do not share a single root which disallows this metric from working for synsets that are not connected. This flag (True by default) creates a fake root that connects all the taxonomies. Set it to false to disable this behavior. For the noun taxonomy, there is usually a default root except for WordNet version 1.6. If you are using wordnet 1.6, a fake root will be added for nouns as well.
            + Return: A score denoting the similarity of the two `Synset` objects, normally greater than 0. None is returned if no connecting path could be found. If a ``Synset`` is compared with itself, the maximum score is returned, which varies depending on the taxonomy depth.
        + `lemma(name, lang='eng')`: Return lemma object that matches the name
        + `lemma_count(lemma)`: Return the frequency count for this Lemma
        + `lemma_from_key(key)`
        + `lemmas(lemma, pos=None, lang='eng')`: Return all Lemma objects with a name matching the specified lemma name and part of speech tag. Matches any part of speech tag if none is specified.
        + `license(lang='eng')`: Return the contents of LICENSE (for omw) use lang=lang to get the license for an individual language
        + `lin_similarity(synset, other, ic, verbose=False)`: __Lin Similarity__: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets. The relationship is given by the equation $2 * IC(lcs) / (IC(s1) + IC(s2))$.
            + `other` (Synset): The `Synset` that this `Synset` is being compared to.
            + `ic` (dict): an information content object (as returned by `nltk.corpus.wordnet_ic.ic()`).
            + Return: A float score denoting the similarity of the two `Synset` objects, in the range 0 to 1.
        + `morphy(form, pos=None, check_exceptions=True)`: Find a possible base form for the given form, with the given part of speech, by checking WordNet's list of exceptional forms, and by recursively stripping affixes for this part of speech until a form in WordNet is found.
        + `of2ss(of)`: take an id and return the synsets
        + `path_similarity(synset, other, verbose=False, simulate_root=True)`: __Path Distance Similarity__: Return a score denoting how similar two word senses are, based on the shortest path that connects the senses in the is-a (hypernym/hypnoym) taxonomy. The score is in the range 0 to 1, except in those cases where a path cannot be found (will only be true for verbs as there are many distinct verb taxonomies), in which case None is returned. A score of 1 represents identity i.e. comparing a sense with itself will return 1.
            + `other` (Synset): The `Synset` that this `Synset` is being compared to.
            + `simulate_root` (dict): The various verb taxonomies do not share a single root which disallows this metric from working for synsets that are not connected. This flag (True by default) creates a fake root that connects all the taxonomies. Set it to false to disable this behavior. For the noun taxonomy, there is usually a default root except for WordNet version 1.6. If you are using wordnet 1.6, a fake root will be added for nouns as well.
            + Return: A score denoting the similarity of the two `Synset` objects, normally between 0 and 1. None is returned if no connecting path could be found. 1 is returned if a `Synset` is compared with itself.
        + `readme(lang='omw')`: Return the contents of README (for omw) use lang=lang to get the readme for an individual language
        + `res_similarity(synset, other, ic, verbose=False)`: __Resnik Similarity__: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node).
            + `other` (Synset): The `Synset` that this `Synset` is being compared to.
            + `ic` (dict): an information content object (as returned by `nltk.corpus.wordnet_ic.ic()`).
            + Return: A float score denoting the similarity of the two `Synset` objects. Synsets whose LCS is the root node of the taxonomy will have a score of 0 (e.g. N['dog'][0] and N['table'][0]).
        + `ss2of(ss)`: return the ID of the synset
        + `synset(name)`: Loading Synsets
        + `synset_from_pos_and_offset(pos, offset)`
        + `synsets(lemma, pos=None, lang='eng', check_exceptions=True)`: Load all synsets with a given lemma and part of speech tag. If no pos is specified, all synsets for all parts of speech will be loaded. If lang is specified, all the synsets associated with the lemma name of that language will be returned.
        + `words(lang='eng')`":  return lemmas of the given language as list of words
        + `wup_similarity(synset, other, verbose=False, simulate_root=True)`: __Wu-Palmer Similarity__: Return a score denoting how similar two word senses are, based on the depth of the two senses in the taxonomy and that of their Least Common Subsumer (most specific ancestor node). Previously, the scores computed by this implementation did _not_ always agree with those given by Pedersen's Perl implementation of WordNet Similarity. However, with the addition of the simulate_root flag (see below), the score for verbs now almost always agree but not always for nouns. <br/>  The LCS does not necessarily feature in the shortest path connecting the two senses, as it is by definition the common ancestor deepest in the taxonomy, not closest to the two senses. Typically, however, it will so feature. Where multiple candidates for the LCS exist, that whose shortest path to the root node is the longest will be selected. Where the LCS has multiple paths to the root, the longer path is used for the purposes of the calculation.
            + `other` (Synset): The `Synset` that this `Synset` is being compared to.
            + `simulate_root` (bool): The various verb taxonomies do not share a single root which disallows this metric from working for synsets that are not connected. This flag (True by default) creates a fake root that connects all the taxonomies. Set it to false to disable this behavior. For the noun taxonomy, there is usually a default root except for WordNet version 1.6. If you are using wordnet 1.6, a fake root will be added for nouns as well.
            + Return: A float score denoting the similarity of the two `Synset` objects, normally greater than zero. If no connecting path between the two senses can be found, None is returned.
    + Data and other attributes:
        + ADJ = 'a'
        + ADJ_SAT = 's'
        + ADV = 'r'
        + MORPHOLOGICAL_SUBSTITUTIONS = {'a': [('er', ''), ('est', ''), ('er', '...
        + NOUN = 'n'
        + VERB = 'v'
    + Methods inherited from `nltk.corpus.reader.api.CorpusReader`:
        + `__repr__(self)`: Return repr(self).
        + `__unicode__ = __str__(/)`: Return str(self).
        + `abspath(fileid)`:  Return the absolute path for the given file.
            + `fileid` (str): The file identifier for the file whose path should be returned.
            + Return: PathPointer
        + `abspaths(fileids=None, include_encoding=False, include_fileid=False)`: Return a list of the absolute paths for all fileids in this corpus; or for the given list of fileids, if specified.
            + `fileids` (None or str or list): Specifies the set of fileids for which paths should be returned.  Can be None, for all fileids; a list of file identifiers, for a specified set of fileids; or a single file identifier, for a single file.  Note that the return value is always a list of paths, even if `fileids` is a single file identifier.
            + `include_encoding`: If true, then return a list of `(path_pointer, encoding)` tuples.
            + Return: list(PathPointer)
        + `encoding(file)`: Return the unicode encoding for the given corpus file, if known. If the encoding is unknown, or if the given file should be processed using byte strings (str), then return None.
        + `ensure_loaded(self)`: Load this corpus (if it has not already been loaded).  This is used by LazyCorpusLoader as a simple method that can be used to make sure a corpus is loaded -- e.g., in case a user wants to do help(some_corpus).
        + `fileids(self)`: Return a list of file identifiers for the fileids that make up this corpus.
        + `open(file)`:  Return an open stream that can be used to read the given file. If the file's encoding is not None, then the stream will automatically decode the file's contents into unicode.
            + `file`: The file identifier of the file to read.
        + `unicode_repr = __repr__(self)`:  Return repr(self).
    + Data descriptors inherited from nltk.corpus.reader.api.CorpusReader:
        + `__dict__`: dictionary for instance variables (if defined)
        + `__weakref__`: list of weak references to the object (if defined)
        + `root`: The directory where this corpus is stored.

+ `wordnet_ic` class
    + Init Signature: `wordnet_ic(root, fileids)`
    + Docstring: A corpus reader for the WordNet information content corpus.
    + Method
        + `__init__(root, fileids)`
            + `root` (PathPointer or str): A path pointer identifying the root directory for this corpus.  If a string is specified, then it will be converted to a `PathPointer` automatically.
            + `fileids`: A list of the files that make up this corpus. This list can either be specified explicitly, as a list of strings; or implicitly, as a regular expression over file paths.  The absolute path for each file will be constructed by joining the reader's root to each file name.
            + `encoding`: The default unicode encoding for the files that make up the corpus.  The value of `encoding` can be any of the following:
                + A string: `encoding` is the encoding name for all files.
                + A dictionary: `encoding[file_id]` is the encoding name for the file whose identifier is `file_id`.  If `file_id` is not in `encoding`, then the file contents will be processed using non-unicode byte strings.
                + A list: `encoding` should be a list of `(regexp, encoding)` tuples.  The encoding for a file whose identifier is `file_id` will be the `encoding` value for the first tuple whose `regexp` matches the `file_id`.  If no tuple's `regexp` matches the `file_id`, the file contents will be processed using non-unicode byte strings.
                + None: the file contents of all files will be processed using non-unicode byte strings.
            + `tagset`: The name of the tagset used by this corpus, to be used for normalizing or converting the POS tags returned by the tagged_...() methods.
        + `ic(icfile)`: Load an information content file from the `wordnet_ic` corpus and return a dictionary.  This dictionary has just two keys, NOUN and VERB, whose values are dictionaries that map from synsets to information content values.
            + `icfile` (str): The name of the wordnet_ic file (e.g. "ic-brown.dat")
            + Return: An information content dictionary
    + Methods inherited from `nltk.corpus.reader.api.CorpusReader`:
        + `__repr__(self)`: Return repr(self).
        + `__unicode__ = __str__(/)`: Return str(self).
        + `abspath(fileid)`: Return the absolute path for the given file.
            + `fileid` (str): The file identifier for the file whose path should be returned.
            + Return: PathPointer
        + `abspaths(fileids=None, include_encoding=False, include_fileid=False)`: Return a list of the absolute paths for all fileids in this corpus; or for the given list of fileids, if specified.
            + `fileids` (None or str or list): Specifies the set of fileids for which paths should be returned.  Can be None, for all fileids; a list of file identifiers, for a specified set of fileids; or a single file identifier, for a single file.  Note that the return value is always a list of paths, even if `fileids` is a single file identifier.
            + `include_encoding`: If true, then return a list of `(path_pointer, encoding)` tuples.
            + return: list(PathPointer)
        + `citation(self)`: Return the contents of the corpus citation.bib file, if it exists.
        + `encoding(file)`: Return the unicode encoding for the given corpus file, if known. If the encoding is unknown, or if the given file should be processed using byte strings (str), then return None.
        + `ensure_loaded(self)`: Load this corpus (if it has not already been loaded).  This is used by LazyCorpusLoader as a simple method that can be used to make sure a corpus is loaded -- e.g., in case a user wants to do help(some_corpus).
        + `fileids(self)`: Return a list of file identifiers for the fileids that make up this corpus.
        + `license(self)`: Return the contents of the corpus LICENSE file, if it exists.
        + `open(file)`: Return an open stream that can be used to read the given file. If the file's encoding is not None, then the stream will automatically decode the file's contents into unicode.
            + `file`: The file identifier of the file to read.
        + `readme(self)`: Return the contents of the corpus README file, if it exists.
        + `unicode_repr = __repr__(self)`: Return repr(self).
    + Data descriptors inherited from `nltk.corpus.reader.api.CorpusReader:`
        + `__dict__`: dictionary for instance variables (if defined)
        + `__weakref__`: list of weak references to the object (if defined)
        + `root`: The directory where this corpus is stored.
    

+ Collocations and Distributional Similarity
    + “You know a word by the company it keeps” [Firth, 1957]
    + Two words that frequently appears in similar contexts are more likely to be semantically related
        + The friends _met at _a_ __café__.
        + Shyam _met_ Ray _at a_ __pizzeria__.
        + Let’s _meet_ up _near the_ __coffee shop__.
        + The secret _meeting at the_ __restaurant__ soon became public.

+ Distributional Similarity: Context
    + Words before, after, within a small window
    + Parts of speech of words before, after, in a small window
    + Specific syntactic relation to the target word
    + Words in the same sentence, same document, …

+ Strength of association between words
    + How frequent are these?
        + Not similar if two words don’t occur together often
    + Also important to see how frequent are individual words
        + ‘the’ is very frequent, so high chances it co-occurs often with every other word
    + Pointwise Mutual Information: $PMI(w,c) = \log [P(w,c) / P(w)P(c)]$

+ How to do it in Python?
    + Use NLTK Collocations and Association measures
        ```python
        import nltk
        from nltk.collocations import *

        bigram_measures = nltk.collocations.BigramAssocMeasures()

        finder = BigramCollocationFinder.from_words(text)
        finder.nbest(bigram_measures.pmi, 10)
        ```
    + finder also has other useful functions, such as frequency filter
        ```python
        finder.apply_freq_filter(10)
        ```

+ `ntlk.collections` Class
    + DESCRIPTION
        + Tools to identify collocations --- words that often appear consecutively --- within corpora. They may also be used to find other associations between word occurrences.
        + See Manning and Schutze [ch. 5](http://nlp.stanford.edu/fsnlp/promo/colloc.pdf) and the [NSP Perl package](http://ngram.sourceforge.net)
        + Finding collocations requires first calculating the frequencies of words and their appearance in the context of other words. Often the collection of words will then requiring filtering to only retain useful content terms. Each ngram of words may then be scored according to some association measure, in order to determine the relative likelihood of each ngram being a collocation.
        + The `BigramCollocationFinder` and `TrigramCollocationFinder` classes provide these functionalities, dependent on being provided a function which scores a ngram given appropriate frequency counts. A number of standard association measures are provided in bigram_measures and trigram_measures.
    + CLASSES: `AbstractCollocationFinder(builtins.object)`
        + `BigramCollocationFinder`
        + `QuadgramCollocationFinder`
        + `TrigramCollocationFinder`

+ `AbstractCollocationFinder(builtins.object)` class
    + Init Signature: `nltk.collections.AbstractCollocationFinder(word_fd, ngram_fd)`
    + Docstring: An abstract base class for collocation finders whose purpose is to collect collocation candidate frequencies, filter and rank them. <br/>
        As a minimum, collocation finders require the frequencies of each word in a corpus, and the joint frequency of word tuples. This data should be provided through nltk.probability.FreqDist objects or an identical interface.
    + Methods
        + `__init__(word_fd, ngram_fd)`: Initialize self.  See help(type(self)) for accurate signature.
        + `above_score(score_fn, min_score)`: Returns a sequence of ngrams, ordered by decreasing score, whose scores each exceed the given minimum score.
        + `apply_freq_filter(min_freq)`: Removes candidate ngrams which have frequency less than min_freq.
        + `apply_ngram_filter(fn)`: Removes candidate ngrams $(w_1, w_2, \ldots)$ where $fn(w_1, w_2, \ldots)$ evaluates to True.
        + `apply_word_filter(fn)`: Removes candidate ngrams $(w_1, w_2, \ldots)$ where any of $(fn(w_1), fn(w_2), \ldots)$ evaluates to True.
        + `nbest(score_fn, n)`: Returns the top n ngrams when scored by the given function.
        + `score_ngrams(score_fn)`: Returns a sequence of (ngram, score) pairs ordered from highest to lowest score, as determined by the scoring function provided.
    + Class methods
        + `from_documents(documents)` from `builtins.type`: Constructs a collocation finder given a collection of documents, each of which is a list (or iterable) of tokens.
    + Data descriptors
        + `__dict__`: dictionary for instance variables (if defined)
        + `__weakref__`: list of weak references to the object (if defined)


+ `BigramCollocationFinder(AbstractCollocationFinder)` classs
    + Init Signature: `ntlk.BigramCollocationFinder(word_fd, bigram_fd, window_size=2)`
    + Doctsring: A tool for the finding and ranking of bigram collocations or other association measures. It is often useful to use from_words() rather than constructing an instance directly.
    + Methods
        + `__init__(word_fd, bigram_fd, window_size=2)`: Construct a BigramCollocationFinder, given FreqDists for appearances of words and (possibly non-contiguous) bigrams.
        + `score_ngram(score_fn, w1, w2)`: Returns the score for a given bigram using the given scoring function.  Following Church and Hanks (1990), counts are scaled by a factor of 1/(window_size - 1).
    + Class methods
        + `from_words(words, window_size=2)` from `builtins.type`: Construct a BigramCollocationFinder for all bigrams in the given sequence.  When window_size > 2, count non-contiguous bigrams, in the style of Church and Hanks's (1990) association ratio.
    + Data and other attributes
        + `default_ws = 2`


+ `QuadgramCollocationFinder(AbstractCollocationFinder)` class
    + Init Signature: `ntlk.QuadgramCollocationFinder(word_fd, quadgram_fd, ii, iii, ixi, ixxi, iixi, ixii)`
    + Docstring: A tool for the finding and ranking of quadgram collocations or other association measures. It is often useful to use from_words() rather than constructing an instance directly.
    + Methods
        + `__init__(word_fd, quadgram_fd, ii, iii, ixi, ixxi, iixi, ixii)`: Construct a QuadgramCollocationFinder, given FreqDists for appearances of words, bigrams, trigrams, two words with one word and two words between them, three words with a word between them in both variations.
        + `score_ngram(score_fn, w1, w2, w3, w4)`
    + Class methods
        + `from_words(words, window_size=4)` from `builtins.type`
    + Data and other attributes
        + `default_ws = 4`


+ `TrigramCollocationFinder(AbstractCollocationFinder)` class
    + Init Signature: `nltk.TrigramCollocationFinder(word_fd, bigram_fd, wildcard_fd, trigram_fd)`
    + Docstring: A tool for the finding and ranking of trigram collocations or other association measures. It is often useful to use from_words() rather than constructing an instance directly.
    + Methods
        + `__init__(word_fd, bigram_fd, wildcard_fd, trigram_fd)`: Construct a TrigramCollocationFinder, given FreqDists for appearances of words, bigrams, two words with any word between them, and trigrams.
        + `bigram_finder()`: Constructs a bigram collocation finder with the bigram and unigram data from this finder. Note that this does not include any filtering applied to this finder.
        + `score_ngram(score_fn, w1, w2, w3)`: Returns the score for a given trigram using the given scoring function.
    + Class methods
        + `from_words(words, window_size=3)` from `builtins.type` Construct a TrigramCollocationFinder for all trigrams in the given sequence.
    + Data and other attributes
        + `default_ws = 3`

+ `NgramAssocMeasures(builtins.object)` class
    + Init Signature: `nltk.NgramAssocMeasures()`
    + Docstring: An abstract class defining a collection of generic association measures.
    + Notes: 
        + Each public method returns a score, taking the following arguments:
            ```python
            score_fn(count_of_ngram,
                    (count_of_n-1gram_1, ..., count_of_n-1gram_j),
                    (count_of_n-2gram_1, ..., count_of_n-2gram_k),
                    ...,
                    (count_of_1gram_1, ..., count_of_1gram_n),
                    count_of_total_words)
            ```
        + Inheriting classes should define a property `_n`, and a method `_contingency` which calculates contingency values from marginals in order for all association measures defined here to be usable.
    + Class methods
        + `chi_sq(*marginals)` from `abc.ABCMeta`: Scores ngrams using Pearson's chi-square as in Manning and Schutze 5.3.3.
        + `jaccard(*marginals)` from `abc.ABCMeta`: Scores ngrams using the Jaccard index.
        + `likelihood_ratio(*marginals)` from `abc.ABCMeta`: Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4.
        + `pmi(*marginals)` from `abc.ABCMeta`: Scores ngrams by pointwise mutual information, as in Manning and Schutze 5.4.
        + `poisson_stirling(*marginals)` from `abc.ABCMeta`: Scores ngrams using the Poisson-Stirling measure.
        + `student_t(*marginals)` from `abc.ABCMeta`: Scores ngrams using Student's t test with independence hypothesis for unigrams, as in Manning and Schutze 5.3.1.
    + Static methods
        + `mi_like(*marginals, **kwargs)`: Scores ngrams using a variant of mutual information. The keyword argument power sets an exponent (default 3) for the numerator. No logarithm of the result is calculated.
        + `raw_freq(*marginals)`: Scores ngrams by their frequency
    + Data descriptors
        + `__dict__`: dictionary for instance variables (if defined)
        + `__weakref__`: list of weak references to the object (if defined)
    + Data and other attributes
        + `__abstractmethods__ = frozenset({'_contingency', '_marginals'})`


+ `BigramAssocMeasures(NgramAssocMeasures)` Class
    + Init Signature: `ntlk.BigramAssocMeasures()`
    + Docstring: A collection of bigram association measures. 
    + Notes:
        + Each association measure is provided as a function with three arguments:
            ```python
            bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)
            ```
        + The arguments constitute the marginals of a contingency table, counting the occurrences of particular events in a corpus. The letter `i` in the suffix refers to the appearance of the word in question, while `x` indicates the appearance of any word. Thus, for example:
            ```python
            n_ii counts (w1, w2), i.e. the bigram being scored
            n_ix counts (w1, *)
            n_xi counts (*, w2)
            n_xx counts (*, *), i.e. any bigram
            ```
        + This may be shown with respect to a contingency table::
            ```
                    w1    ~w1
                ------ ------
            w2 | n_ii | n_oi | = n_xi
                ------ ------
            ~w2 | n_io | n_oo |
                ------ ------
                = n_ix        TOTAL = n_xx
            ```
    + Class methods
        + `chi_sq(n_ii, n_ix_xi_tuple, n_xx) from abc.ABCMeta`: Scores bigrams using chi-square, i.e. phi-sq multiplied by the number of bigrams, as in Manning and Schutze 5.3.3.
        + `fisher(*marginals)` from `abc.ABCMeta`: Scores bigrams using Fisher's Exact Test (Pedersen 1996).  Less sensitive to small counts than PMI or Chi Sq, but also more expensive to compute. Requires scipy.
        + `phi_sq(*marginals)` from `abc.ABCMeta`: Scores bigrams using phi-square, the square of the Pearson correlation coefficient.
    + Static methods
        + `dice(n_ii, n_ix_xi_tuple, n_xx)`: Scores bigrams using Dice's coefficient.
    + Data and other attributes
        + `__abstractmethods__ = frozenset()`

+ `TrigramAssocMeasures(NgramAssocMeasures)` class
    + Init Signature: `nltk.TrigramAssocMeasures()`
    + DocString: A collection of trigram association measures
    + Notes: 
        + Each association measure is provided as a function with four arguments:
            ```python
            trigram_score_fn(n_iii,
                            (n_iix, n_ixi, n_xii),
                            (n_ixx, n_xix, n_xxi),
                            n_xxx)
            ```
        + The arguments constitute the marginals of a contingency table, counting the occurrences of particular events in a corpus. The letter i in the suffix refers to the appearance of the word in question, while x indicates the appearance of any word. Thus, for example:
            + `n_iii` counts $(w_1, w_2, w_3)$, i.e. the trigram being scored
            + `n_ixx` counts $(w_1, *, *)$
            + `n_xxx` counts $(*, *, *)$, i.e. any trigram
    + Data and other attributes defined here:
        + `__abstractmethods__ = frozenset()` 


+ Take Home Concepts
    + Finding similarity between words and text is non-trivial
    + WordNet is a useful resource for semantic relationships between words
    + Many similarity functions exist
    + NLTK is a useful package for many such tasks



### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/6YHYeWgHEeeDRAot5bGaoA.processed/full/360p/index.mp4?Expires=1544054400&Signature=ZlQMdFNtoIH2CDQbvh-URxcoC9-kgLyy9DR0SlA-XGK5t1CLcazL4Iiv3nTR4-bxlfQ0HzkiVsDlTS8zbIlGVHtDzt6ot9rKIPVYOY-J~9ZzqISn0srTUvG-ESWavl9JoI8Ag9Hw39hF3SRaQvrvOMYNkrhWf-oxJaG4tSrtOeg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Semantic Text Similarity" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



## Topic Modeling

### Lecture Notes

+ Documents Exhibit Multiple Topics
    <a href="https://www.coursera.org/learn/python-text-mining/lecture/KiiBl/topic-modeling"> <br/>
        <img src="images/p4-02.png" alt="this is about Seeking Life's Bare Necessities, Bare Genetic Necessities. And as you look through this, you'll notice that some words have been highlighted. So you have words such as genes and genomes that are highlighted in yellow, words such as computer, and predictions, and computer analysis, and computation are in blue. And then you have organism, or survive, or life in pink. This demonstrates that any article you see is more likely to be formed of different topics or sub-units that intermingle very seamlessly in weaving out an article. This is the basis of one of the leading research work that has happened in text manning on topic modeling, and this one particularly is from Latent Dirichlet Allocation. Well, you have three topics. You have genetics that's in yellow, or computation that's in blue, and life-related, life science, let's say, in pink." title="Document with multiple toipcs" height="200">
    </a>

+ Intuition: Documents as a mixture of topics
    <a href="https://www.coursera.org/learn/python-text-mining/lecture/KiiBl/topic-modeling"> <br/>
        <img src="images/p4-03.png" alt="This shows that documents are typically a mixture of topics. So you have topics coming from genetics, computation, or even anatomy. And each of these topics are basically words that are more probable coming from that topic. When you're talking about genes and DNA and so on, you are mostly in the genetics realm, while if you're talking about brain and neuron and nerve, you are in anatomy. If you're talking about computers and numbers and data and so on, you're most likely in computation. So when a new document comes in, in this case this article on seeking life's bare genetic necessities, it comes with it of topic distribution. And so for that particular article, there is some sort of topic distribution over these topics. Assume there are only four topics in the world. Genetics, computation, life sciences, anatomy. Obviously, that's not true. But let's take in this sense that these are the only four topics you have, and this particular article is generated by these four topics in some combination of words. Where anatomy, the green one, is absent, and computation, for example, is the most probable. But then you have genetics also including a percentage and a little bit of life sciences." title="Documents with mixture of topics" height="200">
    </a>

+ More examples of topics
    <a href="https://www.coursera.org/learn/python-text-mining/lecture/KiiBl/topic-modeling"> <br/>
        <img src="images/p4-04.png" alt="text" title="Examples of Tpoics" height="250">
    </a>

+ What is Topic Modeling?
    + A course-level analysis of what’s in a text collection
    + Topic: the subject (theme) of a discourse
    + Topics are represented as a word distribution
    + A document is assumed to be a mixture of topics
    + What’s known:
        + The text collection or corpus
        + Number of topics
    + What’s not known:
        + The actual topics
        + Topic distribution for each document
    + Essentially, text clustering problem
        + Documents and words clustered simultaneously
    + Different topic modeling approaches available
        + Probabilistic Latent Semantic Analysis (PLSA) [Hoffman ’99]
        + Latent Dirichlet Allocation (LDA) [Blei, Ng, and Jordan, ’03]



### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/ldTLaGgHEeeDRAot5bGaoA.processed/full/360p/index.mp4?Expires=1544054400&Signature=PCThz9lSJd6youdaFNWTgGI-xMAuE6252D9rUba1jekqpNYnhRJuG3MWPKVVSnB8Ff0oXX4yPfZzkw8JVT1s3PiiM0euSmb8iYEUGuW5AChexNJSlDfh88MEwvC~MN~nh0et1Gc7X1kojDwvVzdwbEv7AG1Gcy2ZDhVKIY8iU1M_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Topic Modeling" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



## Generative Models and LDA

### Lecture Notes



+ Demo
    ```python

    ```


### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



## Practice Quiz: Practice Quiz

### Lecture Notes



+ Demo
    ```python

    ```


### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



## Information Extraction

### Lecture Notes



+ Demo
    ```python

    ```


### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



## Additional Resources & Readings

### Lecture Notes



+ Demo
    ```python

    ```


### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>



## Quiz: Module 4 Quiz

### Lecture Notes



+ Demo
    ```python

    ```


### Lecture Video

<a href="url" alt="text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


