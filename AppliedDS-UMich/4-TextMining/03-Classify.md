# Module 3: Classification of Text

## Text Classification

### Lecture Notes

+ Which medical speciality does this relate to?
    + Nephrology / Neurology / Podiatry
    + Paragraph1: TINEA PEDIS, or ATHLETE'S FOOT, is a very common fungal skin infection of the foot. It often first appears between the toes. It can be a one-time occurrence or it can be chronic. The fungus, known as Trichophyton, thrives under warm, damp conditions so people whose feet sweat a great deal are more susceptible. It is easily transmitted in showers and pool walkways. Those people with immunosuppressive conditions, such as diabetes mellitus, are also more susceptible to athlete's foot.
    + Paragraph 2: KIDNEY FAILURE, also known as RENAL FAILURE or RENAL INSUFFICIENCY, is a medical condition of impaired kidney function in which the kidneys fail to adequately filter metabolic wastes from the blood.The two main forms are acute kidney injury, which is often reversible with adequate treatment, and chronic kidney disease, which is often not reversible. In both cases, there is usually an underlying cause.
    + Answers: 1) Podiatry, 2) Nephrology

+ What is Classification?
    + Given a set of classes: Nephrology / Neurology / Podiatry
    + Classification: Assign the correct class label to the given input

+ Examples of Text Classification
    + __Topic identification__: Is this news article about Politics, Sports, or Technology?
    + __Spam Detection__: Is this email a spam or not?
    + __Sentiment analysis__: Is this movie review positive or negative?
    + __Spelling correction__: weather or whether? color or colour?

+ Supervised Learning: Humans learn from past experiences, machines learn from past instances!

+ Supervised Classification
    <a href="https://www.coursera.org/learn/python-text-mining/lecture/H05Dd/text-classification"> <br/>
        <img src="images/p3-01.png" alt="So, for example, in a supervised classification task, you have this training phase where information is gathered and a model is built and an inference phase where that model is applied. So, for example, you'll have a set of inputs that we call the labeled input, where we know that this particular instance is positive, this one is negative, and so on. So in this case, let's just do examples. Green and red, or light and dark, or positive and negative. And you did that set that the label set up instances and feed it into a classification algorithm. This classification algorithm will learn which instances appear to be more positive than negative and build a model for what it learns. Once you have the model, you can used it in the inference phase, where you have unlabeled input and then this model will take it and give out labels for those inputs, for those input instances." title="Supervised Classification" height="150">
    </a>

+ Supervised Classification
    + Learn a __classification model__ on properties (“features”) and their importance (“weights”) from labeled instances
        + $X$: Set of attributes or features ${x_1, x_2, \cdots, x_n}$
        + $y$: A “class” label from the label set $Y = {y_1, y_2, \cdots, y_k}$
    + Apply the model on new instances to __predict__ the label

+ Supervised Classification: Phases and Datasets
    <a href="https://www.coursera.org/learn/python-text-mining/lecture/H05Dd/text-classification"> <br/>
        <img src="images/p3-02.png" alt="So when we look at these, there are some terminology of data sets that you would see very commonly. So again, there are two phases, the training phase and the inference phase. The training phase has labeled data set. And in general, in the inference phase you have unlabeled data set. Unlabeled data set is where you have all instances and you have the x defined, but you don't have a y. You don't have a label. Whereas in the labeled data set, you have the x and the y for every instance given to you. However, in training, you don't use the entire label set for training purposes. Because then you will not know how well your model is. So what you want to do is to use a part of it as training data where you actually learn parameters. Learn the model, but leave some aside as a validation data set, or it's sometimes called hold out data set. So that in the training phase, you can learn on the training data but then test, or evaluate, or set parameters on the validation data.   And then, you want another data set to really test how well you do. We are to never use it in training. You don't set your parameters based on that. But you just evaluate on that, so that you can judge whether the model was really good or not on completely unseen data. You have seen all of these concepts in previous courses within this specialization that goes straight. But I want to kind of bring them here so that we have the context in which we're going to talk about." title= "caption" height="200">
    </a>

+ Classification Paradigms
    + When there are only two possible classes; $|Y| = 2$: __Binary Classification__
    + When there are more than two possible classes; $|Y| > 2$: __Multi-class Classification__
    + When data instances can have two or more labels: __Multi-label Classification__

+ Questions to ask in Supervised Learning
    + __Training phase__:
        + What are the features? How do you represent them?
        + What is the classification model / algorithm?
        + What are the model parameters?
    + __Inference phase__:
        + What is the expected performance?
        + What is a good measure?


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/K_Gy1GbFEeeB7Qo5yIjKZg.processed/full/360p/index.mp4?Expires=1543363200&Signature=CRkG7044IWoEgmuDfvKjy-9XLRUKl79Btm~n5PJh5~FC1UNRxt7akQ8YbRIVXM-GcWPQdUnbI~vRQse9XyhYDk4fhmqPqwwqWG4YboKR8BvJGy7PZYiVlylfQRL5TMzuH320YPTEZDbEzcqU9BG3p2YdfzH9IHygNDRp7nztrvo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Text Classification" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Identifying Features from Text

### Lecture Notes

+ Why is Textual Data Unique?
    + Textual data presents a unique set of challenges
    + All the information you need is in the text
    + But features can be pulled out from text at different granularities!

+ [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
    + __Stemming__:
        + Usually refer to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes.
        + The process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form.
        + The stem need not be identical to the morphological root of the word; it is usually sufficient that related words map to the same stem, even if this stem is not in itself a valid root.
        + A stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech.
        + Eg. "cat" as stem of "cats", "catlike" and "catty"
    + __Lemmatization__:
        + usually refer to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the __lemma__.
        + Lemmatisation (or lemmatization) in linguistics: the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form
        + Lemmatisation is the algorithmic process of determining the lemma of a word based on its intended meaning.
        + Examples:
            + "walk" as lemma of 'walk', 'walked', 'walks', 'walking'
            + "good" as lemma of "better"
    + Notes:
        + "walk" as lemma and stem of "walking"
        + "meeting" as base with noun and as form of a verb

+ [Main differences between stemming and lemmatization](https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/)
    + Stemming algorithms work by cutting off the end or the beginning of the word, taking into account a list of common prefixes and suffixes that can be found in an inflected word. This indiscriminate cutting can be successful in some occasions, but not always, and that is why we affirm that this approach presents some limitations. Below we illustrate the method with examples in both English and Spanish.
        <a href="https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/"> <br/>
            <img src="https://blog.bitext.com/hs-fs/hubfs/stemming_v2.png?t=1543243224992&width=372&height=195&name=stemming_v2.png" alt="Stemming" title="Stemming Examples" height="150">
        </a>
    + Lemmatization: take into consideration the morphological analysis of the words. To do so, it is necessary to have detailed dictionaries which the algorithm can look through to link the form back to its lemma. Again, you can see how it works with the same example words.
        <a href="https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/"> <br/>
            <img src="https://blog.bitext.com/hs-fs/hubfs/lemma_v2.png?t=1543243224992&width=840&height=234&name=lemma_v2.png" alt="Lemmatization" title="Lemmatization Examples" height="150">
        </a>

+ Types of Textual Features
    + Words
        + By far the most common class of features
        + Handling commonly-occurring words: Stop words, e.g., "the"
        + Normalization: Make lower case vs. leave as-is; e.g., "US" vs "us"
        + Stemming / Lemmatization
    + Characteristics of words : Capitalization, e.g., "White House" vs "white house"
    + Parts of speech of words in a sentence, e.g. determiner, "weather" vs "whether"
    + Grammatical structure, sentence parsing
    + Grouping words of similar meaning, semantics
        + {buy, purchase}
        + {Mr., Ms., Dr., Prof.}; Numbers / Digits; Dates
    + Depending on classification tasks, features may come from inside words and word sequences
        + bigrams, trigrams, n-grams: “White House”
        + character sub-sequences in words: “ing”, “ion”, …

+ How would you do it? -> Recall lectures from previous week


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/4u6bd2gGEeeDRAot5bGaoA.processed/full/360p/index.mp4?Expires=1543363200&Signature=EVVRqHrSFqIx-TFZwjDgZ8oABB6s8oTtqegsw6yRunz~nnDbbR2yvuJUuCDHyx93FhJ2NGm5p-XBrWPAnuM4uQdNTFVBxSBOBCe~5r34EXAv-CcB5eS7ClMeNChzr1gAEvUjDTA2nYdqgXIqGSkTjqWeRK4niLw671~GJ-5DbsU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Identifying Features from Text" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Naive Bayes Classifiers

### Lecture Notes

+ Case study: Classifying text search queries
    + Suppose you are interested in classifying search queries in three classes: Entertainment, Computer Science, Zoology
    + Most common class of the three is Entertainment.
    + Suppose the query is “Python”
        + Python, the snake (Zoology)
        + Python, the programming language (Computer Science)
        + Python, as in Monty Python (Entertainment)
    + Most common class, given “Python”, is Zoology.
    + Suppose the query is “Python download”
        + Most probable class, given “Python download”, is Computer Science.

+ Probabilistic Model
    + Update the likelihood of the class given new information
    + _Prior Probability_: $Pr(y = Entertainment), Pr(y = CS), Pr(y=Zoology)$
    + _Posterior probability_: $Pr(y = Entertainment|x = “Python”)$

+ Bayes’ Rule
    + Posterior probability = (Prior probability x Likelihood) / (Evidence)
    + $Pr(y | X) = \frac{Pr(y) \times Pr(X|y)}{Pr(X)}$

+ Naïve Bayes Classification
    + $Pr(y=CS | "Python") = \frac{Pr(y=CS) \times Pr("Python" | y = CS)}{Pr("Python)}$
    + $Pr(y=Zoology| "Python") = \frac{Pr(y=Zoology|) \times Pr("Python" | y = Zoology|)}{Pr("Python)}$
    + $Pr(y=CS | "Python") > Pr(y=Zoology | "Python")$ --> $y = CS$
    + Probability Theory: $Pr(y|X) = \frac{Pr(y) \times Pr(X|y)}{Pr(X)}$ 
    + Classification: $y^{\ast} = \arg\max_y Pr(y|X) = \arg\max_y Pr(y) \times Pr(X|y)$
    + __Naïve assumption__: Given the class label, features are assumed to be independent of each other <br/>
        $y^{\ast} = \arg\max_y Pr(y|X) = \arg\max_y Pr(y) \times \prod_{i-1}^n Pr(x_i | y)$
    + Query: "Python download" <br/>
        $y^{\ast} = \arg\max_y Pr(y) \times Pr("Python" | y) \times Pr("download" |y)$

+ Naïve Bayes: What are the parameters?
    + Prior probabilities: $Pr(y)  \forall y \in Y$
    + Likelihood: $Pr(x_i | y) \forall x_i \in X, y \in Y$, $x_i$ = feature, $y$ = label
    + If there are 3 classes $(|Y| = 3)$ and 100 features in $X$, how many parameters does naïve Bayes models have?

+ Naïve Bayes: Learning parameters
    + Prior probabilities: $Pr(y) \forall y \in Y$
        + Remember training data? all queries are labeled
        + Count the number of instances in each class
        + If there are $N$ instances in all, and $n$ out of those are labeled as class $y$ --> $Pr(y) = n / N$
    + Likelihood: $Pr(x_i | y) \forall x_i \in X, y \in Y$
        + Count how many times feature $x_i$ appears in instances labeled as class $y$
        + If there are $p$ instances of class $y$, and $x_i$ appears in $k$ of those, $Pr(x_i | y) = k / p$

+ Example: Counting parameters
    + You are training a naïve Bayes classifier, where the number of possible labels, $|Y| = 3$ and the dimension of the data element, $|X| = 100$, where every feature (dimension) is binary. How many parameters does the naïve Bayes classification model have?
    + A naïve Bayes classifier has two kinds of parameters:
        1. $Pr(y)$ for every $y \in Y$: so if $|Y| = 3$, there are three such parameters.
        2. $Pr(x_i | y)$ for every binary feature $x_i \in X$ and $y \in Y$. Specifically, for a particular feature x_1, the parameters are $Pr(x_1 = 1 | y)$ and $Pr(x_1 = 0 | y)$ for every $y$. So if $|X| = 100$ binary features and $|Y| = 3$, there are $(2 x 100) x 3 = 600$ such features

        Hence in all, there are 603 features.
    + Note that not all of these features are independent. In particular, $Pr(x_i = 0 | y) = 1 - Pr(x_i = 1 | y)$, for every $x_i$ and $y$. So, there are only 300 independent parameters of this kind (as the other 300 parameters are just complements of these). Similarly, the sum of all prior probabilities $Pr(y)$ should be 1. This means there are only 2 independent prior probabilities. In all, for this example, there are 302 independent parameters.

+ [Argmax and Max Calculus](https://www.cs.ubc.ca/~schmidtm/Documents/2016_540_Argmax.pdf)
    + Def: the __argmex__ of a function $f$ on a set $D$ as <br/> 
        $\arg\max_{x \in D} f(x) = \{x | f(x) \geq f(y), \forall y \in D \}$
    + The set of inputs $x$ from the domain $D$ that achieve the highest function value
    + E.g., $\arg\max_{x \in \Re} -x^2 = \{ 0\}$
    + Operations not change the argmax set
        1. $\theta = \text{constant}$: $\arg\max f(x) = \arg\max f(x) + \theta$
        2. $\theta > 0$: $\arg\max f(x) = \arg\max \theta f(x)$
        3. $\theta <> 0$: $\arg\max f(x) = \arg\min \theta f(x)$
        4. \$\arg\max f(x) > 0$: $\arg\max f(x) = \arg\min \frac{1}{f(x)}$
        5. $g$ strictly monotonic: $\arg\max g(f(x)) = \arg\max f(x)$
    + Logarithm (a strictly monotonic function) transform multiplication of probability into addition of log-probabilities: <br/> $\arg\max \prod_{i=1}^n p_i(x) = \arg\max \sum_{i=1}^n \log p_i(x)$

+ Naïve Bayes: Smoothing
    + What happens if $Pr(x_i | y) = 0$?
        + Feature $x_i$ never occurs in documents labeled $y$
        + But then, the posterior probability $Pr(y | x_i)$ will be 0!!
    + Instead, smooth the parameters
    + __Laplace smoothing__ or __Additive smoothing__: Add a dummy count
        + $Pr(x_i | y) = (k+1) / (p+n)$; where $n$ is number of features

+ Take Home Concepts
    + Naïve Bayes is a probabilistic model
    + Naïve, because it assumes features are independent of each other, given the class label -> issue: "White" & "House" for "White House"
    + For text classification problems, naïve Bayes models typically provide very strong baselines
    + Simple model, easy to learn parameters


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/_JUlTGbGEeexMxI6w-Sq3g.processed/full/360p/index.mp4?Expires=1543363200&Signature=PCcVBvVnmfkGJwe2vPk34VQObZ7x-Xe7jwjhOEbbjmfgJkJT1I-OX5w2dEay8sYLmr6Y8baDaFGKjWZBsNlXvIvJtHbGwpEqHfvMzBBbH0tld3~mO6oVj4B1PMP6k5BPklLZ1MNRdG5OC8NcI1alrv7nmD91IdwCGjYAJVF79rU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Naive Bayes Classifiers" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="40px"> 
</a>


## Naive Bayes Variations

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


## Support Vector Machines

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


## Learning Text Classifiers in Python

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


## Notebook: Case Study - Sentiment Analysis




## Demonstration: Case Study - Sentiment Analysis

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


## Quiz: Module 3 Quiz





