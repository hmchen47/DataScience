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


## Naive Bayes Classifiers

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





