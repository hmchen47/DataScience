# Leakage in Data Mining: Formulation, Detection, and Avoidance

Authors: S. Kaufman, S. Rosset, & C. Perlich


## Introduction

+ __Leakage__ in data mining (henceforth leakage) is essentially the introduction of information about the data mining target, which should not be legitimately available to mine from.

+ The introduction of this illegitimate information is unintentional, and facilitated by the data collection, aggregation and preparation process.

+ Leakage is undesirable as it may lead a __modeler__ to learn a suboptimal solution, which would in fact be outperformed in deployment by a leakage-free model that could have otherwise been built. At the very least leakage leads to overestimation of the model's performance.

+ Attempts to fix leakage resulted in the introduction of new leakage which is even harder to deal with.

## Leakage in the KDD Literature

+ Pyle (Data Preparation for Data Mining) 
    + _Anachronisms_: leakage in the context of predictive modeling; something that is out of place in time
    + "too good to be true" performance is "a dead giveaway" of its existence
    + Suggestion: turning to exploratory data analysis in order to find and eliminate leakage sources

+ Nisbet et al., Handbook of Statis-tical Analysis and Data Mining Applications. 2009
    + refer to the issue as "leaks from the future” and claim it is "one of the top 10 data mining mistakes"
    + Two representative examples are: 
        1. An "account number" feature, for the problem of predicting whether a potential customer would open an account at a bank. Obviously, assignment of such an account number is only done after an account has been opened.
        2. An "interviewer name" feature, in a cellular company churn prediction problem. While the information “who inter-viewed the client when they churned” appears innocent enough, it turns out that a specific salesperson was assigned to take over cases where customers had already notified they intend to churn.

+ Kohavi et al.,
    + KDD-cup 2000 organizers‟ report: peeling the onion. ACM SIGKDD Explorations Newsletter.
    + describe the introduction of leaks in data mining competitions as giveaway attributes that predict the target because they are downstream in the data collection process
    + Example in the domain of retail website data analytics: for each page viewed the prediction target is whether the user would leave or stay to view another page
        + Leaking attribute "session length": the total number of pages viewed by the user during this visit to the website
        + Added to each page-view record at the end of the session
        + Solution: replace this attribute with "page number in session" which de-scribes the session length up to the current page, where prediction is required

+ Kohavi et al. 
    + Ten supplementary analyses to improve e-commerce web sites. In Proceedings of the Fifth WEBKDD Workshop, 2003
    + The common business analysis problem of characterizing big spenders among customers
    + The problem is prone to leakage since immediate triggers of the target (e.g. a large purchase or purchase of a diamond) or consequences of the target (e.g. paying a lot of tax) are usually available in collected data and need to be manually identified and removed.
    + Correcting for leakage can become an involved process
    + Example: removing the information "total purchase in jewelry" caused information of "no purchases in any department" to become fictitiously predictive
    + Each customer found in the database is there in the first place due to some purchase, and if this purchase is not in any department (still available), it has to be jewelry (which has been removed).
    + Defining analytical questions that should suffer less from leaks: Characterizing a "migrator" (a user who is a light spender but will become a heavy one) instead of characterizing the "heavy spender"
    + Better to ask analytical questions that have a clear temporal cause-and-effect structure
    + The “use of free shipping”, where a leak is introduced when free shipping is provided as a special offer with large purchases.

+ Rosset et al.
    + 
    + 1st challenge: "Who Reviewed What" to predict whether each user would give a review for each title in 2006, given data up to 2005 and a test set with actual reviews from 2006 was provided
    + 2nd challenge: "How Many Reviews" to predict the number of reviews each title would receive in 2006, also using data given up to 2005
    + Winning submission managed to use the test set for the first problem as the target in a supervised-learning modeling approach for the second problem.
    + Two facts
        + Up to a scaling factor and noise, the expected number of user/review pairs in the first problem's test set in which a title appears is equal to the total number of reviews which that titled received in 2006. This is exactly the target for the second problem, only on different titles.
        + The titles are similar enough to share statistical properties, so from the available dynamics for the first group of titles one can infer the dynamics of the second group's.

+ KDD-Cup 2008 dealt with cancer detection from mammography data
    + "Patient ID" feature (ignored by most competitors) has tremendous and unexpected predictive power
    + Some of these sources were assigned their population with prior knowledge of the patient's condition
    + Leakage was thus facilitated by assigning consecutive patient IDs for data from each source, that is, the merge was done without obfuscating the source.
    + The INFORMS Data Mining Challenge 2008 competition:
        + Addressed the problem of pneumonia diagnosis based on patient information from hospital records.
        + The target was originally embedded as a special value of one or more features in the data given to competitors.
        + The organizers removed these values, however it was possible to identify traces of such removal, constituting the source of leakage in this example

+ Rosset et al.
    + The concept of identifying and harnessing leakage has been openly addressed as one of three key aspects for winning data mining competitions.
    + The intuitive definition of leakage: "The unintentional introduction of predictive information about the target by the data collection, aggregation and preparation process".
    + Leakage might be the cause of many failures of data mining applications
    + A real-life business intelligence project at IBM
        + Potential customers for certain products were identified, among other things, based on keywords found on their websites.
        + Leakage: the website content used for training had been sampled at the point in time where the potential customer has already become a customer, and where the website contained traces of the IBM products purchased, such as the word “Websphere”

+ INFORMS 2010 and IJCNN competitions 2011
    + Leakage continues to plague predictive modeling problems and competitions
    + The INFORMS 2010 Data Mining Challenge
        + Develop a model that predicts stock price movements, over a fixed one-hour horizon, at five minute intervals
        + Provided with intraday trading data showing stock prices, sectoral data, economic data, experts' predictions and indices.
        + Possible to build models that rely on data from the future
        + Having data from the future for the explanatory variables, some of which are highly co-integrated with the target (, and having access to publicly available stock data such as Yahoo/Google Finance was the true driver of success for these models.
        + Verifying future information was not used was impossible, and that it was probable that all models were tainted, as all modelers had been exposed to the test set.
    + The IJCNN 2011 Social Network Challenge
        + The social network in question was Flickr and then to de-anonymize the majority of the data. This allowed them to use edges available from the on-line Flickr network to correctly predict over $60\%$ of edges which were identified, while the rest had to be handled classically using legitimate prediction.


## Formulation

### Preliminaries and Legitimacy



### Leaking Feature



### Leakage in Training Examples



### Discussion



## Avoidance

### Methodology



### External Leakage in Comparisons



## Detection



## (Not) Fixing Leakage



## Conclusion


