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

+ Notations:
    + ${\mathcal W} = ({\mathcal X}, {\mathcal Y})$: multivariate random process
    + $\mathcal Y$: the outcome or target generating process
    + $y$: sample target instances
    + $\mathcal X$: the feature-vector generating process
    + $X$: a feature-vector instance
    + $\bf X$: a feature-vector realization
    + ${\mathcal x} \in {\mathcal X}$: individual feature generating processes
    + $x \in X$: instances
    + $\bf x \in X$: realizations
    + $\bf W_tr$: the training samples, a separate group of samples
    + $u, v$: random variables

+ ${\mathcal W}$-related instances: specific instances $x_0$ and $y_0$ taken from the same instance of ${\mathcal W}$

+ Modeler's goal: statistically infer a target instance from its associated feature-vector instance in $\mathcal W$ and from a separate group of samples of $\mathcal W$

+ Solution: $\hat y = \mathbb M (X, \bf W_tr)$

+ Model's observational inputs for predicting $y$: $X$ and $\bf W_th$

+ Models containing leaks are a subclass of the broader concept of illegitimate or unacceptable models. At this level, __legitimacy__, which is a key concept in our formulation of leakage, is completely abstract. Every modeling problem sets its own rules for what constitutes a legitimate or acceptable solution and different problems, even if using the same data, may have wildly different views on legitimacy.

+ Leakage: a specific form of illegitimacy that is an intrinsic property of the observational inputs of a model

+ $v \in legit\{u\}$: $v$ is $u$-legitimate if $v$ is observable to the client for the purpose of inferring $u$

+ The trivial legitimacy rule: the target itself must never be used for inference: Cond.(1)

    $$y \notin legit\{y\}$$

+ A model contains leaks w.r.t. a target instance if one or more of its observational inputs are $y$-illegitimate. The model inherits the illegitimacy property from the _features_ and _training examples_ it uses.


### Leaking Feature

+ Extend abstract definition of legitimacy to the case of random processes:
    + ${\mathcal u, v}$: some random processes
    + $\mathcal v \in legit\{u\}$: $\mathcal v$ is $\mathcal u$-legitimate; for every pair of instances of $\mathcal u$ and $\mathcal v$, $\mathcal u$ and $\mathcal v$ respectively, which are $\mathcal W$-related
    + Leaking features covered by a simple condition for the absent of leakage: Cond.(2)

        $$\forall x \in {\mathcal X}, x \in legit\{{\mathcal Y}\}$$

    + Any feature made available by the data preparation process is deemed legitimate by the precise formulation of the modeling problem at hand, instance by instance w.r.t. its matching target.

+ No-time-machine requirement
    + Implicitly required that a legitimate model only build on features with information from a time earlier (or sometimes, no later) than that of the target
    + $\mathcal x, y$: random processes over some time axis (not necessarily physical time)
    + Prediction is required by the client for the target process $y$ at times $t_y$, and their $\mathcal W$-related feature process $x$ is observable to the client at times $t_x$.
    + Rule: Cond.(3)

        $$ legit\{y\} \subseteq \{ x \in {\mathcal X} | t_x < t_y\}$$

        + Any legitimate feature w.r.t. the target process is a member of the right hand side set of features.
        + The right hand side is the set of all features whose every instance is observed earlier than its $\mathcal W$-related target instance.
        + Assume that $\mathcal X$ contains all possible features
        + "$\subseteq$": additional legitimacy constraints might be also apply (otherwise "$=$" could be used)

+ Extension: 
    + require features to be observable a sufficient period of time prior to $t_y$ as in (4) below in order to preclude any information that is an immediate trigger of the target.
    + Sometimes it is too limiting to think of the target as pertaining to a point-in-time, only to a rough interval.
    + Example: "heavy spender" from "Ten supplementary analyses to improve e-commerce web sites"
        + With legitimacy defined as (3) (or as (4) when $\tau = 0$) a model may be built that uses the purchase of a diamond to conclude that the customer is a big spender but with sufficiently large this is not allowed.
        + This transforms the problem from identification of “heavy spenders” to the suggested identification of “migrators”.
    + Cond.(4)

        $$legit\{y\} \subseteq \{x \in {\mathcal X} | t_x < t_y - \tau\}$$

+ Memory limitation
    + A model may not use information older than a time relative to that of the target: Cond.(5)

        $$legit\{y\} \subseteq \{x \in {\mathcal X} | t_y - \tau < t_x < t_y\}$$

    + A requirement to use exactly $n$ features from a specified pool $\mathcal X_p$ of preselected features: Cond.(6)

        $$legit\{y\} \subseteq \{[ x_1, \ldots, x_n] | \forall k, x_k \in {\mathcal X_p}, k \text{ unique}\}$$

    + Variant: only the features selected for a specific provided dataset are considered legitimate.
    + Sometimes allow free use of the entire set: Cond.(7)

        $$legit\{y\} \subseteq {\mathcal X_p}$$

    + Combining (7) with (3): Cond.(8)

        $$legit\{y\} \subseteq \{x \in {\mathcal X_p} | t_x < t_y \}$$

+ Most documented cases of leakage mentioned in literature are covered by condition (2) in conjunction with a no-time-machine requirement as in (3).
    + In the trivial example of predicting rainy days, the target is an illegitimate feature since its value is not observable to the client when the prediction is re-quired (say, the previous day).
    + The pneumonia detection database in the INFORMS 2008 challenge implies that a certain combination of missing diagnosis code and some other features is highly informative of the target.
    + Conditions (2) and (3) similarly apply to the account number and interviewer name examples, the session length, the immediate and indirect triggers, and the web-site based features used by IBM


### Leakage in Training Examples

+ Trying to predict the level of a white noise process $\mathcal y_t$ for $t = [101, 102, \ldots, 200]$, clearly a hopeless task.

+ For the purpose of predicting $\mathcal y_t$, itself is a legitimate feature but otherwise, as in (3), only past information is deemed legitimate – so obviously we cannot cheat. 

+ Consider a model trained on examples $\bf W_t$ taken from $t = [1, 2, \ldots, 200]$. The proposed model is ${\hat y}_t = \mathbb{M}(t, {\bf W_{tr}})$, a table containing for each $\bf t$ the target's realized value $\bf y_t$. Strictly speaking, the only feature used by this model, $\mathcal t$, is legitimate.

+ Adding to (2) the following condition for the absence of leakage: For all $y \in Y_{ev}$, [Cond.(9)]

    $$\forall X \in X_{tr}, X \in legit\{y\} \wedge \forall \tilde{y} \in Y_{tr}, \tilde{y} \in legit\{y\}$$

    + $Y_{ev}$: the set of evaluation target instances
    + $Y_{tr}, X_{tr}$: the sets of training targets and feature-vectors
    + $\bf W_{tr}$: the set of training examples formed with realizations of $Y_{tr}, X_{tr}$

+ Interpretation: the information presented for training as constant features embedded into the model, and added to every feature-vector instance the model is called to generate a prediction for.

+ For modeling problems where the usual i.i.d. instances assumption is valid, and when without loss of generality considering all information specific to the instance being predicted as features rather than examples, condition (9) simply reduces to condition (2) since irrelevant observations can always be considered legitimate.

+ When dealing with problems exhibiting non-stationarity, a.k.a. concept-drift, and more specifically the case when samples of the target (or, within a Bayesian framework, the target/feature) are not mutually independent, condition (9) cannot be reduced to condition (2).

+ Example: Available information about the number of reviews given to a group of titles for the "who reviewed what" task is not statistically independent of the number of reviews given to the second group of titles which is the target in the “how many ratings” task.

+ Without proper conditioning on these shared ancestors we have potential dependence, and because most of these ancestors are unobservable, and difficult to find observable proxies for, dependence is bound to occur.

### Discussion




## Avoidance

### Methodology



### External Leakage in Comparisons



## Detection



## (Not) Fixing Leakage



## Conclusion


