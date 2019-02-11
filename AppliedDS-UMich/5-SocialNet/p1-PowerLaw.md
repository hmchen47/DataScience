# Chapter 18. Power Laws and Rich-Get-Richer Phenomena


## Popularity as a Network Phenomenon

+ __Popularity__
    + a phenomenon characterized by extreme imbalances
    + Extreme imbelances: while almost everyone goes through life known only to people in their immediate social circles, a few people achieve wider visibility, and a very, very few attain global name recognition

+ Using the number of in-links to a Web page as a measure of the page’s popularity

+ Web page popularity question: 

    > As a function of k, what fraction of pages on the Web have k in-links?

+ A Simple Hypothesis: The Normal Distribution
    + normal, or Gaussian, distribution — the so-called “bell curve” — used widely throughout probability and statistics.
    + the __Central Limit Theorem__: any sequence of small independent random quantities, then in the limit their sum (or average) will be distributed according to the normal distribution.

+ Web pages
    + model the link structure of the Web
    + each page decides independently at random whether to link to any other given page
    + the number of in-links to a given page is the sum of many independent random quantities (i.e. the presence or absence of a link from each other page)
    + expect it to be normally distributed


## Power Laws

+ The fraction of Web pages that have k in-links is approximately proportional to $1/k^2$.  More precisely, the exponent on k is generally a number slightly larger than $2$.)

+ $1/k^2$ decreases much more slowly as $k$ increases, so pages with very large numbers of in-links are much more common than we’d expect with a normal distribution.

+ Power Law
    + A function that decreases as k to some fixed power
    + E.g., $1/k^2$ in the present case
    + when used to measure the fraction of items having value $k$, it says, qualitatively, that it’s possible to see very large values of $k$.

+ Examples of Power Law
    + the fraction of telephone numbers that receive $k$ calls per day is roughly proportional to $1/k2^$;
    + the fraction of books that are bought by $k$ people is roughly proportional to $1/k^3$; 
    + the fraction of scientific papers that receive $k$ citations in total is roughly proportional to $1/k^3$

+ Power laws seem to dominate in cases where the quantity being measured can be viewed as a type of popularity.

+ One of the first things that’s worth doing is to test whether it’s approximately a power law $1/k^c$ for some $c$, and if so, to estimate the exponent $c$.

+ Model of Power Law Distribution
    + $f(k)$: the fraction of items that have value $k$
    + Verify $f(k) = a/k^c$ approximately holds, for some exponent $c$ and constant of proportionality $a$.
    + Thake the logarithms

        $$\log f(k) = \log a - c \log k$$
        + $c$: the slope
        $\log a$: the $y$-intercept
    + The "log-log" plot: a quick way to see if one’s data exhibits an approximate power-law.
    <a href="http://snap.stanford.edu/class/cs224w-readings/broder00bowtie.pdf"> <br/>
        <img src="https://slideplayer.com/slide/13282114/79/images/37/Degree+Distribution+%28by+Broder+et+al.%29.jpg" alt="Fig. show remarkable agreement with each other, and with similar experiments from data that is over two years old [21]. Indeed, in the case of in-degree, the exponent of the power law is consistently around 2.1, a number reported in [5,21]. The anomalous bump at 120 on the x-axis is due to a large clique formed by a single spammer. In all our log–log plots, straight lines are linear regressions for the best power law fit." title="In-degree distributions show a remarkable similarity over two crawls, run in May and October 1999. Each crawl counts well over 1 billion distinct edges of the Web graph" height="350">
    </a>





## Rich-Get-Richer Models



## The Unpredictability of Rich-Get-Richer Effects



## The Long Tail



## The Effect of Search Tools and Recommendation Systems



## Advanced Material: Analysis of Rich-Get-Richer Processes


