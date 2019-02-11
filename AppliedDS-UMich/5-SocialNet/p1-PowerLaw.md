# Chapter 18. Power Laws and Rich-Get-Richer Phenomena

[Original Chapter](http://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch18.pdf)

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

+ Power laws arise from the feedback introduced by correlated decisions across a population.

+ Simple Model
    + Assume simply that people have a tendency to copy the decisions of people who act before them.
    + The creation of links among Web pages
        1. Pages are created in order, and named $1, 2, 3, \ldots ,N$.
        2. When page $j$ is created, it produces a link to an earlier Web page according to the following probabilistic rule, $p \in [0, 1]$.
            1. With probability $p$, page $j$ chooses a page $i$ uniformly at random from among all earlier pages, and creates a link to this page $i$.
            2. With probability $1−p$, page $j$ instead chooses a page $i$ uniformly at random from among all earlier pages, and creates a link to the page that i points to.
            3. This describes the creation of a single link from page $j$; one can repeat this process to create multiple, independently generated links from page $j$. (However, to keep things simple, we will suppose that each page creates just one outbound link.)
    + The fraction of pages with $k$ in-links will be distributed approximately according to a power law $1/k^c$, where the value of the exponent $c$ depends on the choice of $p$.
    + as $p$ gets smaller, so that copying becomes more frequent, the exponent $c$ gets smaller as well, making one more likely to see extremely popular pages.

+ Rich-get-richer Model
    + the copying mechanism in (2.2) is really an implementation of the  “rich-get-richer” dynamics
    + when copying the decision of a random earlier page, the probability linking to some page $l$ is directly proportional to the total number of pages that currently link to $l$.
    + “Rich-get-richer” rule: The equivalent way to write copying process
        > With probability $1 − p$, page $j$ chooses a page $l$ with probability proportional to $l$’s current number of in-links, and creates a link to $l$.
    + __Preferential attachment__: the probability that page $l$ experiences an increase in popularity is directly proportional to $l$’s current popularity
    + Links formed “preferentially” to pages that already have high popularity.
    + essentially,
    + The more well-known someone is, the more likely you are to hear their name come up in conversation, and hence the more likely you are to end up knowing about them as well.


## The Unpredictability of Rich-Get-Richer Effects

+ The initial stages of its rise to popularity is a relatively fragile thing.

+ The rich-get-richer dynamics of popularity are likely to push it even higher; but getting this rich-get-richer process ignited in the first place seems like a precarious process, full of potential accidents and near-misses.

+ Sensitivity to unpredictable initial fluctuations
    + Information cascades can depend on the outcome of a small number of initial decisions in the population, and a worse technology can win because it reaches a certain critical audience size before its competitors do.
    + The dynamics of popularity suggest that random effects early in the process should play a role here as well.

+ Salgankik, Dodds, and Watts experiment:
    + a music download site, populated with 48 obscure songs of varying quality written by actual performing groups
    + site with a list of the songs and the opportunity to listen to them
    + visitors assigned at random to one of eight “parallel” copies of the site
    + Conclusion: the “market share” of the different songs varied considerably across the different parallel copies, although the best songs never ended up at the bottom and the worst songs never ended up at the top.
    + overall, feedback produced greater inequality in outcomes.
    + the future success of a book, movie, celebrity, or Web site is strongly influenced by these types of feedback effects, and hence may to some extent be inherently unpredictable.

+ Closer Relationships between Power Laws and Information Cascades?
    + information cascades: how a population in which people were aware of earlier decisions made between two alternatives could end up in a cascade, even if each person is making an optimal decision given what they’ve observed.
    + The copying model for power laws
        + a model for popularity including choices among many possible options (e.g. all possible Web pages), rather than just two options.
        + involving a set of people who engage in very limited observation of the population: when you create a new Web page, the model assumes you consult the decision of just one other randomly selected person.
        + based on the idea that later people imitate the decisions of earlier people, but not derive this imitation from a more fundamental model of rational decision-making.



## The Long Tail



## The Effect of Search Tools and Recommendation Systems



## Advanced Material: Analysis of Rich-Get-Richer Processes


