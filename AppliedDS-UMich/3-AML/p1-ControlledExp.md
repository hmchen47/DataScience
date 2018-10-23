# Practical Guide to Controlled Experiments on the Web: Listen to Your Customers not to the HiPPO

Authors: Ron Kohavi, Randal M. Henne, Dan Sommerfield


## Introduction

+ Controlled experiments 
    + Building the appropriate infrastructure can accelerate innovation
    + Stefan Thomke: Experimentation Matters
    + a.k.a. randomized experiments (single-factor or factorial designs), A/B tests (and their generalizations), split tests, Control/Treatment, and parallel flights
    + Control = the "existing" version <br/> Treatment = a new version being evaluated
    + A methodology to reliably evaluate ideas


## Motivating Examples

### Checkout Page at Doctor FootCare

+ The _conversion rate_ of an e-commerce site is the percentage of visits to the website that include a purchase.
    
### Ratings of Microsoft Office Help Articles

+ The initial implementation presented users with a Yes/No widget. The team then modified the widget and offered a 5-star ratings.

+ Motivations for the change
    + The 5-star widget provides finer-grained feedback, which might help better evaluate content writers.
    + The 5-star widget improves usability by exposing users to a single feedback box as opposed to two separate pop-ups (one for Yes/No and another for Why).
    
### Results and ROI

+ Doctor FootCare
    + In reality, the site "upgraded" from the A to B and lost 90% of their revenue.
    + By removing the discount code from the new version (B), conversion-rate increased 6.5% relative to the old version (A).

+ Microsoft’s Office Help
    + The number of ratings plummeted by an order of magnitude, thus significantly missing on goal #2 above.
    + Based on additional tests, it turned out that the two-stage model actually helps in increasing the response rate.
    
## Controlled Experiments

+ A/B test: the simplest controlled experiment
    <a href="https://www.betaout.com/blog/ab-testing-to-drive-better-ecommerce-sale/"> <br/>
        <img src="http://www.betaoutcdn.com/inbound/2013/10/image_1_uv3cr.png" alt="Though this is not a standalone example of A/B testing leading to a huge success. The formula for the ecommerce retail giant Amazon’s success is also testing and only testing, 'There is nothing at Amazon which has not been tested.'  Starting from the product page design to the call to action buttons, checkout process to their delivery and return process, everything has gone through multiple split tests. Not only Amazon, most of the marketers & ecommerce sites nowadays, have turned to A/B testing to identify the most performing option for a variable in their site design and retail process for increased leads and better conversion. For sites with a lot of traffic, A/B testing is the quickest way to iterate and improve. Let’s start with the most common A/B testing variables for ecommerce companies." title= "A/B testing" height="200">
    </a>


### Terminology

+ __Overall Evaluation Criterion__ (OEC)
    + A quantitative measure of the experiment’s objective
    + a.k.a. Response or Dependent Variable in statistics, other synonyms include Outcome, Evaluation metric, Performance metric, or Fitness Function
    + A good OEC should not be short-term focused (e.g., clicks) and should include factors that predict long-term goals, such as predicted lifetime value and repeat visits
    + Some ways to measure what customers want: Ulwick, Anthony. What Customers Want: Using Outcome-Driven Innovation to Create Breakthrough Products and Services. McGraw-Hill, 2005.

+ __Factor__
    + A controllable experimental variable that is thought to influence the OEC. 
    + Assigned Values, Levels or Versions, Variables
    + E.g., A/B tests - single factor with two values: A and B

+ __Variant / Treatment__
    + A user experience being tested by assigning levels to the factors; it is either the Control or one of the Treatments.
    + specifically differentiate between the Control, which is a special variant that designates the existing version being compared against and the new Treatments being tried
    + bug: the experiment is aborted and all users should see the Control variant

+ __Experimentation Unit / Item__: The entity on which observations are made

+ __Null Hypothesis / $H_0$__: the OECs for the variants are not different and that any observed differences during the experiment are due to random fluctuations.

+ __Confidence level__: The probability of failing to reject (i.e., retaining) the null hypothesis when it is true.

+ __Power__: The probability of correctly rejecting the null hypothesis, H0, when it is false. Power measures our ability to detect a difference when it indeed exists.

+ __A/A Test__ / __Null Test__: 
    + Assigning users to one of two groups, but expose them to exactly the same experience.
    + Used to 
        + collect data and assess its variability for power calculations, and 
        + test the experimentation system (the Null hypothesis should be rejected about $5\%$ of the time when a $95\%$ confidence level is used).

+ __Standard Deviation (Std-Dev)__: A measure of variability, typically denoted by $\sigma$.

+ __Standard Error (Std-Err)__: For a statistic, it is the standard deviation of the sampling distribution of the sample statistic. For a mean of $n$ independent observations, it is $\hat{\sigma} / \sqrt{n}$ where $\sigma$ is the estimated standard deviation.


### Hypothesis Testing and Sample Size

+ Accept a Treatment as being statistically significantly different if the test rejects the null hypothesis, which is that the OECs are not different.
+ What is important impact the test:
    1. __Confidence level__: Commonly set to $95\%$, this level implies that 5% of the time we will incorrectly conclude that there is a difference when there is none (Type I error).
    2. __Power__: Commonly desired to be around $80-95\%$, although not directly controlled. If the Null Hypothesis is false, i.e., there is a difference in the OECs, the power is the probability of determining that the difference is statistically significant. (A Type II error is one where we retain the Null Hypothesis when it is false.)
    3. __Standard Error__: The smaller the Std-Err, the more powerful the test. There are three useful ways to reduce the Std-Err:
        1. The estimated OEC is typically a mean of large samples. The Std-Err of a mean decreases proportionally to the square root of the sample size, so increasing the sample size, which usually implies running the experiment longer, reduces the Std-Err and hence increases the power.
        2. Use OEC components that have inherently lower variability, i.e., the Std-Dev, $\sigma$, is smaller.
        3. Lower the variability of the OEC by filtering out users who were not exposed to the variants, yet were still included in the OEC.
    4. The effect, or the difference in OECs for the variants. Larger differences are easier to detect, so great ideas will unlikely be missed. Conversely, if Type I or Type II errors are made, they are more likely when the effects are small.

+ Sample Size: <br/> Approximated formula of the desired sample size, assuming the desired confidence level is $95\%$ and the desired power is $90\%$: 

    $𝑛= (4 r \sigma / \Delta)^2$ 
    
    where $n$ is the sample size, $r$ is the number of variants (assumed to be approximately equal in size), $\sigma$ is the std-dev of the OEC, and $\Delta$ is the minimum difference between the OECs. The factor of $4$ may overestimate by $25\%$ for large $n$, but the approximation suffices for the example.

+ Examples: Revenue <br/>
    E-commerce site and $5\%$ of users who visit during the experiment period end up purchasing with about 75: $E(spending) = 75 \cdot 5\%$.  Assume $\sigma = 30$ and run A/B test to detect $5\%$, then $n = (4 \cdot 2 \cdot 30 / (3.75 \cdot 0.05)) > 1,600,000$

+ Example: $5\%$ conversion rate change (3.b) <br/>
    Purchase (conversion event) modeled as a Bernoulli trial with $p=0.05$ being the probability of a purchase.  $\text{Std-Err} = \sqrt{p(1−𝑝)}$.  Therefore, $4 \cdot 2 \cdot \sqrt{0.05⋅(1−0.05)} / (0.05 \cdot 0.05)^2 < 500,000$. 

+ Example: change tot he checkout procedure (3.c) <br/>
    Assume that $10\%$ of users initiate checkout and that $50\%$ of those users complete it. This user segment is more homogenous and hence the OEC has lower variability. <br/>
    Average conversion rate = $0.5$, $\text{std-dev} = 0.5$.  To detect $5\%$ change, $n = 4 \cdot 2 \cdot 0.5 \cdot (1−0.5) / (0.5 \cdot 0.05)^2 ~ 25,600$.  Therefore, the total number is about $256,000$ since $90\%$ users excluded.

+ When running experiments, it is important to decide in advance on the OEC (a planned comparison); otherwise, there is an increased risk of finding what appear to be significant results by chance (familywise type I error). <br/> __Adjustment__: Fisher’s least-significant-difference, Bonferroni adjustment, Duncan’s test, Scheffé’s test, Tukey’s test, and Dunnett’s test



### Extensions for Online Settings

+ Treatment Ramp-up
    + An experiment can be initiated with a small percentage of users assigned to the treatment(s), and then that percentage can be gradually increased.
    + E.g., run an A/B test at $50\%/50\%$; start with a $99.9\%/0.1\%$ split, then rampup the Treatment from $0.1\%$ to $0.5\%$ to $2.5\%$ to $10\%$ to $50\%$.

+ Automation
    + Once an organization has a clear OEC, it can run experiments to optimize certain areas amenable to automated search.
    + the slots on the home page at Amazon are automatically optimized. If decisions have to be made quickly (e.g., headline optimizations for portal sites), these could be made with lower confidence levels because the cost of mistakes is lower. Multi-armed bandit algorithms and Hoeffding Races can be used for such optimizations.

+ Software Migrations
    + Features or system migrates to new backend, new database or new language -> no change on user experience
    + A/B test with goal to retain $H_0$


### Limitations

1. Quantitative Metrics, but No Explanations
2. Short term vs. Long Term Effects
    + Long-term goals should be part of the OEC.
    + latent conversions: a lag from the time a user is exposed to something and take action.
3. Primacy and Newness Effects
    + primacy effect: change the navigation on a web site, experienced users may be less efficient until they get used to the new navigation
    + newness bias: when a new design or feature is introduced, some users will investigate it, click everywhere; associated with the Hawthorne Effect
    + Both primacy and newness concerns imply that some experiments need to be run for multiple weeks.
4. Features Must be Implemented
    + feature may be a prototype that is being tested against a small portion, or may not cover all edge cases
    + feature must be implemented and be of sufficient quality to expose users to it.
    + Jacob Nielsen correctly points out that paper prototyping can be used for qualitative feedback and quick refinements of designs in early stages.
5. Consistency
    + Users may notice they are getting a different variant than their friends and family.
    + Same user will see multiple variants when using different computers (with different cookies).
6. Parallel Experiments
    + strong interactions are rare in practice -> overrated
    + Pairwise statistical tests can also be done to flag such interactions automatically.
7. Launch Events and Media Announcements.

    
## Implementation Architecture

### Randomization Algorithm

+ Properties
    1. Users must be equally likely to see each variant of an experiment (assuming a 50-50 split).
    2. Repeat assignments of a single user must be consistent.
    3. When multiple experiments are run concurrently, there must be no correlation between experiments.
    4. The algorithm should support monotonic ramp-up , meaning that the percentage of users who see a Treatment can be slowly increased without changing the assignments of users who were already previously assigned to that Treatment.

+ Pseudorandom with caching
    + A good pseudorandom number generator will satisfy the first and third requirements of the randomization algorithm.
    + state introduced: the assignments of users must be cached once they visit the site.
    + server side: storing the assignments for users in some form of database; client side: by storing a user’s assignment in a cookie
    + The fourth requirement (monotonic ramp-up) is particularly difficult to implement using this method.

+ Hash and partition
    + Hash and partition
        + completed stateless
        + Each user is assigned a unique identifier, which is maintained either through a database or a cookie.
        + identifier is appended onto the name or id of the experiment.
    + Very sensitive to the choice of hash function.
        + any _funnels_ (instances where adjacent keys map to the same hash code) then the first property (uniform distribution) will be violated.
        + _characteristics_ (instances where a perturbation of the key produces a predictable perturbation of the hash code), then correlations may occur between experiments.
    + satisfy the second requirement (by definition), satisfying the first and third is more difficult
    + only MD5 generated no correlations between experiments
    + SHA256 requiring a five-way interaction to produce a correlation
    + .NET string hashing function failed to pass even a two-way interaction test


### Assignment Method

+ A piece of software that enables the experimenting website to execute a different code path for different users.
+ Traffic splitting
    + A method that involves implementing each variant of an experiment on a different fleet of servers, be it physical or virtual.
    + Embed the randomization algorithm into a load balancer or proxy server to split traffic between the variants.
    + Requiring no changes to existing code to implement an experiment.
+ Server-side selection
    + API calls embedded into the website’s servers invoke the randomization algorithm and enable branching logic that produces a different user experience for each variant.
    + Extremely general method that supports multiple experiments on any aspect of a website, from visual elements to backend algorithms.
+ Client-side selection
    + JavaScript calls embedded into each web page contact a remote service for assignments
    + Dynamically modify the page to produce the appropriate user experience
    + easier to implement than server-side experiments


## Lesson Learned

### Analysis

+ Mine the Data
    + Rich data is typically collected that can be analyzed using machine learning and data mining techniques.
    + Excluding the population from the analysis showed positive results, and once the bug was fixed, the feature was indeed retested and was positive.

+ Speed Matters
    + experiments at Amazon showed a $1\%$ sales decrease for an additional 100 msec
    + experiments at Google showed increased the time to display search results by 500 msecs reduced revenues by $20\%$ (based on a talk by Marissa Mayer at Web 2.0).

+ Test One Factor at a Time (or Not)
    + testing one factor at a time -> too restrict
    + fractional factorial designs and Taguchi methods -> introducing complexity
    + factorial designs allow for joint optimization of factors, and are therefore superior in theory
    + recommendations
        + Conduct single-factor experiments for gaining insights and decoupled with incremental changes
        + Try some bold bets and very different designs.
            + backend algorithms: easier to try
            + Data mining: isolate areas where the new algorithm is significantly better, leading to interesting insights.
        + factorial designs: Limit the factors and the possible values per factor because users will be fragmented (reducing power) and because testing the combinations for launch is hard.

### Trust and Execution

+ Run Continuous A/A Tests
    1. Are users split according to the planned percentages?
    2. Is the data collected matching the system of record?
    3. Are the results showing non-significant results 95% of the time?

+ Automate Ramp-up and Abort
    + experiments ramp-up in the percentages assigned to the Treatment(s).
    + auto-aborted if a treatment is statistically significantly underperforming relative to the Control
    + auto-abort simply reduces the percentage of users assigned to a treatment to zero.
    + Ramp-up is quite easy to do in online environments, yet hard to do in offline studies.

+ Determine the Minimum Sample Size
    + Decide on the statistical power, the effect you would like to detect, and estimate the variability of the OEC through an A/A test. 
    + Minimum sample size needed for the experiment and the running time based on statistical power
    + common mistake: run experiments underpowered

+ Assign $50\%$ of Users to Treatment
    + common practice: run new variants for only a small percentage of users
    + recommend: Treatment ramp-up
    + Maximize the power of an experiment and minimize the running time: $50\%$ of users see each of the variants in an A/B test
    + Assuming all factors are fixed, a good approximation for the multiplicative increase in running time for an A/B test relative to $50\%/50\%$ is $1/(4𝑝(1−𝑝))$ where the treatment receives portion $p$ of the traffic. 
    + E.g., if an experiment is run at $99\%/1\%$, then it will have to run about $25$ times longer than if it ran at $50\%/50\%$

+ Beware of Day of Week Effects
    + running experiments for at least a week or two, then continuing by multiples of a week so that day-of-week effects can be analyzed
    + generalized to other time-related events: holidays and seasons
    + different geographies: US may not work well in France, Germany, or Japan

+ Power calculation
    + $50\%/50\%$: 5 days
    + $95\%/5\%$: 25 days (4 weeks)
    + $99\%/1\%$: 125 days

### Culture and Business

+ Agree on the OEC Upfront
    + controlled experiments: objectively measure the value of new features for the business
    + OECs can be combined measures, which transform multiple objectives, in the form of experimental observations, into a single metric.
    + Weigh the value of various inputs and decide their relative importance.
    + Assess the _lifetime value_ of users and their actions -> OEC
    + E.g., a search from a new user may be worth more than an additional search from an existing user.

+ Beware of Launching Features that "Do Not Hurt" Users
    + no statistically significant difference between variants
        + truely no difference between the variants
        + no sufficient power to detect the change
    + decision based on "no significant difference" results to launch the change anyway "because it does not hurt anything." -> possible negative experiment but underpowered

+ Weigh the Feature Maintenance Costs
    + Maintenance costs overtaken a statistically significant difference between variants
    + Small increase in OEC may not outweight the cost of maintaining the feature

+ Change to a Data-Driven Culture
    + Running a few online experiments can provide great insights into how customers are using a feature.
    + Running frequent experiments and using experimental results as major input to company decisions and product planning can have a dramatic impact on company culture.
    + classical software development: completely designed prior to implementation.
    + web world: integrate customer feedback directly through prototypes and experimentation
    + OEC + experiment -> real data -> attaining shared goals


## Summary







