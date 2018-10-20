# Practical Guide to Controlled Experiments on the Web: Listen to Your Customers not to the HiPPO

Authors: Ron Kohavi, Randal M. Henne, Dan Sommerfield


## Introduction

+ Controlled experiments 
    + building the appropriate infrastructure can accelerate innovation
    + Stefan Thomke: Experimentation Matters
    + a.k.a. randomized experiments (single-factor or factorial designs), A/B tests (and their generalizations), split tests, Control/Treatment, and parallel flights
    + Control = the "existing" version <br/> Treatment = a new version being evaluated
    + a methodology to reliably evaluate ideas


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



### Terminology


    
### Hypothesis Testing and Sample Size


    
### Extensions for Online Settings


    
### Limitations


    
## Implementation Architecture



### Randomization Algorithm


    
### Assignment Method


    
## Lesson Learned



### Analysis


    
### Trust and Execution


    
### Culture and Business


    
## Summary







