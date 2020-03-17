# Explaining Odds Ratios

Author: Magdalena Szumilas

[Origin](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/)


## What is an odds ratio?

+ Def: A measure of association btw an exposure and an outcome
+ the odds that an outcome will occur given a particular exposure, compared to the odds of the outcome occurring in the absence of that exposure
+ most commonly used in case0control studies
+ able to be used in cross-sectional and cohort study designs
+ logistic regression
  + the regression coefficient (b1): the estimated increase in the log odds of the _outcome per unit increase_ in the value of the _exposure_
  + $\exp{b1}$: the odd ratio associated w/ a one-unit increase in the exposure


## When is it used?

+ used to compare 
  + the relative odds of the occurrence of the outcome of interest, eg, disease or disorder
  + given exposure to the variable of interest, eg, health characteristics, aspect of medical history
+ used to determine whether a particular exposure is a risk factor for a particular outcome, and to compare the magnitude of various risk factors for that outcome
  + $OR = 1$: exposure not affecting odds of outcomes
  + $OR > 1$: exposure associated w/ higher odds of outcome
  + $OR < 1$: exposure associated w/ lower odds of outcome


## What about confidence intervals?

+ confidence interval
  + 95% confidence interval (CI): used to estimate the precision of the OR
  + large CI  $\implies$ low level of precision of the OR
  + small CI $\implies$ higher precision of the OR
  + 95% CI not measuring statistical significance
  + used as a proxy for the presence of statistical significance if not overlap the null value (eg, $OR=1$)
  + inappropriate to interpret OR w/ 95% CI that spans the null value as indicating  evidence for lack of association btw the exposure and outcome


## Confounding

+ confounding
  + Def: non-casual association observed btw a given exposure $\implies$ outcome as a result of the influence of a third variable
  + confounding variable:
    + the third variable
    + causally associated w/ the outcome of interest
    + non-causally or causally associated w/ the exposure
    + not an intermediate variable in the causal pathway btw exposure and outcome
  + methods to address confounding
    + stratification
    + multiple regression

## Example

+ Example
  + task: calculating (a) ORs and (b) 95% CIs
  + [Greenfield et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2583916/):
    + prior suicidal adolescents: $n = 263$
    + using logistic regression to analyze the associations btw baseline variables, such as age, sex, presence of psychiatric disorder, previous hospitalizations, and drug and alcohol use w/ suicidal behavior at 6-month follow-up
  + Calculating ORs

    <table style="font-family: arial,helvetica,sans-serif;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center" width=80%>
      <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Two-by-two frequency table</a></caption>
      <thead>
      <tr>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;"></th>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Outcome Status</th>
      </tr>
      <tr>
        <th colspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;"></th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">+</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">-</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td rowspan="2">Exposure status</td> <td style="text-align: center;">+</td> <td style="text-align: center;">a</td> <td style="text-align: center;">b</td> </tr>
      <tr> <td style="text-align: center;">-</td> <td style="text-align: center;">c</td> <td style="text-align: center;">d</td> </tr>
      </tbody>
    </table>

    + a = number of exposed cases
    + b = number of exposed non-cases
    + c = number of unexposed cases
    + d = number of unexposed non-cases

    \[\begin{align*}
      OR &= \frac{a/c}{b/d} = \frac{ad}{bc} \\\\
      OR &= \frac{(n) \text{ exposed cases } / (n) \text{ unexposed cases }}{(n) \text{ exposed non-cases }/(n) \text{ unexpected non-cases }} \\\\
         &= \frac{(n) \text{ exposed non-cases } \times (n) \text{ unexpected non-cases}}{(n) \text{ exposed cases } \times (n) \text{ unexposed cases }}
    \end{align*}\]

  + numerical study
    + Survey result:
      + 186 of the 263 adolescents previously judged as having experienced a suicidal behavior requiring immediate psychiatric consultation did not exhibit suicidal behavior (non-suicidal, NS) at six months follow-up.
      + Of this group, 86 young people had been assessed as having depression at baseline.
      + Of the 77 young people with persistent suicidal behavior at follow-up (suicidal behavior, SB), 45 had been assessed as having depression at baseline.
    + determining the number for a, b, c, and d:
      + a: number of exposed cases (+ +) = youth w/ persistent SB as having depression at baseline = 45
      + b: number of exposed non-cases (+ -) = youth w/o SB at follow up assessed as having depression at baseline = 86
      + c: number of unexposed cases (- +) = youth w/ persistent SB not assessed as having depression at baseline = 77 - 45 = 32
      + d: number of unexposed-non-cases (- -) = youth w/o SB at follow-up not assessed as having depression at baseline = 186 - 86 = 100
    + the odds of persistent suicidal behavior

      \[ OR = \frac{1/c}{b/d} = \frac{ad}{bc} = \frac{45/32}{86/100} = 1.63 \]

    + calculating 95% confidence intervals

      \[\begin{align*}
        95\% CI &= \exp \left( \in(OR) \pm 1.96 \cdot \sqrt{1/a + 1/b + 1/c + 1/d} \right) \\\\
          & = \exp \left( \in(OR) \pm 1.96 \cdot \sqrt{1/45 + 1/86 + 1/32 + 1/100} \right) = (0.96, 2.80)
      \end{align*}\]

      + $1.0 \in (0.96, 2.80) \implies$ the increased odds (OR = 1.63) of persistent suicidal behavior among adolescents w/ depression at baseline does not reach statistical significance
      + with [reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2583916/) in Table 1, $p = 0.07$
    + the odds of persistent suicidal behavior in this group given presence of borderline personality disorder at baseline was twice that of depression (OR = 3.8, 95% CI = (1.6, 8.7)) and was statistically significant ($p = 0.002$)
  + important points from example
    + presence of a positive OR for an outcome given a particular exposure does not necessarily indicate that this association is statistically significant $\implies$ determined by the confidence intervals and $p$-value
    + overall, depression is strongly linked to suicidal and suicidal attempt w/ a particular size and composition, and in the presence of other variables, the association may not be significant

