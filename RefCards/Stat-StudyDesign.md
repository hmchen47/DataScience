# Statistics: Study Designs

## Overview

+ [Analytic study designs](../Notes/p02-Observational.md#introduction)
  + goal: to identify and evaluate causes or risk factors of diseases or health-related events
  + Types of observational studies
    + case-control and cohort studies
    + cross-section / prevalence studies
  + Temporal design of observational studies (right diagram)

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2998589/" ismap target="_blank">
      <img src="https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=761954&s=23&r=1&c=1" style="margin: 0.1em;" alt="Analytic Study Design" title="Analytic Study Design" width=400>
      <img src="https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=761958&s=23&r=1&c=1" style="margin: 0.1em;" alt="Temporal Design of Observational Studies: Cross-sectional studies are known as prevalence studies and do not have an inherent temporal dimension." title="Temporal Design of Observational Studies: Cross-sectional studies are known as prevalence studies and do not have an inherent temporal dimension." width=300>
    </a>
  </div>

+ [Evidence-based medicine](../Notes/p02-Observational.md#introduction) (EBM)
  + levels and qualifying studies
    + I: High-quality, multicenter or single-center, randomized controlled trial with adequate power; or systematic review of these studies
    + II: Lesser quality, randomized controlled trial; prospective cohort study; or systematic review of these studies
    + III: Retrospective comparative study; case-control study; or systematic review of these studies
    + IV: Case-series
    + V: Expert opinion; case report or clinical example; or evidence based on physiology, bench research, or “first principles”
  + randomized controlled trial (RCT) methodology
  + observational studies
    + providing results similar to RCTs
    + challenging the belief
    + level II and III evidence in EBM
    + recent challenge: vulnerable to influenced by unpredictable confounding factors
    + complement RCTs in hypothesis generation, establishing questions for further RCTs, and defining clinical conditions

+ [Experiment design](../Notes/p01-Bayesian.md#315-design)
  + a natural combination of prediction and decision-making
  + investigator seeking to choose a design to achieve the desired goals
  + technically and computational challenging
  + backwards induction
    + sequential designs to work backward from the end of the study
    + examining all the possible decision points
    + optimize the decision allowing for all the possible circumstances
  + computationally demanding
    + all possible future eventualities
    + approximations


## Cohort Studies

+ [Definition of cohort](../Notes/p02-Observational.md#cohort-study)
  + epidemiology: cohort = a set of people followed over a period of time
  + modern epidemiological definition: a group of people w/ defined characteristics who are followed up to determine incidence of, or mortality from, some specific disease, all causes of death, or some other outcome

### Study Design for Cohort Studies

+ [Cohort study](../Notes/p02-Observational.md#study-design-of-cohort-studies)
  + an outcome or disease-free study population
  + defined by the exposure or event of interest and followed in time until the disease of outcome of interest occurs (diagram)
  + exposure identified before outcome $\implies$ a temporal framework to assess causality and then having the potential to provide the strongest scientific evidence
  + advantages
  + disadvantages

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2998589/" ismap target="_blank">
      <img src="https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=761962&s=23&r=1&c=1" style="margin: 0.1em;" alt="Cohort Study Designs" title="Cohort Study Designs" width=500>
    </a>
  </div>

+ [Prospective cohort studies](../Notes/p02-Observational.md#study-design-of-cohort-studies)
  + carried out from the present into the future
  + designed w/ specific data collection methods
  + advantage: tailored to collect specific exposure data and more complete
  + disadvantage
  + examples: the landmark Framingham Heart study and plastic surgery

+ [Retrospective cohort studies](../Notes/p02-Observational.md#study-design-of-cohort-studies)
  + a.k.a historical cohort studies
  + carried out the present time and looking to the past to examine medical events or outcomes
  + timeliness and inexpensive nature
  + choosing a cohort of subjects based on exposure status at the present time
  + reconstructing measured outcome data (i.e. disease status, event status) for analysis
  + disadvantage
  + advantage: inexpensive, short time period

+ [Case-series](../Notes/p02-Observational.md#study-design-of-cohort-studies)
  + difference w/ cohort studies: the presence of a control, or unexposed, group
  + descriptive studies following one small group of subjects
  + often confused w/ cohort study when only one group of subjects presents
  + clarified unless a second comparative group serving as a control
  + to strengthen an observation from a case-series by selecting appropriate control groups to conduct a cohort or case-control studies


### Methodological Issues for Cohort Studies

+ [Selection of subjects](../Notes/p02-Observational.md#methodological-issues-of-cohort-studies)
  + hallmark: defining the selected group of subjects by exposure status at the start of the investigation
  + characteristic: selecting both the exposed and unexposed groups from the same source population (diagram)
  + excluding subjects not at risk for developing the outcome
  + source population determined by practical considerations, such as sampling

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2998589/" ismap target="_blank">
      <img src="https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=761971&s=23&r=1&c=1" style="margin: 0.1em;" alt="Levels of Subject Selection" title="Levels of Subject Selection" width=500>
    </a>
  </div>

+ [Attrition bias]](../Notes/p02-Observational.md#methodological-issues-of-cohort-studies) (loss to follow-up)
  + long follow-up period $\implies$ minimizing loss of follow-up
  + loss to follow-up: a situation losing contact w/ the subject $\implies$ missing data
  + losing too many subjects $\implies$ the internal validity reduced
  + rule of thumb: the loss to follow-up rate $<20\%$ of the samples

+ [Methods to minimize loss to follow-up](../Notes/p02-Observational.md#methodological-issues-of-cohort-studies)
  + during enrollment
    + exclude subject likely to be lost
    + obtain information to allow future tracking
  + during follow-up: maintain periodic contact


## Cast-Control Studies


### Design Study for Case-Control Studies

+ [Case-control studies](../Notes/p02-Observational.md#study-design-of-cast-control-studies)
  + identifying subjects by outcome at the outset of the investigation
  + outcome of interest: subject undergone a specific type of surgery, experienced a complication, or a diagnosed w/ a disease
  + case-control study design (diagram)
  + suitable for investigating rare outcomes or outcomes w/ long letency period $\impliedby$ subjects selected from the outset by their outcome status
  + advantages
  + disadvantages

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2998589/" ismap target="_blank">
      <img src="https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=761967&s=23&r=1&c=1" style="margin: 0.1em;" alt="Case-Control Study Designs" title="Case-Control Study Designs" width=500>
    </a>
  </div>


### Methodological Issues of Case-Control Studies

+ [Selection of cases](../Notes/p02-Observational.md#methodological-issues-of-case-control-studies)
  + sampling started with
    + explicitly defined inclusion and exclusion criteria
    + criteria ensuring all cases homogeneous
  + selected from various sources
    + validity issues
    + weakening the generalizability of the study findings
  + representative of cases in the target population

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2998589/" ismap target="_blank">
        <img src="https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=761975&s=23&r=1&c=1" style="margin: 0.1em;" alt="Levels of case selection" title="Levels of case selection" width=300>
      </a>
    </div>

+ [Selection of controls](../Notes/p02-Observational.md#methodological-issues-of-case-control-studies)
  + one of the most demanding aspects
  + important principle: the distribution of exposure should be the same among cases and controls
  + same inclusion criteria: the validity depending on the comparability of these two groups

+ [Matching strategy](../Notes/p02-Observational.md#methodological-issues-of-case-control-studies)
  + a method in an attempt to
    + ensure comparability btw cases and controls
    + reduce variability and systematic difference due to background variables not interested
  + individual matching
    + each cases typically individually paired w/ a control subject w.r.t. the background variables
    + exposure to the risk factor of interest compared btw the cases and the controls
  + confounder: variables associated w/ risk factor
  + advantages
  + disadvantages

+ [Multiple controls](../Notes/p02-Observational.md#methodological-issues-of-case-control-studies)
  + rare outcomes: a limited number of cases to select from w/ a large number of controls
  + increasing statistical power $\to$ increasing the sample size
  + improving precision by having about up to 3 or 4 controls per case

+ [Bias in case-control studies](../Notes/p02-Observational.md#methodological-issues-of-case-control-studies)
  + Achilles heel: evaluating exposure status
  + information collected by self-report, interview, or from recorded information
  + susceptible: recall bias, interview bias, or relying on the completeness or accuracy of recorded information
  + bias decreasing the validity $\implies$ carefully addressed and reduced in study design



