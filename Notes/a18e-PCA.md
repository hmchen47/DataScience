# Principal Components Analysis


Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/principal-component-analysis)

[Local notebook](src/a18e-principal-component-analysis.ipynb)


## Introduction

+ Principal Component Analysis and feature engineering
  + a partitioning of the variation in the data
  + a great tool to help to discover important relationship in the data
  + used to create more informative features
  + typically applied to [standardized](https://bit.ly/2S9yMM2) data
  + variation meaning
    + standardized data: correlation
    + unstandardized data: covariance


## Principal Component Analysis

+ Visualization for Principal Component Analysis
  + axes of variation
    + describing the ways the abalone tend to different from one another
    + axes: perpendicular lines along the natural dimensions of the data
    + each axis for one original feature
  + idea of PCA: instead of describing the data w/ the original features, describing it w/ axes of variation
  + dataset: [Abalone data set](https://www.kaggle.com/rodolfomendes/abalone-dataset)
    + physical measurements taken from several thousand Tasmanian abalone
    + only focusing on `Height` and `Diameter` of their shells
  + axes of variation for abalone
    + Size component
      + the longer axis
      + small height and small diameter (lower left) contrasted w/ large height and large diameter (upper right)
    + Shape component
      + the shorter axis
      + small height and large diameter (flat shape) contrasted w/ large height and small diameter (round shape)

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://www.kaggle.com/ryanholbrook/principal-component-analysis" ismap target="_blank">
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/rr8NCDy.png"
          alt   = "Axes of variation with abalone"
          title = "Axes of variation with abalone"
        >
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/XQlRD1q.png"
          alt   = "The principal components become the new features by a rotation of the dataset in the feature space."
          title = "The principal components become the new features by a rotation of the dataset in the feature space."
        >
      </a>
    </div>




## PCA for Feature Engineering








## Example - 1985 Automobiles








## Exercise








