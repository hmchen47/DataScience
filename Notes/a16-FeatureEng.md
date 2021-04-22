# Beyond One-Hot

Title: Beyond One-Hot: An Explanation of Categorical Variables

Author: Will McGinnis

Date: Nov. 29, 2015

[Original](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)


## Introduction

+ Categorical variables
  + data represented a fixed number of possible values
  + value assigned to one of the finite groups
  + ordinal variables: ordering
  + ML algorithms preferring numbers, not strings

+ Concept of dimensionality
  + simple definition: the number of columns in the dataset
  + significant downstream effects on the eventual models
  + curse of dimensionality: probably models stop working properly in high dimensions
  + dataset w/ more dimensions requiring more parameters of the model to understand $\implies$ more rows to reliably learn those parameters
  + fixed number of rows, additional of extra dimensions w/o adding more info for the models $\to$ detrimental effect on the eventual model accuracy

+ Categorical variables and dimensionality
  + conflict: coding categorical variables and dimensionality problem
  + solution
    + ordinal coding: assigning an integer to each category
    + not adding any dimensions
    + implying an order to the variable probably not existed


## Methodology

+ Process overview
  + gathering a dataset for a classification problem w/ categorical variables
  + using some method of coding to convert the X dataset into numeric values
  + using scikit-learn's cross-validation-score and a BernoulliNB() classifier to generate scores for the dataset, repeated 10 times
  + storing the dimensionality of the dataset, mean score, and time to code the data and generate the score

+ UCI dataset repositories
  + [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
  + [Mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom)
  + [Splice Junctions](http://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/)

+ Encoding methods used
  + __Ordinal__: as described above
  + __One-Hot__: one column per category, with a 1 or 0 in each cell for if the row contained that column's category
  + __Binary__: first the categories are encoded as ordinal, then those integers are converted into binary code, then the digits from that binary string are split into separate columns.  This encodes the data in fewer dimensions that one-hot, but with some distortion of the distances.
  + __Sum__: compares the mean of the dependent variable for a given level to the overall mean of the dependent variable over all the levels. That is, it uses contrasts between each of the first k-1 levels and level k In this example, level 1 is compared to all the others, level 2 to all the others, and level 3 to all the others.
  + __Polynomial__: The coefficients taken on by polynomial coding for k=4 levels are the linear, quadratic, and cubic trends in the categorical variable. The categorical variable here is assumed to be represented by an underlying, equally spaced numeric variable. Therefore, this type of encoding is used only for ordered categorical variables with equal spacing.
  + __Backward Difference__: the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable.
  + __Helmert__: The mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels. Hence, the name ‘reverse’ being sometimes applied to differentiate from forward Helmert coding.


## Results

  <table align=center  border="1">
    <caption style="font-size: 1.2em; margin: 0.2em;">Mushrooms</caption>
    <thead>
    <tr style="text-align: left;"><th></th><th>Coding</th><th>Dataset</th><th>Dimensionality</th><th>Avg. Score</th><th>Elapsed Time</th></tr>
    </thead>
    <tbody>
      <tr><th>0</th><td>Ordinal</td><td>Mushroom</td><td>22</td><td>0.810919</td><td>3.653194</td></tr>
      <tr><th>1</th><td>One-Hot Encoded</td><td>Mushroom</td><td>117</td><td>0.813252</td><td>8.193983</td></tr>
      <tr><th>6</th><td>Helmert Coding</td><td>Mushroom</td><td>117</td><td>0.837997</td><td>5.431131</td></tr>
      <tr><th>5</th><td>Backward Difference Coding</td><td>Mushroom</td><td>117</td><td>0.846864</td><td>7.829706</td></tr>
      <tr><th>3</th><td>Sum Coding</td><td>Mushroom</td><td>117</td><td>0.850555</td><td>4.929640</td></tr>
      <tr><th>4</th><td>Polynomial Coding</td><td>Mushroom</td><td>117</td><td>0.855596</td><td>6.136916</td></tr>
      <tr><th>2</th><td>Binary Encoded</td><td>Mushroom</td><td>43</td><td>0.871493</td><td>3.948484</td></tr>
    </tbody>
  </table>

  <table align=center  border="1">
    <caption style="font-size: 1.2em; margin: 0.2em;">Cars</caption>
    <thead>
    <tr style="text-align: left;"><th></th><th>Coding</th><th>Dataset</th><th>Dimensionality</th><th>Avg. Score</th><th>Elapsed Time</th></tr>
    </thead>
    <tbody>
      <tr><th>10</th><td>Sum Coding</td><td>Cars</td><td>21</td><td>0.549347</td><td>1.456738</td></tr>
      <tr><th>13</th><td>Helmert Coding</td><td>Cars</td><td>21</td><td>0.577471</td><td>1.458556</td></tr>
      <tr><th>7</th><td>Ordinal</td><td>Cars</td><td>6</td><td>0.638522</td><td>1.466667</td></tr>
      <tr><th>8</th><td>One-Hot Encoded</td><td>Cars</td><td>21</td><td>0.648694</td><td>1.393966</td></tr>
      <tr><th>11</th><td>Polynomial Coding</td><td>Cars</td><td>21</td><td>0.666130</td><td>1.495264</td></tr>
      <tr><th>12</th><td>Backward Difference Coding</td><td>Cars</td><td>21</td><td>0.697274</td><td>1.499557</td></tr>
      <tr><th>9</th><td>Binary Encoded</td><td>Cars</td><td>12</td><td>0.697911</td><td>1.441609</td></tr>
    </tbody>
  </table>

  <table align=center  border="1">
    <caption style="font-size: 1.2em; margin: 0.2em;">Splice</caption>
  <thead>
  <tr style="text-align: left;"><th></th><th>Coding</th><th>Dataset</th><th>Dimensionality</th><th>Avg. Score</th><th>Elapsed Time</th></tr>
  </thead>
  <tbody>
  <tr><th>14</th><td>Ordinal</td><td>Splice</td><td>61</td><td>0.681816</td><td>5.107389</td></tr>
  <tr><th>17</th><td>Sum Coding</td><td>Splice</td><td>3465</td><td>0.922276</td><td>25.898854</td></tr>
  <tr><th>16</th><td>Binary Encoded</td><td>Splice</td><td>134</td><td>0.935744</td><td>3.352499</td></tr>
  <tr><th>15</th><td>One-Hot Encoded</td><td>Splice</td><td>3465</td><td>0.944839</td><td>2.563578</td></tr>
  </tbody>
  </table>


## Conclusion

+ Conclusion
  + binary coding: consistently performing well w/o significantly increasing dimensionality
  + ordinal coding: worse cases
  + [GitHub source codes for Pandas](https://github.com/wdm0006/categorical_encoding)


# Beyond One-Hot

Title: Beyond One-Hot. 17 Ways of Transforming Categorical Features Into Numeric Features

Author: S. Mazzanti

Date: Dec. 18, 2020

[Original](https://tinyurl.com/aavmpdam)

## Introduction

+ Categorical encoding
  + the process of transforming a categorical column into one (or more) numerical column(s)
  + Python library: [category_encoders](https://github.com/scikit-learn-contrib/category_encoders)

    ```python
    !pip install category_encoders
    
    import category_encoders as ce

    ce.OrdinalEncoder().fit.transform(x)
    ```

## Not All Encodings Created Equal

<figure style="margin: 0.5em; text-align: center;">
  <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
    onclick= "window.open('https://tinyurl.com/aavmpdam')"
    src    = "https://tinyurl.com/38swc7jm"
    alt    = "Classification of encoding methods fpr categorical variables"
    title  = "Classification of encoding methods fpr categorical variables"
  />
</figure>

+ Classification criteria
  + supervised/unsupervised:
    + unsupervised: solely based on the categorical column
    + supervised: based on some function of the original column and a second (numeric) column
  + output dimension
    + output dimension = 1
    + output dimension > 1
  + mapping:
    + unique: always w/ the same output for each level
    + not unique: same level allowed to have different possible outputs


## 17 categorical encoding algorithms

+ OrdinalEncoder
  + each level mapped to an integer, from 1 to L (L is the number of levels)
  + using alphabetical order, but any other custom order is acceptable
  + only a representation of convenience
  + used to save memory, or as intermediate step for other types of encoding

  ```python
  sorted_x = sorted(set(x))
  ordinal_encoding = x.replace(dict(zip(sorted_x, range(1, len(sorted_x) + 1))))
  ```



