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


# Beyond One-Hot: 17 Ways

Title: Beyond One-Hot. 17 Ways of Transforming Categorical Features Into Numeric Features

Author: S. Mazzanti

Date: Dec. 18, 2020

[Original](https://bit.ly/3nfRKfI)

[Original Notebook](https://github.com/smazzanti/beyond_one_hot/blob/main/beyond_one_hot.ipynb)

[Local Notebook](./src/a16-beyond_one_hot.ipynb)

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
    onclick= "window.open('https://bit.ly/3nfRKfI')"
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
  + example python code and results

    ```python
    sorted_x = sorted(set(x))
    ordinal_encoding = x.replace(dict(zip(sorted_x, range(1, len(sorted_x) + 1))))
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://miro.medium.com/max/408/1*Q5OWc3QqZYrhIGQUCNC9lg.png"
        alt    = "Result of ordinal encoder"
        title  = "Result of ordinal encoder"
      />
    </figure>

+ CountEncoder
  + each level mapped to the number of observations carrying that level
  + useful as an indicator of the credibility of each level
  + probably automatically decide to take into account the info brought by the level only its count is above some threshold
  + example python code and results

    ```python
    count_encoding = x.replace(x.value_counts().to_dict())
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://miro.medium.com/max/380/1*uuqrb9F2I3oLUQCNrx2l8g.png"
        alt    = "Result of count encoder"
        title  = "Result of count encoder"
      />
    </figure>

+ OneHotEncoder
  + excellent and the most used
  + each level mapped to a dummy column (i.e., a column of 0/1)
  + input w/ a single column $\to$ output consisting of $L$ columns
  + data applied one-hot encoded $\to$ ready for any predictive algorithm
    + taking one observation for each level
    + observing a target variable, $y$, containing the income of each individual
    + fitting a linear regression (OLS)
  + example python code and results

    ```python
    one_hot_encoding = ordinal_encoding.apply(
      lambda oe: pd.Series(np.diag(np.ones(len(set(x))))[oe - 1].astype(int))
    )
    ```

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://bit.ly/3nfRKfI" ismap target="_blank">
        <img style="margin: 0.1em;" height=150
          src   = "https://miro.medium.com/max/873/1*TEb9Qz-tey_3F8QLoOOnEA.png"
          alt   = "Result of one-hot encoder"
          title = "Result of one-hot encoder"
        >
        <img style="margin: 0.1em;" height=150
          src   = "https://miro.medium.com/max/875/1*dGUozlrwR46YBt12KDVNOQ.png"
          alt   = "Result of one-hot encoder and linear regression"
          title = "Result of one-hot encoder and linear regression"
        >
      </a>
    </div>

+ SumEncoder
  + no so important
  + one of contrast encodings
    + designed to have a specific behavior
    + used in regression problems
    + used to get the regression coefficients
  + used to have the regression coefficients w/ zero-sum
  + example python code and results
    + intercept corresponding to the mean of $y$
    + first level, High_School: y = (50 - 35) = 15
    + last level, PhD: y = (68 - 50) = 18
    + exactly the opposite of the sum of the remaining coefficients (-15-5+2 = -18)

    ```python
    sum_encoding = one_hot_encoding.iloc[:, :-1].apply(
      lambda row: row if row.sum() == 1 else row.replace(0, -1)
      , axis = 1
    )
    ```

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://bit.ly/3nfRKfI" ismap target="_blank">
        <img style="margin: 0.1em;" height=140
          src   = "https://miro.medium.com/max/795/1*9v2JtnKBhOX2lCmA-U6zmA.png"
          alt   = "Result of sum encoder"
          title = "Result of sum encoder"
        >
        <img style="margin: 0.1em;" height=140
          src   = "https://miro.medium.com/max/875/1*jbZdQeGzpfAz0Mwq2Lg64g.png"
          alt   = "Result of sum encoder w/ linear regression"
          title = "Result of sum encoder w/ linear regression"
        >
      </a>
    </div>

+ BackwardDifferenceEncoder
  + one of the contrast encodings
  + useful for ordinal variables
  + designed to compare adjacent levels
  + example python code and results
    + example: (left diagram)
      + ordinal variable: education level
      + relation w/ a numeric variable (e.g., income)
      + comparing each couple of consecutive levels w.r.t the target variable
    + example: (right diagram)
      + intercept coincides w/ the mean of $y$
      + coefficient of Bachelor: 10 = (45 - 35)
      + coefficient of Master: 7 = (52 - 45)

  ```python
  backward_difference_encoding = ordinal_encoding.apply(
    lambda oe: pd.Series(
      [i / len(set(x)) for i in range(1, oe)] + 
      [- i / len(set(x)) for i in range(len(set(x)) - oe, 0, -1)]
    )
  )
  ```

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://bit.ly/3nfRKfI" ismap target="_blank">
      <img style="margin: 0.1em;" height=150
        src   = "https://miro.medium.com/max/723/1*110TWbFjMVD13B0ru0WHEA.png"
        alt   = "Result of backward difference encoder"
        title = "Result of backward difference encoder"
      >
      <img style="margin: 0.1em;" height=150
        src   = "https://miro.medium.com/max/800/1*eIVLcadB4yqQPZbQ6oMcVg.png"
        alt   = "Result of backward difference encoder w/ linear regression"
        title = "Result of backward difference encoder w/ linear regression"
      >
    </a>
  </div>

+ HelmertEncoder
  + similar to BackwardDifferenceEncoder
  + each level compared w/ all the previous levels
  + example python code and results
    + PhD coefficient: 24 = 68 - ((35+45+52)/3)

    ```python
    helmert_encoding = ordinal_encoding.apply(
      lambda oe: pd.Series(
        [0] * (oe - 2) + \
        ([oe - 1] if oe > 1 else []) + [-1] * (len(set(x)) - oe)
      )
    ).div(pd.Series(range(2,len(set(x)) + 1)))
    ```

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://bit.ly/3nfRKfI" ismap target="_blank">
        <img style="margin: 0.1em;" height=150
          src   = "https://bit.ly/3xgJj8d"
          alt   = "Result of Helmert encoder"
          title = "Result of Helmert encoder"
        >
        <img style="margin: 0.1em;" height=150
          src   = "https://bit.ly/3tIRrMT"
          alt   = "Result of Helmert encoder w/ linear regression"
          title = "Result of Helmert encoder w/ linear regression"
        >
      </a>
    </div>

+ PolynomialEncoder
  + one of the contrast encodings
  + designed to quantify linear, quadratic and cubic behavior of the target variable w.r.t the categorical variable
  + assumption: underlying categorical variable w/ levels not only ordinal, but also equally spaced
  + using w/ care
  + example python code and results

    ```python
    def do_polynomial_encoding(order):
      # code from https://github.com/pydata/patsy/blob/master/patsy/contrasts.py
      n = len(set(x))
      scores = np.arange(n)
      scores = np.asarray(scores, dtype=float)
      scores -= scores.mean()
      raw_poly = scores.reshape((-1, 1)) ** np.arange(n).reshape((1, -1))
      q, r = np.linalg.qr(raw_poly)
      q *= np.sign(np.diag(r))
      q /= np.sqrt(np.sum(q ** 2, axis=1))
      q = q[:, 1:]
      return q[order - 1]

    polynomial_encoding = ordinal_encoding.apply(lambda oe: pd.Series(do_polynomial_encoding(oe)))
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 25vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3tKUT9t"
        alt    = "Result of polynomial encoder"
        title  = "Result of polynomial encoder"
      />
    </figure>

+ BinaryEncoder
  + basically the same of OrdinalEncoder
  + difference: the integers converted to binary number, then every proportional digit is one-hot encoded
  + output consisting of dummy columns
  + dimensionality reduction w.r.t one-hot
  + example python code and results

    ```python
    binary_base = ordinal_encoding.apply(
      lambda oe: str(bin(oe))[2:].zfill(len(bin(len(set(x)))) - 2)
    )
    binary_encoding = binary_base.apply(lambda bb: pd.Series(list(bb))).astype(int)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 25vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3dFKDK3"
        alt    = "Result of binary encoder"
        title  = "Result of binary encoder"
      />
    </figure>

+ BaseNEncoder
  + a generalization of the BinaryEncoder
  + BinaryEncoder: base 2
  + BaseNEncoder: base n, w/ $n > 1$
  + question: any practical application?
  + example python code and results

    ```python
    def int2base(n, base):
      # return representation of int n in base base
      out = ''
      while n:
          out += str(int(n % base))
          n //= base
      return out[::-1]

    base_n = ordinal_encoding.apply(lambda oe: int2base(n = oe, base = base))
    base_n_encoding = base_n.apply(
      lambda bn: pd.Series(list(bn.zfill(base_n.apply(len).max())))
    ).astype(int)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3dHRUcD"
        alt    = "Result of base N encoder"
        title  = "Result of base N encoder"
      />
    </figure>

+ HashingEncoder
  + hashing each original level w/ some hashing algorithm
  + outcome: converted to integer and the module of that integer w.r.t. some (big) divisor taken
  + mapped each original string to an integer btw 1 and divisor-1
  + one-hot encoder: the integer obtained by this procedure
  + fundamental property of hashing: hashing trick
    + resulting integer uniformly distributed
    + unlikely mapped into the same integer w/ a divisor big enough
  + example: email spam classifier w/ a logistic regression
    + one-hot encoding for all words contained in the database
    + downside:
      + storing the mapping in the separate dictionary
      + model dimensions changing any time w/ new string appeared
    + solution: hashing trick
      + hashing the input
      + no dictionary anymore
      + fixed output dimension, depending on initial divisor
      + new string likely w/ a different encoding than the existing one
  + example python code and results

    ```python
    def do_hash(string, output_dimension):
      hasher = hashlib.new('sha256')
      hasher.update(bytes(string, 'utf-8'))
      string_hashed = hasher.hexdigest()
      string_hashed_int = int(string_hashed, 16)
      string_hashed_int_remainder = string_hashed_int % output_dimension
      return string_hashed, string_hashed_int, string_hashed_int_remainder

    hashing = x.apply(
      lambda string: pd.Series(
        do_hash(string, output_dimension), 
        index = ['x_hashed', 'x_hashed_int', 'x_hashed_int_remainder']
      )
    )
    hashing_encoding = hashing['x_hashed_int_remainder'].apply(
      lambda rem: pd.Series(np.diag(np.ones(output_dimension))[rem])
    ).astype(int)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 50vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/2RObZoU"
        alt    = "Result of hashing encoder"
        title  = "Result of hashing encoder"
      />
    </figure>

+ Mean weight encodings
  + two variables: categorical ($x$) and numeric ($y$)
  + subjective: transform $x$ into a numeric variable
  + idea: taking the mean of $y$ for each level of $x$
  + formula

    \[ enc_i = mean( y| x = i) \]

  + issue of the approach: some group probably too small or too variable to be reliable
  + supervised encoding overcomes by choosing a middle way btw the group mean and the global mean of $y$

    \[ enc_i = w_i \times mean(y|x = i) + (1 -w) \times mean(y) \]

  + $w_i = 0 / 1$ depending on how "credible" the group mean is
  + TargetEncoder, MEstimateEncoder and JamesSteinEncoder: algorithms differ based on how $w_i$ defined

+ TargetEncoder
  + weight depending on the group numerosity and smoothing parameter
  + smoothing = 0: relying solely on group means
  + smoothing $\nearrow\; \to$ global mean weight $\nearrow\;\to$ stronger regularization
  + example python code and results

    ```python
    y_mean = y.mean()
    y_level_mean = x.replace(y.groupby(x).mean())
    weight = 1 / (1 + np.exp(-(count_encoding - 1) / smoothing))
    target_encoding = y_level_mean * weight + y_mean * (1 - weight)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3tKLkHP"
        alt    = "Result of target encoder"
        title  = "Result of target encoder"
      />
    </figure>

+ MEstimateEncoder
  + resembling TargetEncoder
  + $w_i$ depending on a parameter, $m$
    + how much the global mean should weight in absolute terms
    + considered as a number of observations
    + levels exactly $m$ observations $\implies$ the level mean = the overall mean
  + example python code and results

    ```python
    y_mean = y.mean()
    y_level_mean = x.replace(y.groupby(x).mean())
    weight = count_encoding / (count_encoding + m)
    m_estimate_encoding =  y_level_mean * weight + y_grand_mean * (1 - weight)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3eslWQr"
        alt    = "Result of mean estimate encoder"
        title  = "Result of mean estimate encoder"
      />
    </figure>

+ JamesSteinEncoder
  + trying to set parameters statistically grounded
  + the mean of a group w/ high variance should be trusted less
  + advantages
    + providing better estimates than maximum-likelihood estimator
    + no parameter required
  + example python code and results

    ```python
    y_mean = y.mean()
    y_var = y.var()
    y_level_mean = x.replace(y.groupby(x).mean())
    y_level_var = x.replace(y.groupby(x).var())

    weight = 1 - (y_level_var / (y_var + y_level_var) * (len(set(x)) - 3) / (len(set(x)) - 1))
    james_stein_encoding = y_level_mean * weight + y_mean * (1 - weight)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3sKCxnG"
        alt    = "Result of James Stein encoder"
        title  = "Result of James Stein encoder"
      />
    </figure>

+ GLMMEncoder
  + fitting a Linear Mixed Effect Model on $y$
  + [Linear Mixed Effect Models](https://bit.ly/3auWBEj) designed precisely for handling homogeneous groups of observations
  + idea: fitting a model w/ no regressors (only the intercept) and using the levels as groups
  + output: simply the sum of the intercept and the random effect of the group
  + example python code and results

    ```python
    model = smf.mixedlm(formula = 'y ~ 1', data = y.to_frame(), groups = x).fit()
    intercept = model.params['Intercept']
    random_effect = x.replace({k: float(v) for k, v in model.random_effects.items()})
    glmm_encoding = intercept + random_effect
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/32CBH1Q"
        alt    = "Result of global linear mixed encoder"
        title  = "Result of global linear mixed encoder"
      />
    </figure>

+ WOEEncoder
  + Weight of Evidence encoder
  + employed only for binary target variables
  + two distributions behind Weight of Evidence
    + the distribution of 1s (# of 1s in each group / # of 1s in all $y$)
    + the distribution of 0s (# of 0s in each group / # of 0s in all $y$)
  + heart of the algorithm:
    + dividing the distribution of 1s by the distribution of 0s (for each group)
    + higher value $\to$ more skewed toward 0/1
    + taking the logarithm of the value
  + example python code and results

    ```python
    y_level_ones = x.replace(y.groupby(x).apply(lambda l: (l == 1).sum()))
    y_level_zeros = x.replace(y.groupby(x).apply(lambda l: (l == 0).sum()))
    y_ones = (y == 1).sum()
    y_zeros = (y == 0).sum()
    nominator = y_level_ones / y_ones
    denominator = y_level_zeros / y_zeros
    woe_encoder = np.log(nominator / denominator)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3vaI5cY"
        alt    = "Result of weight of evidence encoder"
        title  = "Result of weight of evidence encoder"
      />
    </figure>

+ LeaveOneOutEncoder
  + unique mapping: risk of overfitting
  + overcome unique mapping issue
  + example python code and results

    ```python
    y_level_except_self = x.to_frame().apply(
      lambda row: y[x == row['x']].drop(row.name).to_list(), 
      axis = 1
    )
    leave_one_out_encoding = y_level_except_self.apply(np.mean)
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3neUBFo"
        alt    = "Result of leave one out encoder"
        title  = "Result of leave one out encoder"
      />
    </figure>

+ CatBoostEncoder
  + a gradient boosting algorithm
  + working extremely well in a wide range of problems
  + Ref: [Transforming categorical features to numerical features](https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html)
  + basically like LeaveOneOutEncoder, but following an on-line approach
  + procedure
    + taking a row somewhere in the middle of the table
    + pretending the row above current one observed previously in time
    + the row below yet to be observed
  + example python code and results

    ```python
    y_mean = y.mean()
    y_level_before_self = x.to_frame().apply(
      lambda row: y[(x == row['x']) & (y.index < row.name)].to_list(), 
      axis = 1
    )
    catboost_encoding = y_level_before_self.apply(
      lambda ylbs: (sum(ylbs) + y_mean * a) / (len(ylbs) + a)
    )
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://bit.ly/3nfRKfI')"
        src    = "https://bit.ly/3aAThrl"
        alt    = "Result of categorical boosting encoder"
        title  = "Result of categorical boosting encoder"
      />
    </figure>



