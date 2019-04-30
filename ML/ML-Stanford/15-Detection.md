# Anomaly Detection

## Density Estimation

### Problem Motivation

#### Lecture Notes

+ Anomaly detection example
	+ Aircraft engine features:
		+ $x_1\;$ = heat generated
		+ $x_2\;$ = vibration intensity
		+ ...
	+ Dataset: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$
	+ New engine: $x_{test}$

	<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
		<div><a href="https://www.ritchieng.com/machine-learning-anomaly-detection/">
			<img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w9_anomaly_recommender/anomaly_detection.png" style="margin: 0.1em;" alt="Scatter plot for the relationship of heat and vibration" title="The relationship of heat vs. vibration" width="350">
		</a></div>
	</div>


+ Density estimation
	+ Dataset: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$
	+ Is $X_{test}$ anomalous?
	+ Given model $p(x)$ to predict $x_{test}$

		$$ \text{Decision } = \begin{cases} \text{anomaly} & \text{if } p(x_{test}) < \epsilon \\ \text{ok} & \text{if }  p(x_{test}) \geq \epsilon  \end{cases}$$

	<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
		<div><a href="https://www.ritchieng.com/machine-learning-anomaly-detection/">
			<img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w9_anomaly_recommender/anomaly_detection2.png" style="margin: 0.1em;" alt="text" title="caption" width="350">
		</a></div>
	</div>

+ Anomaly detection examples
	+ Fraud detection:
		+ $x^{(i)}\;$ = features of user $i$'s activities
		+ Model $p(x)$ from data
		+ Identify unusual users by checking which have $p(x) < \epsilon$
	+ Manufacturing
	+ Monitoring computers in a data center
		+ $x^{(i)}\;$ = features of machine $i$
		+ $x_1\;$ = memory use
		+ $x_2\;$ = number of disk accesses/sec
		+ $x_3\;$ = CPU load
		+ $x_4\;$ = CPU load/network traffic
		+ ...

+ IVQ: Your anomaly detection system flags $x$ as anomalous whenever $p(x) \leq \epsilon$. Suppose your system is flagging too many things as anomalous that are not actually so (similar to supervised learning, these mistakes are called false positives). What should you do?

	1. Try increasing $\epsilon$
	2. Try decreasing $\epsilon$

	Ans: 2


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/16.1-AnomalyDetection-ProblemMotivation-V1.33a3d3f0b22b11e495a62138f9b52d3f/full/360p/index.mp4?Expires=1556755200&Signature=J5r9Pnk3EzfJHNky-MOEJNaAdWYKcaUS3hWu8NSwXJrROgZF3JMkdPJQQK0N4oNttd6MD4ruIwalcfiK5A4s8FyBu0Ykq4noHU5dj3KIwwj4paXXXu88a1UMv19Wh58diXHb2B1BMV7VvAAqgqsGIa9XxpRkm9K04N9TW7RQb~w_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/_BSlVBF8SdGUpVQRfJnR8Q?expiry=1556755200000&hmac=AWMcjAMzac1LNG4ov0IIJjcA6TbbRLhbnAa5HUtwQ7k&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Gaussian Distribution

#### Lecture Notes

+ Gaussian (Normal) Distribution
	+ Say $x \in \mathbb{R}$. If $x$ is a distributed Gaussian with mena $\mu$, variance $\sigma^2$ with $\sigma$ as standard deviation.
	+ Normal distribution: $x \backsim \mathcal{N}(\mu, \sigma^2)$ where `~` means "distributed as"

		$$p(x; \mu, \sigma^2) = \dfrac{1}{\sqrt{2\pi}} \exp \left(- \dfrac{(x - \mu)^2}{2\sigma^2} \right)$$

	<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
		<div><a href="https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/normal-distributions/">
			<img src="https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2013/09/standard-normal-distribution.jpg" style="margin: 0.1em;" alt="One way of figuring out how data are distributed is to plot them in a graph. If the data is evenly distributed, you may come up with a bell curve. A bell curve has a small percentage of the points on both tails and the bigger percentage on the inner part of the curve. In the standard normal model, about 5 percent of your data would fall into the “tails” (colored darker orange in the image below) and 90 percent will be in between. For example, for test scores of students, the normal distribution would show 2.5 percent of students getting very low scores and 2.5 percent getting very high scores. The rest will be in the middle; not too high or too low. The shape of the standard normal distribution looks like this:" title="Standard normal model" width="350">
		</a></div>
	</div>

+ Gaussian distribution example

<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
  <div><a href="https://www.ritchieng.com/machine-learning-anomaly-detection/">
    <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w9_anomaly_recommender/anomaly_detection5.png" style="margin: 0.1em;" alt="Normal distribution with (mean, standard deviation) = (0, 1), (0, 0.5), (0, 2), (3, 0.5)" title="Normal distribution with various means and standard deviations" width="350">
  </a></div>
</div>

+ Parameter estimation
	+ Dataset: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\} \quad x^{(i)} \in \mathbb{R}$
	+ mean ($\mu$): 

		$$\mu = \dfrac{1}{m} \sum_{i=1}^m x^{(i)}$$

	+ Standard deviation($\sigma$): (maximum likelihood estimation form or statistics form)

		$$\sigma^2 = \dfrac{1}{m} \sum_{j=1}^m (x^{(i)} - \mu)^2 \qquad \text{ or } \qquad \sigma^2 = \dfrac{1}{m-1} \sum_{j=1}^{m-1} (x^{(i)} - \mu)^2$$

+ IVQ: The formula for the Gaussian density is:

	$$p(x) = \dfrac{1}{\sqrt{2\pi}\sigma} \exp \left(−\dfrac{(x−\mu)^2}{2\sigma^2} \right)$$

	Which of the following is the formula for the density to the right?

	[A gaussian curve centered at -3 with standard deviation 2]

	1. $p(x) = \dfrac{1}{\sqrt{2\pi} \times 2} \exp(- \dfrac{(x-3)^2}{2 \times 4})$
	2. $p(x) = \dfrac{1}{\sqrt{2\pi} \times 4} \exp(- \dfrac{(x-3)^2}{2 \times 2})$
	3. $p(x) = \dfrac{1}{\sqrt{2\pi} \times 2} \exp(- \dfrac{(x+3)^2}{2 \times 4})$
	4. $p(x) = \dfrac{1}{\sqrt{2\pi} \times 4} \exp(- \dfrac{(x+3)^2}{2 \times 2})$

	Ans: 3


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/16.2-AnomalyDetection-GaussianDistribution.bfc64f20b22b11e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1556755200&Signature=ie7FEzBFcOhhQBmM04zWqqU~IQ4U4jUM0cjjAcEhQsqmkvAT9rjy9kEsqtnVNN4Kk-CgPgrSHPseoJ072DHwTbFUwEoEP7FVb3T205PETz2xhtxQpnvvZHYPT8ZEsNTaryKQ44dUyVkYkyYv~bUFDErBWBNA2bnEsCRg3kx7YMM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/KoYUOLI2SuaGFDiyNormlg?expiry=1556755200000&hmac=J8QoEq5dq1yOtqMFUh_dnjytShkFUDlvJHDHuiyZ6kY&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Algorithm

#### Lecture Notes

+ Density estimation
	+ Training set: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$
	+ Each example is $x \in \mathbb{R}^n$ and independent
	+ Gaussian distribution for each feature: $x_i \backsim \mathcal{N}(\mu_i, \sigma_i^2) \quad \forall i = 1, 2, \dots, n$
	+ the probability density

		$$\begin{array}{rcl} p(x) & =& p(x_1,; \mu_1, \sigma_1^2)p(x_2,; \mu_2, \sigma_2^2)p(x_3,; \mu_3, \sigma_3^2) \dots p(x_n,; \mu_n, \sigma_n^2) \\ &=& \displaystyle \prod_{j=1}^n p(x_j,; \mu_j, \sigma_j^2) \end{array}$$

	+ IVQ: Given a training set $\{x^{(1)}, \dots, x^{(m)}\}$, how would you estimate each $\mu_j$ and $\sigma_j^2$ (Note $\mu_j \in \mathbb{R}, \sigma_j^2 \in \mathbb{R}$.)

		1. $\displaystyle \mu_j = \frac{1}{m}\sum_{i=1}^m x^{(i)},\ \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m(x^{(i)} - \mu)^2$
		2. $\displaystyle \mu_j = \frac{1}{m}\sum_{i=1}^m (x_j^{(i)})^2,\ \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m(x_j^{(i)} - \mu_j)^2$
		3. $\displaystyle \mu_j = \frac{1}{m}\sum_{i=1}^m x_j^{(i)},\ \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m(x^{(i)} - \mu)^2$
		4. $\displaystyle \mu_j = \frac{1}{m}\sum_{i=1}^m x_j^{(i)},\ \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m(x_j^{(i)} - \mu_j)^2$

		Ans: 4


+ Anomaly detection algorithm
	1. Choose features $x_i$ that you think might be indicative of anomalous examples.
	2. Fit parameters $\mu_1, \dots, \mu_n, \sigma_1^2, \dots, \sigma_n^2$ 

		$$\mu_j = \dfrac{1}{m} \sum_{i=1}^m x_j^{(i)} \qquad\qquad \sigma_j^2 = \dfrac{1}{m} \sum_{j=1}^m (x_j^{(i)} - \mu_j)^2$$

		Vectorized form:

		$$\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_n \end{bmatrix} = \dfrac{1}{m} \sum_{i=1}^m x^{(i)}$$

	3. Given new example $x$, compute $p(x)$:

		$$p(x) = \prod_{j-1}^n p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^n \dfrac{1}{\sqrt{2\pi} \sigma_j} \exp \left( - \dfrac{(x_j - \mu_j)^2}{2\sigma^2}  \right)$$

		Anomaly if $p(x) < \epsilon$

+ Anomaly detection example
	+ Height of contour graph = $p(x)$
	+ Set some value of $\epsilon$
	+ The pink shaded area on the contour graph have a low probability hence they’re anomalous

	<div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
		<div><a href="https://www.ritchieng.com/machine-learning-anomaly-detection/">
			<img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w9_anomaly_recommender/anomaly_detection9.png" style="margin: 0.1em;" alt="Anomaly detection example with different mean, standard deviation and epsilon" title="Anomaly detection example" width="450">
		</a></div>
	</div>


#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/16.3-AnomalyDetection-Algorithm.0a287940b22b11e48803b9598c8534ce/full/360p/index.mp4?Expires=1556755200&Signature=SnVPhBoZlrSs17NT280tFKbMSV2LhqGYx3vwEu4uoUVjcpUKncvdV-iCYFYA6fQWj2JP~XstJ4Rl5fNRHOoECWqJuvL4nYXze~~bSpS7OtHS-Vt7yG1yIiZ5VoexK5z1rDBW3mIY9F5UQJQvsk3XnMxL-sLf7SZaisY2GMo9U00_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/JLo-aK-TRta6Pmivk7bWOQ?expiry=1556755200000&hmac=pTFR3N_4fYgkB8zWCln5PoxJGFGDHqQfSM0Oi2lPHrg&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Building an Anomaly Detection System

### Developing and Evaluating an Anomaly Detection System

#### Lecture Notes

+ The importance of real-number evaluation
	+ When developing a learning algorithm (choosing features, etc.), making decisions is much easier if we have a way of evaluating our learning algorithm
	+ Assume we have some labeled data, of anomalous and non-anomalous examples, ($y = 0$ if normal, $y=1$ if anomalous)
	+ Training dataset: $x^{(1)}, x^{(2)}, \dots, x^{(m)}$ (assume normal examples/not anomalous)
	+ Cross validation set: $(x_{cv}^{(1)}, y_{cv}^{(1)}), \dots, (x_{cv}^{(m_{cv})}, y_{cv}^{m_{cv}})$
  + Test set: $(x_{test}^{(1)}, y_{test}^{(1)}), \dots, (x_{test}^{(m_{test})}, y_{test}^{m_{test}})$

+ Example: Aircraft engines motivating example
  + Whole dataset:
    + 10,000 good (normal) engines
    + 20 flawed engines (anomalous)
  + Dataset separation
    + Training set: 6000 good engines ($y=0$) $\implies p(x) = p(x_1; \mu_1, \sigma_1^2) \dots p(x_n; \mu_n, \sigma_n^2)$
    + CV: 2000 good engines ($y=0$), 10 anomalous ($y=1$)
    + Test: 2000 good engines ($y=0), 10 anomalous ($y=1$)
  + Alternative (not recommended)
    + Training set: 6000 good engines ($y=0$)
    + CV: 4000 good engines ($y=0$), 10 anomalous ($y=1$)
    + Test: =4000 good engines ($y=0), 10 anomalous ($y=1$)

+ Algorithm evaluation
  + Fit model $p(x)$ on training set $\{x^{(1)}, x^{(2)}, \dots, x^{(m)} \}$
  + On a cross validation/test example $x$ predict

    $$y = \begin{cases} 1 & \text{if } p(x) < \epsilon \text{ (anomaly)} \\ 0 & \text{if } p(x) \geq \epsilon \text{ (normal)} \end{cases}$$
  + Possible evaluation metrics:
    + True positive, false positive, false negative, true negative
    + Prediction/recall
    + $F_1$-score
  + Can also use cross validation set to choose parameters $\epsilon$ (maximize $F_1$-score)

+ IVQ?: Suppose you have fit a model $p(x)$. When evaluating on the cross validation set or test set, your algorithm predicts:

  $$y = \begin{cases} 1 & \text{if } p(x) < \epsilon \\ 0 & \text{if } p(x) \geq \epsilon \end{cases}$$

  Is classification accuracy a good way to measure the algorithm's performance?

  1. Yes, because we have labels in the cross validation / test sets.
  2. No, because we do not have labels in the cross validation / test sets.
  3. No, because of skewed classes (so an algorithm that always predicts y = 0 will have high accuracy).
  4. No for the cross validation set; yes for the test set.

  Ans: 3



#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/16.4-AnomalyDetection-DevelopingAndEvaluatingAnAnomalyDetectionSystem.232124b0b22b11e4aca907c8d9623f2b/full/360p/index.mp4?Expires=1556755200&Signature=TG19dMp6RPhDSKlWCdYBqAjAP8nbsqulBYcEy1rjbrYC8OYCNgIEJk6xDaDc0LfEBuPJs6UEy6hN2TGfJkdGCngMVgl23kFafUS4lw1hKDsor1fuigVMbM3-jU6BVWqUg9TGmwNcnJtb-OYXv90qS2oe9s4EZkC43w8TepWPieA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/OO6CYLguTTmugmC4Li05KQ?expiry=1556755200000&hmac=0PuuNWo4s7e56fQpLKQBZEJplpoqcSmy146F_h2V0xs&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Anomaly Detection vs. Supervised Learning

#### Lecture Notes

+ Anomaly detection vs. Supervised learning
  + Anomaly detection
    + Very small number of positive examples ($y=1$). (__0-20__ is common)
    + Large number of negative ($y=0$) examples (to fit $p(x)$ with Gaussian distribution)
    + Many different "types" of anomalies.  hard for any algorithm to learn from positive examples what the anomalies look like
    + Future anomalies may look nothing like any of the anomalous examples we've see so far
    + Examples
      + Fraud detection
      + Manufacturing (e.g., aircraft engines)
      + Monitoring machines in a data center
  + Supervised learning
    + Large number of positive and negative examples
    + Enough positive examples for algorithm= to get a sense of what positive examples are like
    + Future positive examples likely to be similar to ones in training set
    + Examples
      + Email spam classification
      + Weather prediction (sunny/rainy/etc.)
      + Cancer classification

+ IVQ: Which of the following problems would you approach with an anomaly detection algorithm (rather than a supervised learning algorithm)? Check all that apply.

  1. You run a power utility (supplying electricity to customers) and want to monitor your electric plants to see if any one of them might be behaving strangely.
  2. You run a power utility and want to predict tomorrow’s expected demand for electricity (so that you can plan to ramp up an appropriate amount of generation capacity).
  3. A computer vision / security application, where you examine video images to see if anyone in your company’s parking lot is acting in an unusual way.
  4. A computer vision application, where you examine an image of a person entering your retail store to determine if the person is male or female.

  Ans: 13



#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/16.5-AnomalyDetection-AnomalyDetectionVsSupervisedLearning-V1.b295d820b22b11e4beb61117ba5cda9e/full/360p/index.mp4?Expires=1556755200&Signature=Fox8p8daSvX8wNpmhS~xFrGxCG03xqpiBvU8Q7tNUcVnG8NSHsNH~9m3Vf~qIB3j5C7cvifHHLDcpAicpJQHO3cEmOOPTFaFAltpZ-jIefQ4BE7XFAWIvH7G4e8DM0G2hYgurZnazRJH6zGqCEU9Qbk4~NI4Na82qLfZmsEmED8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/1O25ppNNShqtuaaTTQoaaQ?expiry=1556755200000&hmac=NSyrf92cCuc6Yb52jAWNWLCZacjr5RxzfuF1pFTKsLg&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Choosing What Features to Use

#### Lecture Notes

+ Non-Gaussian features
  + With non-Gaussian distribution, transform the distribution to Gaussian distribution
  + Examples:
    + $x_1 \;\longleftarrow\; \log(x_1)$
    + $x_2 \;\longleftarrow\; \log(x_2 + c)$
    + $x_3 \;\longleftarrow\; \sqrt{x_3} = x_3^{1/2}$
    + $x_4 \;\longleftarrow\; x_4^{1/3}$

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.ritchieng.com/machine-learning-anomaly-detection/#2c-choosing-what-features-to-use">
      <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w9_anomaly_recommender/anomaly_detection11.png" style="margin: 0.1em;" alt="Non-Gaussian features: Pareto distribution and transform to Gaussian distribution" title="Non-Gaussian features" width="350">
    </a></div>
  </div>

+ Error analysis for anomaly detection
  + Allows us to come up with extra features to extract anomaly
  + Similar to the error analysis procedure for supervised learning
  + Procedure
    + train a complete algorithm
    + run the algorithm on a cross validation set
    + look at the examples it gets wrong
    + see if extra features come up to help the algorithm do better on the examples that it got wrong in the cross-validation set
  + Objective:
    + $p(x)\;$ large for normal examples $x$
    + $p(x)\;$ small for anomalous examples $x$
  + Most common problems
    + $p(x)\;$ is comparable (say, both large) for normal and anomalous examples
  + Diagrams
    + plot with unlabeled data with feature $x_1$ (left diagram)
    + fit the data with Gaussian distribution
    + an anomaly example happened at $x_1 = 2.5$ and buried in the middle of a bunch of normal examples
    + exam the data and observe what went wrong
    + the examination might inspire to come up a new feature $x_2$ to distinguish the anomaly
    + plot the feature $x_2$ with $x_1$, hopefully the anomaly can be identify with the new feature, e.g., $x_2 = 3.5$

  <div style="display:flex;justify-content:center;align-items:center;flex-flow:row wrap;">
    <div><a href="https://www.ritchieng.com/machine-learning-anomaly-detection/#2c-choosing-what-features-to-use">
      <img src="https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w9_anomaly_recommender/anomaly_detection12.png" style="margin: 0.1em;" alt="Error analysis for anomalous detection" title="Error analysis for anomalous detection" width="350">
    </a></div>
  </div>  

+ Example: Monitoring computers in a data center

  Choose features that might take on unusually large or small values in the event of an anomaly
  + $x_1\;$ = memory use of computer
  + $x_2\;$ = number of disk access/sec
  + $x_3\;$ = CPU load
  + $x_4\;$ = network traffic
  + new feature: $x_5 = \dfrac{\text{CPU load}}{\text{network traffic}}$: could be vary huge with large "CPU load" and small "network traffic"
  + new feature: $x_6 =  \dfrac{(\text{CPU load})^2}{\text{network traffic}}$
  
+ IVQ: Suppose your anomaly detection algorithm is performing poorly and outputs a large value of $p(x)$ for many normal examples and for many anomalous examples in your cross validation dataset. Which of the following changes to your algorithm is most likely to help?

  1. Try using fewer features.
  2. Try coming up with more features to distinguish between the normal and the anomalous examples.
  3. Get a larger training set (of normal examples) with which to fit $p(x)$.
  4. Try changing $\epsilon$.

  Ans: 



#### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/16.6-AnomalyDetection-ChoosingWhatFeaturesToUse.cc993cd0b22b11e49f072fa475844d6b/full/360p/index.mp4?Expires=1556755200&Signature=YMQZ4b1U7UuIzqvJb2qXso9Z80~aXQKyDF7DmMZXHaH69zncvKwqtMoIGiib2hsL5~YTWyvsWSwHqWn8t91ZvFgT4Z7iJHkADhnAqUglQ51J5BNBViG6tqmrRdmKOSFUEaq~WkfKXUa-Bjn2-c2ZCGtWVdDKqDQBcWeYmJtJ-8E_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/4QC7hPw-ThSAu4T8Pp4Ucg?expiry=1556755200000&hmac=7DS8lKWUCDtI0chI5UuBVc0RwgXJ-RrfKssCFcd9WEk&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Multivariate Gaussian Distribution (Optional)


### Multivariate Gaussian Distribution

#### Lecture Notes




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


### Anomaly Detection using the Multivariate Gaussian Distribution

#### Lecture Notes




#### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video><br/>


## Review

### Lecture Slides

#### Problem Motivation

Just like in other learning problems, we are given a dataset ${x^{(1)}, x^{(2)},\dots,x^{(m)}}$.

We are then given a new example, $x_{test}$ , and we want to know whether this new example is abnormal/anomalous.

We define a "model" $p(x)$ that tells us the probability the example is not anomalous. We also use a threshold $\epsilon$ (epsilon) as a dividing line so we can say which examples are anomalous and which are not.

A very common application of anomaly detection is detecting fraud:

+ $x^{(i)} = x\;$ features of user i's activities
+ Model $p(x)$ from the data.
+ Identify unusual users by checking which have $p(x) < \epsilon$.

If our anomaly detector is flagging __too many__ anomalous examples, then we need to __decrease__ our threshold $\epsilon$

#### Gaussian Distribution

The Gaussian Distribution is a familiar bell-shaped curve that can be described by a function $N(\mu, \sigma^2)$

Let $x \in \mathbb{R}$. If the probability distribution of $x$ is Gaussian with mean $\mu$, variance $\sigma^2$, then:

$$ x ∼ N(\mu, \sigma^2)$$

The little `∼` or 'tilde' can be read as "distributed as."

The Gaussian Distribution is parameterized by a mean and a variance.

Mu, or $\mu$, describes the center of the curve, called the mean. The width of the curve is described by sigma, or $\sigma$, called the standard deviation.

The full function is as follows:

$$p(x;\mu, \sigma^2) = \dfrac{1}{\sigma \sqrt{(2\pi)}} e^{-\dfrac{1}{2}(\dfrac{x-\mu}{\sigma})^2}$$

We can estimate the parameter $\mu$ from a given dataset by simply taking the average of all the examples:

$$\mu = \dfrac{1}{m}\displaystyle \sum_{i=1}^m x^{(i)}$$

We can estimate the other parameter, $\sigma^2$, with our familiar squared error formula:

$$\sigma^2 = \dfrac{1}{m}\displaystyle \sum_{i=1}^m(x^{(i)} - \mu)^2$$
 

#### Algorithm

Given a training set of examples, $\{x^{(1)}, \dots, x^{(m)}\}$ where each example is a vector, $x \in \mathbb{R}^n$.

$$p(x) = p(x_1; \mu_1, \sigma^2_1)p(x_2;\mu2,\sigma_2^2) \cdots p(x_n;\mu_n,\sigma^2_n)$$

In statistics, this is called an "independence assumption" on the values of the features inside training example $x$.

More compactly, the above expression can be written as follows:

$$p(x) = \prod_{j=1}^n p(x_j; \mu_j,\sigma^2_j)$$


__The algorithm__

Choose features $x_i$ that you think might be indicative of anomalous examples.

Fit parameters $\mu_1,\dots,\mu_n,\sigma_1^2,\dots,\sigma_n^2$

Calculate $\mu_j = \dfrac{1}{m}\displaystyle \sum_{i=1}^m x_j^{(i)}$

Calculate $\sigma^2_j = \dfrac{1}{m}\displaystyle \sum_{i=1}^m(x_j^{(i)} - \mu_j)^2$

Given a new example $x$, compute $p(x)$:

$$p(x) = \prod_{j=1}^n p(x_j; \mu_j,\sigma^2_j) = \prod_{j=1}^n \dfrac{1}{\sqrt{2\pi} \sigma_j} \exp \left( −\dfrac{(x_j−\mu_j)^2}{2\sigma^2_j} \right)$$

Anomaly if $p(x) < \epsilon$

A vectorized version of the calculation for $\mu$ is $\mu = \dfrac{1}{m}\displaystyle \sum_{i=1}^m x^{(i)}$. You can vectorize $\sigma^2$ similarly.


#### Developing and Evaluating an Anomaly Detection System

To evaluate our learning algorithm, we take some labeled data, categorized into anomalous and non-anomalous examples ( $y = 0$ if normal, $y = 1$ if anomalous).

Among that data, take a large proportion of good, non-anomalous data for the training set on which to train $p(x)$.

Then, take a smaller proportion of mixed anomalous and non-anomalous examples (you will usually have many more non-anomalous examples) for your cross-validation and test sets.

For example, we may have a set where $0.2\%$ of the data is anomalous. We take $60\%$ of those examples, all of which are good ($y=0$) for the training set. We then take $20\%$ of the examples for the cross-validation set (with $0.1\%$ of the anomalous examples) and another $20\%$ from the test set (with another 0.1% of the anomalous).

In other words, we split the data 60/20/20 training/CV/test and then split the anomalous examples 50/50 between the CV and test sets.

__Algorithm evaluation:__

Fit model p(x) on training set $\{x^{(1)},\dots,x^{(m)}\}$

On a cross validation/test example x, predict:

+ If $p(x) < \epsilon$ (__anomaly__), then $y=1$
+ If $p(x) \geq \epsilon$ (__normal__), then $y=0$

Possible evaluation metrics (see "Machine Learning System Design" section):

+ True positive, false positive, false negative, true negative.
+ Precision/recall
+ $F_1$ score

Note that we use the cross-validation set to choose parameter $\epsilon$


#### Anomaly Detection vs. Supervised Learning

When do we use anomaly detection and when do we use supervised learning?

Use anomaly detection when...

+ We have a very small number of positive examples (y=1 ... 0-20 examples is common) and a large number of negative (y=0) examples.
+ We have many different "types" of anomalies and it is hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far.

Use supervised learning when...

+ We have a large number of both positive and negative examples. In other words, the training set is more evenly divided into classes.
+ We have enough positive examples for the algorithm to get a sense of what new positives examples look like. The future positive examples are likely similar to the ones in the training set.


#### Choosing What Features to Use

The features will greatly affect how well your anomaly detection algorithm works.

We can check that our features are gaussian by plotting a histogram of our data and checking for the bell-shaped curve.

Some transforms we can try on an example feature x that does not have the bell-shaped curve are:

+ $\log(x)$
+ $\log(x+1)$
+ $\log(x+c)$ for some constant
+ $\sqrt{x}$
+ $x^{1/3}$
 
We can play with each of these to try and achieve the gaussian shape in our data.

There is an __error analysis procedure__ for anomaly detection that is very similar to the one in supervised learning.

Our goal is for $p(x)$ to be large for normal examples and small for anomalous examples.

One common problem is when p(x) is similar for both types of examples. In this case, you need to examine the anomalous examples that are giving high probability in detail and try to figure out new features that will better distinguish the data.

In general, choose features that might take on unusually large or small values in the event of an anomaly.


#### Multivariate Gaussian Distribution (Optional)

The multivariate gaussian distribution is an extension of anomaly detection and may (or may not) catch more anomalies.

Instead of modeling $p(x_1),p(x_2),\dots$ separately, we will model $p(x)$ all in one go. Our parameters will be: $\mu \in \mathbb{R}^n$ and $\sigma \in \mathbb{R}^{n×n}$

$$p(x;\mu,\sigma) = \dfrac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp \left(−1/2(x − \mu)^T \Sigma^{−1}(x−\mu) \right)$$

The important effect is that we can model oblong gaussian contours, allowing us to better fit data that might not fit into the normal circular contours.

Varying $\sigma$ changes the shape, width, and orientation of the contours. Changing $\mu$ will move the center of the distribution.

Check also:

The [Multivariate Gaussian Distribution](http://cs229.stanford.edu/section/gaussians.pdf) Chuong B. Do, October 10, 2008.


### Errata

At the risk of being pedantic, it should be noted that p(x) is not a probability but rather the normalized probability density as parameterized by the feature vector, $x$; therefore, $\epsilon$ is a threshold condition on the probability density. Determination of the actual probability would require integration of this density over the appropriate extent of phase space.

In the Developing and Evaluating an Anomaly Detection System video an alternative way for some people to split the data is to use the same data for the cv and test sets, therefore the number of anomalous engines (y = 1) in each set would be 20 rather than 10 as it states on the slide.


### Quiz: Anomaly Detection





