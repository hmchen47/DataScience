# Machine Learning with Python

## [Scikit-Learn API][000]

### [sklearn.base][001]: Base classes and utility functions¶

Base classes for all estimators.

#### Base classes

| Class | Description | Link |
|-------|-------------|------|
| `base.BaseEstimator` | Base class for all estimators in scikit-learn | [API][002] |
| `base.BiclusterMixin` | Mixin class for all bicluster estimators in scikit-learn | [API][003] |
| `base.ClassifierMixin` | Mixin class for all classifiers in scikit-learn. | [API][004] |
| `base.ClusterMixin` | Mixin class for all cluster estimators in scikit-learn. | [API][005] |
| `base.DensityMixin` | Mixin class for all density estimators in scikit-learn. | [API][006] |
| `base.RegressorMixin` | Mixin class for all regression estimators in scikit-learn. | [API][007] |
| `base.TransformerMixin` | Mixin class for all transformers in scikit-learn. | [API][008] |

#### Functions

| Function | Description | Link |
|----------|-------------|------|
|  `base.clone(estimator[, safe])` | Constructs a new estimator with the same parameters. | [API][09] |
|  `base.is_classifier(estimator)` | Returns True if the given estimator is (probably) a classifier. | [API][010] |
|  `base.is_regressor(estimator)` | Returns True if the given estimator is (probably) a regressor. | [API][011] |
|  `config_context(**new_config)` | Context manager for global scikit-learn configuration | [API][012] |
|  `get_config()` | Retrieve current values for configuration set by set_config | [API][013] |
|  `set_config([assume_finite, working_memory])` | Set global scikit-learn configuration | [API][014] |
|  `show_versions()` | Print useful debugging information | [API][015] |

### [sklearn.calibration][016]: Probability Calibration

+ Calibration of predicted probabilities.

+ User guide: See the [Probability calibration][017] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `calibration.CalibratedClassifierCV([…])` | Probability calibration with isotonic regression or sigmoid. | [API][018] |
| `calibration.calibration_curve(y_true, y_prob)` | Compute true and predicted probabilities for a calibration curve. | [API][019] |


### sklearn.cluster: Clustering

+ The [sklearn.cluster][020] module gathers popular unsupervised clustering algorithms.

User guide: See the [Clustering][021] section for further details.

#### Classes

| Class | Description | Link |
|-------|-------------|------|
| `cluster.AffinityPropagation([damping, …])` | Perform Affinity Propagation Clustering of data. | [API][022] |
| `cluster.AgglomerativeClustering([…])` | Agglomerative Clustering | [API][023] |
| `cluster.Birch([threshold, branching_factor, …])` | Implements the Birch clustering algorithm. | [API][024] |
| `cluster.DBSCAN([eps, min_samples, metric, …])` | Perform DBSCAN clustering from vector array or distance matrix. | [API][025] |
| `cluster.FeatureAgglomeration([n_clusters, …])` | Agglomerate features. | [API][026] |
| `cluster.KMeans([n_clusters, init, n_init, …])` | K-Means clustering | [API][0227] |
| `cluster.MiniBatchKMeans([n_clusters, init, …])` | Mini-Batch K-Means clustering | [API][028] |
| `cluster.MeanShift([bandwidth, seeds, …])` | Mean shift clustering using a flat kernel. | [API][029] |
| `cluster.SpectralClustering([n_clusters, …])` | Apply clustering to a projection to the normalized laplacian. | [API][030] |

#### Functions

| Function | Description | Link |
|----------|-------------|------|
| `cluster.affinity_propagation(S[, …])` | Perform Affinity Propagation Clustering of data | [API][031] |
| `cluster.dbscan(X[, eps, min_samples, …])` | Perform DBSCAN clustering from vector array or distance matrix. | [API][032] |
| `cluster.estimate_bandwidth(X[, quantile, …])` | Estimate the bandwidth to use with the mean-shift algorithm. | [API][033] |
| `cluster.k_means(X, n_clusters[, …])` | K-means clustering algorithm. | [API][034] |
| `cluster.mean_shift(X[, bandwidth, seeds, …])` | Perform mean shift clustering of data using a flat kernel. | [API][035] |
| `cluster.spectral_clustering(affinity[, …])` | Apply clustering to a projection to the normalized laplacian. | [API][036] |
| `cluster.ward_tree(X[, connectivity, …])` | Ward clustering based on a Feature matrix. | [API][037] |


### [sklearn.cluster.bicluster][038]: Biclustering

Spectral biclustering algorithms.

Authors : Kemal Eren License: BSD 3 clause

User guide: See the [Biclustering][039] section for further details.

#### Classes

| Class | Description | Link |
|-------|-------------|------|
| `SpectralBiclustering([n_clusters, method, …])` | Spectral biclustering (Kluger, 2003). | [API][040] |
| `SpectralCoclustering([n_clusters, …])` | Spectral Co-Clustering algorithm (Dhillon, 2001). | [API][041] |


### [sklearn.compose][042]: Composite Estimators

Meta-estimators for building composite models with transformers

In addition to its current contents, this module will eventually be home to refurbished versions of Pipeline and FeatureUnion.

User guide: See the [Pipelines and composite estimators][043] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `compose.ColumnTransformer(transformers[, …])` | Applies transformers to columns of an array or pandas DataFrame. | [API][044] |
| `compose.TransformedTargetRegressor([…])` | Meta-estimator to regress on a transformed target. | [API][045] |
| `compose.make_column_transformer(…)` | Construct a ColumnTransformer from the given transformers. | [API][046] |

### sklearn.covariance: Covariance Estimators

The [sklearn.covariance][047] module includes methods and algorithms to robustly estimate the covariance of features given a set of points. The precision matrix defined as the inverse of the covariance is also estimated. Covariance estimation is closely related to the theory of Gaussian Graphical Models.

User guide: See the [Covariance estimation][048] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `covariance.EmpiricalCovariance([…])` | Maximum likelihood covariance estimator | [API][049] |
| `covariance.EllipticEnvelope([…])` | An object for detecting outliers in a Gaussian distributed dataset. | [API][050] |
| `covariance.GraphicalLasso([alpha, mode, …])` | Sparse inverse covariance estimation with an l1-penalized estimator. | [API][051] |
| `covariance.GraphicalLassoCV([alphas, …])` | Sparse inverse covariance w/ cross-validated choice of the l1 penalty | [API][052] |
| `covariance.LedoitWolf([store_precision, …])` | LedoitWolf Estimator | [API][053] |
| `covariance.MinCovDet([store_precision, …])` | Minimum Covariance Determinant (MCD): robust estimator of covariance. | [API][054] |
| `covariance.OAS([store_precision, …])` | Oracle Approximating Shrinkage Estimator | [API][055] |
| `covariance.ShrunkCovariance([…])` | Covariance estimator with shrinkage | [API][056] |
| `covariance.empirical_covariance(X[, …])` | Computes the Maximum likelihood covariance estimator | [API][057] |
| `covariance.graphical_lasso(emp_cov, alpha[, …])` | l1-penalized covariance estimator | [API][058] |
| `covariance.ledoit_wolf(X[, assume_centered, …])` | Estimates the shrunk Ledoit-Wolf covariance matrix. | [API][059] |
| `covariance.oas(X[, assume_centered])` | Estimate covariance with the Oracle Approximating Shrinkage algorithm. | [API][060] |
| `covariance.shrunk_covariance(emp_cov[, …])` | Calculates a covariance matrix shrunk on the diagonal | [API][061] |


### [sklearn.cross_decomposition][062]: Cross decomposition

User guide: See the [Cross decomposition][063] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `cross_decomposition.CCA([n_components, …])` | CCA Canonical Correlation Analysis. | [API][064] |
| `cross_decomposition.PLSCanonical([…])` | PLSCanonical implements the 2 blocks canonical PLS of the original Wold algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000]. | [API][065] |
| `cross_decomposition.PLSRegression([…])` | PLS regression | [API][066] |
| `cross_decomposition.PLSSVD([n_components, …])` | Partial Least Square SVD | [API][067] |


### sklearn.datasets: Datasets

The [sklearn.datasets][068] module includes utilities to load datasets, including methods to load and fetch popular reference datasets. It also features some artificial data generators.

User guide: See the [Dataset loading utilities][069] section for further details.

#### Loaders

| Function | Description | Link |
|----------|-------------|------|
| `datasets.clear_data_home([data_home])` | Delete all the content of the data home cache. | [API][070] |
| `datasets.dump_svmlight_file(X, y, f[, …])` | Dump the dataset in svmlight / libsvm file format. | [API][071] |
| `datasets.fetch_20newsgroups([data_home, …])` | Load the filenames and data from the 20 newsgroups dataset (classification). | [API][072] |
| `datasets.fetch_20newsgroups_vectorized([…])` | Load the 20 newsgroups dataset and vectorize it into token counts (classification). | [API][073] |
| `datasets.fetch_california_housing([…])` | Load the California housing dataset (regression). | [API][074] |
| `datasets.fetch_covtype([data_home, …])` | Load the covertype dataset (classification). | [API][075] |
| `datasets.fetch_kddcup99([subset, data_home, …])` | Load the kddcup99 dataset (classification). | [API][076] |
| `datasets.fetch_lfw_pairs([subset, …])` | Load the Labeled Faces in the Wild (LFW) pairs dataset (classification). | [API][077] |
| `datasets.fetch_lfw_people([data_home, …])` | Load the Labeled Faces in the Wild (LFW) people dataset (classification). | [API][078] |
| `datasets.fetch_olivetti_faces([data_home, …])` | Load the Olivetti faces data-set from AT&T (classification). | [API][079] |
| `datasets.fetch_openml([name, version, …])` | Fetch dataset from openml by name or dataset id. | [API][080] |
| `datasets.fetch_rcv1([data_home, subset, …])` | Load the RCV1 multilabel dataset (classification). | [API][081] |
| `datasets.fetch_species_distributions([…])` | Loader for species distribution dataset from Phillips et. | [API][082] |
| `datasets.get_data_home([data_home])` | Return the path of the scikit-learn data dir. | [API][083] |
| `datasets.load_boston([return_X_y])` | Load and return the boston house-prices dataset (regression). | [API][084] |
| `datasets.load_breast_cancer([return_X_y])` | Load and return the breast cancer wisconsin dataset (classification). | [API][085] |
| `datasets.load_diabetes([return_X_y])` | Load and return the diabetes dataset (regression). | [API][086] |
| `datasets.load_digits([n_class, return_X_y])` | Load and return the digits dataset (classification). | [API][087] |
| `datasets.load_files(container_path[, …])` | Load text files with categories as subfolder names. | [API][088] |
| `datasets.load_iris([return_X_y])` | Load and return the iris dataset (classification). | [API][089] |
| `datasets.load_linnerud([return_X_y])` | Load and return the linnerud dataset (multivariate regression). | [API][090] |
| `datasets.load_sample_image(image_name)` | Load the numpy array of a single sample image | [API][091] |
| `datasets.load_sample_images()` | Load sample images for image manipulation. | [API][092] |
| `datasets.load_svmlight_file(f[, n_features, …])` | Load datasets in the svmlight / libsvm format into sparse CSR matrix | [API][093] |
| `datasets.load_svmlight_files(files[, …])` | Load dataset from multiple files in SVMlight format | [API][094] |
| `datasets.load_wine([return_X_y])` | Load and return the wine dataset (classification). | [API][095] |
| `datasets.mldata_filename(dataname)` | DEPRECATED: mldata_filename was deprecated in version 0.20 and will be removed in version 0.22 | [API][096] |

#### Samples generator

| Function | Description | Link |
|----------|-------------|------|
| `datasets.make_biclusters(shape, n_clusters)` | Generate an array with constant block diagonal structure for biclustering. | [API][097] |
| `datasets.make_blobs([n_samples, n_features, …])` | Generate isotropic Gaussian blobs for clustering. | [API][098] |
| `datasets.make_checkerboard(shape, n_clusters)` | Generate an array with block checkerboard structure for biclustering. | [API][099] |
| `datasets.make_circles([n_samples, shuffle, …])` | Make a large circle containing a smaller circle in 2d. | [API][100] |
| `datasets.make_classification([n_samples, …])` | Generate a random n-class classification problem. | [API][101] |
| `datasets.make_friedman1([n_samples, …])` | Generate the “Friedman #1” regression problem | [API][102] |
| `datasets.make_friedman2([n_samples, noise, …])` | Generate the “Friedman #2” regression problem | [API][103] |
| `datasets.make_friedman3([n_samples, noise, …])` | Generate the “Friedman #3” regression problem | [API][104] |
| `datasets.make_gaussian_quantiles([mean, …])` | Generate isotropic Gaussian and label samples by quantile | [API][105] |
| `datasets.make_hastie_10_2([n_samples, …])` | Generates data for binary classification used in Hastie et al. | [API][106] |
| `datasets.make_low_rank_matrix([n_samples, …])` | Generate a mostly low rank matrix with bell-shaped singular values | [API][107] |
| `datasets.make_moons([n_samples, shuffle, …])` | Make two interleaving half circles | [API][108] |
| `datasets.make_multilabel_classification([…])` | Generate a random multilabel classification problem. | [API][109] |
| `datasets.make_regression([n_samples, …])` | Generate a random regression problem. | [API][110] |
| `datasets.make_s_curve([n_samples, noise, …])` | Generate an S curve dataset. | [API][111] |
| `datasets.make_sparse_coded_signal(n_samples, …)` | Generate a signal as a sparse combination of dictionary elements. | [API][112] |
| `datasets.make_sparse_spd_matrix([dim, …])` | Generate a sparse symmetric definite positive matrix. | [API][113] |
| `datasets.make_sparse_uncorrelated([…])` | Generate a random regression problem with sparse uncorrelated design | [API][114] |
| `datasets.make_spd_matrix(n_dim[, random_state])` | Generate a random symmetric, positive-definite matrix. | [API][115] |
| `datasets.make_swiss_roll([n_samples, noise, …])` | Generate a swiss roll dataset. | [API][116] |


### sklearn.decomposition: Matrix Decomposition

The [sklearn.decomposition][117] module includes matrix decomposition algorithms, including among others PCA, NMF or ICA. Most of the algorithms of this module can be regarded as dimensionality reduction techniques.

User guide: See the [Decomposing signals in components (matrix factorization problems)][118] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `decomposition.DictionaryLearning([…])` | Dictionary learning | [API][119] |
| `decomposition.FactorAnalysis([n_components, …])` | Factor Analysis (FA) | [API][120] |
| `decomposition.FastICA([n_components, …])` | FastICA: a fast algorithm for Independent Component Analysis. | [API][121] |
| `decomposition.IncrementalPCA([n_components, …])` | Incremental principal components analysis (IPCA). | [API][122] |
| `decomposition.KernelPCA([n_components, …])` | Kernel Principal component analysis (KPCA) | [API][123] |
| `decomposition.LatentDirichletAllocation([…])` | Latent Dirichlet Allocation with online variational Bayes algorithm | [API][124] |
| `decomposition.MiniBatchDictionaryLearning([…])` | Mini-batch dictionary learning | [API][125] |
| `decomposition.MiniBatchSparsePCA([…])` | Mini-batch Sparse Principal Components Analysis | [API][126] |
| `decomposition.NMF([n_components, init, …])` | Non-Negative Matrix Factorization (NMF) | [API][127] |
| `decomposition.PCA([n_components, copy, …])` | Principal component analysis (PCA) | [API][128] |
| `decomposition.SparsePCA([n_components, …])` | Sparse Principal Components Analysis (SparsePCA) | [API][129] |
| `decomposition.SparseCoder(dictionary[, …])` | Sparse coding | [API][130] |
| `decomposition.TruncatedSVD([n_components, …])` | Dimensionality reduction using truncated SVD (aka LSA). | [API][131] |
| `decomposition.dict_learning(X, n_components, …)` | Solves a dictionary learning matrix factorization problem. | [API][132] |
| `decomposition.dict_learning_online(X[, …])` | Solves a dictionary learning matrix factorization problem online. | [API][133] |
| `decomposition.fastica(X[, n_components, …])` | Perform Fast Independent Component Analysis. | [API][134] |
| `decomposition.sparse_encode(X, dictionary[, …])` | Sparse coding | [API][135] |


### [sklearn.discriminant_analysis][136]: Discriminant Analysis

Linear Discriminant Analysis and Quadratic Discriminant Analysis

User guide: See the Linear and Quadratic Discriminant Analysis section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `discriminant_analysis.LinearDiscriminantAnalysis([…])` | Linear Discriminant Analysis | [API][137] |
| `discriminant_analysis.QuadraticDiscriminantAnalysis([…])` | Quadratic Discriminant Analysis | [API][138] |

### [sklearn.dummy][139]: Dummy estimators

User guide: See the [Model evaluation: quantifying the quality of predictions][140] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `dummy.DummyClassifier([strategy, …])` | DummyClassifier is a classifier that makes predictions using simple rules. | [API][141] | 
| `dummy.DummyRegressor([strategy, constant, …])` | DummyRegressor is a regressor that makes predictions using simple rules. | [API][142] | 


### sklearn.ensemble: Ensemble Methods

The [sklearn.ensemble][143] module includes ensemble-based methods for classification, regression and anomaly detection.

User guide: See the [Ensemble methods][144] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `ensemble.AdaBoostClassifier([…])` | An AdaBoost classifier. | [API][145] |
| `ensemble.AdaBoostRegressor([base_estimator, …])` | An AdaBoost regressor. | [API][146] |
| `ensemble.BaggingClassifier([base_estimator, …])` | A Bagging classifier. | [API][147] |
| `ensemble.BaggingRegressor([base_estimator, …])` | A Bagging regressor. | [API][148] |
| `ensemble.ExtraTreesClassifier([…])` | An extra-trees classifier. | [API][149] |
| `ensemble.ExtraTreesRegressor([n_estimators, …])` | An extra-trees regressor. | [API][150] |
| `ensemble.GradientBoostingClassifier([loss, …])` | Gradient Boosting for classification. | [API][151] |
| `ensemble.GradientBoostingRegressor([loss, …])` | Gradient Boosting for regression. | [API][152] |
| `ensemble.IsolationForest([n_estimators, …])` | Isolation Forest Algorithm | [API][153] |
| `ensemble.RandomForestClassifier([…])` | A random forest classifier. | [API][154] |
| `ensemble.RandomForestRegressor([…])` | A random forest regressor. | [API][155] |
| `ensemble.RandomTreesEmbedding([…])` | An ensemble of totally random trees. | [API][156] |
| `ensemble.VotingClassifier(estimators[, …])` | Soft Voting/Majority Rule classifier for unfitted estimators. | [API][157] |

#### partial dependence

Partial dependence plots for tree ensembles.

| Function | Description | Link |
|----------|-------------|------|
| `ensemble.partial_dependence.partial_dependence(…)` | Partial dependence of target_variables. | [API][158] |
| `ensemble.partial_dependence.plot_partial_dependence(…)` | Partial dependence plots for features. | [API][159] |


### sklearn.exceptions: Exceptions and warnings

The [sklearn.exceptions][160] module includes all custom warnings and error classes used across scikit-learn.

| Function | Description | Link |
|----------|-------------|------|
| `exceptions.ChangedBehaviorWarning` | Warning class used to notify the user of any change in the behavior. | [API][161] |
| `exceptions.ConvergenceWarning` | Custom warning to capture convergence problems | [API][162] |
| `exceptions.DataConversionWarning` | Warning used to notify implicit data conversions happening in the code. | [API][163] |
| `exceptions.DataDimensionalityWarning` | Custom warning to notify potential issues with data dimensionality. | [API][164] |
| `exceptions.EfficiencyWarning` | Warning used to notify the user of inefficient computation. | [API][165] |
| `exceptions.FitFailedWarning` | Warning class used if there is an error while fitting the estimator. | [API][166] |
| `exceptions.NotFittedError` | Exception class to raise if estimator is used before fitting. | [API][167] |
| `exceptions.NonBLASDotWarning` | Warning used when the dot operation does not use BLAS. | [API][16]8 |
| `exceptions.UndefinedMetricWarning` | Warning used when the metric is invalid | [API][169] |


### sklearn.feature_extraction: Feature Extraction

The [sklearn.feature_extraction][170] module deals with feature extraction from raw data. It currently includes methods to extract features from text and images.

User guide: See the [Feature extraction][171] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `feature_extraction.DictVectorizer([dtype, …])` | Transforms lists of feature-value mappings to vectors. | [API][172] |
| `feature_extraction.FeatureHasher([…])` | Implements feature hashing, aka the hashing trick | [API][173] |

#### From images

The [sklearn.feature_extraction.image][174] submodule gathers utilities to extract features from images.

| Function | Description | Link |
|----------|-------------|------|
| `feature_extraction.image.extract_patches_2d(…)` | Reshape a 2D image into a collection of patches | [API][175] |
| `feature_extraction.image.grid_to_graph(n_x, n_y)` | Graph of the pixel-to-pixel connections | [API][176] |
| `feature_extraction.image.img_to_graph(img[, …])` | Graph of the pixel-to-pixel gradient connections | [API][177] |
| `feature_extraction.image.reconstruct_from_patches_2d(…)` | Reconstruct the image from all of its patches. | [API][178] |
| `feature_extraction.image.PatchExtractor([…])` | Extracts patches from a collection of images | [API][179] |

#### From text

The [sklearn.feature_extraction.text][180] submodule gathers utilities to build feature vectors from text documents.

| Function | Description | Link |
|----------|-------------|------|
| `feature_extraction.text.CountVectorizer([…])` | Convert a collection of text documents to a matrix of token counts | [API][181] |
| `feature_extraction.text.HashingVectorizer([…])` | Convert a collection of text documents to a matrix of token occurrences | [API][182] |
| `feature_extraction.text.TfidfTransformer([…])` | Transform a count matrix to a normalized tf or tf-idf representation | [API][183] |
| `feature_extraction.text.TfidfVectorizer([…])` | Convert a collection of raw documents to a matrix of TF-IDF features | [API][184] |


### sklearn.feature_selection: Feature Selection

The [sklearn.feature_selection][185] module implements feature selection algorithms. It currently includes univariate filter selection methods and the recursive feature elimination algorithm.

User guide: See the [Feature selection][186] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `feature_selection.GenericUnivariateSelect([…])` | Univariate feature selector with configurable strategy. | [API][187] |
| `feature_selection.SelectPercentile([…])` | Select features according to a percentile of the highest scores. | [API][188] |
| `feature_selection.SelectKBest([score_func, k])` | Select features according to the k highest scores. | [API][189] |
| `feature_selection.SelectFpr([score_func, alpha])` | Filter: Select the pvalues below alpha based on a FPR test. | [API][190] |
| `feature_selection.SelectFdr([score_func, alpha])` | Filter: Select the p-values for an estimated false discovery rate | [API][191] |
| `feature_selection.SelectFromModel(estimator)` | Meta-transformer for selecting features based on importance weights. | [API][192] |
| `feature_selection.SelectFwe([score_func, alpha])` | Filter: Select the p-values corresponding to Family-wise error rate | [API][193] |
| `feature_selection.RFE(estimator[, …])` | Feature ranking with recursive feature elimination. | [API][194] |
| `feature_selection.RFECV(estimator[, step, …])` | Feature ranking with recursive feature elimination and cross-validated selection of the best number of features. | [API][195] |
| `feature_selection.VarianceThreshold([threshold])` | Feature selector that removes all low-variance features. | [API][196] |
| `feature_selection.chi2(X, y)` | Compute chi-squared stats between each non-negative feature and class. | [API][197] |
| `feature_selection.f_classif(X, y)` | Compute the ANOVA F-value for the provided sample. | [API][198] |
| `feature_selection.f_regression(X, y[, center])` | Univariate linear regression tests. | [API][199] |
| `feature_selection.mutual_info_classif(X, y)` | Estimate mutual information for a discrete target variable. | [API][200] |
| `feature_selection.mutual_info_regression(X, y)` | Estimate mutual information for a continuous target variable. | [API][201] |


### sklearn.gaussian_process: Gaussian Processes

The [sklearn.gaussian_process][202] module implements Gaussian Process based regression and classification.

User guide: See the [Gaussian Processes][203] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `gaussian_process.GaussianProcessClassifier([…])` | Gaussian process classification (GPC) based on Laplace approximation. ][API][204] |
| `gaussian_process.GaussianProcessRegressor([…])` | Gaussian process regression (GPR). ][API][205] |
| Kernels: | | |
| `gaussian_process.kernels.CompoundKernel(kernels)` | Kernel which is composed of a set of other kernels. ][API][206] |
| `gaussian_process.kernels.ConstantKernel([…])` | Constant kernel. ][API][207] |
| `gaussian_process.kernels.DotProduct([…])` | Dot-Product kernel. ][API][208] |
| `gaussian_process.kernels.ExpSineSquared([…])` | Exp-Sine-Squared kernel. ][API][209] |
| `gaussian_process.kernels.Exponentiation(…)` | Exponentiate kernel by given exponent. | [API][210] |
| `gaussian_process.kernels.Hyperparameter` | A kernel hyperparameter’s specification in form of a namedtuple. | [API][211] |
| `gaussian_process.kernels.Kernel` | Base class for all kernels. | [API][212] |
| `gaussian_process.kernels.Matern([…])` | Matern kernel. | [API][213] |
| `gaussian_process.kernels.PairwiseKernel([…])` | Wrapper for kernels in sklearn.metrics.pairwise. | [API][214] |
| `gaussian_process.kernels.Product(k1, k2)` | Product-kernel k1 * k2 of two kernels k1 and k2. | [API][215] |
| `gaussian_process.kernels.RBF([length_scale, …])` | Radial-basis function kernel (aka squared-exponential kernel). | [API][216] |
| `gaussian_process.kernels.RationalQuadratic([…])` | Rational Quadratic kernel. | [API][217] |
| `gaussian_process.kernels.Sum(k1, k2)` | Sum-kernel k1 + k2 of two kernels k1 and k2. | [API][218] |
| `gaussian_process.kernels.WhiteKernel([…])` | White kernel. | [API][219] |


### [sklearn.isotonic][220]: Isotonic regression

User guide: See the [Isotonic regression][221] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `isotonic.IsotonicRegression([y_min, y_max, …])` | Isotonic regression model. | [API][222] |
| `isotonic.check_increasing(x, y)` | Determine whether y is monotonically correlated with x. | [API][223] |
| `isotonic.isotonic_regression(y[, …])` | Solve the isotonic regression model: | [API][224] |


### [sklearn.impute][225]: Impute

Transformers for missing value imputation

User guide: See the [Imputation of missing values][226] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `impute.SimpleImputer([missing_values, …])` | Imputation transformer for completing missing values. | [API][227] |
| `impute.MissingIndicator([missing_values, …])` | Binary indicators for missing values. | [API][228] |


### sklearn.kernel_approximation Kernel Approximation

The [sklearn.kernel_approximation][229] module implements several approximate kernel feature maps base on Fourier transforms.

User guide: See the [Kernel Approximation][230] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `kernel_approximation.AdditiveChi2Sampler([…])` | Approximate feature map for additive chi2 kernel. | [API][231] |
| `kernel_approximation.Nystroem([kernel, …])` | Approximate a kernel map using a subset of the training data. | [API][232] |
| `kernel_approximation.RBFSampler([gamma, …])` | Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform. | [API][233] |
| `kernel_approximation.SkewedChi2Sampler([…])` | Approximates feature map of the “skewed chi-squared” kernel by Monte Carlo approximation of its Fourier transform. | [API][234] |


### sklearn.kernel_ridge Kernel Ridge Regression

Module [sklearn.kernel_ridge][235] implements kernel ridge regression.

User guide: See the [Kernel ridge regression][236] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `kernel_ridge.KernelRidge([alpha, kernel, …])` | Kernel ridge regression. | [API][237] |


### sklearn.linear_model: Generalized Linear Models

The [sklearn.linear_model][238] module implements generalized linear models. It includes Ridge regression, Bayesian Regression, Lasso and Elastic Net estimators computed with Least Angle Regression and coordinate descent. It also implements Stochastic Gradient Descent related algorithms.

User guide: See the [Generalized Linear Models][239] section for further details.

| `linear_model.ARDRegression([n_iter, tol, …])` | Bayesian ARD regression. |[API][240] |
| `linear_model.BayesianRidge([n_iter, tol, …])` | Bayesian ridge regression |[API][241] |
| `linear_model.ElasticNet([alpha, l1_ratio, …])` | Linear regression with combined L1 and L2 priors as regularizer. |[API][242] |
| `linear_model.ElasticNetCV([l1_ratio, eps, …])` | Elastic Net model with iterative fitting along a regularization path |[API][243] |
| `linear_model.HuberRegressor([epsilon, …])` | Linear regression model that is robust to outliers. |[API][244] |
| `linear_model.Lars([fit_intercept, verbose, …])` | Least Angle Regression model a.k.a. |[API][245] |
| `linear_model.LarsCV([fit_intercept, …])` | Cross-validated Least Angle Regression model |[API][246] |
| `linear_model.Lasso([alpha, fit_intercept, …])` | Linear Model trained with L1 prior as regularizer (aka the Lasso) |[API][247] |
| `linear_model.LassoCV([eps, n_alphas, …])` | Lasso linear model with iterative fitting along a regularization path |[API][248] |
| `linear_model.LassoLars([alpha, …])` | Lasso model fit with Least Angle Regression a.k.a. |[API][249] |
| `linear_model.LassoLarsCV([fit_intercept, …])` | Cross-validated Lasso, using the LARS algorithm | [API][250] |
| `linear_model.LassoLarsIC([criterion, …])` | Lasso model fit with Lars using BIC or AIC for model selection | [API][251] |
| `linear_model.LinearRegression([…])` | Ordinary least squares Linear Regression. | [API][252] |
| `linear_model.LogisticRegression([penalty, …])` | Logistic Regression (aka logit, MaxEnt) classifier. | [API][253] |
| `linear_model.LogisticRegressionCV([Cs, …])` | Logistic Regression CV (aka logit, MaxEnt) classifier. | [API][254] |
| `linear_model.MultiTaskLasso([alpha, …])` | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer | [API][255] |
| `linear_model.MultiTaskElasticNet([alpha, …])` | Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer | [API][256] |
| `linear_model.MultiTaskLassoCV([eps, …])` | Multi-task L1/L2 Lasso with built-in cross-validation. | [API][257] |
| `linear_model.MultiTaskElasticNetCV([…])` | Multi-task L1/L2 ElasticNet with built-in cross-validation. | [API][258] |
| `linear_model.OrthogonalMatchingPursuit([…])` | Orthogonal Matching Pursuit model (OMP) | [API][259] |
| `linear_model.OrthogonalMatchingPursuitCV([…])` | Cross-validated Orthogonal Matching Pursuit model (OMP) | [API][260] |
| `linear_model.PassiveAggressiveClassifier([…])` | Passive Aggressive Classifier | [API][261] |
| `linear_model.PassiveAggressiveRegressor([C, …])` | Passive Aggressive Regressor | [API][262] |
| `linear_model.Perceptron([penalty, alpha, …])` | Read more in the User Guide. | [API][263] |
| `linear_model.RANSACRegressor([…])` | RANSAC (RANdom SAmple Consensus) algorithm. | [API][264] |
| `linear_model.Ridge([alpha, fit_intercept, …])` | Linear least squares with l2 regularization. | [API][265] |
| `linear_model.RidgeClassifier([alpha, …])` | Classifier using Ridge regression. | [API][266] |
| `linear_model.RidgeClassifierCV([alphas, …])` | Ridge classifier with built-in cross-validation. | [API][267] |
| `linear_model.RidgeCV([alphas, …])` | Ridge regression with built-in cross-validation. | [API][268] |
| `linear_model.SGDClassifier([loss, penalty, …])` | Linear classifiers (SVM, logistic regression, a.o.) with SGD training. | [API][269] |
| `linear_model.SGDRegressor([loss, penalty, …])` | Linear model fitted by minimizing a regularized empirical loss with SGD | [API][270] |
| `linear_model.TheilSenRegressor([…])` | Theil-Sen Estimator: robust multivariate regression model. | [API][271] |
| `linear_model.enet_path(X, y[, l1_ratio, …])` | Compute elastic net path with coordinate descent | [API][272] | 
| `linear_model.lars_path(X, y[, Xy, Gram, …])` | Compute Least Angle Regression or Lasso path using LARS algorithm [1] | [API][273] | 
| `linear_model.lasso_path(X, y[, eps, …])` | Compute Lasso path with coordinate descent | [API][274] | 
| `linear_model.logistic_regression_path(X, y)` | Compute a Logistic Regression model for a list of regularization parameters. | [API][275] | 
| `linear_model.orthogonal_mp(X, y[, …])` | Orthogonal Matching Pursuit (OMP) | [API][276] | 
| `linear_model.orthogonal_mp_gram(Gram, Xy[, …])` | Gram Orthogonal Matching Pursuit (OMP) | [API][277] | 
| `linear_model.ridge_regression(X, y, alpha[, …])` | Solve the ridge equation by the method of normal equations. | [API][278] | 




------------------------------
<!--
[279]: 
[280]: 
[281]: 
[282]: 
[283]: 
[284]: 
[285]: 
[286]: 
[287]: 
[288]: 
[289]: 
[290]: 
[291]: 
[292]: 
[293]: 
[294]: 
[295]: 
[296]: 
[297]: 
[298]: 
[299]: 
[300]: 
[301]: 
[302]: 
[303]: 
[304]: 
[305]: 
[306]: 
[307]: 
[308]: 
[309]: 
[310]: 
[311]: 
[312]: 
[313]: 
[314]: 
[315]: 
[316]: 
[317]: 
[318]: 
[319]: 
[320]: 
[321]: 
[322]: 
[323]: 
[324]: 
[325]: 
[326]: 
[327]: 
[328]: 
[329]: 
[330]: 
[331]: 
[332]: 
[333]: 
[334]: 
[335]: 
[336]: 
[337]: 
[338]: 
[339]: 
[340]: 
[341]: 
[342]: 
[343]: 
[344]: 
[345]: 
[346]: 
[347]: 
[348]: 
[349]: 
[350]: 
[351]: 
[352]: 
[353]: 
[354]: 
[355]: 
[356]: 
[357]: 
[358]: 
[359]: 
[360]: 
[361]: 
[362]: 
[363]: 
[364]: 
[365]: 
[366]: 
[367]: 
[368]: 
[369]: 
[370]: 
[371]: 
[372]: 
[373]: 
[374]: 
[375]: 
[376]: 
[377]: 
[378]: 
[379]: 
[380]: 
[381]: 
[382]: 
[383]: 
[384]: 
[385]: 
[386]: 
[387]: 
[388]: 
[389]: 
[390]: 
[391]: 
[392]: 
[393]: 
[394]: 
[395]: 
[396]: 
[397]: 
[398]: 
[399]: 
[400]: 
[401]: 
[402]: 
[403]: 
[404]: 
[405]: 
[406]: 
[407]: 
[408]: 
[409]: 
[410]: 
[411]: 
[412]: 
[413]: 
[414]: 
[415]: 
[416]: 
[417]: 
[418]: 
[419]: 
[420]: 
[421]: 
[422]: 
[423]: 
[424]: 
[425]: 
[426]: 
[427]: 
[428]: 
[429]: 
[430]: 
[431]: 
[432]: 
[433]: 
[434]: 
[435]: 
[436]: 
[437]: 
[438]: 
[439]: 
[440]: 
[441]: 
[442]: 
[443]: 
[444]: 
[445]: 
[446]: 
[447]: 
[448]: 
[449]: 
[450]: 
[451]: 
[452]: 
[453]: 
[454]: 
[455]: 
[456]: 
[457]: 
[458]: 
[459]: 
[460]: 
[461]: 
[462]: 
[463]: 
[464]: 
[465]: 
[466]: 
[467]: 
[468]: 
[469]: 
[470]: 
[471]: 
[472]: 
[473]: 
[474]: 
[475]: 
[476]: 
[477]: 
[478]: 
[479]: 
[480]: 
[481]: 
[482]: 
[483]: 
[484]: 
[485]: 
[486]: 
[487]: 
[488]: 
[489]: 
[490]: 
[491]: 
[492]: 
[493]: 
[494]: 
[495]: 
[496]: 
[497]: 
[498]: 
[499]: 

-->



[000]: http://scikit-learn.org/stable/modules/classes.html
[001]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base
[002]: http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator
[003]: http://scikit-learn.org/stable/modules/generated/sklearn.base.BiclusterMixin.html#sklearn.base.BiclusterMixin
[004]: http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin
[005]: http://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html#sklearn.base.ClusterMixin
[006]: http://scikit-learn.org/stable/modules/generated/sklearn.base.DensityMixin.html#sklearn.base.DensityMixin
[007]: http://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html#sklearn.base.RegressorMixin
[008]: http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin
[009]: http://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone
[010]: http://scikit-learn.org/stable/modules/generated/sklearn.base.is_classifier.html#sklearn.base.is_classifier
[011]: http://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html#sklearn.base.is_regressor
[012]: http://scikit-learn.org/stable/modules/generated/sklearn.config_context.html#sklearn.config_context
[013]: http://scikit-learn.org/stable/modules/generated/sklearn.get_config.html#sklearn.get_config
[014]: http://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config
[015]: http://scikit-learn.org/stable/modules/generated/sklearn.show_versions.html#sklearn.
[016]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration
[017]: http://scikit-learn.org/stable/modules/calibration.html#calibration
[018]: http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV
[019]: http://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve
[020]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
[021]: http://scikit-learn.org/stable/modules/clustering.html#clustering
[022]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
[023]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
[024]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
[025]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
[026]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration
[027]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
[028]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
[029]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift
[030]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
[031]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.affinity_propagation.html#sklearn.cluster.affinity_propagation
[032]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html#sklearn.cluster.dbscan
[033]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html#sklearn.cluster.estimate_bandwidth
[034]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means
[035]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.mean_shift.html#sklearn.cluster.mean_shift
[036]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html#sklearn.cluster.spectral_clustering
[037]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.ward_tree.html#sklearn.cluster.ward_tree
[038]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster.bicluster
[039]: http://scikit-learn.org/stable/modules/biclustering.html#biclustering
[040]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.bicluster.SpectralBiclustering.html#sklearn.cluster.bicluster.SpectralBiclustering
[041]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.bicluster.SpectralCoclustering.html#sklearn.cluster.bicluster.SpectralCoclustering
[042]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose
[043]: http://scikit-learn.org/stable/modules/compose.html#combining-estimators
[044]: http://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer
[045]: http://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor
[046]: http://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer
[047]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance
[048]: http://scikit-learn.org/stable/modules/covariance.html#covariance
[049]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance
[050]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope
[051]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso
[052]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLassoCV.html#sklearn.covariance.GraphicalLassoCV
[053]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html#sklearn.covariance.LedoitWolf
[054]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html#sklearn.covariance.MinCovDet
[055]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html#sklearn.covariance.OAS
[056]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html#sklearn.covariance.ShrunkCovariance
[057]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html#sklearn.covariance.empirical_covariance
[058]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.graphical_lasso.html#sklearn.covariance.graphical_lasso
[059]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html#sklearn.covariance.ledoit_wolf
[060]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.oas.html#sklearn.covariance.oas
[061]: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html#sklearn.covariance.shrunk_covariance
[062]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition
[063]: http://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition
[064]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html#sklearn.cross_decomposition.CCA
[065]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html#sklearn.cross_decomposition.PLSCanonical
[066]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn.cross_decomposition.PLSRegression
[067]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD
[068]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
[069]: http://scikit-learn.org/stable/datasets/index.html#datasets
[070]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.clear_data_home.html#sklearn.datasets.clear_data_home
[071]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html#sklearn.datasets.dump_svmlight_file
[072]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups
[073]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html#sklearn.datasets.fetch_20newsgroups_vectorized
[074]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing
[075]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype
[076]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html#sklearn.datasets.fetch_kddcup99
[077]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html#sklearn.datasets.fetch_lfw_pairs
[078]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html#sklearn.datasets.fetch_lfw_people
[079]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html#sklearn.datasets.fetch_olivetti_faces
[080]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml
[081]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html#sklearn.datasets.fetch_rcv1
[082]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_species_distributions.html#sklearn.datasets.fetch_species_distributions
[083]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.get_data_home.html#sklearn.datasets.get_data_home
[084]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston
[085]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
[086]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes
[087]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
[088]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html#sklearn.datasets.load_files
[089]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
[090]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud
[091]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html#sklearn.datasets.load_sample_image
[092]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html#sklearn.datasets.load_sample_images
[093]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html#sklearn.datasets.load_svmlight_file
[094]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_files.html#sklearn.datasets.load_svmlight_files
[095]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine
[096]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.mldata_filename.html#sklearn.datasets.mldata_filename
[097]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_biclusters.html#sklearn.datasets.make_biclusters
[098]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
[099]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_checkerboard.html#sklearn.datasets.make_checkerboard
[100]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles
[101]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
[102]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman1
[103]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman2
[104]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman3
[105]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles
[106]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html#sklearn.datasets.make_hastie_10_2
[107]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_low_rank_matrix.html#sklearn.datasets.make_low_rank_matrix
[108]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons
[109]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html#sklearn.datasets.make_multilabel_classification
[110]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression
[111]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html#sklearn.datasets.make_s_curve
[112]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_coded_signal.html#sklearn.datasets.make_sparse_coded_signal
[113]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html#sklearn.datasets.make_sparse_spd_matrix
[114]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html#sklearn.datasets.make_sparse_uncorrelated
[115]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html#sklearn.datasets.make_spd_matrix
[116]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html#sklearn.datasets.make_swiss_roll
[117]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
[118]: http://scikit-learn.org/stable/modules/decomposition.html#decompositions
[119]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning
[120]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis
[121]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA
[122]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA
[123]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA
[124]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
[125]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html#sklearn.decomposition.MiniBatchDictionaryLearning
[126]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA
[127]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF
[128]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
[129]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA
[130]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparseCoder.html#sklearn.decomposition.SparseCoder
[131]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
[132]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning.html#sklearn.decomposition.dict_learning
[133]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning_online.html#sklearn.decomposition.dict_learning_online
[134]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html#sklearn.decomposition.fastica
[135]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.sparse_encode.html#sklearn.decomposition.sparse_encode
[136]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis
[137]: http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
[138]: http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
[139]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy
[140]: http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
[141]: http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier
[142]: http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor
[143]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
[144]: http://scikit-learn.org/stable/modules/ensemble.html#ensemble
[145]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
[146]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
[147]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
[148]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
[149]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
[150]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
[151]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
[152]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
[153]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
[154]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
[155]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
[156]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html#sklearn.ensemble.RandomTreesEmbedding
[157]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier
[158]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.partial_dependence.partial_dependence.html#sklearn.ensemble.partial_dependence.partial_dependence
[159]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.partial_dependence.plot_partial_dependence.html#sklearn.ensemble.partial_dependence.plot_partial_dependence
[160]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions
[161]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ChangedBehaviorWarning.html#sklearn.exceptions.ChangedBehaviorWarning
[162]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ConvergenceWarning.html#sklearn.exceptions.ConvergenceWarning
[163]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataConversionWarning.html#sklearn.exceptions.DataConversionWarning
[164]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataDimensionalityWarning.html#sklearn.exceptions.DataDimensionalityWarning
[165]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.EfficiencyWarning.html#sklearn.exceptions.EfficiencyWarning
[166]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.FitFailedWarning.html#sklearn.exceptions.FitFailedWarning
[167]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html#sklearn.exceptions.NotFittedError
[168]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NonBLASDotWarning.html#sklearn.exceptions.NonBLASDotWarning
[169]: http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html#sklearn.exceptions.UndefinedMetricWarning
[170]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction
[171]: http://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction
[172]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer
[173]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher
[174]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.image
[175]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html#sklearn.feature_extraction.image.extract_patches_2d
[176]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.grid_to_graph.html#sklearn.feature_extraction.image.grid_to_graph
[177]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.img_to_graph.html#sklearn.feature_extraction.image.img_to_graph
[178]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.reconstruct_from_patches_2d.html#sklearn.feature_extraction.image.reconstruct_from_patches_2d
[179]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html#sklearn.feature_extraction.image.PatchExtractor
[180]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
[181]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
[182]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer
[183]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
[184]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
[185]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
[186]: http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
[187]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect
[188]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile
[189]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
[190]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr
[191]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr
[192]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
[193]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe
[194]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
[195]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV
[196]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold
[197]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
[198]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
[199]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
[200]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif
[201]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression
[202]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
[203]: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process
[204]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
[205]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
[206]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.CompoundKernel.html#sklearn.gaussian_process.kernels.CompoundKernel
[207]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html#sklearn.gaussian_process.kernels.ConstantKernel
[208]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.DotProduct.html#sklearn.gaussian_process.kernels.DotProduct
[209]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html#sklearn.gaussian_process.kernels.ExpSineSquared
[210]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Exponentiation.html#sklearn.gaussian_process.kernels.Exponentiation
[211]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Hyperparameter.html#sklearn.gaussian_process.kernels.Hyperparameter
[212]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html#sklearn.gaussian_process.kernels.Kernel
[213]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern
[214]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html#sklearn.gaussian_process.kernels.PairwiseKernel
[215]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Product.html#sklearn.gaussian_process.kernels.Product
[216]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
[217]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html#sklearn.gaussian_process.kernels.RationalQuadratic
[218]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Sum.html#sklearn.gaussian_process.kernels.Sum
[219]: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html#sklearn.gaussian_process.kernels.WhiteKernel
[220]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic
[221]: http://scikit-learn.org/stable/modules/isotonic.html#isotonic
[222]: http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression
[223]: http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.check_increasing.html#sklearn.isotonic.check_increasing
[224]: http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.isotonic_regression.html#sklearn.isotonic.isotonic_regression
[225]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute
[226]: http://scikit-learn.org/stable/modules/impute.html#impute
[227]: http://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
[228]: http://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator
[229]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation
[230]: http://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation
[231]: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler
[232]: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem
[233]: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler
[234]: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html#sklearn.kernel_approximation.SkewedChi2Sampler
[235]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge
[236]: http://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge
[237]: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
[238]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
[239]: http://scikit-learn.org/stable/modules/linear_model.html#linear-model
[240]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression
[241]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
[242]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
[243]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV
[244]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor
[245]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars
[246]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html#sklearn.linear_model.LarsCV
[247]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
[248]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV
[249]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
[250]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV
[251]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC
[252]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
[253]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
[254]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
[255]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso
[256]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet
[257]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV
[258]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV
[259]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit
[260]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV
[261]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier
[262]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor
[263]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
[264]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor
[265]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
[266]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier
[267]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV
[268]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV
[269]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
[270]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
[271]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor
[272]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.enet_path.html#sklearn.linear_model.enet_path
[273]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path
[274]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path
[275]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.logistic_regression_path.html#sklearn.linear_model.logistic_regression_path
[276]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html#sklearn.linear_model.orthogonal_mp
[277]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp_gram.html#sklearn.linear_model.orthogonal_mp_gram
[278]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html#sklearn.linear_model.ridge_regression








