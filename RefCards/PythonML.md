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

| Function | Description | Link |
|----------|-------------|------|
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


### sklearn.manifold: Manifold Learning

The [sklearn.manifold][279] module implements data embedding techniques.

User guide: See the [Manifold learning][280] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `manifold.Isomap([n_neighbors, n_components, …])` | Isomap Embedding | [API][281] |
| `manifold.LocallyLinearEmbedding([…])` | Locally Linear Embedding | [API][282] |
| `manifold.MDS([n_components, metric, n_init, …])` | Multidimensional scaling | [API][283] |
| `manifold.SpectralEmbedding([n_components, …])` | Spectral embedding for non-linear dimensionality reduction. | [API][284] |
| `manifold.TSNE([n_components, perplexity, …])` | t-distributed Stochastic Neighbor Embedding. | [API][285] |
| `manifold.locally_linear_embedding(X, …[, …])` | Perform a Locally Linear Embedding analysis on the data. | [API][286] |
| `manifold.smacof(dissimilarities[, metric, …])` | Computes multidimensional scaling using the SMACOF algorithm. | [API][287] |
| `manifold.spectral_embedding(adjacency[, …])` | Project the sample on the first eigenvectors of the graph Laplacian. | [API][288] |

### sklearn.metrics: Metrics

See the [Model evaluation: quantifying the quality of predictions][289] section and the [Pairwise metrics, Affinities and Kernels][290] section of the user guide for further details.

The [sklearn.metrics][291] module includes score functions, performance metrics and pairwise metrics and distance computations.

#### Model Selection Interface

See the The [scoring parameter: defining model evaluation rules][292] section of the user guide for further details.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.check_scoring(estimator[, scoring, …])` | Determine scorer from user options. | [API][293] |
| `metrics.get_scorer(scoring)` | Get a scorer from string | [API][294] |
| `metrics.make_scorer(score_func[, …])` | Make a scorer from a performance metric or loss function. | [API][295] |

#### Classification metrics

See the [Classification metrics][296] section of the user guide for further details.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.accuracy_score(y_true, y_pred[, …])` | Accuracy classification score. | [API][297] |
| `metrics.auc(x, y[, reorder])` | Compute Area Under the Curve (AUC) using the trapezoidal rule | [API][298] |
| `metrics.average_precision_score(y_true, y_score)` | Compute average precision (AP) from prediction scores | [API][299] |
| `metrics.balanced_accuracy_score(y_true, y_pred)` | Compute the balanced accuracy | [API][300] |
| `metrics.brier_score_loss(y_true, y_prob[, …])` | Compute the Brier score. | [API][301] |
| `metrics.classification_report(y_true, y_pred)` | Build a text report showing the main classification metrics | [API][302] |
| `metrics.cohen_kappa_score(y1, y2[, labels, …])` | Cohen’s kappa: a statistic that measures inter-annotator agreement. | [API][303] |
| `metrics.confusion_matrix(y_true, y_pred[, …])` | Compute confusion matrix to evaluate the accuracy of a classification | [API][304] |
| `metrics.f1_score(y_true, y_pred[, labels, …])` | Compute the F1 score, also known as balanced F-score or F-measure | [API][305] |
| `metrics.fbeta_score(y_true, y_pred, beta[, …])` | Compute the F-beta score | [API][306] |
| `metrics.hamming_loss(y_true, y_pred[, …])` | Compute the average Hamming loss. | [API][307] |
| `metrics.hinge_loss(y_true, pred_decision[, …])` | Average hinge loss (non-regularized) | [API][308] |
| `metrics.jaccard_similarity_score(y_true, y_pred)` | Jaccard similarity coefficient score | [API][309] |
| `metrics.log_loss(y_true, y_pred[, eps, …])` | Log loss, aka logistic loss or cross-entropy loss. |[API][310] |
| `metrics.matthews_corrcoef(y_true, y_pred[, …])` | Compute the Matthews correlation coefficient (MCC) |[API][311] |
| `metrics.precision_recall_curve(y_true, …)` | Compute precision-recall pairs for different probability thresholds |[API][312] |
| `metrics.precision_recall_fscore_support(…)` | Compute precision, recall, F-measure and support for each class |[API][313] |
| `metrics.precision_score(y_true, y_pred[, …])` | Compute the precision |[API][314] |
| `metrics.recall_score(y_true, y_pred[, …])` | Compute the recall |[API][315] |
| `metrics.roc_auc_score(y_true, y_score[, …])` | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. |[API][316] |
| `metrics.roc_curve(y_true, y_score[, …])` | Compute Receiver operating characteristic (ROC) |[API][317] |
| `metrics.zero_one_loss(y_true, y_pred[, …])` | Zero-one classification loss. |[API][318] |


#### Regression metrics

See the [Regression metrics][319] section of the user guide for further details.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.explained_variance_score(y_true, y_pred)` | Explained variance regression score function | [API][320] |
| `metrics.mean_absolute_error(y_true, y_pred)` | Mean absolute error regression loss | [API][321] |
| `metrics.mean_squared_error(y_true, y_pred[, …])` | Mean squared error regression loss | [API][322] |
| `metrics.mean_squared_log_error(y_true, y_pred)` | Mean squared logarithmic error regression loss | [API][323] |
| `metrics.median_absolute_error(y_true, y_pred)` | Median absolute error regression loss | [API][324] |
| `metrics.r2_score(y_true, y_pred[, …])` | R^2 (coefficient of determination) regression score function. | [API][325] |

#### Multilabel ranking metrics

See the [Multilabel ranking][326] metrics section of the user guide for further details.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.coverage_error(y_true, y_score[, …])` | Coverage error measure | [API][327] |
| `metrics.label_ranking_average_precision_score(…)` | Compute ranking-based average precision | [API][328] |
| `metrics.label_ranking_loss(y_true, y_score)` | Compute Ranking loss measure | [API][329] |

#### Clustering metrics

See the [Clustering performance evaluation][330] section of the user guide for further details.

The [sklearn.metrics.cluster][331] submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:

+ supervised, which uses a ground truth class values for each sample.
+ unsupervised, which does not and measures the ‘quality’ of the model itself.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.adjusted_mutual_info_score(…[, …])` | Adjusted Mutual Information between two clusterings. |[API][332] |
| `metrics.adjusted_rand_score(labels_true, …)` | Rand index adjusted for chance. |[API][333] |
| `metrics.calinski_harabaz_score(X, labels)` | Compute the Calinski and Harabaz score. |[API][334] |
| `metrics.davies_bouldin_score(X, labels)` | Computes the Davies-Bouldin score. |[API][335] |
| `metrics.completeness_score(labels_true, …)` | Completeness metric of a cluster labeling given a ground truth. |[API][336] |
| `metrics.cluster.contingency_matrix(…[, …])` | Build a contingency matrix describing the relationship between labels. |[API][337] |
| `metrics.fowlkes_mallows_score(labels_true, …)` | Measure the similarity of two clusterings of a set of points. |[API][338] |
| `metrics.homogeneity_completeness_v_measure(…)` | Compute the homogeneity and completeness and V-Measure scores at once. |[API][339] |
| `metrics.homogeneity_score(labels_true, …)` | Homogeneity metric of a cluster labeling given a ground truth. | [API][340] |
| `metrics.mutual_info_score(labels_true, …)` | Mutual Information between two clusterings. | [API][341] |
| `metrics.normalized_mutual_info_score(…[, …])` | Normalized Mutual Information between two clusterings. | [API][342] |
| `metrics.silhouette_score(X, labels[, …])` | Compute the mean Silhouette Coefficient of all samples. | [API][343] |
| `metrics.silhouette_samples(X, labels[, metric])` | Compute the Silhouette Coefficient for each sample. | [API][344] |
| `metrics.v_measure_score(labels_true, labels_pred)` | V-measure cluster labeling given a ground truth. | [API][345] |

#### Biclustering metrics

See the [Biclustering evaluation][346] section of the user guide for further details.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.consensus_score(a, b[, similarity])` | The similarity of two sets of biclusters. | [API][347] |

#### Pairwise metrics

See the [Pairwise metrics, Affinities and Kernels][348] section of the user guide for further details.

| Function | Description | Link |
|----------|-------------|------|
| `metrics.pairwise.additive_chi2_kernel(X[, Y])` | Computes the additive chi-squared kernel between observations in X and Y | [API][349] |
| `metrics.pairwise.chi2_kernel(X[, Y, gamma])` | Computes the exponential chi-squared kernel X and Y. | [API][350] |
| `metrics.pairwise.cosine_similarity(X[, Y, …])` | Compute cosine similarity between samples in X and Y. | [API][351] |
| `metrics.pairwise.cosine_distances(X[, Y])` | Compute cosine distance between samples in X and Y. | [API][352] |
| `metrics.pairwise.distance_metrics()` | Valid metrics for pairwise_distances. | [API][353] |
| `metrics.pairwise.euclidean_distances(X[, Y, …])` | Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors. | [API][354] |
| `metrics.pairwise.kernel_metrics()` | Valid metrics for pairwise_kernels | [API][355] |
| `metrics.pairwise.laplacian_kernel(X[, Y, gamma])` | Compute the laplacian kernel between X and Y. | [API][356] |
| `metrics.pairwise.linear_kernel(X[, Y, …])` | Compute the linear kernel between X and Y. | [API][357] |
| `metrics.pairwise.manhattan_distances(X[, Y, …])` | Compute the L1 distances between the vectors in X and Y. | [API][358] |
| `metrics.pairwise.pairwise_kernels(X[, Y, …])` | Compute the kernel between arrays X and optional array Y. | [API][359] |
| `metrics.pairwise.polynomial_kernel(X[, Y, …])` | Compute the polynomial kernel between X and Y: | [API][360] |
| `metrics.pairwise.rbf_kernel(X[, Y, gamma])` | Compute the rbf (gaussian) kernel between X and Y: | [API][361] |
| `metrics.pairwise.sigmoid_kernel(X[, Y, …])` | Compute the sigmoid kernel between X and Y: | [API][362] |
| `metrics.pairwise.paired_euclidean_distances(X, Y)` | Computes the paired euclidean distances between X and Y | [API][363] |
| `metrics.pairwise.paired_manhattan_distances(X, Y)` | Compute the L1 distances between the vectors in X and Y. | [API][364] |
| `metrics.pairwise.paired_cosine_distances(X, Y)` | Computes the paired cosine distances between X and Y | [API][365] |
| `metrics.pairwise.paired_distances(X, Y[, metric])` | Computes the paired distances between X and Y. | [API][366] |
| `metrics.pairwise_distances(X[, Y, metric, …])` | Compute the distance matrix from a vector array X and optional Y. | [API][367] |
| `metrics.pairwise_distances_argmin(X, Y[, …])` | Compute minimum distances between one point and a set of points. | [API][368] |
| `metrics.pairwise_distances_argmin_min(X, Y)` | Compute minimum distances between one point and a set of points. | [API][369] |
| `metrics.pairwise_distances_chunked(X[, Y, …])` | Generate a distance matrix chunk by chunk with optional reduction | [API][370] |


### sklearn.mixture: Gaussian Mixture Models

The [sklearn.mixture][371] module implements mixture modeling algorithms.

User guide: See the [Gaussian mixture models][372] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `mixture.BayesianGaussianMixture([…])` | Variational Bayesian estimation of a Gaussian mixture. | [API][373] |
| `mixture.GaussianMixture([n_components, …])` | Gaussian Mixture. | [API][374] |


### [sklearn.model_selection][399]: Model Selection

User guide: See the [Cross-validation: evaluating estimator performance][375], [Tuning the hyper-parameters of an estimator][376] and [Learning curve][377] sections for further details.

#### Splitter Classes

| Function | Description | Link |
|----------|-------------|------|
| `model_selection.GroupKFold([n_splits])` | K-fold iterator variant with non-overlapping groups. | [API][379] |
| `model_selection.GroupShuffleSplit([…])` | Shuffle-Group(s)-Out cross-validation iterator | [API][379] |
| `model_selection.KFold([n_splits, shuffle, …])` | K-Folds cross-validator | [API][380] |
| `model_selection.LeaveOneGroupOut()` | Leave One Group Out cross-validator | [API][381] |
| `model_selection.LeavePGroupsOut(n_groups)` | Leave P Group(s) Out cross-validator | [API][382] |
| `model_selection.LeaveOneOut()` | Leave-One-Out cross-validator | [API][383] |
| `model_selection.LeavePOut(p)` | Leave-P-Out cross-validator | [API][384] |
| `model_selection.PredefinedSplit(test_fold)` | Predefined split cross-validator | [API][385] |
| `model_selection.RepeatedKFold([n_splits, …])` | Repeated K-Fold cross validator. | [API][386] |
| `model_selection.RepeatedStratifiedKFold([…])` | Repeated Stratified K-Fold cross validator. | [API][387] |
| `model_selection.ShuffleSplit([n_splits, …])` | Random permutation cross-validator | [API][388] |
| `model_selection.StratifiedKFold([n_splits, …])` | Stratified K-Folds cross-validator | [API][389] |
| `model_selection.StratifiedShuffleSplit([…])` | Stratified ShuffleSplit cross-validator | [API][390] |
| `model_selection.TimeSeriesSplit([n_splits, …])` | Time Series cross-validator | [API][391] |

#### Splitter Functio23ns

| Function | Description | Link |
|----------|-------------|------|
| `model_selection.check_cv([cv, y, classifier])` | Input checker utility for building a cross-validator | [API][392] |
| `model_selection.train_test_split(*arrays, …)` | Split arrays or matrices into random train and test subsets | [API][393] |

#### Hyper-parameter optimizers

| Function | Description | Link |
|----------|-------------|------|
| `model_selection.GridSearchCV(estimator, …)` | Exhaustive search over specified parameter values for an estimator. | [API][394] |
| `model_selection.ParameterGrid(param_grid)` | Grid of parameters with a discrete number of values for each. | [API][395] |
| `model_selection.ParameterSampler(…[, …])` | Generator on parameters sampled from given distributions. | [API][396] |
| `model_selection.RandomizedSearchCV(…[, …])` | Randomized search on hyper parameters. | [API][397] |
| `model_selection.fit_grid_point(X, y, …[, …])` | Run fit on one set of parameters. | [API][398] |

#### Model validation

| Function | Description | Link |
|----------|-------------|------|
| `model_selection.cross_validate(estimator, X)` | Evaluate metric(s) by cross-validation and also record fit/score times. | [API][400] |
| `model_selection.cross_val_predict(estimator, X)` | Generate cross-validated estimates for each input data point | [API][401] |
| `model_selection.cross_val_score(estimator, X)` | Evaluate a score by cross-validation | [API][402] |
| `model_selection.learning_curve(estimator, X, y)` | Learning curve. | [API][403] |
| `model_selection.permutation_test_score(…)` | Evaluate the significance of a cross-validated score with permutations | [API][404] |
| `model_selection.validation_curve(estimator, …)` | Validation curve. | [API][405] |


### [sklearn.multiclass][410]: Multiclass and multilabel classification

This module implements multiclass learning algorithms:
+ one-vs-the-rest / one-vs-all
+ one-vs-one
+ error correcting output codes

The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. For example, it is possible to use these estimators to turn a binary classifier or a regressor into a multiclass classifier. It is also possible to use these estimators with multiclass estimators in the hope that their accuracy or runtime performance improves.

All classifiers in scikit-learn implement multiclass classification; you only need to use this module if you want to experiment with custom multiclass strategies.

The one-vs-the-rest meta-classifier also implements a predict_proba method, so long as such a method is implemented by the base classifier. This method returns probabilities of class membership in both the single label and multilabel case. Note that in the multilabel case, probabilities are the marginal probability that a given sample falls in the given class. As such, in the multilabel case the sum of these probabilities over all possible labels for a given sample will not sum to unity, as they do in the single label case.

User guide: See the [Multiclass and multilabel algorithms][406] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `multiclass.OneVsRestClassifier(estimator[, …])` | One-vs-the-rest (OvR) multiclass/multilabel strategy | [API][407] |
| `multiclass.OneVsOneClassifier(estimator[, …])` | One-vs-one multiclass strategy | [API][408] |
| `multiclass.OutputCodeClassifier(estimator[, …])` | (Error-Correcting) Output-Code multiclass strategy | [API][409] |


### [sklearn.multioutput][411]: Multioutput regression and classification

This module implements multioutput regression and classification.

The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. The meta-estimator extends single output estimators to multioutput estimators.

User guide: See the [Multiclass and multilabel algorithms][412] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `multioutput.ClassifierChain(base_estimator)` | A multi-label model that arranges binary classifiers into a chain. | [API][413] |
| `multioutput.MultiOutputRegressor(estimator)` | Multi target regression | [API][414] |
| `multioutput.MultiOutputClassifier(estimator)` | Multi target classification | [API][415] |
| `multioutput.RegressorChain(base_estimator[, …])` | A multi-label model that arranges regressions into a chain. | [API][416] |


### sklearn.naive_bayes: Naive Bayes

The [sklearn.naive_bayes][417] module implements Naive Bayes algorithms. These are supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions.

User guide: See the [Naive Bayes][418] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `naive_bayes.BernoulliNB([alpha, binarize, …])` | Naive Bayes classifier for multivariate Bernoulli models. | [API][419] |
| `naive_bayes.GaussianNB([priors, var_smoothing])` | Gaussian Naive Bayes (GaussianNB) | [API][420] |
| `naive_bayes.MultinomialNB([alpha, …])` | Naive Bayes classifier for multinomial models | [API][421] |
| `naive_bayes.ComplementNB([alpha, fit_prior, …])` | The Complement Naive Bayes classifier described in Rennie et al. | [API][422] |


### sklearn.neighbors: Nearest Neighbors

The [sklearn.neighbors][423] module implements the k-nearest neighbors algorithm.

User guide: See the [Nearest Neighbors][424] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `neighbors.BallTree` | BallTree for fast generalized N-point problems | [API][425] |
| `neighbors.DistanceMetric` | DistanceMetric class | [API][426] |
| `neighbors.KDTree` | KDTree for fast generalized N-point problems | [API][427] |
| `neighbors.KernelDensity([bandwidth, …])` | Kernel Density Estimation | [API][428] |
| `neighbors.KNeighborsClassifier([…])` | Classifier implementing the k-nearest neighbors vote. | [API][429] |
| `neighbors.KNeighborsRegressor([n_neighbors, …])` | Regression based on k-nearest neighbors. | [API][430] |
| `neighbors.LocalOutlierFactor([n_neighbors, …])` | Unsupervised Outlier Detection using Local Outlier Factor (LOF) | [API][431] |
| `neighbors.RadiusNeighborsClassifier([…])` | Classifier implementing a vote among neighbors within a given radius | [API][432] |
| `neighbors.RadiusNeighborsRegressor([radius, …])` | Regression based on neighbors within a fixed radius. | [API][433] |
| `neighbors.NearestCentroid([metric, …])` | Nearest centroid classifier. | [API][434] |
| `neighbors.NearestNeighbors([n_neighbors, …])` | Unsupervised learner for implementing neighbor searches. | [API][435] |
| `neighbors.kneighbors_graph(X, n_neighbors[, …])` | Computes the (weighted) graph of k-Neighbors for points in X | [API][436] |
| `neighbors.radius_neighbors_graph(X, radius)` | Computes the (weighted) graph of Neighbors for points in X | [API][437] |


### sklearn.neural_network: Neural network models

The [sklearn.neural_network][438] module includes models based on neural networks.

User guide: See the [Neural network models (supervised)][439] and [Neural network models (unsupervised)][440] sections for further details.

| Function | Description | Link |
|----------|-------------|------|
| `neural_network.BernoulliRBM([n_components, …])` | Bernoulli Restricted Boltzmann Machine (RBM). | [API][441] |
| `neural_network.MLPClassifier([…])` | Multi-layer Perceptron classifier. | [API][442] |
| `neural_network.MLPRegressor([…])` | Multi-layer Perceptron regressor. | [API][443] |


### sklearn.pipeline: Pipeline

The [sklearn.pipeline][444] module implements utilities to build a composite estimator, as a chain of transforms and estimators.

| Function | Description | Link |
|----------|-------------|------|
| `pipeline.FeatureUnion(transformer_list[, …])` | Concatenates results of multiple transformer objects. | [API][445] |
| `pipeline.Pipeline(steps[, memory])` | Pipeline of transforms with a final estimator. | [API][446] |
| `pipeline.make_pipeline(*steps, **kwargs)` | Construct a Pipeline from the given estimators. | [API][447] |
| `pipeline.make_union(*transformers, **kwargs)` | Construct a FeatureUnion from the given transformers. | [API][448] |


### sklearn.preprocessing: Preprocessing and Normalization

The [sklearn.preprocessing][449] module includes scaling, centering, normalization, binarization and imputation methods.

User guide: See the [Preprocessing data][450] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `preprocessing.Binarizer([threshold, copy])` | Binarize data (set feature values to 0 or 1) according to a threshold | [API][451] |
| `preprocessing.FunctionTransformer([func, …])` | Constructs a transformer from an arbitrary callable. | [API][452] |
| `preprocessing.KBinsDiscretizer([n_bins, …])` | Bin continuous data into intervals. | [API][453] |
| `preprocessing.KernelCenterer()` | Center a kernel matrix | [API][454] |
| `preprocessing.LabelBinarizer([neg_label, …])` | Binarize labels in a one-vs-all fashion | [API][455] |
| `preprocessing.LabelEncoder` | Encode labels with value between 0 and n_classes-1. | [API][456] |
| `preprocessing.MultiLabelBinarizer([classes, …])` | Transform between iterable of iterables and a multilabel format | [API][457] |
| `preprocessing.MaxAbsScaler([copy])` | Scale each feature by its maximum absolute value. | [API][458] |
| `preprocessing.MinMaxScaler([feature_range, copy])` | Transforms features by scaling each feature to a given range. | [API][459] |
| `preprocessing.Normalizer([norm, copy])` | Normalize samples individually to unit norm. | [API][460] |
| `preprocessing.OneHotEncoder([n_values, …])` | Encode categorical integer features as a one-hot numeric array. | [API][461] |
| `preprocessing.OrdinalEncoder([categories, dtype])` | Encode categorical features as an integer array. | [API][462] |
| `preprocessing.PolynomialFeatures([degree, …])` | Generate polynomial and interaction features. | [API][463] |
| `preprocessing.PowerTransformer([method, …])` | Apply a power transform featurewise to make data more Gaussian-like. | [API][464] |
| `preprocessing.QuantileTransformer([…])` | Transform features using quantiles information. | [API][465] |
| `preprocessing.RobustScaler([with_centering, …])` | Scale features using statistics that are robust to outliers. | [API][466] |
| `preprocessing.StandardScaler([copy, …])` | Standardize features by removing the mean and scaling to unit variance | [API][467] |
| `preprocessing.add_dummy_feature(X[, value])` | Augment dataset with an additional dummy feature. | [API][468] |
| `preprocessing.binarize(X[, threshold, copy])` | Boolean thresholding of array-like or scipy.sparse matrix | [API][469] |
| `preprocessing.label_binarize(y, classes[, …])` | Binarize labels in a one-vs-all fashion | [API][470] |
| `preprocessing.maxabs_scale(X[, axis, copy])` | Scale each feature to the [-1, 1] range without breaking the sparsity. | [API][471] |
| `preprocessing.minmax_scale(X[, …])` | Transforms features by scaling each feature to a given range. | [API][472] |
| `preprocessing.normalize(X[, norm, axis, …])` | Scale input vectors individually to unit norm (vector length). | [API][473] |
| `preprocessing.quantile_transform(X[, axis, …])` | Transform features using quantiles information. | [API][474] |
| `preprocessing.robust_scale(X[, axis, …])` | Standardize a dataset along any axis | [API][475] |
| `preprocessing.scale(X[, axis, with_mean, …])` | Standardize a dataset along any axis | [API][476] |
| `preprocessing.power_transform(X[, method, …])` | Apply a power transform featurewise to make data more Gaussian-like. | [API][477] |


### sklearn.semi_supervised: Semi-Supervised Learning

The [sklearn.semi_supervised][478] module implements semi-supervised learning algorithms. These algorithms utilized small amounts of labeled data and large amounts of unlabeled data for classification tasks. This module includes Label Propagation.

User guide: See the [Semi-Supervised][479] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `semi_supervised.LabelPropagation([kernel, …])` | Label Propagation classifier | [API][480] |
| `semi_supervised.LabelSpreading([kernel, …])` | LabelSpreading model for semi-supervised learning | [API][481] |


### sklearn.svm: Support Vector Machines

The [sklearn.svm][482] module includes Support Vector Machine algorithms.

User guide: See the [Support Vector Machines][483] section for further details.

#### Estimators

| Function | Description | Link |
|----------|-------------|------|
| `svm.LinearSVC([penalty, loss, dual, tol, C, …])` | Linear Support Vector Classification. | [API][484] |
| `svm.LinearSVR([epsilon, tol, C, loss, …])` | Linear Support Vector Regression. | [API][485] |
| `svm.NuSVC([nu, kernel, degree, gamma, …])` | Nu-Support Vector Classification. | [API][486] |
| `svm.NuSVR([nu, C, kernel, degree, gamma, …])` | Nu Support Vector Regression. | [API][487] |
| `svm.OneClassSVM([kernel, degree, gamma, …])` | Unsupervised Outlier Detection. | [API][488] |
| `svm.SVC([C, kernel, degree, gamma, coef0, …])` | C-Support Vector Classification. | [API][489] |
| `svm.SVR([kernel, degree, gamma, coef0, tol, …])` | Epsilon-Support Vector Regression. | [API][490] |
| `svm.l1_min_c(X, y[, loss, fit_intercept, …])` | Return the lowest bound for C such that for C in (l1_min_C, infinity) the model is guaranteed not to be empty. | [API][491] |

#### Low-level methods

| Function | Description | Link |
|----------|-------------|------|
| `svm.libsvm.cross_validation` | Binding of the cross-validation routine (low-level routine) | [API][492] |
| `svm.libsvm.decision_function` | Predict margin (libsvm name for this is predict_values) | [API][493] |
| `svm.libsvm.fit` | Train the model using libsvm (low-level method) | [API][494] |
| `svm.libsvm.predict` | Predict target values of X given a model (low-level method) | [API][495] |
| `svm.libsvm.predict_proba` | Predict probabilities | [API][496] |


### sklearn.tree: Decision Trees

The [sklearn.tree][497] module includes decision tree-based models for classification and regression.

User guide: See the [Decision Trees][498] section for further details.

| Function | Description | Link |
|----------|-------------|------|
| `tree.DecisionTreeClassifier([criterion, …])` | A decision tree classifier. | [API][499] |
| `tree.DecisionTreeRegressor([criterion, …])` | A decision tree regressor. | [API][500] |
| `tree.ExtraTreeClassifier([criterion, …])` | An extremely randomized tree classifier. | [API][501] |
| `tree.ExtraTreeRegressor([criterion, …])` | An extremely randomized tree regressor. | [API][502] |
| `tree.export_graphviz(decision_tree[, …])` | Export a decision tree in DOT format. | [API][503] |


### sklearn.utils: Utilities

The [sklearn.utils][504] module includes various utilities.

Developer guide: See the [Utilities for Developers][505] page for further details.

| Function | Description | Link |
|----------|-------------|------|
| `utils.testing.mock_mldata_urlopen(*args, …)` | Object that mocks the urlopen function to fake requests to mldata. | [API][506] |
| `utils.arrayfuncs.cholesky_delete` |  | [API][507] |
| `utils.arrayfuncs.min_pos` | Find the minimum value of an array over positive values | [API][508] |
| `utils.as_float_array(X[, copy, force_all_finite])` | Converts an array-like to an array of floats. | [API][509] |
| `utils.assert_all_finite(X[, allow_nan])` | Throw a ValueError if X contains NaN or infinity. | [API][510] |
| `utils.bench.total_seconds(delta)` | helper function to emulate function total_seconds, | [API][511] |
| `utils.check_X_y(X, y[, accept_sparse, …])` | Input validation for standard estimators. | [API][512] |
| `utils.check_array(array[, accept_sparse, …])` | Input validation on an array, list, sparse matrix or similar. | [API][513] |
| `utils.check_consistent_length(*arrays)` | Check that all arrays have consistent first dimensions. | [API][514] |
| `utils.check_random_state(seed)` | Turn seed into a np.random.RandomState instance | [API][515] |
| `utils.class_weight.compute_class_weight(…)` | Estimate class weights for unbalanced datasets. | [API][516] |
| `utils.class_weight.compute_sample_weight(…)` | Estimate sample weights by class for unbalanced datasets. | [API][517] |
| `utils.deprecated([extra])` | Decorator to mark a function or class as deprecated. | [API][518] |
| `utils.estimator_checks.check_estimator(Estimator)` | Check if estimator adheres to scikit-learn conventions. | [API][519] |
| `utils.extmath.safe_sparse_dot(a, b[, …])` | Dot product that handle the sparse matrix case correctly | [API][520] |
| `utils.extmath.randomized_range_finder(A, …)` | Computes an orthonormal matrix whose range approximates the range of A. | [API][521] |
| `utils.extmath.randomized_svd(M, n_components)` | Computes a truncated randomized SVD | [API][522] |
| `utils.extmath.fast_logdet(A)` | Compute log(det(A)) for A symmetric | [API][523] |
| `utils.extmath.density(w, **kwargs)` | Compute density of a sparse vector | [API][524] |
| `utils.extmath.weighted_mode(a, w[, axis])` | Returns an array of the weighted modal (most common) value in a | [API][525] |
| `utils.gen_even_slices(n, n_packs[, n_samples])` | Generator to create n_packs slices going up to n. | [API][526] |
| `utils.graph.single_source_shortest_path_length(…)` | Return the shortest path length from source to all reachable nodes. | [API][527] |
| `utils.graph_shortest_path.graph_shortest_path` | Perform a shortest-path graph search on a positive directed or undirected graph. | [API][528] |
| `utils.indexable(*iterables)` | Make arrays indexable for cross-validation. | [API][529] |
| `utils.multiclass.type_of_target(y)` | Determine the type of data indicated by the target. | [API][530] |
| `utils.multiclass.is_multilabel(y)` | Check if y is in a multilabel format. | [API][531] |
| `utils.multiclass.unique_labels(*ys)` | Extract an ordered array of unique labels | [API][532] |
| `utils.murmurhash3_32` | Compute the 32bit murmurhash3 of key at seed. | [API][533] |
| `utils.resample(*arrays, **options)` | Resample arrays or sparse matrices in a consistent way | [API][534] |
| `utils.safe_indexing(X, indices)` | Return items or rows from X using indices. | [API][535] |
| `utils.safe_mask(X, mask)` | Return a mask which is safe to use on X. | [API][536] |
| `utils.safe_sqr(X[, copy])` | Element wise squaring of array-likes and sparse matrices. | [API][537] |
| `utils.shuffle(*arrays, **options)` | Shuffle arrays or sparse matrices in a consistent way | [API][538] |
| `utils.sparsefuncs.incr_mean_variance_axis(X, …)` | Compute incremental mean and variance along an axix on a CSR or CSC matrix. | [API][539] |
| `utils.sparsefuncs.inplace_column_scale(X, scale)` | Inplace column scaling of a CSC/CSR matrix. | [API][540] |
| `utils.sparsefuncs.inplace_row_scale(X, scale)` | Inplace row scaling of a CSR or CSC matrix. | [API][541] |
| `utils.sparsefuncs.inplace_swap_row(X, m, n)` | Swaps two rows of a CSC/CSR matrix in-place. | [API][542] |
| `utils.sparsefuncs.inplace_swap_column(X, m, n)` | Swaps two columns of a CSC/CSR matrix in-place. | [API][543] |
| `utils.sparsefuncs.mean_variance_axis(X, axis)` | Compute mean and variance along an axix on a CSR or CSC matrix | [API][544] |
| `utils.sparsefuncs.inplace_csr_column_scale(X, …)` | Inplace column scaling of a CSR matrix. | [API][545] |
| `utils.sparsefuncs_fast.inplace_csr_row_normalize_l1` | Inplace row normalize using the l1 norm | [API][546] |
| `utils.sparsefuncs_fast.inplace_csr_row_normalize_l2` | Inplace row normalize using the l2 norm | [API][547] |
| `utils.random.sample_without_replacement` | Sample integers without replacement. | [API][548] |
| `utils.validation.check_is_fitted(estimator, …)` | Perform is_fitted validation for estimator. | [API][549] |
| `utils.validation.check_memory(memory)` | Check that memory is joblib.Memory-like. | [API][550] |
| `utils.validation.check_symmetric(array[, …])` | Make sure that array is 2D, square and symmetric. | [API][551] |
| `utils.validation.column_or_1d(y[, warn])` | Ravel column or 1d numpy array, else raises an error | [API][552] |
| `utils.validation.has_fit_parameter(…)` | Checks whether the estimator’s fit method supports the given parameter. | [API][553] |
| `utils.testing.assert_in` | Just like self.assertTrue(a in b), but with a nicer default message. | [API][554] |
| `utils.testing.assert_not_in` | Just like self.assertTrue(a not in b), but with a nicer default message. | [API][555] |
| `utils.testing.assert_raise_message(…)` | Helper function to test the message raised in an exception. | [API][556] |
| `utils.testing.all_estimators([…])` | Get a list of all estimators from sklearn. | [API][557] |

#### Utilities from joblib:

+ Library Import 
    ```python
    
    ```

| Function | Description | Link |
|----------|-------------|------|
| `utils.Memory([location, backend, cachedir, …])` | A context object for caching a function’s return value each time it is called with the same input arguments. | [API][558] |
| `utils.Parallel([n_jobs, backend, verbose, …])` | Helper class for readable parallel mapping. | [API][559] |
| `utils.cpu_count()` | Return the number of CPUs. | [API][560] |
| `utils.delayed(function[, check_pickle])` | Decorator used to capture the arguments of a function. | [API][561] |
| `utils.parallel_backend(backend[, n_jobs])` | Change the default backend used by Parallel inside a with block. | [API][562] |






## Preprocessing

+ Library Import 
    ```python
    from sklearn.preprocessing import MinMaxScaler              # scaler

    # Text Mining
    from sklearn.feature_extraction.text import CountVectorizer # countvec
    from sklearn.feature_extraction.text import TfidfVectorizer # tfidfvec
    ```

| Function | Description | Link |
|----------|-------------|------|
| `MinMaxScaler(feature_range=(0, 1), copy=True)` | Transforms features by scaling each feature to a given range | [Supervised ML][565] |
| `scaler.fit_transform(X, y=None, **fit_params)` | Fit to data, then transform it | [Supervised ML][565] |
| `scaler.transform(X)` | Scaling features of X according to feature_range | [Supervised ML][565] |
| `countvec = CountVectorizer(input='content', *args)` | Convert a collection of text documents to a matrix of token counts | [Text Demo][586] |
| `countvec.fit(raw_documents, y=None)` | Learn a vocabulary dictionary of all tokens in the raw documents | [Text Demo][586] |
| `countvec.get_feature_names()` | Array mapping from feature integer indices to feature name | [Text Demo][586] |
| `countvec.transform(raw_documents)` | Transform documents to document-term matrix | [Text Demo][586] |
| `tfidfvec = TfidfVectorizer(input='content', *args)` | Convert a collection of raw documents to a matrix of TF-IDF features | [Text Demo][586] |
| `tfidfvec.fit(raw_documents, y=None)` | Learn vocabulary and idf from training set | [Text Demo][586] |
| `tfidfvec.get_feature_names(self)` | Array mapping from feature integer indices to feature name | [Text Demo][586] |
| `tfidfvec.transform(raw_documents, copy=True)` | Transform documents to document-term matrix | [Text Demo][586] |


## Model Selcetion

+ Library Import 
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import validation_curve
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import GridSearchCV
    ```

| Function | Description | Link |
|----------|-------------|------|
| `train_test_split(*arrays, **options)` | Split arrays or matrices into random train and test subsets | [Example][563], [Text][585] |
| `cross_val_score(estimator, X, *args)` | Evaluate a score by cross-validation | [Cross-Valiation][572] |
| `validation_curve(estimator, X, y, param_name, param_range, *args*)` | Compute scores for an estimator with different values of a specified parameter | [Cross-Validation][572] |
| `cross_val_score(estimator, X, *args*)` | Evaluate a score by cross-validation | [Model Selection]][580] |
| `cross_val_predict(estimator, X, y=None, *arg*)` | Generate cross-validated estimates for each input data point | [Text][585] |
| `GridSearchCV(estimator, param_grid, *args)` | Exhaustive search over specified parameter values for an estimator | [Model Selection][580] |
| `GridSearchCV.decision_function(X)` | Call decision_function on the estimator with the best found parameters | [Model Selection][580] |


## Classification

+ Library Import 
    ```python
    # Numeric Analysis
    from sklearn.neighbors import KNeighborsClassifier          # knn
    from sklearn.svm import SVC                                 # svc
    from sklearn.svm import LinearSVC                           # linsvc
    from sklearn.tree import DecisionTreeClassifier             # dtclf
    from sklearn.dummy import DummyClassifier                   # dummy
    from sklearn.naive_bayes import GaussianNB                  # nbclf
    from sklearn.ensemble import GradientBoostingClassifier     # gbdtclf
    from sklearn.neural_network import MLPClassifier            # mlpclf

    # Text Mining
    from sklearn.naive_bayes import MultinomialNB               # mnnbclf
    from nltk.classify import NaiveBayesClassifier              # nltknbclf
    from nltk.classify import SklearnClassifier                 # skclf
    ```

| Function | Description | Link |
|----------|-------------|------|
| `knn = KNeighborsClassifier(n_neighbors=5, **kwargs)` | Classifier implementing the k-nearest neighbors vote | [KNN][564] |
| `knn.fit(X, y)` | Fit the model using X as training data and y as target values | [KNN][564] |
| `knn.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [KNN][564] |
| `knn.predict(X)` | Predict the class labels for the provided data | [KNN][564] |
| `svc = SVC(C=1.0, kernel='rbf', *args)` | C-Support Vector Classification | [SVM][571], [Text][585] |
| `svc.fit(X, y, sample_weight=None)` | Fit the SVM model according to the given training data  | [SVM][571], [Text][585] |
| `svc.predict(X)` | Perform classification on samples in X.  For an one-class model, $+1$ or $-1$ is returned  | [SVM][571], [Text][585] |
| `svc.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels  | [SVM][571] |
| `linsvc = LinearSVC(penalty='l2', *args)` | Linear Support Vector Classification  | [SVM][571] |
| `linsvc.fit(X, y, sample_weight=None)` | Fit the model according to the given training data  | [SVM][571] |
| `linsvc.predict(X)` | Predict class labels for samples in X  | [SVM][571] |
| `linsvc.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels  | [SVM][571] |
| `dtclf = DecisionTreeClassifier(criterion='gini', *args)` | A decision tree classifier | [Decision Tree][573] |
| `dtclf.fit(X, y, *args)`  | Build a decision tree classifier from the training set $(X, y)$. | [Decision Tree][573] |
| `dtclf.predict(X, check_input=True)` | Predict class or regression value for X | [Decision Tree][573] |
| `dtclf.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [Decision Tree][573] |
| `dummy = DummyClassifier(strategy='stratified', random_state=None, constant=None)` | DummyClassifier is a classifier that makes predictions using simple rules | [Model Evaluation][574] |
| `dummy.fit(X, y, sample_weight=None)` | Fit the random classifier | [Model Evaluation][574] |
| `dummy.predict(X)` | Perform classification on test vectors X | [Model Evaluation][574] |
| `dummy.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [Model Evaluation][574] |
| `nbclf = GaussianNB(priors=None)` | Gaussian Naive Bayes | [Naive Bayes][581] |
| `mnnbclf.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)` | Naive Bayes classifier for multinomial models | [Text][585] |
| `nbclf.fit(X, y, sample_weight=None)` | Fit Gaussian Naive Bayes according to `X`, `y` | [Naive Bayes][581], [Text][585] |
| `nbclf.predict(X)` | Perform classification on an array of test vectors `X` | [Naive Bayes][581], [Text][585] |
| `nbclf.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [Naive Bayes][581] |
| `rfclf = RandomForestClassifier(n_estimators=10, criterion='gini', *args)` | A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. | [Random Forest][582] |
| `rfclf.fit(X, y, sample_weight=None)` | Build a forest of trees from the training set (X, y) | [Random Forest][582] |
| `rfclf.predict(X)` | The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates. | [Random Forest][582] |
| `rfclf.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [Random Forest][582] |
| `gbdtclf = GradientBoostingClassifier(loss='deviance', *args)` | GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. | [GBDT][583] |
| `gbdtclf..fit(X, y, *args)` | Fit the gradient boosting model | [GBDT][583] |
| `gbdtclf..predict(X)` | Predict class for X | [GBDT][583] |
| `gbdtclf..score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [GBDT][583] |
| `mlpclf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', *args)` | Multi-layer Perceptron classifier; optimize the log-loss function using LBFGS or stochastic gradient descent | [NN][584] |
| `mlpclf.fit(X, y)` | Fit the model to data matrix X and target(s) y | [NN][584] |
| `mlpclf..predict(X)` | Predict using the multi-layer perceptron classifier | [NN][584] |
| `mlpclf.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [NN][584] |
| `nltknbclf = NaiveBayesClassifier(label_probdist, feature_probdist)` | A Naive Bayes classifier | [Text][585] |
| `nltknbclf.train(labeled_featuresets, *args)` | model training | [Text][585] |
| `nltknbclf.classify(featureset)` | return the most appropriate label for the given featureset | [Text][585] |
| `nltknbclf.classify_many(featuresets)` | Apply `self.classify()` to each element of `featuresets` | [Text][585] |
| `nltknbclf.labels()` | return the list of category labels used by this classifier | [Text][585] |
| `nltknbclf.show_most_informative_features(n=10)` | Display most $n$ informative features | [Text][585] |
| `nltk.classify.util.accuracy(classifier, gold)` | Accuracy of model | [Text][585] |
| `skclf = SklearnClassifier(estimator, dtype=<class 'float'>, sparse=True)` | Wrapper for scikit-learn classifiers | [Text][585] |
| `slclf.train(labeled_featuresets)` | Train (fit) the scikit-learn estimator | [Text][585] |




## Regression

+ Library Import
    ```python
    from sklearn.neighbors import KNeighborsRegressor           # knnreg
    from sklearn.linear_model import LinearRegression           # linreg
    from sklearn.linear_model import Ridge                      # linridge
    from sklearn.linear_model import Lasso                      # linlasso
    from sklearn.preprocessing import PolynomialFeatures        # poly
    from sklearn.linear_model import LogisticRegression         # logreg
    from sklearn.neural_network import MLPRegressor             # mlpreg
    ```

| Function | Description | Link |
|----------|-------------|------|
| `knnreg = KNeighborsRegressor(n_neighbors=5, *args)` | Regression based on k-nearest neighbors | [KNN][567] |
| `knnreg.fit(X, y)` | Fit the model using X as training data and y as target values| [KNN][567] |
| `knnreg.predict(X)` | Predict the target for the provided data | [KNN][567] |
| `knnreg.score(X, y, sample_weight=None)` | Returns the coefficient of determination $R^2$ of the prediction | [KNN][567] |
| `linreg = LinearRegression(fit_intercept=True, *args)` | Ordinary least squares Linear Regression | [Least Square][568] |
| `linreg.fit(X, y, sample_weight=None)` | Fit linear model | [Least Square][568] |
| `linreg.predict(X)` | Predict using the linear model | [Least Square][568] |
| `linreg.score(X, y, sample_weight=None)` | Returns the coefficient of determination $R^2$ of the prediction | [Least Square][568] |
| `linridge = Ridge(alpha=1.0, *args*)` | Linear least squares with l2 regularization | [Ridge][569] |
| `linridge.fit(X, y, sample_weight=None)` | Fit Ridge regression model | [Ridge][569] |
| `linridge.predict(X)` | Predict using the linear model | [Ridge][569] |
| `linridge.score(X, y, sample_weight=None)` | Returns the coefficient of determination $R^2$ of the prediction | [Ridge][569] |
| `linlasso = Lasso(alpha=1.0, *args*)` | Linear Model trained with L1 prior as regularizer | [Lasso][569] |
| `linlasso.fit(X, y, check_input=True)` | Fit model with coordinate descent | [Lasso][569] |
| `linlasso.predict(X)` | Predict using the linear model | [Lasso][569] |
| `linlasso.score(X, y, sample_weight=None)` | Returns the coefficient of determination $R^2$ of the prediction | [Lasso][569] |
| `poly = PolynomialFeatures(degree=2, *args)` | Generate polynomial and interaction features | [Poly][569] |
| `poly.fit_transform(X, y=None, **fit_params)` | Fit to data, then transform it. | [Poly][569] |
| `logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0. *args)` | Logistic Regression (aka logit, MaxEnt) classifier | [Logistic][570] |
| `logreg.LogisticRegression.fit(X, y, sample_weight=None)` | Fit the model according to the given training data | [Logistic][570] |
| `logreg.predict(X)` | Predict class labels for samples in X | [Logistic][570] |
| `logreg.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels | [Logistic][570] |
| `logreg.decision_function(X)` | The confidence score for a sample is the signed distance of that sample to the hyperplane | [Decision Functions][576] |
| `logreg.predict_proba(X)` | Probability estimates. The returned estimates for all classes are ordered by the label of classes. | [Decision Functions][576] |
| `mplreg = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', *args)` | Multi-layer Perceptron regressor; optimize the squared-loss using LBFGS or stochastic gradient descent | [NN][584] |
| `mlpreg.fit(X, y)` | Fit the model to data matrix X and target(s) `y` | [NN][584] |
| `mlpreg.predict(X)` | Predict using the multi-layer perceptron model | [NN][584] |
| `mlpreg.score(X, y, sample_weight=None)` | Returns the coefficient of determination $R^2$ of the prediction | [NN][584] |


## Clustering

+ Import Modules
    ```python

    ```

| Function | Description | Link |
|----------|-------------|------|



## Utilities, Metrics and Datasets

+ Import Modules
    ```python
    from sklearn.datasets import make_classification        # datasets
    from sklearn.datasets import make_blobs
    from sklearn.datasets import make_regression
    from sklearn.datasets import make_friedman1
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_digits

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, precision_recall_curve
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.metrics import mean_squared_error, r2_score
    ```

| Function | Description | Link |
|----------|-------------|------|
| `make_classification(n_samples=100, n_features=20, *args)` | Generate a random n-class classification problem | [Datasets][566] |
| `make_blobs(n_samples=100, n_features=2, *args)` | Generate isotropic Gaussian blobs for clustering | [Datasets][566] |
| `make_regression(n_samples=100, n_features=100, *args)` | nerate a random regression problem | [Datasets][566] |
| `make_friedman1(n_samples=100, n_features=10, noise=0.0, random_state=None)` | Generate the "Friedman #1" regression problem | [Datasets][566] |
| `load_breast_cancer(return_X_y=False)` | Load and return the breast cancer wisconsin dataset (classification) | [Datasets][566] |
| `load_iris(return_X_y=False)` | Load and return the iris dataset (classification) | [Decision Tree][573] |
| `confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)` | Compute confusion matrix to evaluate the accuracy of a classification | [Model Evaluation][574], [Confusion Matrix][575] |
| `accuracy_score(y_true, y_pred, *args*)` | Accuracy classification score | [Evaluation Metrics][575] |
| `precision_score(y_true, y_pred, *args*)` | The precision is the ratio $tp / (tp + fp)$ where $tp$ is the number of true positives and $fp$ the number of false positives. | [Evaluation Metrics][575] |
| `recall_score(y_true, y_pred, *args*)` | The recall is the ratio $tp / (tp + fn)$ where $tp$ is the number of true positives and $fn$ the number of false negatives. | [Evaluation Metrics][575] |
| `f1_score(y_true, y_pred, *args*)` | The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. | [Evaluation Metrics][575], [Text][585] |
| `classification_report(y_true, y_pred, *args*)` | Build a text report showing the main classification metrics | [Evaluation Metrics][575] |
| `precision_recall_curve(y_true, probas_pred, *args*)` | Compute precision-recall pairs for different probability thresholds | [Decision Functions][576] |
| `roc_curve(y_true, y_score, *args*)` | Compute Receiver operating characteristic (ROC) | [ROC Curves][577] |
| `auc(x, y, reorder=False)` | Compute Area Under the Curve (AUC) using the trapezoidal rule | [ROC Curves][577] |
| `mean_squared_error(y_true, y_pred, *args*)` | Mean squared error regression loss | [Regression Evaluation][578] |
| `r2_score(y_true, y_pred, *args)` | $R^2$ (coefficient of determination) regression score function | [Regression Evaluation][578] |
| `roc_auc_score(y_true, y_score, *args)` | Compute Area Under the Curve (AUC) from prediction scores | [Model Selection][580] |


## Semantic Analysis

### Import Files

```python
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.collocations import AbstractCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import QuadgramAssocMeasures
from nltk.collocations import TrigramAssocMeasures
```

### `wordnet` class

| Function | Description | Link |
|----------|-------------|------|
| `wordnet(root, omw_reader)` | Construct a new wordnet corpus reader, with the given root directory | [Semantic][587] |
| `synset(name)` | Loading Synsets | [Semantic][587] |
| `ic(corpus, weight_senses_equally=False, smoothing=1.0)` | Creates an information content lookup dictionary from a corpus | [Semantic][587] |
| `jcn_similarity(synset1, synset2, ic, verbose=False)` | __Jiang-Conrath Similarity__: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets. | [Semantic][587] |
| `lch_similarity(synset1, synset2, *args*)` | __Leacock Chodorow Similarity__: Return a score denoting how similar two word senses are, based on the shortest path that connects the senses (as above) and the maximum depth of the taxonomy in which the senses occur.  | [Semantic][587] |
| `lin_similarity(synset1, synset2, ic, verbose=False)` | __Lin Similarity__: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets.  | [Semantic][587] |
| `path_similarity(synset1, synset2, *args)` | __Path Distance Similarity__: Return a score denoting how similar two word senses are, based on the shortest path that connects the senses in the is-a (hypernym/hypnoym) taxonomy. The score is in the range 0 to 1, except in those cases where a path cannot be found (will only be true for verbs as there are many distinct verb taxonomies), in which case None is returned. A score of 1 represents identity i.e. comparing a sense with itself will return 1.
| `res_similarity(synset1, synset2, ic, verbose=False)` | __Resnik Similarity__: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node)  | [Semantic][587] |
| `wup_similarity(synset1, synset2, verbose=False, simulate_root=True)`: __Wu-Palmer Similarity__: Return a score denoting how similar two word senses are, based on the depth of the two senses in the taxonomy and that of their Least Common Subsumer (most specific ancestor node).  | [Semantic][587] |
| `lemma(name, lang='eng')` | Return lemma object that matches the name | [Semantic][587] |
| `lemma_count(lemma)` | Return the frequency count for this Lemma | [Semantic][587] |
| `lemma_from_key(key)` | Return the Lemma from given key | [Semantic][587] |
| `lemmas(lemma, pos=None, lang='eng')` | Return all Lemma objects with a name matching the specified lemma name and part of speech tag | [Semantic][587] |


### `wordnet_ic` class

| Function | Description | Link |
|----------|-------------|------|
| `wordnet_ic(root, fileids)` | A corpus reader for the WordNet information content corpus | [Semantic][587] |
| `ic(icfile)`: Load an information content file from the wordnet_ic corpus and return a dictionary | [Semantic][587] |


### `AbstractCollocationFinder` class

| Function | Description | Link |
|----------|-------------|------|
| `AbstractCollocationFinder(word_fd, ngram_fd)` | An abstract base class for collocation finders whose purpose is to collect collocation candidate frequencies, filter and rank them | [Semantic][587] |
| `above_score(score_fn, min_score)` | Returns a sequence of ngrams, ordered by decreasing score, whose scores each exceed the given minimum score. | [Semantic][587] |
| `apply_freq_filter(min_freq)`| Removes candidate ngrams which have frequency less than min_freq. | [Semantic][587] |
| `apply_ngram_filter(fn)` | Removes candidate ngrams $(w_1, w_2, \ldots)$ where $fn(w_1, w_2, \ldots)$ evaluates to True. | [Semantic][587] |
| `apply_word_filter(fn)`| Removes candidate ngrams $(w_1, w_2, \ldots)$ where any of $(fn(w_1), fn(w_2), \ldots)$ evaluates to True. | [Semantic][587] |
| `nbest(score_fn, n)`| Returns the top n ngrams when scored by the given function. | [Semantic][587] |
| `score_ngrams(score_fn)`| Returns a sequence of (ngram, score) pairs ordered from highest to lowest score, as determined by the scoring function provided. | [Semantic][587] |


### `BigramCollocationFinder(AbstractCollocationFinder)` class

| Function | Description | Link |
|----------|-------------|------|
| `BigramCollocationFinder(word_fd, bigram_fd, window_size=2)` | A tool for the finding and ranking of bigram collocations or other association measures
| `score_ngram(score_fn, w1, w2)` | Returns the score for a given bigram using the given scoring function | [Semantic][587] |
| `from_words(words, window_size=2)` | from `builtins.type`; Construct a BigramCollocationFinder for all bigrams in the given sequence | [Semantic][587] |


### `TrigramCollocationFinder(AbstractCollocationFinder)` class

| Function | Description | Link |
|----------|-------------|------|
| `TrigramCollocationFinder(word_fd, bigram_fd, wildcard_fd, trigram_fd)` | A tool for the finding and ranking of trigram collocations or other association measure | [Semantic][587] |
| `bigram_finder()` | Constructs a bigram collocation finder with the bigram and unigram data from this finder | [Semantic][587] |
| `score_ngram(score_fn, w1, w2, w3)` | Returns the score for a given trigram using the given scoring function. | [Semantic][587] |

### `QuadgramCollocationFinder(AbstractCollocationFinder)` class

| Function | Description | Link |
|----------|-------------|------|
| `QuadgramCollocationFinder(word_fd, quadgram_fd, ii, iii, ixi, ixxi, iixi, ixii)` | A tool for the finding and ranking of quadgram collocations or other association measures
| `score_ngram(score_fn, w1, w2, w3, w4)` | Returns the score for a given quadgram using the given scoring function | [Semantic][587] |
| `from_words(words, window_size=4)` | from `builtins.type`; Construct a QuadgramCollocationFinder for all quadgrams in the given sequence | [Semantic][587] |

### `NgramAssocMeasures(builtins.object)` class

| Function | Description | Link |
|----------|-------------|------|
| `nltk.NgramAssocMeasures()` | An abstract class defining a collection of generic association measures | [Semantic][587] |
| `chi_sq(*marginals)` | Scores ngrams using Pearson's chi-square as in Manning and Schutze 5.3.3. | [Semantic][587] |
| `jaccard(*marginals)` | Scores ngrams using the Jaccard index. | [Semantic][587] |
| `likelihood_ratio(*marginals)` |  Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4. | [Semantic][587] |
| `pmi(*marginals)` |  Scores ngrams by pointwise mutual information, as in Manning and Schutze 5.4. | [Semantic][587] |
| `poisson_stirling(*marginals)` |  Scores ngrams using the Poisson-Stirling measure. | [Semantic][587] |
| `student_t(*marginals)` | Scores ngrams using Student's t test with independence hypothesis for unigrams, as in Manning and Schutze 5.3.1. | [Semantic][587] |
| `mi_like(*marginals, **kwargs)` | Scores ngrams using a variant of mutual information | [Semantic][587] |
| `raw_freq(*marginals)` | Scores ngrams by their frequency | [Semantic][587] |


### `BigramAssocMeasures(NgramAssocMeasures)` Class

| Function | Description | Link |
|----------|-------------|------|
| `ntlk.BigramAssocMeasures()` | A collection of bigram association measures| [Semantic][587] |
| `bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)` | association measure| [Semantic][587] |
| `chi_sq(n_ii, n_ix_xi_tuple, n_xx)` | Scores bigrams using chi-square| [Semantic][587] |
| `fisher(*marginals)` | Scores bigrams using Fisher's Exact Test (Pedersen 1996)| [Semantic][587] |
| `phi_sq(*marginals)` | Scores bigrams using phi-square, the square of the Pearson correlation coefficient | [Semantic][587] |


### `TrigramAssocMeasures(NgramAssocMeasures)` class

| Function | Description | Link |
|----------|-------------|------|
| `nltk.TrigramAssocMeasures()` | A collection of trigram association measures | [Semantic][587] |
| `trigram_score_fn(n_iii, (n_iix, n_ixi, n_xii), (n_ixx, n_xix, n_xxi), n_xxx)` | associate measure | [Semantic][587] |


## Toic Modeling

### Import Files

```python
import gensim
from gensim import corpora, models
```

### `Dictionary(gensim.utils.SaveLoad, collections.abc.Mapping)` class

| Function | Description | Link |
|----------|-------------|------|
| `corpora.Dictionary(documents=None, prune_at=2000000)` | Dictionary encapsulates the mapping between normalized words and their integer ids | [Topic][588] |
| `doc2bow(document, allow_update=False, return_missing=False)` | Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples | [Topic][588] |
| `doc2idx(document, unknown_word_index=-1)` | Convert `document` (a list of words) into a list of indexes = list of `token_id` | [Topic][588] |
| `filter_extremes(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None)` | Filter out tokens in the dictionary by their frequency | [Topic][588] |
| `filter_n_most_frequent(remove_n)` | Filter out the 'remove_n' most frequent tokens that appear in the documents | [Topic][588] |
| `filter_tokens(bad_ids=None, good_ids=None)` | Remove the selected `bad_ids` tokens from `~gensim.corpora.dictionary.Dictionary` | [Topic][588] |
| `iteritems()`, `iterkeys = __iter__()`, `itervalues()`, `keys()` | Get all stored ids | [Topic][588] |
| `merge_with(other)` | Merge another dictionary into this dictionary, mapping the same tokens to the same ids and new tokens to new ids | [Topic][588] |
| `save_as_text(fname, sort_by_word=True)` | Save `~gensim.corpora.dictionary.Dictionary` to a text file | [Topic][588] |
| `from_corpus(corpus, id2word=None)` | Create `~gensim.corpora.dictionary.Dictionary` from an existing corpus | [Topic][588] |
| `from_documents(documents)` | Create `~gensim.corpora.dictionary.Dictionary` from `documents` | [Topic][588] |
| `load_from_text(fname)` | Load a previously stored `~gensim.corpora.dictionary.Dictionary` from a text file | [Topic][588] |
| `save(fname_or_handle, *args)` | Save the object to a file | [Topic][588] |
| `load(fname, mmap=None)` | Load an object previously saved using `~gensim.utils.SaveLoad.save` from a file | [Topic][588] |


### `LdaModel(gensim.interfaces.TransformationABC, gensim.models.basemodel.BaseTopicModel)` class

| Function | Description | Link |
|----------|-------------|------|
| `gensim.models.ldamodel.LdaModel(corpus=None, *args)` | Train and use Online Latent Dirichlet Allocation (OLDA) models as presented in Hoffman et al. : "[Online Learning for Latent Dirichlet Allocation](https://www.di.ens.fr/~fbach/mdhnips2010.pdf)" | [Topic][588] |
| `diff(other, *args)`: Calculate the difference in topic distributions between two models: `self` and `other` | [Topic][588] |
| `do_estep(chunk, state=None)` | Perform inference on a chunk of documents, and accumulate the collected sufficient statistics | [Topic][588] |
| `do_mstep(rho, other, extra_pass=False)` | Maximization step: use linear interpolation between the existing topics and collected sufficient statistics in `other` to update the topics | [Topic][588] |
| `get_document_topics(bow, *args)` | Get the topic distribution for the given document | [Topic][588] |
| `get_term_topics(word_id, minimum_probability=None)` | Get the most relevant topics to the given word | [Topic][588] |
| `get_topic_terms(topicid, topn=10)` | Get the representation for a single topic | [Topic][588] |
| `get_topics()` | Get the term-topic matrix learned during inference | [Topic][588] |
| `inference(chunk, collect_sstats=False)` |  Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights) for each document in the chunk | [Topic][588] |
| `init_dir_prior(prior, name)` | Initialize priors for the Dirichlet distribution | [Topic][588] |
| `log_perplexity(chunk, total_docs=None)` | Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus | [Topic][588] |
| `save(fname, *args, **kwargs)` | Save the model to a file | [Topic][588] |
| `show_topic(topicid, topn=10)` | Get the representation for a single topic | [Topic][588] |
| `show_topics(num_topics=10, *args)` | Get a representation for selected topics | [Topic][588] |
| `sync_state()` | Propagate the states topic probabilities to the inner object's attribute | [Topic][588] |
| `top_topics(corpus=None, *args)` | Get the topics with the highest coherence score the coherence for each topic | [Topic][588] |
| `update(corpus, *args)` | Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until the maximum number of allowed iterations is reached | [Topic][588] |
| `update_alpha(gammat, rho)` | Update parameters for the Dirichlet prior on the per-document topic weights | [Topic][588] |
| `update_eta(lambdat, rho)` | Update parameters for the Dirichlet prior on the per-topic word weights | [Topic][588] |
| `load(fname, *args, **kwargs)` | Load a previously saved `gensim.models.ldamodel.LdaModel` from file | [Topic][588] |
| `print_topic(topicno, topn=10)` | Get a single topic as a formatted string | [Topic][588] |
| `print_topics(num_topics=20, num_words=10)` | Get the most significant topics (alias for `show_topics()` method) | [Topic][588] |


### `LdaState(gensim.utils.SaveLoad)` class

| Function | Description | Link |
|----------|-------------|------|
| `gensim.models.ldamodel.LdaState(eta, shape, dtype=<class 'numpy.float32'>)` | Encapsulate information for distributed computation of `~gensim.models.ldamodel.LdaModel` objects | [Topic][588] |
| `blend(rhot, other, targetsize=None)` | Merge the current state with another one using a weighted average for the sufficient statistics | [Topic][588] |
| `blend2(rhot, other, targetsize=None)` | Merge the current state with another one using a weighted sum for the sufficient statistics | [Topic][588] |
| `get_Elogbeta()` | Get the log (posterior) probabilities for each topic | [Topic][588] |
| `get_lambda()` | Get the parameters of the posterior over the topics, also referred to as "the topics" | [Topic][588] |
| `merge(other)` | Merge the result of an E step from one node with that of another node (summing up sufficient statistics) | [Topic][588] |
| `reset()` | Prepare the state for a new EM iteration (reset sufficient stats) | [Topic][588] |
| `load(fname, *args, **kwargs)` | Load a previously stored state from disk | [Topic][588] |
| `save(fname_or_handle, *args*)` | Save the object to a file | [Topic][588] |
| `dirichlet_expectation(...)`, `dirichlet_expectation(alpha)` | Expected value of log(theta) where theta is drawn from a Dirichlet distribution | [Topic][588] |
| `logsumexp(...)`, `logsumexp(x)` | Log of sum of exponentials, using `~gensim._matutils._logsumexp_2d` | [Topic][588] |
| `mean_absolute_difference(...)`, `mean_absolute_difference(a, b)` | Mean absolute difference between two arrays, using `~gensim._matutils._mean_absolute_difference` | [Topic][588] |
| `update_dir_prior(prior, N, logphat, rho)` | Update a given prior using Newton's method, described in J. Huang: "[Maximum Likelihood Estimation of Dirichlet Distribution Parameters](http://jonathan-huang.org/research/dirichlet/dirichlet.pdf)" | [Topic][588] |
| 



------------------------------
<!-- 
[588]: ../AppliedDS-UMich/4-TextMining/04-Modeling.md#generative-models-and-lda
[589]: 
[590]: 
[591]: 
[592]: 
[593]: 
[594]: 
[595]: 
[596]: 
[597]: 
[598]: 
[599]: 
[600]: 
[601]: 
[602]: 
[603]: 
[604]: 
[605]: 
[606]: 
[607]: 
[608]: 
[609]: 
[610]: 
[611]: 
[612]: 
[613]: 
[614]: 
[615]: 
[616]: 
[617]: 
[618]: 
[619]: 
[620]: 
[621]: 
[622]: 
[623]: 
[624]: 
[625]: 
[626]: 
[627]: 
[628]: 
[629]: 
[630]: 
[631]: 
[632]: 
[633]: 
[634]: 
[635]: 
[636]: 
[637]: 
[638]: 
[639]: 
[640]: 
[641]: 
[642]: 
[643]: 
[644]: 
[645]: 
[646]: 
[647]: 
[648]: 
[649]: 
[650]: 
[651]: 
[652]: 
[653]: 
[654]: 
[655]: 
[656]: 
[657]: 
[658]: 
[659]: 
[660]: 
[661]: 
[662]: 
[663]: 
[664]: 
[665]: 
[666]: 
[667]: 
[668]: 
[669]: 
[670]: 
[671]: 
[672]: 
[673]: 
[674]: 
[675]: 
[676]: 
[677]: 
[678]: 
[679]: 
[680]: 
[681]: 
[682]: 
[683]: 
[684]: 
[685]: 
[686]: 
[687]: 
[688]: 
[689]: 
[690]: 
[691]: 
[692]: 
[693]: 
[694]: 
[695]: 
[696]: 
[697]: 
[698]: 
[699]: 

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
[279]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
[280]: http://scikit-learn.org/stable/modules/manifold.html#manifold
[281]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap
[282]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding
[283]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
[284]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html#sklearn.manifold.SpectralEmbedding
[285]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE
[286]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding
[287]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html#sklearn.manifold.smacof
[288]: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.spectral_embedding.html#sklearn.manifold.spectral_embedding
[289]: http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
[290]: http://scikit-learn.org/stable/modules/metrics.html#metrics
[291]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
[292]: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
[293]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.check_scoring.html#sklearn.metrics.check_scoring
[294]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer
[295]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
[296]: http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
[297]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
[298]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
[299]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
[300]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
[301]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss
[302]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
[303]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
[304]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
[305]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
[306]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score
[307]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss
[308]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss
[309]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score
[310]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
[311]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
[312]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve
[313]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
[314]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
[315]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
[316]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
[317]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
[318]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html#sklearn.metrics.zero_one_loss
[319]: http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
[320]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
[321]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
[322]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
[323]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error
[324]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
[325]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
[326]: http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
[327]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.coverage_error.html#sklearn.metrics.coverage_error
[328]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score
[329]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html#sklearn.metrics.label_ranking_loss
[330]: http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
[331]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster
[332]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
[333]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score
[334]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabaz_score.html#sklearn.metrics.calinski_harabaz_score
[335]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score
[336]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
[337]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html#sklearn.metrics.cluster.contingency_matrix
[338]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score
[339]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html#sklearn.metrics.homogeneity_completeness_v_measure
[340]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
[341]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score
[342]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
[343]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
[344]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples
[345]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
[346]: http://scikit-learn.org/stable/modules/biclustering.html#biclustering-evaluation
[347]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.consensus_score.html#sklearn.metrics.consensus_score
[348]: http://scikit-learn.org/stable/modules/metrics.html#metrics
[349]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.additive_chi2_kernel.html#sklearn.metrics.pairwise.additive_chi2_kernel
[350]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.chi2_kernel.html#sklearn.metrics.pairwise.chi2_kernel
[351]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity
[352]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances
[353]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
[354]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances
[355]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
[356]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html#sklearn.metrics.pairwise.laplacian_kernel
[357]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html#sklearn.metrics.pairwise.linear_kernel
[358]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html#sklearn.metrics.pairwise.manhattan_distances
[359]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html#sklearn.metrics.pairwise.pairwise_kernels
[360]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html#sklearn.metrics.pairwise.polynomial_kernel
[361]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel
[362]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html#sklearn.metrics.pairwise.sigmoid_kernel
[363]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_euclidean_distances.html#sklearn.metrics.pairwise.paired_euclidean_distances
[364]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_manhattan_distances.html#sklearn.metrics.pairwise.paired_manhattan_distances
[365]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_cosine_distances.html#sklearn.metrics.pairwise.paired_cosine_distances
[366]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html#sklearn.metrics.pairwise.paired_distances
[367]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
[368]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html#sklearn.metrics.pairwise_distances_argmin
[369]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html#sklearn.metrics.pairwise_distances_argmin_min
[370]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html#sklearn.metrics.pairwise_distances_chunked
[371]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture
[372]: http://scikit-learn.org/stable/modules/mixture.html#mixture
[373]: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
[374]: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
[375]: http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
[376]: http://scikit-learn.org/stable/modules/grid_search.html#grid-search
[377]: http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
[378]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold
[379]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit
[380]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold
[381]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut
[382]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut
[383]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut
[384]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html#sklearn.model_selection.LeavePOut
[385]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html#sklearn.model_selection.PredefinedSplit
[386]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold
[387]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold
[388]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
[389]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
[390]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
[391]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit
[392]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html#sklearn.model_selection.check_cv
[393]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
[394]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
[395]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid
[396]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler
[397]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
[398]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.fit_grid_point.html#sklearn.model_selection.fit_grid_point
[399]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
[400]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
[401]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict
[402]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
[403]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
[404]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html#sklearn.model_selection.permutation_test_score
[405]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve
[406]: http://scikit-learn.org/stable/modules/multiclass.html#multiclass
[407]: http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier
[408]: http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier
[409]: http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html#sklearn.multiclass.OutputCodeClassifier
[410]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass
[411]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput
[412]: http://scikit-learn.org/stable/modules/multiclass.html#multiclass
[413]: http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain
[414]: http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor
[415]: http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier
[416]: http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain
[417]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes
[418]: http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
[419]: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
[420]: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
[421]: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
[422]: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB
[423]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
[424]: http://scikit-learn.org/stable/modules/neighbors.html#neighbors
[425]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree
[426]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
[427]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
[428]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
[429]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
[430]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
[431]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
[432]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier
[433]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor
[434]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid
[435]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
[436]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html#sklearn.neighbors.kneighbors_graph
[437]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.radius_neighbors_graph.html#sklearn.neighbors.radius_neighbors_graph
[438]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network
[439]: http://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised
[440]: http://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#neural-networks-unsupervised
[441]: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM
[442]: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
[443]: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
[444]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
[445]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion
[446]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
[447]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline
[448]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_union.html#sklearn.pipeline.make_union
[449]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
[450]: http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
[451]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
[452]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer
[453]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer
[454]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html#sklearn.preprocessing.KernelCenterer
[455]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
[456]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
[457]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer
[458]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
[459]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
[460]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
[461]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
[462]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
[463]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
[464]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
[465]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
[466]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
[467]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
[468]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.add_dummy_feature.html#sklearn.preprocessing.add_dummy_feature
[469]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html#sklearn.preprocessing.binarize
[470]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#sklearn.preprocessing.label_binarize
[471]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html#sklearn.preprocessing.maxabs_scale
[472]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale
[473]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize
[474]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform
[475]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale
[476]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale
[477]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform
[478]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised
[479]: http://scikit-learn.org/stable/modules/label_propagation.html#semi-supervised
[480]: http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation
[481]: http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading
[482]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
[483]: http://scikit-learn.org/stable/modules/svm.html#svm
[484]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
[485]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
[486]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC
[487]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR
[488]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
[489]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
[490]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
[491]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.l1_min_c.html#sklearn.svm.l1_min_c
[492]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.cross_validation.html#sklearn.svm.libsvm.cross_validation
[493]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.decision_function.html#sklearn.svm.libsvm.decision_function
[494]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html#sklearn.svm.libsvm.fit
[495]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.predict.html#sklearn.svm.libsvm.predict
[496]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.predict_proba.html#sklearn.svm.libsvm.predict_proba
[497]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
[498]: http://scikit-learn.org/stable/modules/tree.html#tree
[499]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
[500]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
[501]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier
[502]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor
[503]: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
[504]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils
[505]: http://scikit-learn.org/stable/developers/utilities.html#developers-utils
[506]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.mock_mldata_urlopen.html#sklearn.utils.testing.mock_mldata_urlopen
[507]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.cholesky_delete.html#sklearn.utils.arrayfuncs.cholesky_delete
[508]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.min_pos.html#sklearn.utils.arrayfuncs.min_pos
[509]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.as_float_array.html#sklearn.utils.as_float_array
[510]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.assert_all_finite.html#sklearn.utils.assert_all_finite
[511]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.bench.total_seconds.html#sklearn.utils.bench.total_seconds
[512]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y
[513]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html#sklearn.utils.check_array
[514]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_consistent_length.html#sklearn.utils.check_consistent_length
[515]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state
[516]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html#sklearn.utils.class_weight.compute_class_weight
[517]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html#sklearn.utils.class_weight.compute_sample_weight
[518]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated
[519]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator
[520]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.safe_sparse_dot.html#sklearn.utils.extmath.safe_sparse_dot
[521]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_range_finder.html#sklearn.utils.extmath.randomized_range_finder
[522]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd
[523]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.fast_logdet.html#sklearn.utils.extmath.fast_logdet
[524]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.density.html#sklearn.utils.extmath.density
[525]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.weighted_mode.html#sklearn.utils.extmath.weighted_mode
[526]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_even_slices.html#sklearn.utils.gen_even_slices
[527]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.graph.single_source_shortest_path_length.html#sklearn.utils.graph.single_source_shortest_path_length
[528]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.graph_shortest_path.graph_shortest_path.html#sklearn.utils.graph_shortest_path.graph_shortest_path
[529]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.indexable.html#sklearn.utils.indexable
[530]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html#sklearn.utils.multiclass.type_of_target
[531]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.is_multilabel.html#sklearn.utils.multiclass.is_multilabel
[532]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.unique_labels.html#sklearn.utils.multiclass.unique_labels
[533]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.murmurhash3_32.html#sklearn.utils.murmurhash3_32
[534]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample
[535]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_indexing.html#sklearn.utils.safe_indexing
[536]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_mask.html#sklearn.utils.safe_mask
[537]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_sqr.html#sklearn.utils.safe_sqr
[538]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html#sklearn.utils.shuffle
[539]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.incr_mean_variance_axis.html#sklearn.utils.sparsefuncs.incr_mean_variance_axis
[540]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_column_scale.html#sklearn.utils.sparsefuncs.inplace_column_scale
[541]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_row_scale.html#sklearn.utils.sparsefuncs.inplace_row_scale
[542]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_row.html#sklearn.utils.sparsefuncs.inplace_swap_row
[543]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_column.html#sklearn.utils.sparsefuncs.inplace_swap_column
[544]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.mean_variance_axis.html#sklearn.utils.sparsefuncs.mean_variance_axis
[545]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_csr_column_scale.html#sklearn.utils.sparsefuncs.inplace_csr_column_scale
[546]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1
[547]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2
[548]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.random.sample_without_replacement.html#sklearn.utils.random.sample_without_replacement
[549]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted
[550]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_memory.html#sklearn.utils.validation.check_memory
[551]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_symmetric.html#sklearn.utils.validation.check_symmetric
[552]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.column_or_1d.html#sklearn.utils.validation.column_or_1d
[553]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.has_fit_parameter.html#sklearn.utils.validation.has_fit_parameter
[554]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.assert_in.html#sklearn.utils.testing.assert_in
[555]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.assert_not_in.html#sklearn.utils.testing.assert_not_in
[556]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.assert_raise_message.html#sklearn.utils.testing.assert_raise_message
[557]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.all_estimators.html#sklearn.utils.testing.all_estimators
[558]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.Memory.html#sklearn.utils.Memory
[559]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.Parallel.html#sklearn.utils.Parallel
[560]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.cpu_count.html#sklearn.utils.cpu_count
[561]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.delayed.html#sklearn.utils.delayed
[562]: http://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel_backend.html#sklearn.utils.parallel_backend
[563]: ../AppliedDS-UMich/3-AML/01-FundML.md#an-example-machine-learning-problem
[564]: ../AppliedDS-UMich/3-AML/01-FundML.md#k-nearest-neighbors-classification
[565]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#introduction-to-supervised-machine-learning
[566]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#supervised-learning-datasets
[567]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#k-nearest-neighbors-classification-and-regression
[568]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#linear-regression-least-squares
[569]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#linear-regression-ridge-lasso-and-polynomial-regression
[570]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#logistic-regression
[571]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#linear-classifiers-support-vector-machines
[572]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#cross-validation
[573]: ../AppliedDS-UMich/3-AML/02-Supervised1.md#decision-trees
[574]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#model-evaluation--selection
[575]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#confusion-matrices--basic-evaluation-metrics
[576]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#classifier-decision-functions
[577]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#precision-recall-and-roc-curves
[578]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#multi-class-evaluation
[579]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#regression-evaluation
[580]: ../AppliedDS-UMich/3-AML/03-Evaluation.md#model-selection-optimizing-classifiers-for-different-evaluation-metrics
[581]: ../AppliedDS-UMich/3-AML/04-Supervised2.md#naive-bayes-classifiers
[582]: ../AppliedDS-UMich/3-AML/04-Supervised2.md#random-forests
[583]: ../AppliedDS-UMich/3-AML/04-Supervised2.md#gradient-boosted-decision-trees
[584]: ../AppliedDS-UMich/3-AML/04-Supervised2.md#neural-networks
[585]: ../AppliedDS-UMich/4-TextMining/03-Classify.md#learning-text-classifiers-in-python
[586]: ../AppliedDS-UMich/4-TextMining/03-Classify.md#demonstration-case-study---sentiment-analysis
[587]: ../AppliedDS-UMich/4-TextMining/04-Modeling.md#semantic-text-similarity
