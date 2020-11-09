# Sklearn: Scikit-learn for Machine Learning

## sklearn.base: Base classes and utility functions

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base">sklearn.base</a>: Base classes and utility functions</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Class/Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Basic classes </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator" title="sklearn.base.BaseEstimator"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.BaseEstimator</span></code></a></p></td>
    <td><p>Base class for all estimators in scikit-learn</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.BiclusterMixin.html#sklearn.base.BiclusterMixin" title="sklearn.base.BiclusterMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.BiclusterMixin</span></code></a></p></td>
    <td><p>Mixin class for all bicluster estimators in scikit-learn</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin" title="sklearn.base.ClassifierMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.ClassifierMixin</span></code></a></p></td>
    <td><p>Mixin class for all classifiers in scikit-learn.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html#sklearn.base.ClusterMixin" title="sklearn.base.ClusterMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.ClusterMixin</span></code></a></p></td>
    <td><p>Mixin class for all cluster estimators in scikit-learn.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.DensityMixin.html#sklearn.base.DensityMixin" title="sklearn.base.DensityMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.DensityMixin</span></code></a></p></td>
    <td><p>Mixin class for all density estimators in scikit-learn.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html#sklearn.base.RegressorMixin" title="sklearn.base.RegressorMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.RegressorMixin</span></code></a></p></td>
    <td><p>Mixin class for all regression estimators in scikit-learn.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin" title="sklearn.base.TransformerMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.TransformerMixin</span></code></a></p></td>
    <td><p>Mixin class for all transformers in scikit-learn.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectorMixin.html#sklearn.feature_selection.SelectorMixin" title="sklearn.feature_selection.SelectorMixin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectorMixin</span></code></a></p></td>
    <td><p>Transformer mixin that performs feature selection given a support mask</p></td>
    </tr>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Functions </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone" title="sklearn.base.clone"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.clone</span></code></a>(estimator,&nbsp;*[,&nbsp;safe])</p></td>
    <td><p>Constructs a new estimator with the same parameters.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.is_classifier.html#sklearn.base.is_classifier" title="sklearn.base.is_classifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.is_classifier</span></code></a>(estimator)</p></td>
    <td><p>Return True if the given estimator is (probably) a classifier.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html#sklearn.base.is_regressor" title="sklearn.base.is_regressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">base.is_regressor</span></code></a>(estimator)</p></td>
    <td><p>Return True if the given estimator is (probably) a regressor.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html#sklearn.config_context" title="sklearn.config_context"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">config_context</span></code></a>(**new_config)</p></td>
    <td><p>Context manager for global scikit-learn configuration</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.get_config.html#sklearn.get_config" title="sklearn.get_config"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">get_config</span></code></a>()</p></td>
    <td><p>Retrieve current values for configuration set by <a https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config" title="sklearn.set_config"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">set_config</span></code></a></p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config" title="sklearn.set_config"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">set_config</span></code></a>([assume_finite,<br/>&nbsp;&nbsp;working_memory,&nbsp;…])</p></td>
    <td><p>Set global scikit-learn configuration</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.show_versions.html#sklearn.show_versions" title="sklearn.show_versions"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">show_versions</span></code></a>()</p></td>
    <td><p>Print useful debugging information”</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.calibration: Probability Calibration

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration">sklearn.calibration</a>: Probability Calibration</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV" title="sklearn.calibration.CalibratedClassifierCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">calibration.CalibratedClassifierCV</span></code></a><br/>&nbsp;&nbsp;([…])</p></td>
  <td><p>Probability calibration with isotonic regression or logistic regression.</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve" title="sklearn.calibration.calibration_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">calibration.calibration_curve</span></code></a><br/>&nbsp;&nbsp;(y_true,&nbsp;y_prob,&nbsp;*)</p></td>
  <td><p>Compute true and predicted probabilities for a calibration curve.</p></td>
  </tr>
  </tbody>
</table><br/>


## sklearn.cluster: Clustering


<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster">sklearn.cluster</a>: Clustering</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Class/Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Classes </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation" title="sklearn.cluster.AffinityPropagation"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.AffinityPropagation</span></code></a>(*[,&nbsp;damping,&nbsp;…])</p></td>
    <td><p>Perform Affinity Propagation Clustering of data.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering" title="sklearn.cluster.AgglomerativeClustering"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.AgglomerativeClustering</span></code></a>([…])</p></td>
    <td><p>Agglomerative Clustering</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch" title="sklearn.cluster.Birch"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.Birch</span></code></a>(*[,&nbsp;threshold,&nbsp;…])</p></td>
    <td><p>Implements the Birch clustering algorithm.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN" title="sklearn.cluster.DBSCAN"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.DBSCAN</span></code></a>([eps,&nbsp;min_samples,&nbsp;metric,&nbsp;…])</p></td>
    <td><p>Perform DBSCAN clustering from vector array or distance matrix.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration" title="sklearn.cluster.FeatureAgglomeration"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.FeatureAgglomeration</span></code></a>([n_clusters,&nbsp;…])</p></td>
    <td><p>Agglomerate features.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans" title="sklearn.cluster.KMeans"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.KMeans</span></code></a>([n_clusters,&nbsp;init,&nbsp;n_init,&nbsp;…])</p></td>
    <td><p>K-Means clustering.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans" title="sklearn.cluster.MiniBatchKMeans"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.MiniBatchKMeans</span></code></a>([n_clusters,&nbsp;init,&nbsp;…])</p></td>
    <td><p>Mini-Batch K-Means clustering.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift" title="sklearn.cluster.MeanShift"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.MeanShift</span></code></a>(*[,&nbsp;bandwidth,&nbsp;seeds,&nbsp;…])</p></td>
    <td><p>Mean shift clustering using a flat kernel.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS" title="sklearn.cluster.OPTICS"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.OPTICS</span></code></a>(*[,&nbsp;min_samples,&nbsp;max_eps,&nbsp;…])</p></td>
    <td><p>Estimate clustering structure from vector array.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering" title="sklearn.cluster.SpectralClustering"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.SpectralClustering</span></code></a>([n_clusters,&nbsp;…])</p></td>
    <td><p>Apply clustering to a projection of the normalized Laplacian.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html#sklearn.cluster.SpectralBiclustering" title="sklearn.cluster.SpectralBiclustering"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.SpectralBiclustering</span></code></a>([n_clusters,&nbsp;…])</p></td>
    <td><p>Spectral biclustering (Kluger, 2003).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html#sklearn.cluster.SpectralCoclustering" title="sklearn.cluster.SpectralCoclustering"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.SpectralCoclustering</span></code></a>([n_clusters,&nbsp;…])</p></td>
    <td><p>Spectral Co-Clustering algorithm (Dhillon, 2001).</p></td>
    </tr>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Functions </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.affinity_propagation.html#sklearn.cluster.affinity_propagation" title="sklearn.cluster.affinity_propagation"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.affinity_propagation</span></code></a>(S,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Perform Affinity Propagation Clustering of data</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_dbscan.html#sklearn.cluster.cluster_optics_dbscan" title="sklearn.cluster.cluster_optics_dbscan"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.cluster_optics_dbscan</span></code></a>(*,&nbsp;…)</p></td>
    <td><p>Performs DBSCAN extraction for an arbitrary epsilon.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_xi.html#sklearn.cluster.cluster_optics_xi" title="sklearn.cluster.cluster_optics_xi"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.cluster_optics_xi</span></code></a>(*,&nbsp;reachability,&nbsp;…)</p></td>
    <td><p>Automatically extract clusters according to the Xi-steep method.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.compute_optics_graph.html#sklearn.cluster.compute_optics_graph" title="sklearn.cluster.compute_optics_graph"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.compute_optics_graph</span></code></a>(X,&nbsp;*,&nbsp;…)</p></td>
    <td><p>Computes the OPTICS reachability graph.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html#sklearn.cluster.dbscan" title="sklearn.cluster.dbscan"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.dbscan</span></code></a>(X[,&nbsp;eps,&nbsp;min_samples,&nbsp;…])</p></td>
    <td><p>Perform DBSCAN clustering from vector array or distance matrix.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html#sklearn.cluster.estimate_bandwidth" title="sklearn.cluster.estimate_bandwidth"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.estimate_bandwidth</span></code></a>(X,&nbsp;*[,&nbsp;quantile,&nbsp;…])</p></td>
    <td><p>Estimate the bandwidth to use with the mean-shift algorithm.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means" title="sklearn.cluster.k_means"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.k_means</span></code></a>(X,&nbsp;n_clusters,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>K-means clustering algorithm.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.mean_shift.html#sklearn.cluster.mean_shift" title="sklearn.cluster.mean_shift"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.mean_shift</span></code></a>(X,&nbsp;*[,&nbsp;bandwidth,&nbsp;seeds,&nbsp;…])</p></td>
    <td><p>Perform mean shift clustering of data using a flat kernel.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html#sklearn.cluster.spectral_clustering" title="sklearn.cluster.spectral_clustering"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.spectral_clustering</span></code></a>(affinity,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Apply clustering to a projection of the normalized Laplacian.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cluster.ward_tree.html#sklearn.cluster.ward_tree" title="sklearn.cluster.ward_tree"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cluster.ward_tree</span></code></a>(X,&nbsp;*[,&nbsp;connectivity,&nbsp;…])</p></td>
    <td><p>Ward clustering based on a Feature matrix.</p></td>
    </tr>
  </tbody>
</table><br/>

## sklearn.compose: Composite Estimators

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose">sklearn.compose</a>: Composite Estimators</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer" title="sklearn.compose.ColumnTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">compose.ColumnTransformer</span></code></a><br/>&nbsp;&nbsp;(transformers,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Applies transformers to columns of an array or pandas DataFrame.</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor" title="sklearn.compose.TransformedTargetRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">compose.TransformedTargetRegressor</span></code></a><br/>&nbsp;&nbsp;([…])</p></td>
  <td><p>Meta-estimator to regress on a transformed target.</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer" title="sklearn.compose.make_column_transformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">compose.make_column_transformer</span></code></a>(…)</p></td>
  <td><p>Construct a ColumnTransformer from the given transformers.</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector" title="sklearn.compose.make_column_selector"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">compose.make_column_selector</span></code></a><br/>&nbsp;&nbsp;([pattern,&nbsp;…])</p></td>
  <td><p>Create a callable to select columns to be used with <codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ColumnTransformer</span></code>.</p></td>
  </tr>
  </tbody>
</table><br/>


## sklearn.covariance: Covariance Estimators

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance">sklearn.covariance</a>: Covariance Estimators</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance" title="sklearn.covariance.EmpiricalCovariance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.EmpiricalCovariance</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Maximum likelihood covariance estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope" title="sklearn.covariance.EllipticEnvelope"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.EllipticEnvelope</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>An object for detecting outliers in a Gaussian distributed dataset.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso" title="sklearn.covariance.GraphicalLasso"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.GraphicalLasso</span></code></a>([alpha,&nbsp;mode,&nbsp;…])</p></td>
    <td><p>Sparse inverse covariance estimation with an l1-penalized estimator.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLassoCV.html#sklearn.covariance.GraphicalLassoCV" title="sklearn.covariance.GraphicalLassoCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.GraphicalLassoCV</span></code></a>(*[,&nbsp;alphas,&nbsp;…])</p></td>
    <td><p>Sparse inverse covariance w/ cross-validated choice of the l1 penalty.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html#sklearn.covariance.LedoitWolf" title="sklearn.covariance.LedoitWolf"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.LedoitWolf</span></code></a>(*[,&nbsp;store_precision,&nbsp;…])</p></td>
    <td><p>LedoitWolf Estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html#sklearn.covariance.MinCovDet" title="sklearn.covariance.MinCovDet"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.MinCovDet</span></code></a>(*[,&nbsp;store_precision,&nbsp;…])</p></td>
    <td><p>Minimum Covariance Determinant (MCD): robust estimator of covariance.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html#sklearn.covariance.OAS" title="sklearn.covariance.OAS"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.OAS</span></code></a>(*[,&nbsp;store_precision,&nbsp;…])</p></td>
    <td><p>Oracle Approximating Shrinkage Estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html#sklearn.covariance.ShrunkCovariance" title="sklearn.covariance.ShrunkCovariance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.ShrunkCovariance</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Covariance estimator with shrinkage</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html#sklearn.covariance.empirical_covariance" title="sklearn.covariance.empirical_covariance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.empirical_covariance</span></code></a>(X,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Computes the Maximum likelihood covariance estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.graphical_lasso.html#sklearn.covariance.graphical_lasso" title="sklearn.covariance.graphical_lasso"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.graphical_lasso</span></code></a>(emp_cov,&nbsp;alpha,&nbsp;*)</p></td>
    <td><p>l1-penalized covariance estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html#sklearn.covariance.ledoit_wolf" title="sklearn.covariance.ledoit_wolf"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.ledoit_wolf</span></code></a>(X,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Estimates the shrunk Ledoit-Wolf covariance matrix.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.oas.html#sklearn.covariance.oas" title="sklearn.covariance.oas"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.oas</span></code></a>(X,&nbsp;*[,&nbsp;assume_centered])</p></td>
    <td><p>Estimate covariance with the Oracle Approximating Shrinkage algorithm.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html#sklearn.covariance.shrunk_covariance" title="sklearn.covariance.shrunk_covariance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.shrunk_covariance</span></code></a>(emp_cov[,&nbsp;…])</p></td>
    <td><p>Calculates a covariance matrix shrunk on the diagonal</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html#sklearn.covariance.empirical_covariance" title="sklearn.covariance.empirical_covariance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.empirical_covariance</span></code></a>(X,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Computes the Maximum likelihood covariance estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.graphical_lasso.html#sklearn.covariance.graphical_lasso" title="sklearn.covariance.graphical_lasso"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.graphical_lasso</span></code></a>(emp_cov,&nbsp;alpha,&nbsp;*)</p></td>
    <td><p>l1-penalized covariance estimator</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html#sklearn.covariance.ledoit_wolf" title="sklearn.covariance.ledoit_wolf"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.ledoit_wolf</span></code></a>(X,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Estimates the shrunk Ledoit-Wolf covariance matrix.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.oas.html#sklearn.covariance.oas" title="sklearn.covariance.oas"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.oas</span></code></a>(X,&nbsp;*[,&nbsp;assume_centered])</p></td>
    <td><p>Estimate covariance with the Oracle Approximating Shrinkage algorithm.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html#sklearn.covariance.shrunk_covariance" title="sklearn.covariance.shrunk_covariance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">covariance.shrunk_covariance</span></code></a>(emp_cov[,&nbsp;…])</p></td>
    <td><p>Calculates a covariance matrix shrunk on the diagonal</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.cross_decomposition: Cross decomposition

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition">sklearn.cross_decomposition</a>: Cross decomposition</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html#sklearn.cross_decomposition.CCA" title="sklearn.cross_decomposition.CCA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cross_decomposition.CCA</span></code></a>([n_components,&nbsp;…])</p></td>
  <td><p>CCA Canonical Correlation Analysis.</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html#sklearn.cross_decomposition.PLSCanonical" title="sklearn.cross_decomposition.PLSCanonical"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cross_decomposition.PLSCanonical</span></code></a>([…])</p></td>
  <td><p>PLSCanonical implements the 2 blocks canonical PLS of the original Wold algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn.cross_decomposition.PLSRegression" title="sklearn.cross_decomposition.PLSRegression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cross_decomposition.PLSRegression</span></code></a>([…])</p></td>
  <td><p>PLS regression</p></td>
  </tr>
  <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD" title="sklearn.cross_decomposition.PLSSVD"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">cross_decomposition.PLSSVD</span></code></a>([n_components,&nbsp;…])</p></td>
  <td><p>Partial Least Square SVD</p></td>
  </tr>
  </tbody>
</table><br/>


## sklearn.datasets: Datasets

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets">sklearn.datasets</a>: Datasets</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Class/Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Basic classes </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.clear_data_home.html#sklearn.datasets.clear_data_home" title="sklearn.datasets.clear_data_home"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.clear_data_home</span></code></a>([data_home])</p></td>
    <td><p>Delete all the content of the data home cache.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html#sklearn.datasets.dump_svmlight_file" title="sklearn.datasets.dump_svmlight_file"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.dump_svmlight_file</span></code></a>(X,&nbsp;y,&nbsp;f,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Dump the dataset in svmlight / libsvm file format.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups" title="sklearn.datasets.fetch_20newsgroups"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_20newsgroups</span></code></a>(*[,&nbsp;data_home,&nbsp;…])</p></td>
    <td><p>Load the filenames and data from the 20 newsgroups dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html#sklearn.datasets.fetch_20newsgroups_vectorized" title="sklearn.datasets.fetch_20newsgroups_vectorized"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_20newsgroups_vectorized</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Load the 20 newsgroups dataset and vectorize it into token counts (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing" title="sklearn.datasets.fetch_california_housing"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_california_housing</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Load the California housing dataset (regression).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype" title="sklearn.datasets.fetch_covtype"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_covtype</span></code></a>(*[,&nbsp;data_home,&nbsp;…])</p></td>
    <td><p>Load the covertype dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html#sklearn.datasets.fetch_kddcup99" title="sklearn.datasets.fetch_kddcup99"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_kddcup99</span></code></a>(*[,&nbsp;subset,&nbsp;…])</p></td>
    <td><p>Load the kddcup99 dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html#sklearn.datasets.fetch_lfw_pairs" title="sklearn.datasets.fetch_lfw_pairs"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_lfw_pairs</span></code></a>(*[,&nbsp;subset,&nbsp;…])</p></td>
    <td><p>Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html#sklearn.datasets.fetch_lfw_people" title="sklearn.datasets.fetch_lfw_people"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_lfw_people</span></code></a>(*[,&nbsp;data_home,&nbsp;…])</p></td>
    <td><p>Load the Labeled Faces in the Wild (LFW) people dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html#sklearn.datasets.fetch_olivetti_faces" title="sklearn.datasets.fetch_olivetti_faces"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_olivetti_faces</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Load the Olivetti faces data-set from AT&amp;T (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml" title="sklearn.datasets.fetch_openml"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_openml</span></code></a>([name,&nbsp;version,&nbsp;…])</p></td>
    <td><p>Fetch dataset from openml by name or dataset id.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html#sklearn.datasets.fetch_rcv1" title="sklearn.datasets.fetch_rcv1"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_rcv1</span></code></a>(*[,&nbsp;data_home,&nbsp;subset,&nbsp;…])</p></td>
    <td><p>Load the RCV1 multilabel dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_species_distributions.html#sklearn.datasets.fetch_species_distributions" title="sklearn.datasets.fetch_species_distributions"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.fetch_species_distributions</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Loader for species distribution dataset from Phillips et.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.get_data_home.html#sklearn.datasets.get_data_home" title="sklearn.datasets.get_data_home"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.get_data_home</span></code></a>([data_home])</p></td>
    <td><p>Return the path of the scikit-learn data dir.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston" title="sklearn.datasets.load_boston"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_boston</span></code></a>(*[,&nbsp;return_X_y])</p></td>
    <td><p>Load and return the boston house-prices dataset (regression).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer" title="sklearn.datasets.load_breast_cancer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_breast_cancer</span></code></a>(*[,&nbsp;return_X_y,&nbsp;…])</p></td>
    <td><p>Load and return the breast cancer wisconsin dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes" title="sklearn.datasets.load_diabetes"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_diabetes</span></code></a>(*[,&nbsp;return_X_y,&nbsp;as_frame])</p></td>
    <td><p>Load and return the diabetes dataset (regression).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits" title="sklearn.datasets.load_digits"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_digits</span></code></a>(*[,&nbsp;n_class,&nbsp;…])</p></td>
    <td><p>Load and return the digits dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html#sklearn.datasets.load_files" title="sklearn.datasets.load_files"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_files</span></code></a>(container_path,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Load text files with categories as subfolder names.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris" title="sklearn.datasets.load_iris"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_iris</span></code></a>(*[,&nbsp;return_X_y,&nbsp;as_frame])</p></td>
    <td><p>Load and return the iris dataset (classification).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud" title="sklearn.datasets.load_linnerud"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_linnerud</span></code></a>(*[,&nbsp;return_X_y,&nbsp;as_frame])</p></td>
    <td><p>Load and return the physical excercise linnerud dataset.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html#sklearn.datasets.load_sample_image" title="sklearn.datasets.load_sample_image"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_sample_image</span></code></a>(image_name)</p></td>
    <td><p>Load the numpy array of a single sample image</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html#sklearn.datasets.load_sample_images" title="sklearn.datasets.load_sample_images"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_sample_images</span></code></a>()</p></td>
    <td><p>Load sample images for image manipulation.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html#sklearn.datasets.load_svmlight_file" title="sklearn.datasets.load_svmlight_file"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_svmlight_file</span></code></a>(f,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Load datasets in the svmlight / libsvm format into sparse CSR matrix</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_files.html#sklearn.datasets.load_svmlight_files" title="sklearn.datasets.load_svmlight_files"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_svmlight_files</span></code></a>(files,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Load dataset from multiple files in SVMlight format</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine" title="sklearn.datasets.load_wine"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.load_wine</span></code></a>(*[,&nbsp;return_X_y,&nbsp;as_frame])</p></td>
    <td><p>Load and return the wine dataset (classification).</p></td>
    </tr>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Samples generator </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_biclusters.html#sklearn.datasets.make_biclusters" title="sklearn.datasets.make_biclusters"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_biclusters</span></code></a>(shape,&nbsp;n_clusters,&nbsp;*)</p></td>
    <td><p>Generate an array with constant block diagonal structure for biclustering.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs" title="sklearn.datasets.make_blobs"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_blobs</span></code></a>([n_samples,&nbsp;n_features,&nbsp;…])</p></td>
    <td><p>Generate isotropic Gaussian blobs for clustering.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_checkerboard.html#sklearn.datasets.make_checkerboard" title="sklearn.datasets.make_checkerboard"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_checkerboard</span></code></a>(shape,&nbsp;n_clusters,&nbsp;*)</p></td>
    <td><p>Generate an array with block checkerboard structure for biclustering.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles" title="sklearn.datasets.make_circles"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_circles</span></code></a>([n_samples,&nbsp;shuffle,&nbsp;…])</p></td>
    <td><p>Make a large circle containing a smaller circle in 2d.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification" title="sklearn.datasets.make_classification"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_classification</span></code></a>([n_samples,&nbsp;…])</p></td>
    <td><p>Generate a random n-class classification problem.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman1" title="sklearn.datasets.make_friedman1"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_friedman1</span></code></a>([n_samples,&nbsp;…])</p></td>
    <td><p>Generate the “Friedman #1” regression problem</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html#sklearn.datasets.make_friedman2" title="sklearn.datasets.make_friedman2"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_friedman2</span></code></a>([n_samples,&nbsp;noise,&nbsp;…])</p></td>
    <td><p>Generate the “Friedman #2” regression problem</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman3.html#sklearn.datasets.make_friedman3" title="sklearn.datasets.make_friedman3"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_friedman3</span></code></a>([n_samples,&nbsp;noise,&nbsp;…])</p></td>
    <td><p>Generate the “Friedman #3” regression problem</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles" title="sklearn.datasets.make_gaussian_quantiles"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_gaussian_quantiles</span></code></a>(*[,&nbsp;mean,&nbsp;…])</p></td>
    <td><p>Generate isotropic Gaussian and label samples by quantile</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html#sklearn.datasets.make_hastie_10_2" title="sklearn.datasets.make_hastie_10_2"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_hastie_10_2</span></code></a>([n_samples,&nbsp;…])</p></td>
    <td><p>Generates data for binary classification used in Hastie et al.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_low_rank_matrix.html#sklearn.datasets.make_low_rank_matrix" title="sklearn.datasets.make_low_rank_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_low_rank_matrix</span></code></a>([n_samples,&nbsp;…])</p></td>
    <td><p>Generate a mostly low rank matrix with bell-shaped singular values</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons" title="sklearn.datasets.make_moons"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_moons</span></code></a>([n_samples,&nbsp;shuffle,&nbsp;…])</p></td>
    <td><p>Make two interleaving half circles</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html#sklearn.datasets.make_multilabel_classification" title="sklearn.datasets.make_multilabel_classification"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_multilabel_classification</span></code></a>([…])</p></td>
    <td><p>Generate a random multilabel classification problem.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression" title="sklearn.datasets.make_regression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_regression</span></code></a>([n_samples,&nbsp;…])</p></td>
    <td><p>Generate a random regression problem.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html#sklearn.datasets.make_s_curve" title="sklearn.datasets.make_s_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_s_curve</span></code></a>([n_samples,&nbsp;noise,&nbsp;…])</p></td>
    <td><p>Generate an S curve dataset.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_coded_signal.html#sklearn.datasets.make_sparse_coded_signal" title="sklearn.datasets.make_sparse_coded_signal"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_sparse_coded_signal</span></code></a>(n_samples,&nbsp;…)</p></td>
    <td><p>Generate a signal as a sparse combination of dictionary elements.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html#sklearn.datasets.make_sparse_spd_matrix" title="sklearn.datasets.make_sparse_spd_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_sparse_spd_matrix</span></code></a>([dim,&nbsp;…])</p></td>
    <td><p>Generate a sparse symmetric definite positive matrix.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html#sklearn.datasets.make_sparse_uncorrelated" title="sklearn.datasets.make_sparse_uncorrelated"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_sparse_uncorrelated</span></code></a>([…])</p></td>
    <td><p>Generate a random regression problem with sparse uncorrelated design</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html#sklearn.datasets.make_spd_matrix" title="sklearn.datasets.make_spd_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_spd_matrix</span></code></a>(n_dim,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Generate a random symmetric, positive-definite matrix.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html#sklearn.datasets.make_swiss_roll" title="sklearn.datasets.make_swiss_roll"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">datasets.make_swiss_roll</span></code></a>([n_samples,&nbsp;noise,&nbsp;…])</p></td>
    <td><p>Generate a swiss roll dataset.</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.decomposition: Matrix Decomposition

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition">sklearn.decomposition</a>: Matrix Decomposition</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning" title="sklearn.decomposition.DictionaryLearning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.DictionaryLearning</span></code></a>([…])</p></td>
    <td><p>Dictionary learning</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis" title="sklearn.decomposition.FactorAnalysis"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.FactorAnalysis</span></code></a>([n_components,&nbsp;…])</p></td>
    <td><p>Factor Analysis (FA)</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA" title="sklearn.decomposition.FastICA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.FastICA</span></code></a>([n_components,&nbsp;…])</p></td>
    <td><p>FastICA: a fast algorithm for Independent Component Analysis.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA" title="sklearn.decomposition.IncrementalPCA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.IncrementalPCA</span></code></a>([n_components,&nbsp;…])</p></td>
    <td><p>Incremental principal components analysis (IPCA).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA" title="sklearn.decomposition.KernelPCA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.KernelPCA</span></code></a>([n_components,&nbsp;…])</p></td>
    <td><p>Kernel Principal component analysis (KPCA)</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation" title="sklearn.decomposition.LatentDirichletAllocation"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.LatentDirichletAllocation</span></code></a>([…])</p></td>
    <td><p>Latent Dirichlet Allocation with online variational Bayes algorithm</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html#sklearn.decomposition.MiniBatchDictionaryLearning" title="sklearn.decomposition.MiniBatchDictionaryLearning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.MiniBatchDictionaryLearning</span></code></a>([…])</p></td>
    <td><p>Mini-batch dictionary learning</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA" title="sklearn.decomposition.MiniBatchSparsePCA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.MiniBatchSparsePCA</span></code></a>([…])</p></td>
    <td><p>Mini-batch Sparse Principal Components Analysis</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF" title="sklearn.decomposition.NMF"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.NMF</span></code></a>([n_components,&nbsp;init,&nbsp;…])</p></td>
    <td><p>Non-Negative Matrix Factorization (NMF)</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA" title="sklearn.decomposition.PCA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.PCA</span></code></a>([n_components,&nbsp;copy,&nbsp;…])</p></td>
    <td><p>Principal component analysis (PCA).</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA" title="sklearn.decomposition.SparsePCA"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.SparsePCA</span></code></a>([n_components,&nbsp;…])</p></td>
    <td><p>Sparse Principal Components Analysis (SparsePCA)</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparseCoder.html#sklearn.decomposition.SparseCoder" title="sklearn.decomposition.SparseCoder"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.SparseCoder</span></code></a>(dictionary,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Sparse coding</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD" title="sklearn.decomposition.TruncatedSVD"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">decomposition.TruncatedSVD</span></code></a>([n_components,&nbsp;…])</p></td>
    <td><p>Dimensionality reduction using truncated SVD (aka LSA).</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.discriminant_analysis: Discriminant Analysis

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis">sklearn.discriminant_analysis</a>: Discriminant Analysis</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis" title="sklearn.discriminant_analysis.LinearDiscriminantAnalysis"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">discriminant_analysis.LinearDiscriminantAnalysis</span></code></a>(*)</p></td>
    <td><p>Linear Discriminant Analysis</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis" title="sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">discriminant_analysis.QuadraticDiscriminantAnalysis</span></code></a>(*)</p></td>
    <td><p>Quadratic Discriminant Analysis</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.dummy: Dummy estimators

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy">sklearn.dummy</a>: Dummy estimators</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <trstyle="vertical-align: middle;"><td><p><astyle="vertical-align: middle;" https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier" title="sklearn.dummy.DummyClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">dummy.DummyClassifier</span></code></a>(*[,&nbsp;strategy,&nbsp;…])</p></td>
  <td><p>DummyClassifier is a classifier that makes predictions using simple rules.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><astyle="vertical-align: middle;" https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor" title="sklearn.dummy.DummyRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">dummy.DummyRegressor</span></code></a>(*[,&nbsp;strategy,&nbsp;…])</p></td>
  <td><p>DummyRegressor is a regressor that makes predictions using simple rules.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.ensemble: Ensemble Methods

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble">sklearn.ensemble</a>: Ensemble Methods</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier" title="sklearn.ensemble.AdaBoostClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.AdaBoostClassifier</span></code></a>([…])</p></td>
    <td><p>An AdaBoost classifier.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor" title="sklearn.ensemble.AdaBoostRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.AdaBoostRegressor</span></code></a>([base_estimator,&nbsp;…])</p></td>
    <td><p>An AdaBoost regressor.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier" title="sklearn.ensemble.BaggingClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.BaggingClassifier</span></code></a>([base_estimator,&nbsp;…])</p></td>
    <td><p>A Bagging classifier.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor" title="sklearn.ensemble.BaggingRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.BaggingRegressor</span></code></a>([base_estimator,&nbsp;…])</p></td>
    <td><p>A Bagging regressor.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier" title="sklearn.ensemble.ExtraTreesClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.ExtraTreesClassifier</span></code></a>([…])</p></td>
    <td><p>An extra-trees classifier.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor" title="sklearn.ensemble.ExtraTreesRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.ExtraTreesRegressor</span></code></a>([n_estimators,&nbsp;…])</p></td>
    <td><p>An extra-trees regressor.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier" title="sklearn.ensemble.GradientBoostingClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.GradientBoostingClassifier</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Gradient Boosting for classification.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor" title="sklearn.ensemble.GradientBoostingRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.GradientBoostingRegressor</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Gradient Boosting for regression.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest" title="sklearn.ensemble.IsolationForest"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.IsolationForest</span></code></a>(*[,&nbsp;n_estimators,&nbsp;…])</p></td>
    <td><p>Isolation Forest Algorithm.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier" title="sklearn.ensemble.RandomForestClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.RandomForestClassifier</span></code></a>([…])</p></td>
    <td><p>A random forest classifier.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor" title="sklearn.ensemble.RandomForestRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.RandomForestRegressor</span></code></a>([…])</p></td>
    <td><p>A random forest regressor.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html#sklearn.ensemble.RandomTreesEmbedding" title="sklearn.ensemble.RandomTreesEmbedding"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.RandomTreesEmbedding</span></code></a>([…])</p></td>
    <td><p>An ensemble of totally random trees.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier" title="sklearn.ensemble.StackingClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.StackingClassifier</span></code></a>(estimators[,&nbsp;…])</p></td>
    <td><p>Stack of estimators with a final classifier.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html#sklearn.ensemble.StackingRegressor" title="sklearn.ensemble.StackingRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.StackingRegressor</span></code></a>(estimators[,&nbsp;…])</p></td>
    <td><p>Stack of estimators with a final regressor.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier" title="sklearn.ensemble.VotingClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.VotingClassifier</span></code></a>(estimators,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Soft Voting/Majority Rule classifier for unfitted estimators.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor" title="sklearn.ensemble.VotingRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.VotingRegressor</span></code></a>(estimators,&nbsp;*[,&nbsp;…])</p></td>
    <td><p>Prediction voting regressor for unfitted estimators.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor" title="sklearn.ensemble.HistGradientBoostingRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.HistGradientBoostingRegressor</span></code></a>([…])</p></td>
    <td><p>Histogram-based Gradient Boosting Regression Tree.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier" title="sklearn.ensemble.HistGradientBoostingClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">ensemble.HistGradientBoostingClassifier</span></code></a>([…])</p></td>
    <td><p>Histogram-based Gradient Boosting Classification Tree.</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.exceptions: Exceptions and warnings

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions">sklearn.exceptions</a>: Exceptions and warnings</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ChangedBehaviorWarning.html#sklearn.exceptions.ChangedBehaviorWarning" title="sklearn.exceptions.ChangedBehaviorWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.ChangedBehaviorWarning</span></code></a></p></td>
    <td><p>Warning class used to notify the user of any change in the behavior.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ConvergenceWarning.html#sklearn.exceptions.ConvergenceWarning" title="sklearn.exceptions.ConvergenceWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.ConvergenceWarning</span></code></a></p></td>
    <td><p>Custom warning to capture convergence problems</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataConversionWarning.html#sklearn.exceptions.DataConversionWarning" title="sklearn.exceptions.DataConversionWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.DataConversionWarning</span></code></a></p></td>
    <td><p>Warning used to notify implicit data conversions happening in the code.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataDimensionalityWarning.html#sklearn.exceptions.DataDimensionalityWarning" title="sklearn.exceptions.DataDimensionalityWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.DataDimensionalityWarning</span></code></a></p></td>
    <td><p>Custom warning to notify potential issues with data dimensionality.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.EfficiencyWarning.html#sklearn.exceptions.EfficiencyWarning" title="sklearn.exceptions.EfficiencyWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.EfficiencyWarning</span></code></a></p></td>
    <td><p>Warning used to notify the user of inefficient computation.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.FitFailedWarning.html#sklearn.exceptions.FitFailedWarning" title="sklearn.exceptions.FitFailedWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.FitFailedWarning</span></code></a></p></td>
    <td><p>Warning class used if there is an error while fitting the estimator.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html#sklearn.exceptions.NotFittedError" title="sklearn.exceptions.NotFittedError"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.NotFittedError</span></code></a></p></td>
    <td><p>Exception class to raise if estimator is used before fitting.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NonBLASDotWarning.html#sklearn.exceptions.NonBLASDotWarning" title="sklearn.exceptions.NonBLASDotWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.NonBLASDotWarning</span></code></a></p></td>
    <td><p>Warning used when the dot operation does not use BLAS.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html#sklearn.exceptions.UndefinedMetricWarning" title="sklearn.exceptions.UndefinedMetricWarning"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">exceptions.UndefinedMetricWarning</span></code></a></p></td>
    <td><p>Warning used when the metric is invalid</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.experimental: Experimental

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental">sklearn.experimental</a>: Experimental</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer" title="sklearn.feature_extraction.DictVectorizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.DictVectorizer</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Transforms lists of feature-value mappings to vectors.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher" title="sklearn.feature_extraction.FeatureHasher"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.FeatureHasher</span></code></a>([…])</p></td>
    <td><p>Implements feature hashing, aka the hashing trick.</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.feature_extraction: Feature Extraction

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction">sklearn.feature_extraction</a>: Feature Extraction</caption>
  <thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
  </thead>
  <tbody>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Basics </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer" title="sklearn.feature_extraction.DictVectorizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.DictVectorizer</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Transforms lists of feature-value mappings to vectors.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher" title="sklearn.feature_extraction.FeatureHasher"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.FeatureHasher</span></code></a>([…])</p></td>
    <td><p>Implements feature hashing, aka the hashing trick.</p></td>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> From images </td> </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html#sklearn.feature_extraction.image.extract_patches_2d" title="sklearn.feature_extraction.image.extract_patches_2d"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.image.extract_patches_2d</span></code></a>(…)</p></td>
    <td><p>Reshape a 2D image into a collection of patches</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.grid_to_graph.html#sklearn.feature_extraction.image.grid_to_graph" title="sklearn.feature_extraction.image.grid_to_graph"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.image.grid_to_graph</span></code></a>(n_x,&nbsp;n_y)</p></td>
    <td><p>Graph of the pixel-to-pixel connections</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.img_to_graph.html#sklearn.feature_extraction.image.img_to_graph" title="sklearn.feature_extraction.image.img_to_graph"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.image.img_to_graph</span></code></a>(img,&nbsp;*)</p></td>
    <td><p>Graph of the pixel-to-pixel gradient connections</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.reconstruct_from_patches_2d.html#sklearn.feature_extraction.image.reconstruct_from_patches_2d" title="sklearn.feature_extraction.image.reconstruct_from_patches_2d"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.image.<br/>reconstruct_from_patches_2d</span></code></a>(…)</p></td>
    <td><p>Reconstruct the image from all of its patches.</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html#sklearn.feature_extraction.image.PatchExtractor" title="sklearn.feature_extraction.image.PatchExtractor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.image.PatchExtractor</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Extracts patches from a collection of images</p></td>
    </tr>
    <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> From text </td> </tr>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer" title="sklearn.feature_extraction.text.CountVectorizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.text.CountVectorizer</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Convert a collection of text documents to a matrix of token counts</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer" title="sklearn.feature_extraction.text.HashingVectorizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.text.HashingVectorizer</span></code></a>(*)</p></td>
    <td><p>Convert a collection of text documents to a matrix of token occurrences</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer" title="sklearn.feature_extraction.text.TfidfTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.text.TfidfTransformer</span></code></a>(*)</p></td>
    <td><p>Transform a count matrix to a normalized tf or tf-idf representation</p></td>
    </tr>
    <tr style="vertical-align:middle"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer" title="sklearn.feature_extraction.text.TfidfVectorizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_extraction.text.TfidfVectorizer</span></code></a>(*[,&nbsp;…])</p></td>
    <td><p>Convert a collection of raw documents to a matrix of TF-IDF features.</p></td>
    </tr>
  </tbody>
</table><br/>


## sklearn.feature_selection: Feature Selection 

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection">sklearn.feature_selection</a>: Feature Selection</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect" title="sklearn.feature_selection.GenericUnivariateSelect"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.GenericUnivariateSelect</span></code></a>([…])</p></td>
  <td><p>Univariate feature selector with configurable strategy.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile" title="sklearn.feature_selection.SelectPercentile"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectPercentile</span></code></a>([…])</p></td>
  <td><p>Select features according to a percentile of the highest scores.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest" title="sklearn.feature_selection.SelectKBest"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectKBest</span></code></a>([score_func,&nbsp;k])</p></td>
  <td><p>Select features according to the k highest scores.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr" title="sklearn.feature_selection.SelectFpr"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectFpr</span></code></a>([score_func,&nbsp;alpha])</p></td>
  <td><p>Filter: Select the pvalues below alpha based on a FPR test.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr" title="sklearn.feature_selection.SelectFdr"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectFdr</span></code></a>([score_func,&nbsp;alpha])</p></td>
  <td><p>Filter: Select the p-values for an estimated false discovery rate</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel" title="sklearn.feature_selection.SelectFromModel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectFromModel</span></code></a>(estimator,&nbsp;*)</p></td>
  <td><p>Meta-transformer for selecting features based on importance weights.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe" title="sklearn.feature_selection.SelectFwe"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.SelectFwe</span></code></a>([score_func,&nbsp;alpha])</p></td>
  <td><p>Filter: Select the p-values corresponding to Family-wise error rate</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE" title="sklearn.feature_selection.RFE"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.RFE</span></code></a>(estimator,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Feature ranking with recursive feature elimination.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV" title="sklearn.feature_selection.RFECV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.RFECV</span></code></a>(estimator,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold" title="sklearn.feature_selection.VarianceThreshold"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.VarianceThreshold</span></code></a>([threshold])</p></td>
  <td><p>Feature selector that removes all low-variance features.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2" title="sklearn.feature_selection.chi2"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.chi2</span></code></a>(X,&nbsp;y)</p></td>
  <td><p>Compute chi-squared stats between each non-negative feature and class.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif" title="sklearn.feature_selection.f_classif"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.f_classif</span></code></a>(X,&nbsp;y)</p></td>
  <td><p>Compute the ANOVA F-value for the provided sample.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression" title="sklearn.feature_selection.f_regression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.f_regression</span></code></a>(X,&nbsp;y,&nbsp;*[,&nbsp;center])</p></td>
  <td><p>Univariate linear regression tests.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif" title="sklearn.feature_selection.mutual_info_classif"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.mutual_info_classif</span></code></a>(X,&nbsp;y,&nbsp;*)</p></td>
  <td><p>Estimate mutual information for a discrete target variable.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression" title="sklearn.feature_selection.mutual_info_regression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">feature_selection.mutual_info_regression</span></code></a>(X,&nbsp;y,&nbsp;*)</p></td>
  <td><p>Estimate mutual information for a continuous target variable.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.gaussian_process: Gaussian Processes

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction">sklearn.gaussian_process</a>: Gaussian Processes</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> General </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier" title="sklearn.gaussian_process.GaussianProcessClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.GaussianProcessClassifier</span></code></a>([…])</p></td>
  <td><p>Gaussian process classification (GPC) based on Laplace approximation.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor" title="sklearn.gaussian_process.GaussianProcessRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.GaussianProcessRegressor</span></code></a>([…])</p></td>
  <td><p>Gaussian process regression (GPR).</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Kernels </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.CompoundKernel.html#sklearn.gaussian_process.kernels.CompoundKernel" title="sklearn.gaussian_process.kernels.CompoundKernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.CompoundKernel</span></code></a>(kernels)</p></td>
  <td><p>Kernel which is composed of a set of other kernels.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html#sklearn.gaussian_process.kernels.ConstantKernel" title="sklearn.gaussian_process.kernels.ConstantKernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.ConstantKernel</span></code></a>([…])</p></td>
  <td><p>Constant kernel.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.DotProduct.html#sklearn.gaussian_process.kernels.DotProduct" title="sklearn.gaussian_process.kernels.DotProduct"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.DotProduct</span></code></a>([…])</p></td>
  <td><p>Dot-Product kernel.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html#sklearn.gaussian_process.kernels.ExpSineSquared" title="sklearn.gaussian_process.kernels.ExpSineSquared"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.ExpSineSquared</span></code></a>([…])</p></td>
  <td><p>Exp-Sine-Squared kernel (aka periodic kernel).</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Exponentiation.html#sklearn.gaussian_process.kernels.Exponentiation" title="sklearn.gaussian_process.kernels.Exponentiation"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.Exponentiation</span></code></a>(…)</p></td>
  <td><p>The Exponentiation kernel takes one base kernel and a scalar parameter <spanstyle="vertical-align: middle;"><mjx-containerstyle="vertical-align: middle;" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="0" style="font-size: 113.1%; position: relative;"><mjx-mathstyle="vertical-align: middle;" aria-hidden="true"><mjx-mistyle="vertical-align: middle;"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>p</mi></math></mjx-assistive-mml></mjx-container></span> and combines them via</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Hyperparameter.html#sklearn.gaussian_process.kernels.Hyperparameter" title="sklearn.gaussian_process.kernels.Hyperparameter"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.Hyperparameter</span></code></a></p></td>
  <td><p>A kernel hyperparameter’s specification in form of a namedtuple.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html#sklearn.gaussian_process.kernels.Kernel" title="sklearn.gaussian_process.kernels.Kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.Kernel</span></code></a></p></td>
  <td><p>Base class for all kernels.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern" title="sklearn.gaussian_process.kernels.Matern"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.Matern</span></code></a>([…])</p></td>
  <td><p>Matern kernel.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html#sklearn.gaussian_process.kernels.PairwiseKernel" title="sklearn.gaussian_process.kernels.PairwiseKernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.PairwiseKernel</span></code></a>([…])</p></td>
  <td><p>Wrapper for kernels in sklearn.metrics.pairwise.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Product.html#sklearn.gaussian_process.kernels.Product" title="sklearn.gaussian_process.kernels.Product"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.Product</span></code></a>(k1,&nbsp;k2)</p></td>
  <td><p>The <codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">Product</span></code> kernel takes two kernels <spanstyle="vertical-align: middle;"><mjx-containerstyle="vertical-align: middle;" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="1" style="font-size: 113.1%; position: relative;"><mjx-mathstyle="vertical-align: middle;" aria-hidden="true"><mjx-msub><mjx-mistyle="vertical-align: middle;" noic="true"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mnstyle="vertical-align: middle;" size="s"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>k</mi><mn>1</mn></msub></math></mjx-assistive-mml></mjx-container></span> and <spanstyle="vertical-align: middle;"><mjx-containerstyle="vertical-align: middle;" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="2" style="font-size: 113.1%; position: relative;"><mjx-mathstyle="vertical-align: middle;" aria-hidden="true"><mjx-msub><mjx-mistyle="vertical-align: middle;" noic="true"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mnstyle="vertical-align: middle;" size="s"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>k</mi><mn>2</mn></msub></math></mjx-assistive-mml></mjx-container></span> and combines them via</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF" title="sklearn.gaussian_process.kernels.RBF"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.RBF</span></code></a>([length_scale,&nbsp;…])</p></td>
  <td><p>Radial-basis function kernel (aka squared-exponential kernel).</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html#sklearn.gaussian_process.kernels.RationalQuadratic" title="sklearn.gaussian_process.kernels.RationalQuadratic"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.RationalQuadratic</span></code></a>([…])</p></td>
  <td><p>Rational Quadratic kernel.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Sum.html#sklearn.gaussian_process.kernels.Sum" title="sklearn.gaussian_process.kernels.Sum"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.Sum</span></code></a>(k1,&nbsp;k2)</p></td>
  <td><p>The <codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">Sum</span></code> kernel takes two kernels <spanstyle="vertical-align: middle;"><mjx-containerstyle="vertical-align: middle;" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="3" style="font-size: 113.1%; position: relative;"><mjx-mathstyle="vertical-align: middle;" aria-hidden="true"><mjx-msub><mjx-mistyle="vertical-align: middle;" noic="true"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mnstyle="vertical-align: middle;" size="s"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>k</mi><mn>1</mn></msub></math></mjx-assistive-mml></mjx-container></span> and <spanstyle="vertical-align: middle;"><mjx-containerstyle="vertical-align: middle;" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="4" style="font-size: 113.1%; position: relative;"><mjx-mathstyle="vertical-align: middle;" aria-hidden="true"><mjx-msub><mjx-mistyle="vertical-align: middle;" noic="true"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mnstyle="vertical-align: middle;" size="s"><mjx-cstyle="vertical-align: middle;"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>k</mi><mn>2</mn></msub></math></mjx-assistive-mml></mjx-container></span> and combines them via</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html#sklearn.gaussian_process.kernels.WhiteKernel" title="sklearn.gaussian_process.kernels.WhiteKernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">gaussian_process.kernels.WhiteKernel</span></code></a>([…])</p></td>
  <td><p>White kernel.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.impute: Impute

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute">sklearn.impute</a>: Impute</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer" title="sklearn.impute.SimpleImputer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">impute.SimpleImputer</span></code></a>(*[,&nbsp;missing_values,&nbsp;…])</p></td>
  <td><p>Imputation transformer for completing missing values.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer" title="sklearn.impute.IterativeImputer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">impute.IterativeImputer</span></code></a>([estimator,&nbsp;…])</p></td>
  <td><p>Multivariate imputer that estimates each feature from all the others.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator" title="sklearn.impute.MissingIndicator"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">impute.MissingIndicator</span></code></a>(*[,&nbsp;missing_values,&nbsp;…])</p></td>
  <td><p>Binary indicators for missing values.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer" title="sklearn.impute.KNNImputer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">impute.KNNImputer</span></code></a>(*[,&nbsp;missing_values,&nbsp;…])</p></td>
  <td><p>Imputation for completing missing values using k-Nearest Neighbors.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.inspection: inspection

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection">sklearn.inspection</a>: Inspection</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> General </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html#sklearn.inspection.partial_dependence" title="sklearn.inspection.partial_dependence"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">inspection.partial_dependence</span></code></a>(estimator,&nbsp;X,&nbsp;…)</p></td>
  <td><p>Partial dependence of <codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">features</span></code>.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance" title="sklearn.inspection.permutation_importance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">inspection.permutation_importance</span></code></a>(estimator,&nbsp;…)</p></td>
  <td><p>Permutation importance for feature evaluation <a https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#rd9e56ef97513-bre" id="id2"><span>[Rd9e56ef97513-BRE]</span></a>.</p></td>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Plotting </td> </tr>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay" title="sklearn.inspection.PartialDependenceDisplay"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">inspection.PartialDependenceDisplay</span></code></a>(…)</p></td>
  <td><p>Partial Dependence Plot (PDP) visualization.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.isotonic: Isotonic regression

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic">sklearn.isotonic</a>: Isotonic regression</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression" title="sklearn.isotonic.IsotonicRegression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">isotonic.IsotonicRegression</span></code></a>(*[,&nbsp;y_min,&nbsp;…])</p></td>
  <td><p>Isotonic regression model.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.check_increasing.html#sklearn.isotonic.check_increasing" title="sklearn.isotonic.check_increasing"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">isotonic.check_increasing</span></code></a>(x,&nbsp;y)</p></td>
  <td><p>Determine whether y is monotonically correlated with x.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.isotonic_regression.html#sklearn.isotonic.isotonic_regression" title="sklearn.isotonic.isotonic_regression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">isotonic.isotonic_regression</span></code></a>(y,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Solve the isotonic regression model.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.kernel_approximation: Kernel Approximation

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation">kernel_approximation</a>: Kernel Approximation</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler" title="sklearn.kernel_approximation.AdditiveChi2Sampler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">kernel_approximation.AdditiveChi2Sampler</span></code></a>(*)</p></td>
  <td><p>Approximate feature map for additive chi2 kernel.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem" title="sklearn.kernel_approximation.Nystroem"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">kernel_approximation.Nystroem</span></code></a>([kernel,&nbsp;…])</p></td>
  <td><p>Approximate a kernel map using a subset of the training data.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler" title="sklearn.kernel_approximation.RBFSampler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">kernel_approximation.RBFSampler</span></code></a>(*[,&nbsp;gamma,&nbsp;…])</p></td>
  <td><p>Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html#sklearn.kernel_approximation.SkewedChi2Sampler" title="sklearn.kernel_approximation.SkewedChi2Sampler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">kernel_approximation.SkewedChi2Sampler</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Approximates feature map of the “skewed chi-squared” kernel by Monte Carlo approximation of its Fourier transform.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.kernel_ridge: Kernel Ridge Regression

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge">sklearn.kernel_ridge</a>: Kernel Ridge Regression</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge" title="sklearn.kernel_ridge.KernelRidge"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">kernel_ridge.KernelRidge</span></code></a>([alpha,&nbsp;kernel,&nbsp;…])</p></td>
  <td><p>Kernel ridge regression.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.linear_model: Linear Models

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model">sklearn.linear_model</a>: Linear Models</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Linear classifiers </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression" title="sklearn.linear_model.LogisticRegression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LogisticRegression</span></code></a>([penalty,&nbsp;…])</p></td>
  <td><p>Logistic Regression (aka logit, MaxEnt) classifier.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV" title="sklearn.linear_model.LogisticRegressionCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LogisticRegressionCV</span></code></a>(*[,&nbsp;Cs,&nbsp;…])</p></td>
  <td><p>Logistic Regression CV (aka logit, MaxEnt) classifier.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier" title="sklearn.linear_model.PassiveAggressiveClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.PassiveAggressiveClassifier</span></code></a>(*)</p></td>
  <td><p>Passive Aggressive Classifier</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron" title="sklearn.linear_model.Perceptron"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.Perceptron</span></code></a>(*[,&nbsp;penalty,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>Read more in the <a href="linear_model.html#perceptron"><spanstyle="vertical-align: middle;">User Guide</span></a>.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier" title="sklearn.linear_model.RidgeClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.RidgeClassifier</span></code></a>([alpha,&nbsp;…])</p></td>
  <td><p>Classifier using Ridge regression.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV" title="sklearn.linear_model.RidgeClassifierCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.RidgeClassifierCV</span></code></a>([alphas,&nbsp;…])</p></td>
  <td><p>Ridge classifier with built-in cross-validation.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier" title="sklearn.linear_model.SGDClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.SGDClassifier</span></code></a>([loss,&nbsp;penalty,&nbsp;…])</p></td>
  <td><p>Linear classifiers (SVM, logistic regression, etc.) with SGD training.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Classical linear regressors </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression" title="sklearn.linear_model.LinearRegression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LinearRegression</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Ordinary least squares Linear Regression.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge" title="sklearn.linear_model.Ridge"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.Ridge</span></code></a>([alpha,&nbsp;fit_intercept,&nbsp;…])</p></td>
  <td><p>Linear least squares with l2 regularization.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV" title="sklearn.linear_model.RidgeCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.RidgeCV</span></code></a>([alphas,&nbsp;…])</p></td>
  <td><p>Ridge regression with built-in cross-validation.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor" title="sklearn.linear_model.SGDRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.SGDRegressor</span></code></a>([loss,&nbsp;penalty,&nbsp;…])</p></td>
  <td><p>Linear model fitted by minimizing a regularized empirical loss with SGD</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Regressors with variable selection </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet" title="sklearn.linear_model.ElasticNet"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.ElasticNet</span></code></a>([alpha,&nbsp;l1_ratio,&nbsp;…])</p></td>
  <td><p>Linear regression with combined L1 and L2 priors as regularizer.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV" title="sklearn.linear_model.ElasticNetCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.ElasticNetCV</span></code></a>(*[,&nbsp;l1_ratio,&nbsp;…])</p></td>
  <td><p>Elastic Net model with iterative fitting along a regularization path.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars" title="sklearn.linear_model.Lars"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.Lars</span></code></a>(*[,&nbsp;fit_intercept,&nbsp;…])</p></td>
  <td><p>Least Angle Regression model a.k.a.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html#sklearn.linear_model.LarsCV" title="sklearn.linear_model.LarsCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LarsCV</span></code></a>(*[,&nbsp;fit_intercept,&nbsp;…])</p></td>
  <td><p>Cross-validated Least Angle Regression model.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso" title="sklearn.linear_model.Lasso"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.Lasso</span></code></a>([alpha,&nbsp;fit_intercept,&nbsp;…])</p></td>
  <td><p>Linear Model trained with L1 prior as regularizer (aka the Lasso)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV" title="sklearn.linear_model.LassoCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LassoCV</span></code></a>(*[,&nbsp;eps,&nbsp;n_alphas,&nbsp;…])</p></td>
  <td><p>Lasso linear model with iterative fitting along a regularization path.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars" title="sklearn.linear_model.LassoLars"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LassoLars</span></code></a>([alpha,&nbsp;…])</p></td>
  <td><p>Lasso model fit with Least Angle Regression a.k.a.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV" title="sklearn.linear_model.LassoLarsCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LassoLarsCV</span></code></a>(*[,&nbsp;fit_intercept,&nbsp;…])</p></td>
  <td><p>Cross-validated Lasso, using the LARS algorithm.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC" title="sklearn.linear_model.LassoLarsIC"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.LassoLarsIC</span></code></a>([criterion,&nbsp;…])</p></td>
  <td><p>Lasso model fit with Lars using BIC or AIC for model selection</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit" title="sklearn.linear_model.OrthogonalMatchingPursuit"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.OrthogonalMatchingPursuit</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Orthogonal Matching Pursuit model (OMP)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV" title="sklearn.linear_model.OrthogonalMatchingPursuitCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.OrthogonalMatchingPursuitCV</span></code></a>(*)</p></td>
  <td><p>Cross-validated Orthogonal Matching Pursuit model (OMP).</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Bayesian regressors </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression" title="sklearn.linear_model.ARDRegression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.ARDRegression</span></code></a>(*[,&nbsp;n_iter,&nbsp;tol,&nbsp;…])</p></td>
  <td><p>Bayesian ARD regression.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge" title="sklearn.linear_model.BayesianRidge"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.BayesianRidge</span></code></a>(*[,&nbsp;n_iter,&nbsp;tol,&nbsp;…])</p></td>
  <td><p>Bayesian ridge regression.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Multi-task linear regressors with variable selection </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet" title="sklearn.linear_model.MultiTaskElasticNet"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.MultiTaskElasticNet</span></code></a>([alpha,&nbsp;…])</p></td>
  <td><p>Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV" title="sklearn.linear_model.MultiTaskElasticNetCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.MultiTaskElasticNetCV</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Multi-task L1/L2 ElasticNet with built-in cross-validation.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso" title="sklearn.linear_model.MultiTaskLasso"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.MultiTaskLasso</span></code></a>([alpha,&nbsp;…])</p></td>
  <td><p>Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV" title="sklearn.linear_model.MultiTaskLassoCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.MultiTaskLassoCV</span></code></a>(*[,&nbsp;eps,&nbsp;…])</p></td>
  <td><p>Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Outlier-robust regressors </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor" title="sklearn.linear_model.HuberRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.HuberRegressor</span></code></a>(*[,&nbsp;epsilon,&nbsp;…])</p></td>
  <td><p>Linear regression model that is robust to outliers.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor" title="sklearn.linear_model.RANSACRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.RANSACRegressor</span></code></a>([…])</p></td>
  <td><p>RANSAC (RANdom SAmple Consensus) algorithm.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor" title="sklearn.linear_model.TheilSenRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.TheilSenRegressor</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Theil-Sen Estimator: robust multivariate regression model.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Generalized linear models (GLM) for regression </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor" title="sklearn.linear_model.PoissonRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.PoissonRegressor</span></code></a>(*[,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>Generalized Linear Model with a Poisson distribution.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor" title="sklearn.linear_model.TweedieRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.TweedieRegressor</span></code></a>(*[,&nbsp;power,&nbsp;…])</p></td>
  <td><p>Generalized Linear Model with a Tweedie distribution.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor" title="sklearn.linear_model.GammaRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.GammaRegressor</span></code></a>(*[,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>Generalized Linear Model with a Gamma distribution.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Miscellaneous </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor" title="sklearn.linear_model.PassiveAggressiveRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.PassiveAggressiveRegressor</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Passive Aggressive Regressor</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.enet_path.html#sklearn.linear_model.enet_path" title="sklearn.linear_model.enet_path"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.enet_path</span></code></a>(X,&nbsp;y,&nbsp;*[,&nbsp;l1_ratio,&nbsp;…])</p></td>
  <td><p>Compute elastic net path with coordinate descent.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path" title="sklearn.linear_model.lars_path"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.lars_path</span></code></a>(X,&nbsp;y[,&nbsp;Xy,&nbsp;Gram,&nbsp;…])</p></td>
  <td><p>Compute Least Angle Regression or Lasso path using LARS algorithm [1]</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path_gram.html#sklearn.linear_model.lars_path_gram" title="sklearn.linear_model.lars_path_gram"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.lars_path_gram</span></code></a>(Xy,&nbsp;Gram,&nbsp;*,&nbsp;…)</p></td>
  <td><p>lars_path in the sufficient stats mode [1]</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path" title="sklearn.linear_model.lasso_path"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.lasso_path</span></code></a>(X,&nbsp;y,&nbsp;*[,&nbsp;eps,&nbsp;…])</p></td>
  <td><p>Compute Lasso path with coordinate descent</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html#sklearn.linear_model.orthogonal_mp" title="sklearn.linear_model.orthogonal_mp"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.orthogonal_mp</span></code></a>(X,&nbsp;y,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Orthogonal Matching Pursuit (OMP)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp_gram.html#sklearn.linear_model.orthogonal_mp_gram" title="sklearn.linear_model.orthogonal_mp_gram"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.orthogonal_mp_gram</span></code></a>(Gram,&nbsp;Xy,&nbsp;*)</p></td>
  <td><p>Gram Orthogonal Matching Pursuit (OMP)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html#sklearn.linear_model.ridge_regression" title="sklearn.linear_model.ridge_regression"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">linear_model.ridge_regression</span></code></a>(X,&nbsp;y,&nbsp;alpha,&nbsp;*)</p></td>
  <td><p>Solve the ridge equation by the method of normal equations.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.manifold: Manifold Learning

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold">sklearn.manifold</a>: Manifold Learning</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap" title="sklearn.manifold.Isomap"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">manifold.Isomap</span></code></a>(*[,&nbsp;n_neighbors,&nbsp;…])</p></td>
  <td><p>Isomap Embedding</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding" title="sklearn.manifold.LocallyLinearEmbedding"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">manifold.LocallyLinearEmbedding</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Locally Linear Embedding</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS" title="sklearn.manifold.MDS"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">manifold.MDS</span></code></a>([n_components,&nbsp;metric,&nbsp;n_init,&nbsp;…])</p></td>
  <td><p>Multidimensional scaling</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html#sklearn.manifold.SpectralEmbedding" title="sklearn.manifold.SpectralEmbedding"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">manifold.SpectralEmbedding</span></code></a>([n_components,&nbsp;…])</p></td>
  <td><p>Spectral embedding for non-linear dimensionality reduction.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE" title="sklearn.manifold.TSNE"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">manifold.TSNE</span></code></a>([n_components,&nbsp;perplexity,&nbsp;…])</p></td>
  <td><p>t-distributed Stochastic Neighbor Embedding.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.metrics: Metrics

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold">sklearn.metrics</a>: Metrics</caption>
<thead>
  <tr style="font-size: 1.2em;">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
  </tr>
</thead>
<tbody>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Model Selection Interface </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.check_scoring.html#sklearn.metrics.check_scoring" title="sklearn.metrics.check_scoring"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.check_scoring</span></code></a>(estimator[,&nbsp;scoring,&nbsp;…])</p></td>
  <td><p>Determine scorer from user options.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer" title="sklearn.metrics.get_scorer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.get_scorer</span></code></a>(scoring)</p></td>
  <td><p>Get a scorer from string.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer" title="sklearn.metrics.make_scorer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.make_scorer</span></code></a>(score_func,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Make a scorer from a performance metric or loss function.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Classification metrics </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score" title="sklearn.metrics.accuracy_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.accuracy_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Accuracy classification score.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc" title="sklearn.metrics.auc"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.auc</span></code></a>(x,&nbsp;y)</p></td>
  <td><p>Compute Area Under the Curve (AUC) using the trapezoidal rule</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score" title="sklearn.metrics.average_precision_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.average_precision_score</span></code></a>(y_true,&nbsp;…)</p></td>
  <td><p>Compute average precision (AP) from prediction scores</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score" title="sklearn.metrics.balanced_accuracy_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.balanced_accuracy_score</span></code></a>(y_true,&nbsp;…)</p></td>
  <td><p>Compute the balanced accuracy</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss" title="sklearn.metrics.brier_score_loss"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.brier_score_loss</span></code></a>(y_true,&nbsp;y_prob,&nbsp;*)</p></td>
  <td><p>Compute the Brier score.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report" title="sklearn.metrics.classification_report"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.classification_report</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Build a text report showing the main classification metrics.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score" title="sklearn.metrics.cohen_kappa_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.cohen_kappa_score</span></code></a>(y1,&nbsp;y2,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Cohen’s kappa: a statistic that measures inter-annotator agreement.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix" title="sklearn.metrics.confusion_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.confusion_matrix</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Compute confusion matrix to evaluate the accuracy of a classification.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.dcg_score.html#sklearn.metrics.dcg_score" title="sklearn.metrics.dcg_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.dcg_score</span></code></a>(y_true,&nbsp;y_score,&nbsp;*[,&nbsp;k,&nbsp;…])</p></td>
  <td><p>Compute Discounted Cumulative Gain.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score" title="sklearn.metrics.f1_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.f1_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute the F1 score, also known as balanced F-score or F-measure</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score" title="sklearn.metrics.fbeta_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.fbeta_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*,&nbsp;beta)</p></td>
  <td><p>Compute the F-beta score</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss" title="sklearn.metrics.hamming_loss"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.hamming_loss</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute the average Hamming loss.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss" title="sklearn.metrics.hinge_loss"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.hinge_loss</span></code></a>(y_true,&nbsp;pred_decision,&nbsp;*)</p></td>
  <td><p>Average hinge loss (non-regularized)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score" title="sklearn.metrics.jaccard_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.jaccard_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Jaccard similarity coefficient score</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss" title="sklearn.metrics.log_loss"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.log_loss</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;eps,&nbsp;…])</p></td>
  <td><p>Log loss, aka logistic loss or cross-entropy loss.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef" title="sklearn.metrics.matthews_corrcoef"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.matthews_corrcoef</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Compute the Matthews correlation coefficient (MCC)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html#sklearn.metrics.multilabel_confusion_matrix" title="sklearn.metrics.multilabel_confusion_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.multilabel_confusion_matrix</span></code></a>(y_true,&nbsp;…)</p></td>
  <td><p>Compute a confusion matrix for each class or sample</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html#sklearn.metrics.ndcg_score" title="sklearn.metrics.ndcg_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.ndcg_score</span></code></a>(y_true,&nbsp;y_score,&nbsp;*[,&nbsp;k,&nbsp;…])</p></td>
  <td><p>Compute Normalized Discounted Cumulative Gain.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve" title="sklearn.metrics.precision_recall_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.precision_recall_curve</span></code></a>(y_true,&nbsp;…)</p></td>
  <td><p>Compute precision-recall pairs for different probability thresholds</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support" title="sklearn.metrics.precision_recall_fscore_support"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.precision_recall_fscore_support</span></code></a>(…)</p></td>
  <td><p>Compute precision, recall, F-measure and support for each class</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score" title="sklearn.metrics.precision_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.precision_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute the precision</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score" title="sklearn.metrics.recall_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.recall_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute the recall</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score" title="sklearn.metrics.roc_auc_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.roc_auc_score</span></code></a>(y_true,&nbsp;y_score,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve" title="sklearn.metrics.roc_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.roc_curve</span></code></a>(y_true,&nbsp;y_score,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute Receiver operating characteristic (ROC)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html#sklearn.metrics.zero_one_loss" title="sklearn.metrics.zero_one_loss"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.zero_one_loss</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Zero-one classification loss.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Regression metrics </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score" title="sklearn.metrics.explained_variance_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.explained_variance_score</span></code></a>(y_true,&nbsp;…)</p></td>
  <td><p>Explained variance regression score function</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error" title="sklearn.metrics.max_error"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.max_error</span></code></a>(y_true,&nbsp;y_pred)</p></td>
  <td><p>max_error metric calculates the maximum residual error.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error" title="sklearn.metrics.mean_absolute_error"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mean_absolute_error</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Mean absolute error regression loss</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error" title="sklearn.metrics.mean_squared_error"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mean_squared_error</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Mean squared error regression loss</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error" title="sklearn.metrics.mean_squared_log_error"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mean_squared_log_error</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Mean squared logarithmic error regression loss</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error" title="sklearn.metrics.median_absolute_error"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.median_absolute_error</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Median absolute error regression loss</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score" title="sklearn.metrics.r2_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.r2_score</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>R^2 (coefficient of determination) regression score function.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html#sklearn.metrics.mean_poisson_deviance" title="sklearn.metrics.mean_poisson_deviance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mean_poisson_deviance</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Mean Poisson deviance regression loss.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html#sklearn.metrics.mean_gamma_deviance" title="sklearn.metrics.mean_gamma_deviance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mean_gamma_deviance</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Mean Gamma deviance regression loss.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html#sklearn.metrics.mean_tweedie_deviance" title="sklearn.metrics.mean_tweedie_deviance"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mean_tweedie_deviance</span></code></a>(y_true,&nbsp;y_pred,&nbsp;*)</p></td>
  <td><p>Mean Tweedie deviance regression loss.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Multilabel ranking metrics </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.coverage_error.html#sklearn.metrics.coverage_error" title="sklearn.metrics.coverage_error"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.coverage_error</span></code></a>(y_true,&nbsp;y_score,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Coverage error measure</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score" title="sklearn.metrics.label_ranking_average_precision_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.label_ranking_average_precision_score</span></code></a>(…)</p></td>
  <td><p>Compute ranking-based average precision</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html#sklearn.metrics.label_ranking_loss" title="sklearn.metrics.label_ranking_loss"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.label_ranking_loss</span></code></a>(y_true,&nbsp;y_score,&nbsp;*)</p></td>
  <td><p>Compute Ranking loss measure</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Clustering metrics </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score" title="sklearn.metrics.adjusted_mutual_info_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.adjusted_mutual_info_score</span></code></a>(…[,&nbsp;…])</p></td>
  <td><p>Adjusted Mutual Information between two clusterings.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score" title="sklearn.metrics.adjusted_rand_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.adjusted_rand_score</span></code></a>(labels_true,&nbsp;…)</p></td>
  <td><p>Rand index adjusted for chance.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score" title="sklearn.metrics.calinski_harabasz_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.calinski_harabasz_score</span></code></a>(X,&nbsp;labels)</p></td>
  <td><p>Compute the Calinski and Harabasz score.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score" title="sklearn.metrics.davies_bouldin_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.davies_bouldin_score</span></code></a>(X,&nbsp;labels)</p></td>
  <td><p>Computes the Davies-Bouldin score.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score" title="sklearn.metrics.completeness_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.completeness_score</span></code></a>(labels_true,&nbsp;…)</p></td>
  <td><p>Completeness metric of a cluster labeling given a ground truth.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html#sklearn.metrics.cluster.contingency_matrix" title="sklearn.metrics.cluster.contingency_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.cluster.contingency_matrix</span></code></a>(…[,&nbsp;…])</p></td>
  <td><p>Build a contingency matrix describing the relationship between labels.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score" title="sklearn.metrics.fowlkes_mallows_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.fowlkes_mallows_score</span></code></a>(labels_true,&nbsp;…)</p></td>
  <td><p>Measure the similarity of two clusterings of a set of points.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html#sklearn.metrics.homogeneity_completeness_v_measure" title="sklearn.metrics.homogeneity_completeness_v_measure"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.homogeneity_completeness_v_measure</span></code></a>(…)</p></td>
  <td><p>Compute the homogeneity and completeness and V-Measure scores at once.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score" title="sklearn.metrics.homogeneity_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.homogeneity_score</span></code></a>(labels_true,&nbsp;…)</p></td>
  <td><p>Homogeneity metric of a cluster labeling given a ground truth.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score" title="sklearn.metrics.mutual_info_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.mutual_info_score</span></code></a>(labels_true,&nbsp;…)</p></td>
  <td><p>Mutual Information between two clusterings.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score" title="sklearn.metrics.normalized_mutual_info_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.normalized_mutual_info_score</span></code></a>(…[,&nbsp;…])</p></td>
  <td><p>Normalized Mutual Information between two clusterings.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score" title="sklearn.metrics.silhouette_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.silhouette_score</span></code></a>(X,&nbsp;labels,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute the mean Silhouette Coefficient of all samples.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples" title="sklearn.metrics.silhouette_samples"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.silhouette_samples</span></code></a>(X,&nbsp;labels,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute the Silhouette Coefficient for each sample.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score" title="sklearn.metrics.v_measure_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.v_measure_score</span></code></a>(labels_true,&nbsp;…[,&nbsp;beta])</p></td>
  <td><p>V-measure cluster labeling given a ground truth.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Biclustering metrics </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.consensus_score.html#sklearn.metrics.consensus_score" title="sklearn.metrics.consensus_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.consensus_score</span></code></a>(a,&nbsp;b,&nbsp;*[,&nbsp;similarity])</p></td>
  <td><p>The similarity of two sets of biclusters.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Pairwise metrics </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.additive_chi2_kernel.html#sklearn.metrics.pairwise.additive_chi2_kernel" title="sklearn.metrics.pairwise.additive_chi2_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.additive_chi2_kernel</span></code></a>(X[,&nbsp;Y])</p></td>
  <td><p>Computes the additive chi-squared kernel between observations in X and Y</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.chi2_kernel.html#sklearn.metrics.pairwise.chi2_kernel" title="sklearn.metrics.pairwise.chi2_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.chi2_kernel</span></code></a>(X[,&nbsp;Y,&nbsp;gamma])</p></td>
  <td><p>Computes the exponential chi-squared kernel X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity" title="sklearn.metrics.pairwise.cosine_similarity"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.cosine_similarity</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Compute cosine similarity between samples in X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances" title="sklearn.metrics.pairwise.cosine_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.cosine_distances</span></code></a>(X[,&nbsp;Y])</p></td>
  <td><p>Compute cosine distance between samples in X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics" title="sklearn.metrics.pairwise.distance_metrics"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.distance_metrics</span></code></a>()</p></td>
  <td><p>Valid metrics for pairwise_distances.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances" title="sklearn.metrics.pairwise.euclidean_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.euclidean_distances</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html#sklearn.metrics.pairwise.haversine_distances" title="sklearn.metrics.pairwise.haversine_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.haversine_distances</span></code></a>(X[,&nbsp;Y])</p></td>
  <td><p>Compute the Haversine distance between samples in X and Y</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics" title="sklearn.metrics.pairwise.kernel_metrics"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.kernel_metrics</span></code></a>()</p></td>
  <td><p>Valid metrics for pairwise_kernels</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html#sklearn.metrics.pairwise.laplacian_kernel" title="sklearn.metrics.pairwise.laplacian_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.laplacian_kernel</span></code></a>(X[,&nbsp;Y,&nbsp;gamma])</p></td>
  <td><p>Compute the laplacian kernel between X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html#sklearn.metrics.pairwise.linear_kernel" title="sklearn.metrics.pairwise.linear_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.linear_kernel</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Compute the linear kernel between X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html#sklearn.metrics.pairwise.manhattan_distances" title="sklearn.metrics.pairwise.manhattan_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.manhattan_distances</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Compute the L1 distances between the vectors in X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html#sklearn.metrics.pairwise.nan_euclidean_distances" title="sklearn.metrics.pairwise.nan_euclidean_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.nan_euclidean_distances</span></code></a>(X)</p></td>
  <td><p>Calculate the euclidean distances in the presence of missing values.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html#sklearn.metrics.pairwise.pairwise_kernels" title="sklearn.metrics.pairwise.pairwise_kernels"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.pairwise_kernels</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Compute the kernel between arrays X and optional array Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html#sklearn.metrics.pairwise.polynomial_kernel" title="sklearn.metrics.pairwise.polynomial_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.polynomial_kernel</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Compute the polynomial kernel between X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel" title="sklearn.metrics.pairwise.rbf_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.rbf_kernel</span></code></a>(X[,&nbsp;Y,&nbsp;gamma])</p></td>
  <td><p>Compute the rbf (gaussian) kernel between X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html#sklearn.metrics.pairwise.sigmoid_kernel" title="sklearn.metrics.pairwise.sigmoid_kernel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.sigmoid_kernel</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Compute the sigmoid kernel between X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_euclidean_distances.html#sklearn.metrics.pairwise.paired_euclidean_distances" title="sklearn.metrics.pairwise.paired_euclidean_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.paired_euclidean_distances</span></code></a>(X,&nbsp;Y)</p></td>
  <td><p>Computes the paired euclidean distances between X and Y</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_manhattan_distances.html#sklearn.metrics.pairwise.paired_manhattan_distances" title="sklearn.metrics.pairwise.paired_manhattan_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.paired_manhattan_distances</span></code></a>(X,&nbsp;Y)</p></td>
  <td><p>Compute the L1 distances between the vectors in X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_cosine_distances.html#sklearn.metrics.pairwise.paired_cosine_distances" title="sklearn.metrics.pairwise.paired_cosine_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.paired_cosine_distances</span></code></a>(X,&nbsp;Y)</p></td>
  <td><p>Computes the paired cosine distances between X and Y</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html#sklearn.metrics.pairwise.paired_distances" title="sklearn.metrics.pairwise.paired_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise.paired_distances</span></code></a>(X,&nbsp;Y,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Computes the paired distances between X and Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances" title="sklearn.metrics.pairwise_distances"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise_distances</span></code></a>(X[,&nbsp;Y,&nbsp;metric,&nbsp;…])</p></td>
  <td><p>Compute the distance matrix from a vector array X and optional Y.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html#sklearn.metrics.pairwise_distances_argmin" title="sklearn.metrics.pairwise_distances_argmin"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise_distances_argmin</span></code></a>(X,&nbsp;Y,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Compute minimum distances between one point and a set of points.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html#sklearn.metrics.pairwise_distances_argmin_min" title="sklearn.metrics.pairwise_distances_argmin_min"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise_distances_argmin_min</span></code></a>(X,&nbsp;Y,&nbsp;*)</p></td>
  <td><p>Compute minimum distances between one point and a set of points.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html#sklearn.metrics.pairwise_distances_chunked" title="sklearn.metrics.pairwise_distances_chunked"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.pairwise_distances_chunked</span></code></a>(X[,&nbsp;Y,&nbsp;…])</p></td>
  <td><p>Generate a distance matrix chunk by chunk with optional reduction</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Plotting </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix" title="sklearn.metrics.plot_confusion_matrix"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.plot_confusion_matrix</span></code></a>(estimator,&nbsp;X,&nbsp;…)</p></td>
  <td><p>Plot Confusion Matrix.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html#sklearn.metrics.plot_precision_recall_curve" title="sklearn.metrics.plot_precision_recall_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.plot_precision_recall_curve</span></code></a>(…[,&nbsp;…])</p></td>
  <td><p>Plot Precision Recall Curve for binary classifiers.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html#sklearn.metrics.plot_roc_curve" title="sklearn.metrics.plot_roc_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">metrics.plot_roc_curve</span></code></a>(estimator,&nbsp;X,&nbsp;y,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Plot Receiver operating characteristic (ROC) curve.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.mixture: Gaussian Mixture Models

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture">sklearn.mixture</a>: Gaussian Mixture Models</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture" title="sklearn.mixture.BayesianGaussianMixture"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">mixture.BayesianGaussianMixture</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Variational Bayesian estimation of a Gaussian mixture.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture" title="sklearn.mixture.GaussianMixture"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">mixture.GaussianMixture</span></code></a>([n_components,&nbsp;…])</p></td>
  <td><p>Gaussian Mixture.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.model_selection: Model Selection

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection">sklearn.model_selection</a>: Model Selection</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Splitter Classes </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold" title="sklearn.model_selection.GroupKFold"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.GroupKFold</span></code></a>([n_splits])</p></td>
  <td><p>K-fold iterator variant with non-overlapping groups.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit" title="sklearn.model_selection.GroupShuffleSplit"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.GroupShuffleSplit</span></code></a>([…])</p></td>
  <td><p>Shuffle-Group(s)-Out cross-validation iterator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold" title="sklearn.model_selection.KFold"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.KFold</span></code></a>([n_splits,&nbsp;shuffle,&nbsp;…])</p></td>
  <td><p>K-Folds cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut" title="sklearn.model_selection.LeaveOneGroupOut"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.LeaveOneGroupOut</span></code></a></p></td>
  <td><p>Leave One Group Out cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut" title="sklearn.model_selection.LeavePGroupsOut"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.LeavePGroupsOut</span></code></a>(n_groups)</p></td>
  <td><p>Leave P Group(s) Out cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut" title="sklearn.model_selection.LeaveOneOut"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.LeaveOneOut</span></code></a></p></td>
  <td><p>Leave-One-Out cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html#sklearn.model_selection.LeavePOut" title="sklearn.model_selection.LeavePOut"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.LeavePOut</span></code></a>(p)</p></td>
  <td><p>Leave-P-Out cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html#sklearn.model_selection.PredefinedSplit" title="sklearn.model_selection.PredefinedSplit"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.PredefinedSplit</span></code></a>(test_fold)</p></td>
  <td><p>Predefined split cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold" title="sklearn.model_selection.RepeatedKFold"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.RepeatedKFold</span></code></a>(*[,&nbsp;n_splits,&nbsp;…])</p></td>
  <td><p>Repeated K-Fold cross validator.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold" title="sklearn.model_selection.RepeatedStratifiedKFold"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.RepeatedStratifiedKFold</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Repeated Stratified K-Fold cross validator.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit" title="sklearn.model_selection.ShuffleSplit"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.ShuffleSplit</span></code></a>([n_splits,&nbsp;…])</p></td>
  <td><p>Random permutation cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold" title="sklearn.model_selection.StratifiedKFold"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.StratifiedKFold</span></code></a>([n_splits,&nbsp;…])</p></td>
  <td><p>Stratified K-Folds cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit" title="sklearn.model_selection.StratifiedShuffleSplit"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.StratifiedShuffleSplit</span></code></a>([…])</p></td>
  <td><p>Stratified ShuffleSplit cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit" title="sklearn.model_selection.TimeSeriesSplit"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.TimeSeriesSplit</span></code></a>([n_splits,&nbsp;…])</p></td>
  <td><p>Time Series cross-validator</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Splitter Functions </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html#sklearn.model_selection.check_cv" title="sklearn.model_selection.check_cv"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.check_cv</span></code></a>([cv,&nbsp;y,&nbsp;classifier])</p></td>
  <td><p>Input checker utility for building a cross-validator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split" title="sklearn.model_selection.train_test_split"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.train_test_split</span></code></a>(*arrays,&nbsp;…)</p></td>
  <td><p>Split arrays or matrices into random train and test subsets</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Hyper-parameter optimizers </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" title="sklearn.model_selection.GridSearchCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.GridSearchCV</span></code></a>(estimator,&nbsp;…)</p></td>
  <td><p>Exhaustive search over specified parameter values for an estimator.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid" title="sklearn.model_selection.ParameterGrid"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.ParameterGrid</span></code></a>(param_grid)</p></td>
  <td><p>Grid of parameters with a discrete number of values for each.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler" title="sklearn.model_selection.ParameterSampler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.ParameterSampler</span></code></a>(…[,&nbsp;…])</p></td>
  <td><p>Generator on parameters sampled from given distributions.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV" title="sklearn.model_selection.RandomizedSearchCV"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.RandomizedSearchCV</span></code></a>(…[,&nbsp;…])</p></td>
  <td><p>Randomized search on hyper parameters.</p></td>
  </tr>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Model validation </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate" title="sklearn.model_selection.cross_validate"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.cross_validate</span></code></a>(estimator,&nbsp;X)</p></td>
  <td><p>Evaluate metric(s) by cross-validation and also record fit/score times.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict" title="sklearn.model_selection.cross_val_predict"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.cross_val_predict</span></code></a>(estimator,&nbsp;X)</p></td>
  <td><p>Generate cross-validated estimates for each input data point</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score" title="sklearn.model_selection.cross_val_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.cross_val_score</span></code></a>(estimator,&nbsp;X)</p></td>
  <td><p>Evaluate a score by cross-validation</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve" title="sklearn.model_selection.learning_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.learning_curve</span></code></a>(estimator,&nbsp;X,&nbsp;…)</p></td>
  <td><p>Learning curve.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html#sklearn.model_selection.permutation_test_score" title="sklearn.model_selection.permutation_test_score"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.permutation_test_score</span></code></a>(…)</p></td>
  <td><p>Evaluate the significance of a cross-validated score with permutations</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve" title="sklearn.model_selection.validation_curve"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">model_selection.validation_curve</span></code></a>(estimator,&nbsp;…)</p></td>
  <td><p>Validation curve.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.multiclass: Multiclass and multilabel classification

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass">sklearn.multiclass</a>: Multiclass and multilabel classification</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier" title="sklearn.multiclass.OneVsRestClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">multiclass.OneVsRestClassifier</span></code></a>(estimator,&nbsp;*)</p></td>
  <td><p>One-vs-the-rest (OvR) multiclass/multilabel strategy</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier" title="sklearn.multiclass.OneVsOneClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">multiclass.OneVsOneClassifier</span></code></a>(estimator,&nbsp;*)</p></td>
  <td><p>One-vs-one multiclass strategy</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html#sklearn.multiclass.OutputCodeClassifier" title="sklearn.multiclass.OutputCodeClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">multiclass.OutputCodeClassifier</span></code></a>(estimator,&nbsp;*)</p></td>
  <td><p>(Error-Correcting) Output-Code multiclass strategy</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.naive_bayes: Naive Bayes

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes">sklearn.naive_bayes</a>: Naive Bayes</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB" title="sklearn.naive_bayes.BernoulliNB"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">naive_bayes.BernoulliNB</span></code></a>(*[,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>Naive Bayes classifier for multivariate Bernoulli models.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB" title="sklearn.naive_bayes.CategoricalNB"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">naive_bayes.CategoricalNB</span></code></a>(*[,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>Naive Bayes classifier for categorical features</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB" title="sklearn.naive_bayes.ComplementNB"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">naive_bayes.ComplementNB</span></code></a>(*[,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>The Complement Naive Bayes classifier described in Rennie et al.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB" title="sklearn.naive_bayes.GaussianNB"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">naive_bayes.GaussianNB</span></code></a>(*[,&nbsp;priors,&nbsp;…])</p></td>
  <td><p>Gaussian Naive Bayes (GaussianNB)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB" title="sklearn.naive_bayes.MultinomialNB"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">naive_bayes.MultinomialNB</span></code></a>(*[,&nbsp;alpha,&nbsp;…])</p></td>
  <td><p>Naive Bayes classifier for multinomial models</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.neighbors: Nearest Neighbors

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors">sklearn.neighbors</a>: Nearest Neighbors</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree" title="sklearn.neighbors.BallTree"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.BallTree</span></code></a>(X[,&nbsp;leaf_size,&nbsp;metric])</p></td>
  <td><p>BallTree for fast generalized N-point problems</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric" title="sklearn.neighbors.DistanceMetric"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.DistanceMetric</span></code></a></p></td>
  <td><p>DistanceMetric class</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree" title="sklearn.neighbors.KDTree"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.KDTree</span></code></a>(X[,&nbsp;leaf_size,&nbsp;metric])</p></td>
  <td><p>KDTree for fast generalized N-point problems</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity" title="sklearn.neighbors.KernelDensity"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.KernelDensity</span></code></a>(*[,&nbsp;bandwidth,&nbsp;…])</p></td>
  <td><p>Kernel Density Estimation.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier" title="sklearn.neighbors.KNeighborsClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.KNeighborsClassifier</span></code></a>([…])</p></td>
  <td><p>Classifier implementing the k-nearest neighbors vote.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor" title="sklearn.neighbors.KNeighborsRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.KNeighborsRegressor</span></code></a>([n_neighbors,&nbsp;…])</p></td>
  <td><p>Regression based on k-nearest neighbors.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsTransformer.html#sklearn.neighbors.KNeighborsTransformer" title="sklearn.neighbors.KNeighborsTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.KNeighborsTransformer</span></code></a>(*[,&nbsp;mode,&nbsp;…])</p></td>
  <td><p>Transform X into a (weighted) graph of k nearest neighbors</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor" title="sklearn.neighbors.LocalOutlierFactor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.LocalOutlierFactor</span></code></a>([n_neighbors,&nbsp;…])</p></td>
  <td><p>Unsupervised Outlier Detection using Local Outlier Factor (LOF)</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier" title="sklearn.neighbors.RadiusNeighborsClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.RadiusNeighborsClassifier</span></code></a>([…])</p></td>
  <td><p>Classifier implementing a vote among neighbors within a given radius</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor" title="sklearn.neighbors.RadiusNeighborsRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.RadiusNeighborsRegressor</span></code></a>([radius,&nbsp;…])</p></td>
  <td><p>Regression based on neighbors within a fixed radius.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsTransformer.html#sklearn.neighbors.RadiusNeighborsTransformer" title="sklearn.neighbors.RadiusNeighborsTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.RadiusNeighborsTransformer</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Transform X into a (weighted) graph of neighbors nearer than a radius</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid" title="sklearn.neighbors.NearestCentroid"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.NearestCentroid</span></code></a>([metric,&nbsp;…])</p></td>
  <td><p>Nearest centroid classifier.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors" title="sklearn.neighbors.NearestNeighbors"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.NearestNeighbors</span></code></a>(*[,&nbsp;n_neighbors,&nbsp;…])</p></td>
  <td><p>Unsupervised learner for implementing neighbor searches.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html#sklearn.neighbors.NeighborhoodComponentsAnalysis" title="sklearn.neighbors.NeighborhoodComponentsAnalysis"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neighbors.NeighborhoodComponentsAnalysis</span></code></a>([…])</p></td>
  <td><p>Neighborhood Components Analysis</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.neural_network: Neural network models

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network">sklearn.neural_network</a>: Neural network models</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM" title="sklearn.neural_network.BernoulliRBM"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neural_network.BernoulliRBM</span></code></a>([n_components,&nbsp;…])</p></td>
  <td><p>Bernoulli Restricted Boltzmann Machine (RBM).</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier" title="sklearn.neural_network.MLPClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neural_network.MLPClassifier</span></code></a>([…])</p></td>
  <td><p>Multi-layer Perceptron classifier.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor" title="sklearn.neural_network.MLPRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">neural_network.MLPRegressor</span></code></a>([…])</p></td>
  <td><p>Multi-layer Perceptron regressor.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.pipeline: Pipeline

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline">sklearn.pipeline</a>: Pipeline</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion" title="sklearn.pipeline.FeatureUnion"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">pipeline.FeatureUnion</span></code></a>(transformer_list,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Concatenates results of multiple transformer objects.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline" title="sklearn.pipeline.Pipeline"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">pipeline.Pipeline</span></code></a>(steps,&nbsp;*[,&nbsp;memory,&nbsp;verbose])</p></td>
  <td><p>Pipeline of transforms with a final estimator.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline" title="sklearn.pipeline.make_pipeline"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">pipeline.make_pipeline</span></code></a>(*steps,&nbsp;**kwargs)</p></td>
  <td><p>Construct a Pipeline from the given estimators.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_union.html#sklearn.pipeline.make_union" title="sklearn.pipeline.make_union"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">pipeline.make_union</span></code></a>(*transformers,&nbsp;**kwargs)</p></td>
  <td><p>Construct a FeatureUnion from the given transformers.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.preprocessing: Preprocessing and Normalization

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing">sklearn.preprocessing</a>: Preprocessing and Normalization</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer" title="sklearn.preprocessing.Binarizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.Binarizer</span></code></a>(*[,&nbsp;threshold,&nbsp;copy])</p></td>
  <td><p>Binarize data (set feature values to 0 or 1) according to a threshold</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer" title="sklearn.preprocessing.FunctionTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.FunctionTransformer</span></code></a>([func,&nbsp;…])</p></td>
  <td><p>Constructs a transformer from an arbitrary callable.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer" title="sklearn.preprocessing.KBinsDiscretizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.KBinsDiscretizer</span></code></a>([n_bins,&nbsp;…])</p></td>
  <td><p>Bin continuous data into intervals.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html#sklearn.preprocessing.KernelCenterer" title="sklearn.preprocessing.KernelCenterer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.KernelCenterer</span></code></a>()</p></td>
  <td><p>Center a kernel matrix</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer" title="sklearn.preprocessing.LabelBinarizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.LabelBinarizer</span></code></a>(*[,&nbsp;neg_label,&nbsp;…])</p></td>
  <td><p>Binarize labels in a one-vs-all fashion</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder" title="sklearn.preprocessing.LabelEncoder"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.LabelEncoder</span></code></a></p></td>
  <td><p>Encode target labels with value between 0 and n_classes-1.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer" title="sklearn.preprocessing.MultiLabelBinarizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.MultiLabelBinarizer</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Transform between iterable of iterables and a multilabel format</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler" title="sklearn.preprocessing.MaxAbsScaler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.MaxAbsScaler</span></code></a>(*[,&nbsp;copy])</p></td>
  <td><p>Scale each feature by its maximum absolute value.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler" title="sklearn.preprocessing.MinMaxScaler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.MinMaxScaler</span></code></a>([feature_range,&nbsp;copy])</p></td>
  <td><p>Transform features by scaling each feature to a given range.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer" title="sklearn.preprocessing.Normalizer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.Normalizer</span></code></a>([norm,&nbsp;copy])</p></td>
  <td><p>Normalize samples individually to unit norm.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder" title="sklearn.preprocessing.OneHotEncoder"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.OneHotEncoder</span></code></a>(*[,&nbsp;categories,&nbsp;…])</p></td>
  <td><p>Encode categorical features as a one-hot numeric array.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder" title="sklearn.preprocessing.OrdinalEncoder"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.OrdinalEncoder</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Encode categorical features as an integer array.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures" title="sklearn.preprocessing.PolynomialFeatures"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.PolynomialFeatures</span></code></a>([degree,&nbsp;…])</p></td>
  <td><p>Generate polynomial and interaction features.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer" title="sklearn.preprocessing.PowerTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.PowerTransformer</span></code></a>([method,&nbsp;…])</p></td>
  <td><p>Apply a power transform featurewise to make data more Gaussian-like.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer" title="sklearn.preprocessing.QuantileTransformer"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.QuantileTransformer</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Transform features using quantiles information.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler" title="sklearn.preprocessing.RobustScaler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.RobustScaler</span></code></a>(*[,&nbsp;…])</p></td>
  <td><p>Scale features using statistics that are robust to outliers.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler" title="sklearn.preprocessing.StandardScaler"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.StandardScaler</span></code></a>(*[,&nbsp;copy,&nbsp;…])</p></td>
  <td><p>Standardize features by removing the mean and scaling to unit variance</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.add_dummy_feature.html#sklearn.preprocessing.add_dummy_feature" title="sklearn.preprocessing.add_dummy_feature"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.add_dummy_feature</span></code></a>(X[,&nbsp;value])</p></td>
  <td><p>Augment dataset with an additional dummy feature.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html#sklearn.preprocessing.binarize" title="sklearn.preprocessing.binarize"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.binarize</span></code></a>(X,&nbsp;*[,&nbsp;threshold,&nbsp;copy])</p></td>
  <td><p>Boolean thresholding of array-like or scipy.sparse matrix</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#sklearn.preprocessing.label_binarize" title="sklearn.preprocessing.label_binarize"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.label_binarize</span></code></a>(y,&nbsp;*,&nbsp;classes)</p></td>
  <td><p>Binarize labels in a one-vs-all fashion</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html#sklearn.preprocessing.maxabs_scale" title="sklearn.preprocessing.maxabs_scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.maxabs_scale</span></code></a>(X,&nbsp;*[,&nbsp;axis,&nbsp;copy])</p></td>
  <td><p>Scale each feature to the [-1, 1] range without breaking the sparsity.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale" title="sklearn.preprocessing.minmax_scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.minmax_scale</span></code></a>(X[,&nbsp;…])</p></td>
  <td><p>Transform features by scaling each feature to a given range.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize" title="sklearn.preprocessing.normalize"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.normalize</span></code></a>(X[,&nbsp;norm,&nbsp;axis,&nbsp;…])</p></td>
  <td><p>Scale input vectors individually to unit norm (vector length).</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform" title="sklearn.preprocessing.quantile_transform"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.quantile_transform</span></code></a>(X,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Transform features using quantiles information.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale" title="sklearn.preprocessing.robust_scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.robust_scale</span></code></a>(X,&nbsp;*[,&nbsp;axis,&nbsp;…])</p></td>
  <td><p>Standardize a dataset along any axis</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale" title="sklearn.preprocessing.scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.scale</span></code></a>(X,&nbsp;*[,&nbsp;axis,&nbsp;with_mean,&nbsp;…])</p></td>
  <td><p>Standardize a dataset along any axis</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform" title="sklearn.preprocessing.power_transform"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">preprocessing.power_transform</span></code></a>(X[,&nbsp;method,&nbsp;…])</p></td>
  <td><p>Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like.</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.random_projection: Random projection

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection">sklearn.random_projection</a>: Random projection</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection" title="sklearn.random_projection.GaussianRandomProjection"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">random_projection.GaussianRandomProjection</span></code></a>([…])</p></td>
  <td><p>Reduce dimensionality through Gaussian random projection</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html#sklearn.random_projection.SparseRandomProjection" title="sklearn.random_projection.SparseRandomProjection"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">random_projection.SparseRandomProjection</span></code></a>([…])</p></td>
  <td><p>Reduce dimensionality through sparse random projection</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.semi_supervised: Semi-Supervised Learning

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised">sklearn.semi_supervised</a>: Semi-Supervised Learning</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation" title="sklearn.semi_supervised.LabelPropagation"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">semi_supervised.LabelPropagation</span></code></a>([kernel,&nbsp;…])</p></td>
  <td><p>Label Propagation classifier</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading" title="sklearn.semi_supervised.LabelSpreading"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">semi_supervised.LabelSpreading</span></code></a>([kernel,&nbsp;…])</p></td>
  <td><p>LabelSpreading model for semi-supervised learning</p></td>
  </tr>
</tbody>
</table><br/>


## sklearn.svm: Support Vector Machines

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm">sklearn.svm</a>: Support Vector Machines</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr><td colspan="2" style="text-align: center; font-weight: bolder; line-height: 1.6; vertical-align: middle; font-size: 1.2em; background-color: lightgrey; color: darkblue;"> Estimators </td> </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC" title="sklearn.svm.LinearSVC"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.LinearSVC</span></code></a>([penalty,&nbsp;loss,&nbsp;dual,&nbsp;tol,&nbsp;C,&nbsp;…])</p></td>
  <td><p>Linear Support Vector Classification.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR" title="sklearn.svm.LinearSVR"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.LinearSVR</span></code></a>(*[,&nbsp;epsilon,&nbsp;tol,&nbsp;C,&nbsp;loss,&nbsp;…])</p></td>
  <td><p>Linear Support Vector Regression.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC" title="sklearn.svm.NuSVC"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.NuSVC</span></code></a>(*[,&nbsp;nu,&nbsp;kernel,&nbsp;degree,&nbsp;gamma,&nbsp;…])</p></td>
  <td><p>Nu-Support Vector Classification.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR" title="sklearn.svm.NuSVR"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.NuSVR</span></code></a>(*[,&nbsp;nu,&nbsp;C,&nbsp;kernel,&nbsp;degree,&nbsp;gamma,&nbsp;…])</p></td>
  <td><p>Nu Support Vector Regression.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM" title="sklearn.svm.OneClassSVM"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.OneClassSVM</span></code></a>(*[,&nbsp;kernel,&nbsp;degree,&nbsp;gamma,&nbsp;…])</p></td>
  <td><p>Unsupervised Outlier Detection.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC" title="sklearn.svm.SVC"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.SVC</span></code></a>(*[,&nbsp;C,&nbsp;kernel,&nbsp;degree,&nbsp;gamma,&nbsp;…])</p></td>
  <td><p>C-Support Vector Classification.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR" title="sklearn.svm.SVR"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.SVR</span></code></a>(*[,&nbsp;kernel,&nbsp;degree,&nbsp;gamma,&nbsp;coef0,&nbsp;…])</p></td>
  <td><p>Epsilon-Support Vector Regression.</p></td>
  </tr>
  <td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.svm.l1_min_c.html#sklearn.svm.l1_min_c" title="sklearn.svm.l1_min_c"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">svm.l1_min_c</span></code></a>(X,&nbsp;y,&nbsp;*[,&nbsp;loss,&nbsp;fit_intercept,&nbsp;…])</p></td>
</tbody>
</table><br/>


## sklearn.tree: Decision Trees

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree">sklearn.tree</a>: Decision Trees</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier" title="sklearn.tree.DecisionTreeClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.DecisionTreeClassifier</span></code></a>(*[,&nbsp;criterion,&nbsp;…])</p></td>
  <td><p>A decision tree classifier.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor" title="sklearn.tree.DecisionTreeRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.DecisionTreeRegressor</span></code></a>(*[,&nbsp;criterion,&nbsp;…])</p></td>
  <td><p>A decision tree regressor.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier" title="sklearn.tree.ExtraTreeClassifier"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.ExtraTreeClassifier</span></code></a>(*[,&nbsp;criterion,&nbsp;…])</p></td>
  <td><p>An extremely randomized tree classifier.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor" title="sklearn.tree.ExtraTreeRegressor"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.ExtraTreeRegressor</span></code></a>(*[,&nbsp;criterion,&nbsp;…])</p></td>
  <td><p>An extremely randomized tree regressor.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz" title="sklearn.tree.export_graphviz"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.export_graphviz</span></code></a>(decision_tree[,&nbsp;…])</p></td>
  <td><p>Export a decision tree in DOT format.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html#sklearn.tree.export_text" title="sklearn.tree.export_text"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.export_text</span></code></a>(decision_tree,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Build a text report showing the rules of a decision tree.</p></td>
  </tr>
  <td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree" title="sklearn.tree.plot_tree"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">tree.plot_tree</span></code></a>(decision_tree,&nbsp;*[,&nbsp;…])</p></td>
</tbody>
</table><br/>


## sklearn.utils: Utilities

<table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 60vw;" cellspacing="0" cellpadding="5" border="1">
<caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils">sklearn.utils</a>: Utilities</caption>
<thead>
<tr style="font-size: 1.2em;">
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Function</th>
  <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:90%;">Description</th>
</tr>
</thead>
<tbody>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.min_pos.html#sklearn.utils.arrayfuncs.min_pos" title="sklearn.utils.arrayfuncs.min_pos"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.arrayfuncs.min_pos</span></code></a></p></td>
  <td><p>Find the minimum value of an array over positive values</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.as_float_array.html#sklearn.utils.as_float_array" title="sklearn.utils.as_float_array"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.as_float_array</span></code></a>(X,&nbsp;*[,&nbsp;copy,&nbsp;…])</p></td>
  <td><p>Converts an array-like to an array of floats.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.assert_all_finite.html#sklearn.utils.assert_all_finite" title="sklearn.utils.assert_all_finite"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.assert_all_finite</span></code></a>(X,&nbsp;*[,&nbsp;allow_nan])</p></td>
  <td><p>Throw a ValueError if X contains NaN or infinity.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch" title="sklearn.utils.Bunch"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.Bunch</span></code></a>(**kwargs)</p></td>
  <td><p>Container object exposing keys as attributes</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y" title="sklearn.utils.check_X_y"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.check_X_y</span></code></a>(X,&nbsp;y[,&nbsp;accept_sparse,&nbsp;…])</p></td>
  <td><p>Input validation for standard estimators.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html#sklearn.utils.check_array" title="sklearn.utils.check_array"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.check_array</span></code></a>(array[,&nbsp;accept_sparse,&nbsp;…])</p></td>
  <td><p>Input validation on an array, list, sparse matrix or similar.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_scalar.html#sklearn.utils.check_scalar" title="sklearn.utils.check_scalar"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.check_scalar</span></code></a>(x,&nbsp;name,&nbsp;target_type,&nbsp;*)</p></td>
  <td><p>Validate scalar parameters type and value.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_consistent_length.html#sklearn.utils.check_consistent_length" title="sklearn.utils.check_consistent_length"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.check_consistent_length</span></code></a>(*arrays)</p></td>
  <td><p>Check that all arrays have consistent first dimensions.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state" title="sklearn.utils.check_random_state"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.check_random_state</span></code></a>(seed)</p></td>
  <td><p>Turn seed into a np.random.RandomState instance</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html#sklearn.utils.class_weight.compute_class_weight" title="sklearn.utils.class_weight.compute_class_weight"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.class_weight.compute_class_weight</span></code></a>(…)</p></td>
  <td><p>Estimate class weights for unbalanced datasets.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html#sklearn.utils.class_weight.compute_sample_weight" title="sklearn.utils.class_weight.compute_sample_weight"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.class_weight.compute_sample_weight</span></code></a>(…)</p></td>
  <td><p>Estimate sample weights by class for unbalanced datasets.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated" title="sklearn.utils.deprecated"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.deprecated</span></code></a>([extra])</p></td>
  <td><p>Decorator to mark a function or class as deprecated.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator" title="sklearn.utils.estimator_checks.check_estimator"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.estimator_checks.check_estimator</span></code></a>(Estimator)</p></td>
  <td><p>Check if estimator adheres to scikit-learn conventions.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks" title="sklearn.utils.estimator_checks.parametrize_with_checks"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.estimator_checks.parametrize_with_checks</span></code></a>(…)</p></td>
  <td><p>Pytest specific decorator for parametrizing estimator checks.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_html_repr.html#sklearn.utils.estimator_html_repr" title="sklearn.utils.estimator_html_repr"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.estimator_html_repr</span></code></a>(estimator)</p></td>
  <td><p>Build a HTML representation of an estimator.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.safe_sparse_dot.html#sklearn.utils.extmath.safe_sparse_dot" title="sklearn.utils.extmath.safe_sparse_dot"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.extmath.safe_sparse_dot</span></code></a>(a,&nbsp;b,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Dot product that handle the sparse matrix case correctly</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_range_finder.html#sklearn.utils.extmath.randomized_range_finder" title="sklearn.utils.extmath.randomized_range_finder"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.extmath.randomized_range_finder</span></code></a>(A,&nbsp;*,&nbsp;…)</p></td>
  <td><p>Computes an orthonormal matrix whose range approximates the range of A.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd" title="sklearn.utils.extmath.randomized_svd"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.extmath.randomized_svd</span></code></a>(M,&nbsp;n_components,&nbsp;*)</p></td>
  <td><p>Computes a truncated randomized SVD</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.fast_logdet.html#sklearn.utils.extmath.fast_logdet" title="sklearn.utils.extmath.fast_logdet"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.extmath.fast_logdet</span></code></a>(A)</p></td>
  <td><p>Compute log(det(A)) for A symmetric</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.density.html#sklearn.utils.extmath.density" title="sklearn.utils.extmath.density"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.extmath.density</span></code></a>(w,&nbsp;**kwargs)</p></td>
  <td><p>Compute density of a sparse vector</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.weighted_mode.html#sklearn.utils.extmath.weighted_mode" title="sklearn.utils.extmath.weighted_mode"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.extmath.weighted_mode</span></code></a>(a,&nbsp;w,&nbsp;*[,&nbsp;axis])</p></td>
  <td><p>Returns an array of the weighted modal (most common) value in a</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_even_slices.html#sklearn.utils.gen_even_slices" title="sklearn.utils.gen_even_slices"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.gen_even_slices</span></code></a>(n,&nbsp;n_packs,&nbsp;*[,&nbsp;n_samples])</p></td>
  <td><p>Generator to create n_packs slices going up to n.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph.single_source_shortest_path_length.html#sklearn.utils.graph.single_source_shortest_path_length" title="sklearn.utils.graph.single_source_shortest_path_length"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.graph.single_source_shortest_path_length</span></code></a>(…)</p></td>
  <td><p>Return the shortest path length from source to all reachable nodes.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph_shortest_path.graph_shortest_path.html#sklearn.utils.graph_shortest_path.graph_shortest_path" title="sklearn.utils.graph_shortest_path.graph_shortest_path"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.graph_shortest_path.graph_shortest_path</span></code></a></p></td>
  <td><p>Perform a shortest-path graph search on a positive directed or undirected graph.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.indexable.html#sklearn.utils.indexable" title="sklearn.utils.indexable"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.indexable</span></code></a>(*iterables)</p></td>
  <td><p>Make arrays indexable for cross-validation.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.metaestimators.if_delegate_has_method.html#sklearn.utils.metaestimators.if_delegate_has_method" title="sklearn.utils.metaestimators.if_delegate_has_method"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.metaestimators.if_delegate_has_method</span></code></a>(…)</p></td>
  <td><p>Create a decorator for methods that are delegated to a sub-estimator</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html#sklearn.utils.multiclass.type_of_target" title="sklearn.utils.multiclass.type_of_target"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.multiclass.type_of_target</span></code></a>(y)</p></td>
  <td><p>Determine the type of data indicated by the target.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.is_multilabel.html#sklearn.utils.multiclass.is_multilabel" title="sklearn.utils.multiclass.is_multilabel"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.multiclass.is_multilabel</span></code></a>(y)</p></td>
  <td><p>Check if <codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">y</span></code> is in a multilabel format.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.unique_labels.html#sklearn.utils.multiclass.unique_labels" title="sklearn.utils.multiclass.unique_labels"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.multiclass.unique_labels</span></code></a>(*ys)</p></td>
  <td><p>Extract an ordered array of unique labels</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.murmurhash3_32.html#sklearn.utils.murmurhash3_32" title="sklearn.utils.murmurhash3_32"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.murmurhash3_32</span></code></a></p></td>
  <td><p>Compute the 32bit murmurhash3 of key at seed.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample" title="sklearn.utils.resample"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.resample</span></code></a>(*arrays,&nbsp;**options)</p></td>
  <td><p>Resample arrays or sparse matrices in a consistent way</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils._safe_indexing.html#sklearn.utils._safe_indexing" title="sklearn.utils._safe_indexing"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils._safe_indexing</span></code></a>(X,&nbsp;indices,&nbsp;*[,&nbsp;axis])</p></td>
  <td><p>Return rows, items or columns of X using indices.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_mask.html#sklearn.utils.safe_mask" title="sklearn.utils.safe_mask"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.safe_mask</span></code></a>(X,&nbsp;mask)</p></td>
  <td><p>Return a mask which is safe to use on X.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_sqr.html#sklearn.utils.safe_sqr" title="sklearn.utils.safe_sqr"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.safe_sqr</span></code></a>(X,&nbsp;*[,&nbsp;copy])</p></td>
  <td><p>Element wise squaring of array-likes and sparse matrices.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html#sklearn.utils.shuffle" title="sklearn.utils.shuffle"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.shuffle</span></code></a>(*arrays,&nbsp;**options)</p></td>
  <td><p>Shuffle arrays or sparse matrices in a consistent way</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.incr_mean_variance_axis.html#sklearn.utils.sparsefuncs.incr_mean_variance_axis" title="sklearn.utils.sparsefuncs.incr_mean_variance_axis"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.incr_mean_variance_axis</span></code></a>(X,&nbsp;…)</p></td>
  <td><p>Compute incremental mean and variance along an axix on a CSR or CSC matrix.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_column_scale.html#sklearn.utils.sparsefuncs.inplace_column_scale" title="sklearn.utils.sparsefuncs.inplace_column_scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.inplace_column_scale</span></code></a>(X,&nbsp;scale)</p></td>
  <td><p>Inplace column scaling of a CSC/CSR matrix.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_row_scale.html#sklearn.utils.sparsefuncs.inplace_row_scale" title="sklearn.utils.sparsefuncs.inplace_row_scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.inplace_row_scale</span></code></a>(X,&nbsp;scale)</p></td>
  <td><p>Inplace row scaling of a CSR or CSC matrix.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_row.html#sklearn.utils.sparsefuncs.inplace_swap_row" title="sklearn.utils.sparsefuncs.inplace_swap_row"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.inplace_swap_row</span></code></a>(X,&nbsp;m,&nbsp;n)</p></td>
  <td><p>Swaps two rows of a CSC/CSR matrix in-place.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_column.html#sklearn.utils.sparsefuncs.inplace_swap_column" title="sklearn.utils.sparsefuncs.inplace_swap_column"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.inplace_swap_column</span></code></a>(X,&nbsp;m,&nbsp;n)</p></td>
  <td><p>Swaps two columns of a CSC/CSR matrix in-place.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.mean_variance_axis.html#sklearn.utils.sparsefuncs.mean_variance_axis" title="sklearn.utils.sparsefuncs.mean_variance_axis"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.mean_variance_axis</span></code></a>(X,&nbsp;axis)</p></td>
  <td><p>Compute mean and variance along an axix on a CSR or CSC matrix</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_csr_column_scale.html#sklearn.utils.sparsefuncs.inplace_csr_column_scale" title="sklearn.utils.sparsefuncs.inplace_csr_column_scale"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs.inplace_csr_column_scale</span></code></a>(X,&nbsp;…)</p></td>
  <td><p>Inplace column scaling of a CSR matrix.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1" title="sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs_fast.inplace_csr_row_normalize_l1</span></code></a></p></td>
  <td><p>Inplace row normalize using the l1 norm</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2" title="sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.sparsefuncs_fast.inplace_csr_row_normalize_l2</span></code></a></p></td>
  <td><p>Inplace row normalize using the l2 norm</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.random.sample_without_replacement.html#sklearn.utils.random.sample_without_replacement" title="sklearn.utils.random.sample_without_replacement"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.random.sample_without_replacement</span></code></a></p></td>
  <td><p>Sample integers without replacement.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted" title="sklearn.utils.validation.check_is_fitted"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.validation.check_is_fitted</span></code></a>(estimator)</p></td>
  <td><p>Perform is_fitted validation for estimator.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_memory.html#sklearn.utils.validation.check_memory" title="sklearn.utils.validation.check_memory"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.validation.check_memory</span></code></a>(memory)</p></td>
  <td><p>Check that <codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">memory</span></code> is joblib.Memory-like.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_symmetric.html#sklearn.utils.validation.check_symmetric" title="sklearn.utils.validation.check_symmetric"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.validation.check_symmetric</span></code></a>(array,&nbsp;*[,&nbsp;…])</p></td>
  <td><p>Make sure that array is 2D, square and symmetric.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.column_or_1d.html#sklearn.utils.validation.column_or_1d" title="sklearn.utils.validation.column_or_1d"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.validation.column_or_1d</span></code></a>(y,&nbsp;*[,&nbsp;warn])</p></td>
  <td><p>Ravel column or 1d numpy array, else raises an error</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.has_fit_parameter.html#sklearn.utils.validation.has_fit_parameter" title="sklearn.utils.validation.has_fit_parameter"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.validation.has_fit_parameter</span></code></a>(…)</p></td>
  <td><p>Checks whether the estimator’s fit method supports the given parameter.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.all_estimators.html#sklearn.utils.all_estimators" title="sklearn.utils.all_estimators"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.all_estimators</span></code></a>([type_filter])</p></td>
  <td><p>Get a list of all estimators from sklearn.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel_backend.html#sklearn.utils.parallel_backend" title="sklearn.utils.parallel_backend"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.parallel_backend</span></code></a>(backend[,&nbsp;n_jobs,&nbsp;…])</p></td>
  <td><p>Change the default backend used by Parallel inside a with block.</p></td>
  </tr>
  <tr style="vertical-align: middle;"><td><p><a https://scikit-learn.org/stable/modules/generated/sklearn.utils.register_parallel_backend.html#sklearn.utils.register_parallel_backend" title="sklearn.utils.register_parallel_backend"><codestyle="vertical-align: middle;"><spanstyle="vertical-align: middle;">utils.register_parallel_backend</span></code></a>(name,&nbsp;factory)</p></td>
  <td><p>Register a new Parallel backend factory.</p></td>
  </tr>
</tbody>
</table><br/>



