# Unsupervised Machine Learning

## Unsupervised Learning Notebook

+ [Web notebook Launching Page](https://www.coursera.org/learn/python-machine-learning/notebook/KDjxZ/unsupervised-learning-notebook)
+ [Web Notebook](https://hub.coursera-notebooks.org/user/elkljxyoytcwjbmkgctrtg/notebooks/UnsupervisedLearning.ipynb)
+ [Local notebook](notebooks/UnsupervisedLearning.ipynb)
+ [Local python code](notebooks/UnsupervisedLearning.py)

## Introduction

+ Introduction: Unsupervised Learning
    + Unsupervised learning involves tasks that operate on datasets __without__ labeled responses or target values.
    + Instead, the goal is to capture interesting structure or information.
    + Applications of unsupervised learning:
        + Visualize structure of a complex dataset.
        + Density estimation to predict probabilities of events.
        + Compress and summarize the data.
        + Extract features for supervised learning.
        + Discover important clusters or outliers.

+ Web Clustering Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-01.png" alt="Here's an example of an unsupervised method called clustering. Suppose you're in charge of running a website that allows people to buy products from your company. And the site gets thousands of visits per day. As people access the site by clicking links to products or typing in search terms, their interactions are logged by the web server that creates a large log file. It might be useful for your business to understand who's using your site by grouping users according to their shopping behavior. For example, there might be a group of more expert users who use more advanced features to find something very specific. While another group of non-expert users might just enjoy browsing a broader set of items. By clustering users into groups, you might gain some insight into who your typical customers are and what site features different types of users find important. You could use insights from clustering users to improve the site's features for different groups or to recommend products to specific groups that would be more likely to buy them. " title= "Web Clustering Example" height="150">
    </a>

+ Two major types of unsupervised learning methods
    + Transformations
        + Processes that extract or compute information
    + Clustering
        + Find groups in the data
        + Assign every point in the dataset to one of the groups


+ Transformations: Density Estimation
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-02.png" alt="Okay, let's look at some important unsupervised learning methods that transform the input data in useful ways. One method called density estimation is used when you have a set of measurements scattered throughout an area. And you want to create what you can think of as a smooth version over the whole area that gives a general estimate for how likely it would be to observe a particular measurement in some area of that space. " title= "Transformations: Density Estimation" height="150">
    </a>

+ Density Estimation Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-03.png" alt="For example, in a medical application related to diagnosing diabetes, density estimation with one variable might be used to estimate the distribution of a specific test score. The plasma glucose concentration number from a blood test for people who have a particular form of diabetes. With this density estimate we can estimate the probability that anyone with that medical condition has a particular glucose score. Even if that specific score wasn't seen in the original dataset. " title= "Density Estimation Example: diabetic" height="150">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/fig5-04.png" alt="We could then compare this to the range of glucose levels for people who do not have that condition, which is shown by the red line here. Often, density estimates are then used in further machine learning stages as part of providing features for classification or regression. The more technical way to say this is that density estimation calculates a continuous probability density over the feature space, given a set of discrete samples in that feature space.  " title= "Density Estimation Example: non-diabetic" height="150">
    </a>

+ Kernel Density Example
    <a href="http://www.digital-geography.com/csv-heatmap-leaflet/"> <br/>
        <img src="images/fig5-05.png" alt="With this density estimate, we can estimate how likely any given combination of features is to occur. In Scikit-Learn, you can use the kernel density class in the sklearn.neighbors module to perform one widely used form of density estimation called kernel density estimation. Kernel density's especially popular for use in creating heat maps with geospatial data like this one. " title= "Kernel Density Example" height="150">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/8O0xDUFoEee3MRIl4lCYSA.processed/full/360p/index.mp4?Expires=1538784000&Signature=IouwnCBJvtm7tds2irYlEwq~KkFXqQ8RyAiZiwhB~UPJu9F~Cu75IxYtdIV3QiIwUETdpBVLa8bkEDU6AjkKb~5hVHjDunNzTzB7yBVporyuQTt2uiBLXLrEiO2V658rb1Qm0-HocicG9AXIFXl9EcvYU6ucv-3sevC5uWoTcKw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Introduction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Dimensionality Reduction and Manifold Learning

+ Dimensionality Reduction
    + Finds an approximate version of your dataset using fewer features
    + Used for exploring and visualizing a dataset to understand grouping or relationships
    + Often visualized using a 2-dimensional scatterplot
    + Also used for compression, finding features for supervised learning
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-06.png" alt="Another very important family of unsupervised learning methods that fall into the transformation category are known as dimensionality reduction algorithms. As the name suggests, this kind of transform takes your original dataset that might contain say, 200 features and finds an approximate version of dataset that uses, say, only 10 dimensions. One very common need for dimensionality reduction arises when first exploring a dataset, to understand how the samples may be grouped or related to each other by visualizing it using a two-dimensional scatterplot. " title= "Dimensionality Reduction" height="150">
    </a>
    + The one-dimensional approximation is obtained by projecting the original points onto the diagonal line and using their position on that line as the new single feature.

+ Simple PCA Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-07.png" alt="One very important form of dimensionality reduction is called principal component analysis, or PCA. Intuitively, what PCA does is take your cloud of original data points and finds a rotation of it. So the dimensions are statistically uncorrelated. PCA then typically drops all but the most informative initial dimensions that capture most of the variation in the original dataset. Here's a simple example of what I mean with a synthetic two-dimensional dataset. Here, if we have two original features that are highly correlated represented by this cloud of points, PCA will rotate the data so the direction of highest variance - called the first principal component, which is along the long direction of the cloud, becomes the first dimension. It will then find the direction at right angles that maximally captures the remaining variance. This is the second principle component. In two dimensions, there's only one possible such direction at right angles of the first principal component, but for higher dimensions, there would be infinitely many. With more than two dimensions, the process of finding successive principal components at right angles to the previous ones would continue until the desired number of principal components is reached. One result of applying PCA is that we now know the best one-dimensional approximation to the original two-dimensional data. In other words, we can take any data point that used two features before - x and y - and approximate it using just one feature, namely its location when projected onto the first principal component. " title= "Simple PCA Example" height="150">
    </a>

+ Dimensionality Reduction with PCA in scikit-learn
    ```python
    # #### Using PCA to find the first two principal components of the breast cancer dataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

    # Before applying PCA, each feature should be centered (zero mean) and with unit variance
    X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  

    pca = PCA(n_components = 2).fit(X_normalized)

    X_pca = pca.transform(X_normalized)
    print(X_cancer.shape, X_pca.shape)
    # (569, 30) (569, 2)

    # #### Plotting the PCA-transformed version of the breast cancer dataset
    from adspy_shared_utilities import plot_labelled_scatter
    plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Breast Cancer Dataset PCA (n_components = 2)');
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-01.png" alt="Here's an example of using scikit learn to apply PCA to a higher dimensional dataset; the breast cancer dataset. To perform PCA, we import the PCA class from sklearn.decomposition. It's important to first transform the dataset so that each features range of values has zero mean and unit variance. And we can do this using the fit and transform methods of the standard scalar class, as shown here. We then create the PCA object, specify that we want to retain just the first two principal components to reduce the dimensionality to just two columns and call the fit method using our normalized data. This will set up PCA so that it learns the right rotation of the dataset. We can then apply this properly prepared PCA object to project all the points in our original input dataset to this new two-dimensional space. Notice here since we're not doing supervised learning in evaluating a model against a test set, we don't have to split our dataset into training and test sets. You see that if we take the shape of the array that's returned from PCA, it's transformed our original dataset with 30 features into a new array that has just two columns, essentially expressing each original data point in terms of two new features representing the position of the data point in this new two-dimensional PCA space. We can then create a scatterplot that uses these two new features to see how the data forms clusters. In this example, we've used the dataset that has labels for supervised learning; namely, the malignant and benign labels on cancer cells. So we can see how well PCA serves to find clusters in the data. Here's the result of plotting all the 30 feature data samples using the two new features computed with PCA. We can see that the malignant and benign cells do indeed tend to cluster into two groups in the space. In fact, we could now apply a linear classifier to this two-dimensional representation of the original dataset and we can see that it would likely do fairly well. This illustrates another use of dimensionality reduction methods like PCA to find informative features that could then be used in a later supervised learning stage. " title= "Dimensionality Reduction with PCA in scikit-learn" height="150">
    </a>

+ Visualizing PCA Components
    ```python
    # #### Plotting the magnitude of each feature value for the first two principal components
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(pca.components_, interpolation = 'none', cmap = 'plasma')
    feature_names = list(cancer.feature_names)

    plt.gca().set_xticks(np.arange(-.5, len(feature_names)));
    plt.gca().set_yticks(np.arange(0.5, 2));
    plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12);
    plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12);

    plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0, 
        pca.components_.max()], pad=0.65);      # Fig 2

    # #### PCA on the fruit dataset (for comparison)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # each feature should be centered (zero mean) and with unit variance
    X_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  

    pca = PCA(n_components = 2).fit(X_normalized)
    X_pca = pca.transform(X_normalized)

    from adspy_shared_utilities import plot_labelled_scatter
    plot_labelled_scatter(X_pca, y_fruits, ['apple','mandarin','orange','lemon'])

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Fruits Dataset PCA (n_components = 2)'); # Fig 3
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-02.png" alt="We can create a heat map that visualizes the first two principal components of the breast cancer dataset to get an idea of what feature groupings each component is associated with. Note that we can get the arrays representing the two principal component axes that define the PCA space using the PCA.components_attribute that's filled in after the PCA fit method is used on the data. We can see that the first principle component is all positive, showing a general correlation between all 30 features. In other words, they tend to vary up and down together. The second principle component has a mixture of positive and negative signs; but in particular, we can see a cluster of negatively signed features that co-vary together and in the opposite direction of the remaining features. Looking at the names, it makes sense the subset wold co-vary together. We see the pair mean texture and worst texture and the pair mean radius and worst radius varying together and so on. " title= "Fig.2: Visualizing PCA Components" height="150">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> 
        <img src="images/plt5-03.png" alt="Fig.3" title= "Fig. 3" height="150">
    </a>

+ The "Swiss Roll" Dataset
    <a href="http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering"> <br/>
        <img src="http://scikit-learn.org/stable/_images/sphx_glr_plot_ward_structured_vs_unstructured_0011.png" alt="PCA gives a good initial tool for exploring a dataset, but may not be able to find more subtle groupings that produce better visualizations for more complex datasets. There is a family of unsupervised algorithms called Manifold Learning Algorithms that are very good at finding low dimensional structure in high dimensional data and are very useful for visualizations. One classic example of a low dimensional subset in a high dimensional space is this data set in three dimensions, where the points all lie on a two-dimensional sheet with an interesting shape. This lower dimensional sheet within a higher dimensional space is called the manifold. PCA is not sophisticated enough to find this interesting structure. " title= "The Swiss Roll Dataset" height="150">
    </a>

+ Multidimensional scaling (MDS) attempts to find a distance-preserving low-dimensional projection
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/fig5-09.png" alt="One widely used manifold learning method is called multi-dimensional scaling, or MDS. There are many flavors of MDS, but they all have the same general goal; to visualize a high dimensional dataset and project it onto a lower dimensional space - in most cases, a two-dimensional page - in a way that preserves information about how the points in the original data space are close to each other. In this way, you can find and visualize clustering behavior in your high dimensional data. " title= "Multidimensional scaling (MDS) attempts to find a distance-preserving low-dimensional projection" height="150">
    </a>

+ Notebook: MDS on the Fruit Dataset
    ```python
    # #### Multidimensional scaling (MDS) on the fruit dataset
    from adspy_shared_utilities import plot_labelled_scatter
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import MDS

    # each feature should be centered (zero mean) and with unit variance
    X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  

    mds = MDS(n_components = 2)

    X_fruits_mds = mds.fit_transform(X_fruits_normalized)

    plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
    plt.xlabel('First MDS feature')
    plt.ylabel('Second MDS feature')
    plt.title('Fruit sample dataset MDS');  # Fig.3

    # #### Multidimensional scaling (MDS) on the breast cancer dataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import MDS
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

    # each feature should be centered (zero mean) and with unit variance
    X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  

    mds = MDS(n_components = 2)

    X_mds = mds.fit_transform(X_normalized)

    from adspy_shared_utilities import plot_labelled_scatter
    plot_labelled_scatter(X_mds, y_cancer, ['malignant', 'benign'])

    plt.xlabel('First MDS dimension')
    plt.ylabel('Second MDS dimension')
    plt.title('Breast Cancer Dataset MDS (n_components = 2)');  # Fig.4
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-03.png" alt="Using a technique like MDS and scikit learn is quite similar to using PCA. Like with PCA, each feature should be normalized so its feature values have zero mean and unit variants. After importing the MDS class from sklearn.manifold and transforming the input data, you create the MDS object, specifying the number of components - typically set to two dimensions for visualization. You then fit the object using the transform data, which will learn the mapping and then you can apply the MDS mapping to the transformed data. Here's an example of applying MDS to the fruit dataset. And you can see it does a pretty good job of visualizing the fact that the different fruit types do indeed tend to cluster into groups. " title= "Fig.3: MDS on the Fruit Dataset (n_components = 2)" height="150">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/plt5-04.png" alt="xxx" title= "Fig.4: MDS on the Fruit Dataset" height="150">
    </a>

+ t-SNE: a powerful manifold learning method that finds a 2D projection preserving information about neighbors
    <a href="https://lvdmaaten.github.io/tsne/"> <br/>
        <img src="images/fig5-10.png" alt="An especially powerful manifold learning algorithm for visualizing your data is called t-SNE. t-SNE finds a two-dimensional representation of your data, such that the distances between points in the 2D scatterplot match as closely as possible the distances between the same points in the original high dimensional dataset. In particular, t-SNE gives much more weight to preserving information about distances between points that are neighbors. Here's an example of t-SNE applied to the images in the handwritten digits dataset. You can see that this two-dimensional plot preserves the neighbor relationships between images that are similar in terms of their pixels. For example, the cluster for most of the digit eight samples is closer to the cluster for the digits three and five, in which handwriting can appear more similar than to say the digit one, whose cluster is much farther away. " title= "t-SNE: a powerful manifold learning method that finds a 2D projection preserving information about neighbors" height="150">
    </a>

+ Notebook: t-SNE on the Fruit Dataset
    ```python
    # #### t-SNE on the fruit dataset
    from sklearn.manifold import TSNE

    tsne = TSNE(random_state = 0)

    X_tsne = tsne.fit_transform(X_fruits_normalized)

    plot_labelled_scatter(X_tsne, y_fruits, 
        ['apple', 'mandarin', 'orange', 'lemon'])
    plt.xlabel('First t-SNE feature')
    plt.ylabel('Second t-SNE feature')
    plt.title('Fruits dataset t-SNE');  # Fig.6

    # #### t-SNE on the breast cancer dataset
    # Although not shown in the lecture video, this example is included for comparison, showing the results of running 
    # t-SNE on the breast cancer dataset.  See the reading "How to Use t-SNE effectively" for further details on how 
    # the visualizations from t-SNE are affected by specific parameter settings.
    tsne = TSNE(random_state = 0)

    X_tsne = tsne.fit_transform(X_normalized)

    plot_labelled_scatter(X_tsne, y_cancer, 
        ['malignant', 'benign'])
    plt.xlabel('First t-SNE feature')
    plt.ylabel('Second t-SNE feature')
    plt.title('Breast cancer dataset t-SNE');   # Fig.7
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-06.png" alt="And here's an example of applying t-SNE on the fruit dataset. The code is very similar to applying MDS and essentially just replaces MDS with t-SNE. The interesting thing here is that t-SNE does a poor job of finding structure in this rather small and simple fruit dataset, which reminds us that we should try at least a few different approaches when visualizing data using manifold learning to see which works best for a particular dataset. t-SNE tends to work better on datasets that have more well-defined local structure; in other words, more clearly defined patterns of neighbors. " title= "Fig.6: t-SNE on the Fruit Dataset" height="150">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-06.png" alt="And here's an example of applying t-SNE on the fruit dataset. The code is very similar to applying MDS and essentially just replaces MDS with t-SNE. The interesting thing here is that t-SNE does a poor job of finding structure in this rather small and simple fruit dataset, which reminds us that we should try at least a few different approaches when visualizing data using manifold learning to see which works best for a particular dataset. t-SNE tends to work better on datasets that have more well-defined local structure; in other words, more clearly defined patterns of neighbors. " title= "Fig.7: t-SNE on the Breast Cancer Dataset" height="150">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/_lwQeEFoEeeR4AqenwJvyA.processed/full/360p/index.mp4?Expires=1538784000&Signature=M-yHQE4YNhQkqE2YWjSFXFsr1yxYToboercbWNU-FwtkpQKiAuVgKlP4HzsiXqnZsCVW1TUmlgNvbeq3q1aomtD~8VpIQopqcrxiVp1aG2Jq9kGQnYsuK2yWVuxtwd4CYzOHkhZ3K76JpnRwiOohk6jeUBpsQgIjDuznYPJSu3s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Dimensionality Reduction and Manifold Learning" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Clustering

+ Clustering:
    + Finding a way to divide a dataset into groups ('clusters')
    + Data points within the same cluster should be 'close' or 'similar' in some way.
    + Data points in different clusters should be 'far apart' or 'different'
    + Clustering algorithms output a cluster membership index for each data point:
        + Hard clustering: each data point belongs to exactly one cluster
        + Soft (or fuzzy) clustering: each data point is assigned a weight, score, or probability of membership for each cluster
    <a href="https://brilliant.org/wiki/k-means-clustering/"> <br/>
        <img src="https://ds055uzetaobb.cloudfront.net/image_optimizer/ff1732816ba08239c0d3b200c3a9708070885705.jpg" alt="K-means algorithm iteratively minimizes the distances between every data point and its centroid in order to find the most optimal solution for all the data points. 1) $k$ random points of the data set are chosen to be centroids. 2) Distances between every data point and the $K$ centroids are calculated and stored. 3) Based on distance calculates, each point is assigned to the nearest cluster 4) New cluster centroid positions are updated: similar to finding a mean in the point locations. 5) If the centroid locations changed, the process repeats from step 2, until the calculated new center stays the same, which signals that the clusters members and centroids are now set. -- Aside from transformations, the other family of unsupervised learning methods are the clustering methods. The goal of clustering is to find a way to divide up a data set into groups called clusters. So that groups with similar data instances are assigned to the same cluster, while very dissimilar objects are assigned to different clusters. If new data points were being added over time, some clustering algorithms could also predict which cluster a new data instance should be assigned to. Similar to classification, but without being able to train the clustering model using label examples in advanced. " title= "t-SNE: a powerful manifold learning method that finds a 2D projection preserving information about neighbors." height="250">
    </a>

+ K-means Clustering Algorithm
    + __Initialization__: Pick the number of clusters k you want to find.Then pick k random points to serve as an initialguess for the cluster centers.
    + __Step A__: Assign each data point to the nearest cluster center.
    + __Step B__: Update each cluster center by replacing it withthe mean of all points assigned to that cluster (in step A).
    + __Repeat steps A and B__: until the centers converge to a stable solution.
    + Typically running 10 different random initialization
    <a href="https://www.naftaliharris.com/blog/visualizing-k-means-clustering/"> <br/>
        <img src="images/fig5-10.png" alt="One of the most widely used clustering algorithms is called k-means clustering. K-means clustering finds k cluster centers in different regions of the feature space that it thinks represent very different groups. You need to specify the value of k ahead of time, which is one of the draw backs of k-means. For some problems, we may know the number of classes the data should fall into, but for many other tasks, we might not. K-means operates by first randomly picking locations for the k-cluster centers. Then it goes back and forth between two steps. In the first step, given the locations of existing cluster centers, it assigns each data point to a cluster center based on its distance from the center. In other words, it assigns each data point to the closest center. Then in the second step, it adjusts the locations of each cluster center. It does this by setting the new cluster center to the mean of the positions of all the data points in that cluster. Somewhat magically, after running this alternating process for a while, coherent clusters do start to form. And the cluster centers and the corresponding cluster assignment for each data point eventually settled down to something stable. Now one aspect of k means is that different random starting points for the cluster centers often result in very different clustering solutions. So typically, the k-means algorithm is run in scikit-learn with ten different random initializations. And the solution occurring the most number of times is chosen. " title= "K-means Clustering Algorithm" height="150">
    </a>

+ K-means Example
    + Step 1A: 
        + We want three clusters, so three centers are chosen randomly.
        + Data points are colored according to the closest center.
    + Step 1B:
        + Each center is then updated…
        + … using the mean of all points assigned to that cluster.
    + Step 2A:
        + Data points are colored (again) according to the closest center.
    + Step 2B: 
        + Re-calculate all cluster centers.
    + Converged: 
        + After repeating these steps for several more iterations…
        + The centers converge to a stable solution!
        + These centers define the final clusters.
    <a href="https://www.naftaliharris.com/blog/visualizing-k-means-clustering/"> <br/>
        <img src="images/fig5-11.png" alt="Here is a step by step example. We first choose three locations in the space randomly to be the cluster centers. Then we assign each data point to the cluster with the nearest center. " title= "K-means Example: Step 1A" height="100">
    </a>
    <a href="https://www.naftaliharris.com/blog/visualizing-k-means-clustering/"> 
        <img src="images/fig5-12.png" alt="Now for each cluster, we compute the mean location of all points in the cluster and use that as the new cluster center for the next iteration. " title= "K-means Example: Step 1B" height="100">
    </a>
    <a href="https://www.naftaliharris.com/blog/visualizing-k-means-clustering/"> 
        <img src="images/fig5-13.png" alt="Here's the second iteration of the first and second steps. " title= "K-means Example: Step 2A" height="100">
    </a>
    <a href="https://www.naftaliharris.com/blog/visualizing-k-means-clustering/">
        <img src="images/fig5-14.png" alt="Here's the second iteration of the first and second steps. " title= "K-means Example: Step 2B" height="100">
    </a>
    <a href="https://www.naftaliharris.com/blog/visualizing-k-means-clustering/">
        <img src="images/fig5-15.png" alt="Eventually, after 20 or 50 or 100 steps, things settle down to converge on one solution, as shown here. " title= "K-means Example: Converged" height="100">
    </a>

+ k-means Example in Scikit-Learn
    ```python
    # This example from the lecture video creates an artificial dataset with make_blobs, then 
    # applies k-means to find 3 clusters, and plots the points in each cluster identified by a 
    # corresponding color.
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from adspy_shared_utilities import plot_labelled_scatter

    X, y = make_blobs(random_state = 10)

    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)

    plot_labelled_scatter(X, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'])   # Fig.8
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/plt5-08.png" alt="K-means clustering is simple to apply in scikit learning. You import the k-means class from sklearn cluster create the k-means object set into value of k by specifying the n cluster parameter, and then calling the fit method on the dataset to run the algorithm. One distinction should be made here between clustering algorithms that can predict which center new data points should be assigned to, and those that cannot make such predictions. K-means supports the predict method, and so we can call the fit and predict methods separately. Later methods we'll look at like agglomerative clustering do not and must perform the fit and predict in a single step, as we'll see. " title= "Fig.8: k-means Example in Scikit-Learn" height="150">
    </a>

+ k-means Output on the Fruits Dataset
    ```python
    # Example showing k-means used to find 4 clusters in the fruits dataset.  Note that in 
    # general, it's important to scale the individual features before applying k-means clustering.
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from adspy_shared_utilities import plot_labelled_scatter
    from sklearn.preprocessing import MinMaxScaler

    fruits = pd.read_table('fruit_data_with_colors.txt')
    X_fruits = fruits[['mass','width','height', 'color_score']].as_matrix()
    y_fruits = fruits[['fruit_label']] - 1

    X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)  

    kmeans = KMeans(n_clusters = 4, random_state = 0)
    kmeans.fit(X_fruits_normalized)

    plot_labelled_scatter(X_fruits_normalized, kmeans.labels_, 
        ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])   # Fig.9
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-09.png" alt="Here's the output from the notebook code showing the result supplied to the fruits dataset, where we know the value of k ahead of time. Note that kmeans is very sensitive to the range of future values. So if your data has features with very different ranges, it's important to normalize using min-max scaling, as we did for some supervised learning methods. " title= "Fig.9: k-means Output on the Fruits Dataset" height="150">
    </a>
    + Can you interpret how these clusters correspond with the true fruit labels?

+ Limitations of k-means
    + Works well for simple clusters that are same size, well-separated, globular shapes.
    + Does not do well with irregular, complex clusters.
    + Variants of k-means like k-medoids can work with categorical features.

+ Agglomerative Clustering Example
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"><br/>
        <img src="images/fig5-16.png" alt="Here's a visual example of how agglomerative clustering might proceed on a sample dataset until three clusters are reached. In Stage 1, each data point is in its own cluster, shown by the circles around the points. In Stage 2, the two most similar clusters, which at this stage amounts to defining the closest points are merged. And this process is continued, as denoted by the expanding and closed regions that denote each cluster. " title= "Agglomerative Clustering Example" height="150">
    </a>

+ Linkage Criteria for Agglomerative Clustering
    + Ward's method: Least increase in total variance(around cluster centroids)
    + Average linkage: Average distance between clusters
    + Complete linkage: Max distance between clusters
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"><br/>
        <img src="images/fig5-17.png" alt="You can choose how the agglomerative clustering algorithm determines the most similar cluster by specifying one of several possible linkage criteria. In scikit-learn, the following three linkage criteria are available, ward, average, and complete. Ward's method chooses to merge the two clusters that give the smallest increase in total variance within all clusters. Average linkage merges the two clusters that have the smallest average distance between points. Complete linkage, which is also known as maximum linkage, merges the two clusters that have the smallest maximum distance between their points. In general, Ward's method works well on most data sets, and that's our usual method of choice. In some cases, if you expect the sizes of the clusters to be very different, for example, that one cluster is much larger than the rest. It's worth trying average and complete linkage criteria as well. " title= "Linkage Criteria for Agglomerative Clustering" height="150">
    </a>

+ Agglomerative Clustering in Scikit-Learn
    ```python
    # ### Agglomerative clustering
    from sklearn.datasets import make_blobs
    from sklearn.cluster import AgglomerativeClustering
    from adspy_shared_utilities import plot_labelled_scatter

    X, y = make_blobs(random_state = 10)

    cls = AgglomerativeClustering(n_clusters = 3)
    cls_assignment = cls.fit_predict(X)

    plot_labelled_scatter(X, cls_assignment, 
            ['Cluster 1', 'Cluster 2', 'Cluster 3'])    # Fig.10
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"><br/>
        <img src="images/plt5-10.png" alt="To perform agglomerative clustering in scikit-learn, you import the agglomerative clustering class from sklearn cluster. When initializing the object, you specify the n clusters parameter that causes the algorithm to stop when it has reach that number of clusters. You call the fit predict method using the data set as input and they return the set of cluster assignments for the data points as shown here. " title= "Fig.10: Agglomerative Clustering in Scikit-Learn" height="150">
    </a>

+ Hierarchical Clustering
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"><br/>
        <img src="images/fig5-18.png" alt="One of the nice things about agglomerative clustering is that it automatically arranges the data into a hierarchy as an effect of the algorithm, reflecting the order and cluster distance at which each data point is assigned to successive clusters. This hierarchy can be useful to visualize using what's called a dendrogram, which can be used even with higher dimensional data. Here's the dendogram corresponding to the Ward's method clustering of the previous data set example. The data points are at the bottom and are numbered. The y axis represents cluster distance, namely, the distance that two clusters are apart in the data space. The data points form the leaves of the tree at the bottom, and the new node parent in the tree is added as each pair of successive clusters is merged. The height of the node parent along the y axis captures how far apart the two clusters were when they merged, with the branch going up representing the new merged cluster. Note that you can tell how far apart the merged clusters are by the length of each branch of the tree. This property of a dendogram can help us figure out the right number of clusters. In general, we want clusters that have highly similar items within each cluster, but that are far apart from other clusters. For example, we can see that going from three clusters to two happens at a fairly high Y value. Which means the clusters that were merged were a significant distance apart. We might want to avoid choosing two clusters and stick with three clusters that don't involve forcing a merge for clusters that have very dissimilar items in them. " title= "Linkage Criteria for Agglomerative Clustering" height="150">
    </a>

+ Dendrogram Example
    ```python
    # This dendrogram plot is based on the dataset created in the previous step 
    # with make_blobs, but for clarity, only 10 samples have been selected for 
    # this example, as plotted here:
    X, y = make_blobs(random_state = 10, n_samples = 10)
    plot_labelled_scatter(X, y, 
            ['Cluster 1', 'Cluster 2', 'Cluster 3'])    # Fig.11
    print(X)

    # And here's the dendrogram corresponding to agglomerative clustering of the 
    # 10 points above using Ward's method.  The index 0..9 of the points 
    # corresponds to the index of the points in the X array above.  For example, 
    # point 0 (5.69, -9.47) and point 9 (5.43, -9.76) are the closest two points 
    # and are clustered first.
    from scipy.cluster.hierarchy import ward, dendrogram
    plt.figure()
    dendrogram(ward(X))
    plt.show()      # Fig.12
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-11.png" alt="Scikit-learn doesn't provide the ability to plot dendrograms, but SciPy does. SciPy handles clustering a little differently than scikit-learn, but here is an example. We first import the dendrogram in word functions from the scipy.cluster hierarchy module. The word function returns an array that specifies the distances spanned during the agglomerative clustering. This word function returns a linkage array, which can then be passed to the dendogram function to plot the tree. " title= "Fig.11: Dendrogram Example with make_blobs" height="150">
    </a>
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/plt5-12.png" alt="Scikit-learn doesn't provide the ability to plot dendrograms, but SciPy does. SciPy handles clustering a little differently than scikit-learn, but here is an example. We first import the dendrogram in word functions from the scipy.cluster hierarchy module. The word function returns an array that specifies the distances spanned during the agglomerative clustering. This word function returns a linkage array, which can then be passed to the dendogram function to plot the tree. " title= "Fig.12: Dendrogram Example using Ward's method" height="150">
    </a>

+ DBSCAN Clustering
    + Unlike k-means, you don't need to specify # of clusters
    + Relatively efficient –can be used with large datasets
    + Identifies likely noise points
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/fig5-19.png" alt="Typically, making use of this hierarchy is most useful when the underlying data itself follows some kind of hierarchical process so the tree is easily interpreted. For example, hierarchical clustering is especially useful for genetic and other biological data where the levels represent stages of mutation or evolution. But there are other data sets where both k-means clustering and agglomerative clustering don't perform well. So we're now going to give an overview of a third clustering method called DBSCAN. DBSCAN is an acronym that stands for density-based spatial clustering of applications with noise. One advantage of DBSCAN is that you don't need to specify the number of clusters in advance. Another advantage is that it works well with datasets that have more complex cluster shapes. It can also find points that are outliers that shouldn't reasonably be assigned to any cluster. DBSCAN is relatively efficient and can be used for large datasets. The main idea behind DBSCAN is that clusters represent areas in the data space that are more dense with data points, while being separated by regions that are empty or at least much less densely populated. The two main parameters for DBSCAN are min samples and eps. All points that lie in a more dense region are called core samples. For a given data point, if there are min sample of other data points that lie within a distance of eps, that given data points is labeled as a core sample. Then, all core samples that are with a distance of eps units apart are put into the same cluster. In addition to points being categorized as core samples, points that don't end up belonging to any cluster are considered as noise. While points that are within a distance of eps units from core points, but not core points themselves, are termed boundary points. " title= "Dendrogram Example" height="150">
    </a>

+ DBSCAN Example in Scikit-Learn
    + DBSCAN: density-based spatial clustering of applications with noise
    ```python
    # ### DBSCAN clustering
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs

    X, y = make_blobs(random_state = 9, n_samples = 25)

    dbscan = DBSCAN(eps = 2, min_samples = 2)

    cls = dbscan.fit_predict(X)
    print("Cluster membership values:\n{}".format(cls))

    plot_labelled_scatter(X, cls + 1, 
            ['Noise', 'Cluster 0', 'Cluster 1', 'Cluster 2']) # Fig.13
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/plt5-13.png" alt="Here's an example of DBSCAN applied to a sample data set. As with the other clustering methods, DBSCAN is imported from the Scikit-Learn cluster module. And just like with a agglomerative clustering, DBSCAN doesn't make cluster assignments from new data. So we use the fit predict method to cluster and get the cluster assignments back in one step. One consequence of not having the right settings of eps and min samples for your particular dataset might be that the cluster memberships returned by DBSCAN may all be assigned the label -1, which indicates noise. Bsically, the EPS setting does implicitly control the number of clusters that are found. With DBSCAN, if you've scaled your data using a standard scalar or min-max scalar to make sure the feature values have comparable ranges, finding an appropriate value for eps is a bit easer to do. One final note, make sure that when you use the cluster assignments from DBSCAN, you check for and handle the -1 noise value appropriately. Since this negative value might cause problems, for example, if the cluster assignment is used as an index into another array later on. " title= "Fig.13: Dendrogram Example" height="150">
    </a>

+ Clustering Evaluation
    + With ground truth, existing labels can be used to evaluate cluster quality.
    + Without ground truth, evaluation can difficult: multiple clusteringsmay be plausible for a dataset.
    + Consider task-based evaluation: Evaluate clustering according to performance on a task that doeshave an objective basis for comparison.
    + Example: the effectiveness of clustering-based features for a supervised learning task.
    + Some evaluation heuristics exist (e.g. silhouette) but these can be unreliable.
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction">
        <img src="images/fig5-20.png" alt="Unlike supervised learning, where we have existing labels or target values to use for evaluating the effectiveness of the learning method, it can be difficult to evaluate unsupervised learning algorithms automatically. Since there's typically no ground truth to compare against. In some cases, as in the breast cancer example, we may have existing labels that can be used to evaluate the quality of the clusters by comparing the assignment of a data point to a cluster with the label assigned to the same data point. But there are many cases where labels are not available. In addition, in the case of clustering, for example, there's ambiguity, in a sense that there are typically multiple clusterings that could be plausibly assigned to a given data set. And none of them is obviously better than another unless we have some additional criteria. Such as, performance on the specific application task that does have an objective evaluation to use as a basis for comparison. " title= "Clustering Evaluation" height="150">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/EKwFq0FpEeeR4AqenwJvyA.processed/full/360p/index.mp4?Expires=1538784000&Signature=GfcwoMKAblhSPQyH~uPsYtUOBD4hlzaLtp6YpIezDo2cFBR~ejHta8TpLNVP0MRhI9Lnbq4mHjCowmsZHwHdsH~F5XSMi8-23PmWLhMZ5ttIgRBmjsBdNEqoiUOsKtadriFByfGMaApZkyig5~14RU1gu3C04hNFowLPd9wUUPo_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Clustering" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## How to Use t-SNE Effectively

+ Wattenberg, et al., "[How to Use t-SNE Effectively](http://doi.org/10.23915/distill.00002)", Distill, 2016. 
+ http://distill.pub/2016/misread-tsne/#citation

## How Machines Make Sense of Big Data: an Introduction to Clustering Algorithms

Gleesen, Peter. "[How Machines Make Sense of Big Data: an Introduction to Clustering Algorithms](https://medium.freecodecamp.com/how-machines-make-sense-of-big-data-an-introduction-to-clustering-algorithms-4bd97d4fbaba)", freeCodeCamp, 2017.

### Introduction

+ Us humans take it for granted how good we are categorizing and making sense of large volumes of data pretty quickly.

+ Humans are generally fairly efficient at making sense of whatever data.

+ Three clustering algorithms that machines can use to quickly make sense of large datasets.

### K-means clustering

+ When: an idea of how many groups you’re expecting to find a priori.

+ How:
    1. randomly assign each observation into one of k categories, then calculates the mean of each category
    2. reassign each observation to the category with the closest mean before recalculating the means
    3. repeats over and over until no more reassignments are necessary

+ Example:
    + Take a group of 12 football (or ‘soccer’) players who have each scored a certain number of goals this season (say in the range 3–30). Let’s divide them into separate clusters — say three.
    + Procedure
        1. randomly split the players into three groups and calculate the means of each
            + Group 1: Player A (5 goals), Player B (20 goals), Player C (11 goals) -> Group Mean = (5 + 20 + 11) / 3 = 12
            + Group 2: Player D (5 goals), Player E (3 goals), Player F (19 goals) -> Group Mean = 9
            + Group 3: Player G (30 goals), Player H (3 goals), Player I (15 goals) -> Group Mean = 16
        2. For each player, reassign them to the group with the closest mean. Then recalculate the group means.
            + Group 1 (Old Mean = 12): Player C (11 goals) -> New Mean = 11
            + Group 2 (Old Mean = 9): Player A (5 goals), Player D (5 goals), Player E (3 goals), Player H (3 goals) -> New Mean = 4
            + Group 3 (Old Mean = 16): Player G (30 goals), Player I (15 goals), Player B (20 goals), Player F (19 goals) -> New Mean = 21
        3. Repeat Step 2 over and over until the group means no longer change
            + Group 1 (Old Mean = 11): Player C (11 goals), Player I (15 goals) -> Final Mean = 13
            + Group 2 (Old Mean = 4): Player A (5 goals), Player D (5 goals), Player E (3 goals), Player H (3 goals) -> Final Mean = 4
            + Group 3 (Old Mean = 21): Player G (30 goals), Player B (20 goals), Player F (19 goals) -> Final Mean = 23
    + The clusters could correspond to the players’ positions on the field — such as defenders, midfielders and attackers.
    + Given data on a range of performance statistics, a machine could do a reasonable job of estimating the positions of players from any team sport — useful for sports analytics, and indeed any other purpose where classification of a dataset into predefined groups can provide relevant insights.

+ Finer details:
    + The initial method of ‘seeding’ the clusters can be done in one of several ways.
    + Alternative: 
        + Seed the clusters with just one player each, then start assigning players to the nearest cluster.
        + The returned clusters are more sensitive to the initial seeding step, reducing repeatability in highly variable datasets.
    + limitation:
        + Provide a priori assumptions about how many clusters you’re expecting to find. 
        + Methods to assess the fit of a particular set of clusters. E.g., the Within-Cluster Sum-of-Squares, a measure of the variance within each cluster. 
        + The ‘better’ the clusters, the lower the overall WCSS.

### Hierarchical clustering

+ When: uncover the underlying relationships between your observations

+ How:
    1. A distance matrix is computed, where the value of cell $(i, j)$ is a distance metric between observations $i$ and $j$.
    2. Pair the closest two observations and calculate their average. Form a new distance matrix, merging the paired observations into a single object. From this distance matrix, pair up the closest two observations and calculate their average.
    3. Repeat until all observations are grouped together.

+ Example:
    + Selection of whale and dolphin species.
    + The typical body lengths for these six species
    + Dataset
        | Species            | Initials | Length(m) |
        |:-------------------|---------:|----------:|
        | Bottlenose Dolphin |     BD |       3.0 |
        | Risso's Dolphin    |     RD |       3.6 |
        | Pilot Whale        |     PW |       6.5 |
        | Killer Whale       |     KW |       7.5 |
        | Humpback Whale     |     HW |      15.0 |
        | Fin Whale          |     FW |      20.0 |
    + Procedure:
        1. compute a distance matrix between each species. <br/>
            The difference in length between any pair of species can be looked up by reading the value at the intersection of the relevant row and column.
            |    | BD   | RD   | PW   | KW   | HW |
            |:---|-----:|-----:|-----:|-----:|---:|
            | RD |  0.6 |      |      |      |      |
            | PW |  3.5 |  2.9 |      |      |      |
            | KW |  4.5 |  3.9 |  1.0 |      |      |
            | HW | 12.0 | 11.4 |  8.5 |  7.5 |      |
            | FW | 17.0 | 16.4 | 13.5 | 12.5 |  5.0 |
        2. Pair up the two closest species. Here the Bottlenose & Risso’s Dolphins, with an average length of 3.3m.
        3. Repeat Step 1 by recalculating the distance matrix, but this time merge the Bottlenose & Risso’s Dolphins into a single object with length 3.3m.
            |    | [BD, RD]  | PW  | KW  | HW |
            |:---|----------:|----:|----:|---:|
            | PW |      3.2  |     |     |    |
            | KW |      4.2  | 1.0 |     |     |
            | HW |     11.7  | 8.5 | 7.5 |     |
            | FW |     16.7  |13.5 |12.5 | 5.0 |
        4. Repeat Step 2 with this new distance matrix. Here, the smallest distance is between the Pilot & Killer Whales, so we pair them up and take their average — which gives us 7.0m.
        5. Repeat Step 1 — recalculate the distance matrix, but now we’ve merged the Pilot & Killer Whales into a single object of length 7.0m.
            |          | [BD, RD] | [PW, KW] | HW  |
            |:---------|---------:|---------:|----:|
            | [PW, KW] |      3.7 |          |     |
            | HW       |     11.7 |      8.0 |     |
            | FW       |     16.7 |     13.0 | 5.0 |
        6. Repeat Step 2 with this distance matrix. The smallest distance (3.7m) is between the two merged objects — so now we merge them into an even bigger object, and take the average (which is 5.2m).
        7. Repeat Step 1 and compute a new distance matrix, having merged the Bottlenose & Risso’s Dolphins with the Pilot & Killer Whales.
            |    | [[BD, RD] , [PW, KW]]  |  HW |
            |:---|-----------------------:|----:|
            | HW |                   9.8  |     |
            | FW |                  14.8  | 5.0 |
        8. repeat Step 2. The smallest distance (5.0m) is between the Humpback & Fin Whales, so we merge them into a single object, and take the average (17.5m).
        9. Repeat Step 1 — compute the distance matrix, having merged the Humpback & Fin Whales.
            |          | [[BD, RD] , [PW, KW]] |
            |:---------|----------------------:|
            | [HW, FW] |                  12.3 |
        10. repeat Step 2 — there is only one distance (12.3m) in this matrix, so we pair everything into one big object. <br/> 
            [[[BD, RD],[PW, KW]],[HW, FW]]
    <a href="https://medium.freecodecamp.org/how-machines-make-sense-of-big-data-an-introduction-to-clustering-algorithms-4bd97d4fbaba"> <br/>
        <img src="https://cdn-images-1.medium.com/max/1000/1*jwd6EHmjOtkH9RiQSJ_ZnQ.png" alt="It has a nested structure (think JSON), which allows it to be drawn up as a tree-like graph, or dendrogram. It reads in much the same way a family tree might. The nearer two observations are on the tree, the more similar or closely-related they are taken to be. The structure of the dendrogram gives us insight into how our dataset is structured. In our example, we see two main branches, with Humpback Whale and Fin Whale on one side, and the Bottlenose Dolphin/Risso’s Dolphin and Pilot Whale/Killer Whale on the other." title="A no-frills dendrogram generated at R-Fiddle.org" height="250">
    </a>
    + Hierarchical clustering has applications in Data Mining and Machine Learning contexts.
    + This approach requires no assumptions about the number of clusters you’re looking for.
    + If we draw a horizontal line at height = 10, we’d intersect the two main branches, splitting the dendrogram into two sub-graphs. 
    + If we cut at height = 2, we’d be splitting the dendrogram into three clusters.

+ Finer details:
    + three aspects in which hierarchical clustering algorithms
        + Most fundamental used an agglomerative process: start with one giant cluster, and then proceed to divide the data into smaller and smaller clusters until you’re left with isolated data points.
        + Methods to calculate the distance matrices
        + __Linkage criterion__: Clusters are linked according to how close they are to one another, but the way in which we define ‘close’ is flexible.
    + Define the distance between two clusters to be the minimum (or maximum) distance between any of their points.
    <a href="https://medium.freecodecamp.org/how-machines-make-sense-of-big-data-an-introduction-to-clustering-algorithms-4bd97d4fbaba"> <br/>
        <img src="https://cdn-images-1.medium.com/max/1000/1*4aWCKqBrD8BbEeiNzc2gwg.png" alt="Red/Blue: centroid linkage; Red/Green: minimum linkage; Green/Blue: maximum linkage. For example, each cluster is made up of several discrete points. We could define the distance between two clusters to be the minimum (or maximum) distance between any of their points — as illustrated in the figure below. There are still other ways of defining the linkage criterion, which may be suitable in different contexts." title="caption" height="200">
    </a>


### Graph Community Detection

+ When: data that can be represented as a network, or ‘graph’.

+ How:
    + A graph community is very generally defined as a subset of vertices which are more connected to each other than with the rest of the network.
    + Algorithms: Edge Betweenness, Modularity-Maximsation, Walktrap, Clique Percolation, Leading Eigenvector ...

+ Example
    + Graph theory: a fascinating branch of mathematics that lets us model complex systems as an abstract collection of ‘dots’ (or vertices) connected by ‘lines’ (or edges)
    + Social networks: the vertices represent people, and edges connect vertices who are friends/followers
    + Any system can be modelled as a network if you can justify a method to meaningfully connect different components.
    + Eight websites I most recently visited, linked according to whether their respective Wikipedia articles link out to one another.
    <a href="https://medium.freecodecamp.org/how-machines-make-sense-of-big-data-an-introduction-to-clustering-algorithms-4bd97d4fbaba"> <br/>
        <img src="https://cdn-images-1.medium.com/max/1000/1*qd41Vp7sw98vq4PyxVE_DQ.png" alt="The vertices are colored according to their community membership, and sized according to their centrality. See how Google and Twitter are the most central?" title="Graph plotted with ‘igraph’ package" height="200">
    </a>
    + The yellow vertices are generally reference/look-up sites; the blue vertices are all used for online publishing (of articles, tweets, or code); and the red vertices include YouTube, which was of course founded by former PayPal employees.
    + The real power of networks comes from their mathematical analysis.
    + _Adjacency matrix_ of the network
        |          | GH | Gl | M | P | Q | T | W | Y |
        |----------|----|----|---|---|---|---|---|---|
        | GitHub   |  0 |  1 | 0 | 0 | 0 | 1 | 0 | 0 |
        | Google   |  1 |  0 | 1 | 1 | 1 | 1 | 1 | 1 |
        | Medium   |  0 |  1 | 0 | 0 | 0 | 1 | 0 | 0 |
        | PayPal   |  0 |  1 | 0 | 0 | 0 | 1 | 0 | 1 |
        | Quora    |  0 |  1 | 0 | 0 | 0 | 1 | 1 | 0 |
        | Twitter  |  1 |  1 | 1 | 1 | 1 | 0 | 0 | 1 |
        | Wikipedia|  0 |  1 | 0 | 0 | 1 | 0 | 0 | 0 |
        | YouTube  |  0 |  1 | 0 | 1 | 0 | 1 | 0 | 0 |
    + The value at the intersection of each row and column records whether there is an edge between that pair of vertices.
    + Encoded within the adjacency matrix are all the properties of this network — it gives us the key to start unlocking all manner of valuable insights.
    + For a start, summing any column (or row) gives you the degree of each vertex — i.e., how many others it is connected to, commonly denoted with $k$.
    + Modularity (measure of the structure of networks or graphs) of any given clustering of the network:
        + Summing the degrees of every vertex and dividing by two gives you L, the number of edges (or ‘links’) in the network. The number of rows/columns gives us N, the number of vertices (or ‘nodes’) in the network.
        + Knowing just $k, L, N$ and the value of each cell in the adjacency matrix $A$
        + Use the modularity score to assess the ‘quality’ of this clustering.
        + A higher score will show we’ve split the network into ‘accurate’ communities, whereas a low score suggests our clusters are more random than insightful.
    <a href="https://medium.freecodecamp.org/how-machines-make-sense-of-big-data-an-introduction-to-clustering-algorithms-4bd97d4fbaba"> <br/>
        <img src="https://cdn-images-1.medium.com/max/1000/1*6_kSe1q4nDbvnnghF4yJwA.png" alt="Say we’ve clustered the network into a number of communities. We can use the modularity score to assess the ‘quality’ of this clustering. A higher score will show we’ve split the network into ‘accurate’ communities, whereas a low score suggests our clusters are more random than insightful. The image below illustrates this." title="Modularity serves as a measure of the ‘quality’ of a partition" height="150">
    </a>
    + Formula:

        $$M = \frac{1}{2L} \sum_{i,j=1}^N (A_{ij} - \frac{k_i k_j}{2L}) \delta c_i c_j$$
        + $M$: modularity
        + $1/2L$: divide everything that follows by 2L, i.e., twice the number of edges in the network
        + summing up everything to the right and iterate over every row and column in the adjacency matrix $A$, the $i, j = 1$ and the $N$ work much like nested for-loops in programming
    + Coding
        ```python
        sum = 0
        for i in range(1,N):
            for j in range(1,N):
                ans = #stuff with i and j as indices
                sum += ans
        ```
    + `#stuff with i and j` details
        + The bit in brackets tells us to subtract $( k_i k_j ) / 2L$ from $A_{ij}$
        + $A_{ij}$: the value in the adjacency matrix at row $i$, column $j$
        + $k_i$ and $k_j$: the degrees of each vertex — found by adding up the entries in row $i$ and column $j$ respectively
        + The difference between the network’s real structure and the expected structure it would have if randomly reassembled.
    + $\delta c_i, c_j$ is the fancy-sounding but totally harmless Kronecker-delta function
        ```python
        def Kronecker_Delta(ci, cj):
            if ci == cj:
                return 1
            else:
                return 0
        Kronecker_Delta("A","A")    #returns 1
        Kronecker_Delta("A","B")    #returns 0
        ```
    + The Kronecker-delta function takes two arguments, and returns $1$ if they are identical, otherwise, zero.
    + If vertices $i$ and $j$ have been put in the same cluster, then $\delta c_i, c_j = 1$. Otherwise, if they are in different clusters, the function returns zero.
    + For the nested $\sum$, the outcome is highest when there are lots of ‘unexpected’ edges connecting vertices assigned to the same cluster.
    + Dividing by $2L$ bounds the upper value of modularity at $1$.
    + Modularity scores near to or below zero indicate the current clustering of the network is really no use.
    + The higher the modularity, the better the clustering of the network into separate communities.
    + By maximizing modularity, find the best way of clustering the network.
    + brute force: computationally impossible beyond a very limited sample size
    + Eight vertices: 4140 different ways of clustering
    + Heuristic method:
        + Fast-Greedy Modularity-Maximization
        + Analogous to the agglomerative hierarchical clustering algorithm
    + ‘Mod-Max’ merges communities according to changes in modularity
        1. initially assigning every vertex to its own community, and calculating the modularity of the whole network, $M$.
        2. for each community pair linked by at least a single edge, the algorithm calculates the resultant change in modularity $\Delta M$ if the two communities were merged into one.
        3. Repeat steps 1 and 2 — each time merging the pair of communities for which doing so produces the biggest gain in $\Delta M$, then recording the new clustering pattern and its associated modularity score $M$.
        4. Stop when all the vertices are grouped into one giant cluster. Identify the clustering pattern that returned the highest value of $M$.

+ Finer details
    + Graph theory: NP-hard problems
    + Community detection is a major focus of current research in graph theory, and there are plenty of alternatives to Modularity-Maximization
    + resolution limit: not find communities below a certain size
    + Mod-Max approach: produce a wide ‘plateau’ of many similar high modularity scores
    + Edge-Betweenness:
        + a divisive algorithm, starting with all vertices grouped in one giant cluster
        + iteratively remove the least ‘important’ edges in the network, until all vertices are left isolated.
    + Clique Percolation: take into account possible overlap between graph communities
    + random-walks across the graph, and then spectral clustering methods which start delving into the eigendecomposition of the adjacency matrix and other matrices derived therefrom.


### Conclusion

+ Machine Learning: an extraordinarily ambitious field of research, in which massively complex problems require solving in as accurate and as efficient a way possible





