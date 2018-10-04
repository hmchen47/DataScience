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
    ```
    <a href="https://www.coursera.org/learn/python-machine-learning/lecture/XIt7x/introduction"> <br/>
        <img src="images/plt5-06.png" alt="And here's an example of applying t-SNE on the fruit dataset. The code is very similar to applying MDS and essentially just replaces MDS with t-SNE. The interesting thing here is that t-SNE does a poor job of finding structure in this rather small and simple fruit dataset, which reminds us that we should try at least a few different approaches when visualizing data using manifold learning to see which works best for a particular dataset. t-SNE tends to work better on datasets that have more well-defined local structure; in other words, more clearly defined patterns of neighbors. " title= "Fi.6: t-SNE on the Fruit Dataset" height="150">
    </a>


### Lecture Video

<a href="https://d3c33hcgiwev3.cloudfront.net/_lwQeEFoEeeR4AqenwJvyA.processed/full/360p/index.mp4?Expires=1538784000&Signature=M-yHQE4YNhQkqE2YWjSFXFsr1yxYToboercbWNU-FwtkpQKiAuVgKlP4HzsiXqnZsCVW1TUmlgNvbeq3q1aomtD~8VpIQopqcrxiVp1aG2Jq9kGQnYsuK2yWVuxtwd4CYzOHkhZ3K76JpnRwiOohk6jeUBpsQgIjDuznYPJSu3s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" alt="Dimensionality Reduction and Manifold Learning" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Clustering

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="Clustering" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## How to Use t-SNE Effectively

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="How to Use t-SNE Effectively" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## How Machines Make Sense of Big Data: an Introduction to Clustering Algorithms

+ Demo
    ```python

    ```

### Lecture Video

<a href="url" alt="How Machines Make Sense of Big Data: an Introduction to Clustering Algorithms" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

