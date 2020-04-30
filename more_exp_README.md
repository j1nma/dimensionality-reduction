## Dimensionality Reduction

### Installation
```sh
$  pip3 install -r requirements.txt
```

---

### Running

Custom hyperparameters in a textfile i.e. _"./configs/config.txt"_.

```sh
$  python3 experiments.py ./configs/config.txt
```

A _results_ folder will contain a timestamp directory with the latest results.

---

### Datasets
* Iris (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) 

The first rows of this dataset's dataframe are shown below:
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/df_head.png)

This dataset is often used as a benchmark for multiple machine learning fields.
It is composed of 4 features of flower petals (dimensionality = 4). There are three possible classes of petals,
with 50 samples per class.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/Iris.svg)

From the image above one can see at a glance a histogram of each feature with respect to the three available classes.
Some features distinguish the labels more than others, for instance, the setosa type is more easily identified when
conditioning on petal width.

* Breast Cancer (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

The first rows of this dataset's dataframe are shown below:
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/df_head.png)

This dataset represents characteristics of cell nuclei from images of breast mass.
It is composed of 569 samples of 30 features for each cell nuclei (dimensionality = 30). There are 2 possible classes for a breast mass,
'malignant' or 'benign'. Some characteristics are: texture, concavity, symmetry.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/BreastCancerWisconsin.svg)

As in the previous database, from the image above one can see a histogram of each feature with respect to the two
possible outcomes.
In the same manner, the difference in overlaps along the features can suggest a certain covariance between the classes.
These last histograms were generated based on algorithms from https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21.

---

### Techniques
* PCA
* t-SNE
* Multi Dimensional Scaling (MDS)

---

### Results
#### Description of the results from the visualisations
**Iris Dataset**

_PCA_
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/PCA/20200428_005929/PCA_c=2.svg)
The cumulative explained variation for 2 principal components for this 2D visualization was 0.958. 
The versicolour and virginica classes are not as identifiable as the setosa variation.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/PCA/20200428_010206/PCA_c=3.svg)
The cumulative explained variation for 3 principal components for this 3D visualization was 0.995, a 4% improvement.
In other words, the loss of information is 0.5%.
In this case, the classes appear to be more separated, but the 3D visualization is not as clear as its 2D version.


_t-SNE_
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/tSNE/20200428_010226/tSNE_c=2.svg)
The time taken to calculate this t-SNE reduction was 0.678 seconds. Similar to the last 2D visualization, the setosa
class is more identifiable. Also, there are some outliers from the remaining classes in each others areas.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/tSNE/20200428_010232/tSNE_c=3.svg)
The time taken to calculate this t-SNE reduction with an extra dimension 1.080 seconds, a 60% increase.
As in the previous case, the visualization may result more confusing than the one with 1 dimension less.


_Multi Dimensional Scaling (MDS)_
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/MDS/20200428_005743/MDS_c=2.svg)
This visualization resembles its PCA variation, though it seems to expand more area, compared to 
the t-SNE visualization which clearly places setosa far apart form the others.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/iris/MDS/20200428_005807/MDS_c=3.svg)
This visualization with an extra dimension seems to be a better representation of the
data, with the datapoints arranged as in 'layers' in 3D space.



**Breast Cancer Dataset**

_PCA_
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/PCA/20200428_012935/PCA_c=2.svg)
The cumulative explained variation for 2 principal components for this 2D visualization was 0.632. 
The malignant and benign data points are dispersed around distinct areas,
with a certain section that borders both.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/PCA/20200428_012952/PCA_c=3.svg)
The cumulative explained variation for 3 principal components for this 3D visualization was 0.726, a 15% improvement.
In other words, the loss of information is about 30%.
In this case, outliers seem to be more clear, but the perspective does not make
the visualization more clear than its 2D version.


_t-SNE_
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/tSNE/20200428_013012/tSNE_c=2.svg)
The time taken to calculate this t-SNE reduction was 2.711 seconds. This 2D visualization places the classes
further apart in general terms, though some malignant samples are around
benign points.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/tSNE/20200428_013020/tSNE_c=3.svg)
This 3D variation displays clusters for each label, and took 5.932 seconds to calculate, more than double the effort.
Though not clear from the current perspective, the extra component seems to separate
the classes on the z-index.


_Multi Dimensional Scaling (MDS)_
![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/MDS/20200428_013035/MDS_c=2.svg)
This MDS visualization splits the data in two general areas, with data points
spreading from each cluster, which seem to be the best representations of the characteristics in the histograms
from above.

![](/Users/JuanmaAlonso/FHTKWien/Machine-Learning/dimensionality-reduction/results/breast-cancer-wisconsin/MDS/20200428_013042/MDS_c=3.svg)
This 3D visualization adds another component to the previous visualization,
still portraying a cluster-like separation, but the sparsity of most datapoints
does not provide a clearer picture than the 2D alternative.