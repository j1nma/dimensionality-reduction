## Dimensionality Reduction
This work is experiments with three dimensionality reduction techniques in Python.

### Installation
$ pip3 install -r requirements.txt

### Running

Custom hyperparameters in a textfile i.e. _"./configs/config.txt"_.
$ python3 experiments.py ./configs/config.txt

A _results_ folder will contain a timestamp directory with the latest results.

This work is a experiment with a number of algorithms on several datasets.
The aim is to get a feeling of how well each of these algorithms works, 
and whether there are differences depending on the dataset.

### Datasets
* Iris (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) 
* Breast Cancer (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

### Techniques
* PCA
* t-SNE
* Multi Dimensional Scaling (MDS)

### Report
Dimensionality-Reduction-Alonso.pdf
