import argparse
import sys
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def log(logfile, s):
    """ Log a string into a file and print it. """
    with open(logfile, 'a', encoding='utf8') as f:
        f.write(str(s))
        f.write("\n")
    print(s)


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-d",
        "--dataset",
        default="Iris",
        help="Name of the dataset to use: Iris, BreastCancer."
    )
    parser.add_argument(
        "-t",
        "--technique",
        default="PCA",
        help="Name of the dimensionality reduction technique: PCA, tSNE, MDS."
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=1910299034,
        help="Random seed."
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default='results/'
    )

    return parser


def experiments(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + timestamp + '/'

    # Create results directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Create results directory
    os.mkdir(outdir)

    # Logging
    logfile = outdir + 'log.txt'
    # f = open(logfile, 'w')
    # f.close()

    log(logfile, "Directory " + outdir + " created.")

    # Set dataset
    if args.dataset == 'Iris':
        dataset = load_iris()
        dataset_name = "Iris"
        original_labels = ['setosa', 'versicolour', 'virginica']
    elif args.dataset == 'BreastCancer':
        dataset = load_breast_cancer()
        dataset_name = "Breast Cancer Wisconsin"
        original_labels = ['benign', 'malignant']
    else:
        raise ("Dataset not found")

    data = dataset.data
    labels = dataset.target

    # Add labels and feature names
    df = pd.DataFrame(data, columns=dataset.feature_names)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))
    for i in range(0, len(original_labels)):
        df['label'].replace(str(i), original_labels[i], inplace=True)

    # Dataset analysis
    log(logfile, 'Size of the data: {} and labels: {}'.format(data.shape, labels.shape))
    log(logfile, 'Size of the reshaped dataframe: {}'.format(df.shape))
    log(logfile, df.head())
    log(logfile, df.tail())

    # Normalization of features
    data = df.loc[:, dataset.feature_names].values
    data = StandardScaler().fit_transform(data)

    # feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
    # normalised_df = pd.DataFrame(data, columns=feat_cols)
    normalised_df = pd.DataFrame(data, columns=dataset.feature_names)
    log(logfile, normalised_df.tail())

    # Set technique
    transformed_df = {}
    n_components = -1
    if args.technique == 'PCA':
        if args.dataset == 'Iris':
            n_components = 3
        else:
            n_components = 2
        pca = PCA(n_components=n_components)
        data_transformed = pca.fit_transform(data)
        transformed_df = pd.DataFrame(data=data_transformed, columns=['PC ' + str(i + 1) for i in range(n_components)])
        log(logfile, 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    elif args.technique == 'tSNE':
        technique = 1
        # technique = load_breast_cancer() TODO
    elif args.technique == 'MDS':
        technique = 1
        # technique = load_breast_cancer() TODO
    else:
        raise ("Technique not found")

    log(logfile, transformed_df.tail())

    # Plotting
    if n_components == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.set_zlabel('PC 3', fontsize=15)
    elif n_components == 2:
        plt.figure()
        plt.figure(figsize=(10, 10))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('PC 1', fontsize=15)
    plt.ylabel('PC 2', fontsize=15)

    plt.title("{} of {} Dataset".format(args.technique, dataset_name), fontsize=20)
    colors = ['r', 'g', 'b']
    for label, color in zip(original_labels, colors):
        indicesToKeep = df['label'] == label
        if args.dataset == 'Iris':
            ax.scatter(transformed_df.loc[indicesToKeep, 'PC 1'],
                       transformed_df.loc[indicesToKeep, 'PC 2'],
                       transformed_df.loc[indicesToKeep, 'PC 3'], s=50)
        else:
            plt.scatter(transformed_df.loc[indicesToKeep, 'PC 1'],
                        transformed_df.loc[indicesToKeep, 'PC 2'], c=color, s=50)

    plt.legend(original_labels, prop={'size': 15}, loc="lower right")

    plt.savefig(outdir + '{}_c={}.svg'.format(args.technique, n_components), format="svg")


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
