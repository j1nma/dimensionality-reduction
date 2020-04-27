import argparse
import sys
import os
import datetime
from sklearn.datasets import load_iris, load_breast_cancer


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

    # Create results directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        print("Directory ", args.outdir, " created.")

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + timestamp + '/'
    os.mkdir(outdir)
    print("Directory", outdir, "created.")

    # Set data dictionary
    data = {}
    if args.dataset == 'Iris':
        data = load_iris()
    elif args.dataset == 'BreastCancer':
        data = load_breast_cancer()
    else:
        raise ("Dataset not found")

    print(data.target[[10, 50, 85]])
    print(list(data.target_names))

    # TODO: clean datasets?

    # TODO: choose technique

    # Dataset analysis
    # if args.name == 'Congress':
    # print(train_df.info())
    # c_palette = ['tab:blue', 'tab:orange']
    # categorical_summarized(train_df, y='class', hue='el-salvador-aid', palette=c_palette)
    # else:
    # print(train_df.describe())

    # if args.test == 'True':
    #     # Perform predictions on testing set to save to CV
    #     do_experiment(train_dataset,
    #                   data_dictionary['dataset_name'],
    #                   int(args.seed),
    #                   list(args.k_neighbours),
    #                   list(args.n_trees),
    #                   outdir,
    #                   int(args.perceptron_iterations),
    #                   float(args.perceptron_learning_rate),
    #                   int(args.kfold),
    #                   int(args.max_depth),
    #                   test_input_samples_encoded,
    #                   ple)
    # else:
    #     # Perform training and testing split among training set
    #     do_experiment(train_dataset,
    #                   data_dictionary['dataset_name'],
    #                   int(args.seed),
    #                   list(args.k_neighbours),
    #                   list(args.n_trees),
    #                   outdir,
    #                   int(args.perceptron_iterations),
    #                   float(args.perceptron_learning_rate),
    #                   int(args.kfold),
    #                   int(args.max_depth),
    #                   do_k_fold=True)

    # do_experiment(datasets.load_digits(),
    #               "Digits",
    #               int(args.seed),
    #               list(args.k_neighbours),
    #               list(args.n_trees),
    #               outdir,
    #               int(args.perceptron_iterations),
    #               float(args.perceptron_learning_rate),
    #               int(args.kfold),
    #               int(args.max_depth))

    # for dataset in extract_music_data(args.dataPath):
    #     do_experiment(dataset,
    #                   str(dataset.name),
    #                   int(args.seed),
    #                   list(args.k_neighbours),
    #                   list(args.n_trees),
    #                   outdir,
    #                   int(args.perceptron_iterations),
    #                   float(args.perceptron_learning_rate),
    #                   int(args.kfold),
    #                   int(args.max_depth))

    # Logging
    logfile = outdir + 'log.txt'
    f = open(logfile, 'w')
    f.close()


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
