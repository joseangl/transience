#!/usr/bin/env python3

import sys
import optparse
# import tempfile
# import shutil
import utils
import preprocess
from glob import glob, iglob
import numpy as np
from os.path import basename, join
from dctw import DCTW
import joblib


_DEFAULT_MAX_ITERS = 100
_DEFAULT_EPOCHS_PER_ITER = 1
_DEFAULT_VALIDATION_SPLIT = 0.2
_DEFAULT_ACTIVATION = 'lrelu'  # As suggest4ed by Trigeorgis et al. (2018)
_DEFAULT_DROPOUT = 0.0
_DEFAULT_LR = 1e-4
_DEFAULT_BATCH_SIZE = 500  # Best value according to Wang et al. (2016)
_DEFAULT_NETWORK_ARCH = '200#100#100'  # As suggested by Trigeorgis et al. (2018)
_DEFAULT_LATENT_SHARED_DIM = 50  # As Wang et al (2015) suggests
_DEFAULT_LATENT_PRIVATE_DIM = 0
_DEFAULT_DTW_DISTANCE = 'sqeuclidean'
_DEFAULT_SIMILARITIY = 'cca'
_DEFAULT_WIN = 1
_DEFAULT_AUTOENCODER_LOSS_WEIGHT = 0.0
_DEFAULT_NOISE_STD = 0.0


_ACTIVATION_TYPES = ['linear', 'sigmoid', 'tanh', 'relu', 'elu', 'lrelu']
_SIMILARITY_FUNCS = ['cca', 'mmi']


def print_usage():
    print("Usage: {} OPTIONS <out_network_file> <view1_dir> <view2_dir>\n".format(sys.argv[0]))
    print("Where:")
    print("\tout_network_file: file where the DCCA network is saved.")
    print("\tview1_dir: directory with the Numpy files containing the features for view1 (e.g. sensor features).")
    print("\tview2_dir: directory with the Numpy files containing the features for view2 (e.g. speech features).\n")
    print("OPTIONS are:")
    print("\t--val_split: fraction of the training data to be used as validation data [Default={:.1f}]".format(_DEFAULT_VALIDATION_SPLIT))
    print("\t--activation: activation function used for the hidden layers [Types available: {}][Default={}]".format(','.join(_ACTIVATION_TYPES), _DEFAULT_ACTIVATION))
    print("\t--learning_rate: learning rate for the Adam algorithm [Default= {:.2e}]".format(_DEFAULT_LR))
    print("\t--batch_size: size of the minibatches for training [Default= {}]".format(_DEFAULT_BATCH_SIZE))
    print("\t--max_iters: maximum number of training iterations (1 iteration= DCCA training + DTW alignment) [Default= {}]".format(_DEFAULT_MAX_ITERS))
    print("\t--epochs_per_iter: number of epochs the DCCA network is trained in each iteration [Default= {}]".format(_DEFAULT_EPOCHS_PER_ITER))
    print("\t--dropout: dropout probability (0.0 means no dropout is applied) [Default= {:.2f}]".format(_DEFAULT_DROPOUT))
    print("\t--network_arch: number of hidden layers and units per layer [Default= \'{}\']".format(_DEFAULT_NETWORK_ARCH))
    print("\t--latent_shared_dim: the dimensionality of the shared latent space [Default= {}]".format(_DEFAULT_LATENT_SHARED_DIM))
    print("\t--latent_private_dim: the dimensionality of the private latent spaces [Default= {}]".format(_DEFAULT_LATENT_PRIVATE_DIM))
    print("\t--dtw_distance: metric used for DTW [Types available: see scipy's cdist] [Default= {}]".format(_DEFAULT_DTW_DISTANCE))
    print("\t--similarity: loss function used to compute the similarity in the embbeding space [Types available: {}][Default={}]".format(','.join(_SIMILARITY_FUNCS), _DEFAULT_SIMILARITIY))
    print("\t--autoencoder_loss_weight: if > 0, an autoencoder type loss is implemented, where this parameter controls the importance of this loss [Default={}]".format(_DEFAULT_AUTOENCODER_LOSS_WEIGHT))
    print("\t--view1_win: lenght of the sliding window used for view1 [Default= {}]".format(_DEFAULT_WIN))
    print("\t--view2_win: lenght of the sliding window used for view2 [Default= {}]".format(_DEFAULT_WIN))
    print("\t--view1_pca_ncomps: percentage between 0 and 1 indicating the number of components retained by PCA when applied to view1 [Default= None (i.e. do not apply PCA)]")
    print("\t--view2_pca_ncomps: percentage between 0 and 1 indicating the number of components retained by PCA when applied to view2 [Default= None (i.e. do not apply PCA)]")
    print("\t--noise_std: standard deviation of the zero-mean Gaussian noise added to the inputs (i.e. 0 means no noise is added) [Default= {}]".format(_DEFAULT_NOISE_STD))


def parse_args():
    # Parse the program args
    p = optparse.OptionParser()
    p.add_option("--val_split", type="float", default=_DEFAULT_VALIDATION_SPLIT)
    p.add_option("--activation", default=_DEFAULT_ACTIVATION)
    p.add_option("--learning_rate", type="float", default=_DEFAULT_LR)
    p.add_option("--batch_size", type="int", default=_DEFAULT_BATCH_SIZE)
    p.add_option("--max_iters", type="int", default=_DEFAULT_MAX_ITERS)
    p.add_option("--epochs_per_iter", type="int", default=_DEFAULT_EPOCHS_PER_ITER)
    p.add_option("--dropout", type="float", default=_DEFAULT_DROPOUT)
    p.add_option("--network_arch", default=_DEFAULT_NETWORK_ARCH)
    p.add_option("--latent_shared_dim", type="int", default=_DEFAULT_LATENT_SHARED_DIM)
    p.add_option("--latent_private_dim", type="int", default=_DEFAULT_LATENT_PRIVATE_DIM)
    p.add_option("--dtw_distance", default=_DEFAULT_DTW_DISTANCE)
    p.add_option("--similarity", default=_DEFAULT_SIMILARITIY)
    p.add_option("--autoencoder_loss_weight", type="float", default=_DEFAULT_AUTOENCODER_LOSS_WEIGHT)
    p.add_option("--view1_win", type="int", default=_DEFAULT_WIN)
    p.add_option("--view2_win", type="int", default=_DEFAULT_WIN)
    p.add_option("--view1_pca_ncomps", type="float", default=None)
    p.add_option("--view2_pca_ncomps", type="float", default=None)
    p.add_option("--noise_std", type="float", default=_DEFAULT_NOISE_STD)
    opt, args = p.parse_args()
    return opt, args[0], args[1], args[2]


# def split_parallel_dataset(dir1, dir2, val_split):
#     files = np.array([basename(f) for f in glob(join(dir1, '*.npy'))])
#     num_files = len(files)
#     index = np.arange(num_files)
#     np.random.shuffle(index)
#     cut = round(num_files * val_split)

#     # Copy the validation files
#     val_dir_1 = tempfile.mkdtemp()
#     val_dir_2 = tempfile.mkdtemp()
#     for f in files[index[:cut]]:
#         shutil.copy(join(dir1, f), val_dir_1)
#         shutil.copy(join(dir2, f), val_dir_2)

#     # Copy the training files
#     train_dir_1 = tempfile.mkdtemp()
#     train_dir_2 = tempfile.mkdtemp()
#     for f in files[index[cut:]]:
#         shutil.copy(join(dir1, f), train_dir_1)
#         shutil.copy(join(dir2, f), train_dir_2)
#     return train_dir_1, train_dir_2, val_dir_1, val_dir_2


def split_parallel_dataset(dir1, dir2, val_split):
    files = np.array([basename(f) for f in iglob(join(dir1, '*.npy'))])
    if val_split > 0.0:
        num_files = len(files)
        index = np.arange(num_files)
        np.random.shuffle(index)
        cut = round(num_files * val_split)
        validation_files = files[index[:cut]].tolist()
        train_files = files[index[cut:]].tolist()
    else:
        train_files = files
        validation_files = None
    return train_files, validation_files


def get_feature_dim(directory):
    dim = None
    files = glob(join(directory, '*.npy'))
    if len(files) > 0:
        data = np.load(files[0])
        dim = data.shape[1]
    return dim


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    # Parse the command line arguments
    (opt, network_file, view1_dir, view2_dir) = parse_args()

    # # Split the files into training and validation
    # if opt.val_split > 0.0:
    #     train_dir_1, train_dir_2, val_dir_1, val_dir_2 = split_parallel_dataset(view1_dir, view2_dir, opt.val_split)
    #     remove_dirs = True
    # else:
    #     train_dir_1 = view1_dir
    #     train_dir_2 = view2_dir
    #     val_dir_1 = val_dir_2 = None
    #     remove_dirs = False

    # Split the files into training and validation
    train_files, validation_files = split_parallel_dataset(view1_dir, view2_dir, opt.val_split)

    # Read the data and extract the features
    view1_raw_data = utils.load_numpy_dataset(view1_dir, train_files)
    view1_pipeline = preprocess.create_preprocessing_pipeline(opt.view1_win, opt.view1_pca_ncomps)
    view1_train_data = preprocess.preprocess_data(view1_pipeline, view1_raw_data)

    view2_raw_data = utils.load_numpy_dataset(view2_dir, train_files)
    view2_pipeline = preprocess.create_preprocessing_pipeline(opt.view2_win, opt.view2_pca_ncomps)
    view2_train_data = preprocess.preprocess_data(view2_pipeline, view2_raw_data)

    view1_validation_data = view2_validation_data = None
    if validation_files:
        view1_raw_data = utils.load_numpy_dataset(view1_dir, validation_files)
        view1_validation_data = preprocess.preprocess_data(view1_pipeline, view1_raw_data)
        view2_raw_data = utils.load_numpy_dataset(view2_dir, validation_files)
        view2_validation_data = preprocess.preprocess_data(view2_pipeline, view2_raw_data)

    if opt.network_arch == '':
        hidden_layers = []
    else:
        hidden_layers = opt.network_arch.split('#')

    # Create and train the model
    model = DCTW(True)
    view1_feature_dim = view1_train_data[0].shape[1]
    view2_feature_dim = view2_train_data[0].shape[1]
    model.build(view1_feature_dim, view2_feature_dim, opt.latent_shared_dim, opt.latent_private_dim, network_architecture=hidden_layers, activation=opt.activation,
                similarity=opt.similarity, cycle_loss=opt.autoencoder_loss_weight, dropout=opt.dropout, lr=opt.learning_rate)
    model.fit(view1_train_data, view2_train_data, view1_validation_data, view2_validation_data, max_iters=opt.max_iters, epochs_per_dtw=opt.epochs_per_iter,
              batch_size=opt.batch_size, dtw_metric=opt.dtw_distance, noise_std=opt.noise_std)

    # Save the model
    model.save(network_file)

    # Save the parameters of the preprocessing pipeline
    joblib.dump((view1_pipeline, view2_pipeline), '{}.joblib'.format(network_file))

    # if remove_dirs:
    #     shutil.rmtree(train_dir_1)
    #     shutil.rmtree(train_dir_2)
    #     shutil.rmtree(val_dir_1)
    #     shutil.rmtree(val_dir_2)
