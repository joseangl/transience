#!/usr/bin/env python3

import sys
import optparse
from glob import iglob
import numpy as np
import os
import os.path
import joblib
import preprocess
from transience import DCTW
import librosa.display
import matplotlib.pyplot as plt


def print_usage():
    print("Usage: {} OPTIONS <network_file> <view1_in_dir> <view2_in_dir> <view1_out_dir> <view2_out_dir>\n".format(sys.argv[0]))
    print("Where:")
    print("\tout_network_file: file where the DCCA network is stored.")
    print("\tview1_in_dir: directory with the Numpy files containing the unaligned sequences for view1 (e.g. speech features).")
    print("\tview2_in_dir: directory with the Numpy files containing the unaligned sequences for view1 (e.g. sensor features).")
    print("\tview1_out_dir: directory where the aligned view1 sequences are saved.")
    print("\tview2_in_dir: directory where the aligned view2 sequences are saved.\n")
    print("OPTIONS are:")
    print("\t--plot_dir: directory where the plots with the DTW alignments are saved [Default=None]")


def parse_args():
    # Parse the program args
    p = optparse.OptionParser()
    p.add_option("--plot_dir", default=None)
    opt, args = p.parse_args()
    return opt, args[0], args[1], args[2], args[3], args[4]


def plot_alignment(filepath, cost_matrix, path1, path2):
    librosa.display.specshow(cost_matrix, x_axis='frames', y_axis='frames')
    plt.title('DTW alignment')
    plt.plot(path1, path2, label='Optimal path', color='y')
    plt.savefig(filepath)
    plt.clf()


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print_usage()
        sys.exit(1)

    # Parse the command line arguments
    (opt, network_file, view1_in_dir, view2_in_dir, view1_out_dir, view2_out_dir) = parse_args()

    # Load the aligner network
    aligner = DCTW()
    aligner.build_from_file(network_file)

    # Load the pipelines for both views
    view1_pipeline, view2_pipeline = joblib.load('{}.joblib'.format(network_file))

    if not os.path.exists(view1_out_dir):
        os.mkdir(view1_out_dir)
    if not os.path.exists(view2_out_dir):
        os.mkdir(view2_out_dir)

    if opt.plot_dir is not None:
        if not os.path.exists(opt.plot_dir):
            os.mkdir(opt.plot_dir)

    for f in iglob(os.path.join(view1_in_dir, '*.npy')):
        filename = os.path.basename(f)
        f2 = os.path.join(view2_in_dir, filename)
        print("Aligning {}...".format(filename))

        # Ensure that the file also exists in dir2
        if not os.path.isfile(f2):
            continue

        x = np.load(f)
        y = np.load(f2)
        path_x, path_y, cost_matrix = aligner.align(preprocess.preprocess_data(view1_pipeline, x),
                                                    preprocess.preprocess_data(view2_pipeline, y))

        np.save(os.path.join(view1_out_dir, filename), x[path_x, :])
        np.save(os.path.join(view2_out_dir, filename), y[path_y, :])

        if opt.plot_dir:
            filename_no_ext = os.path.splitext(filename)[0]
            plot_file = os.path.join(opt.plot_dir, filename_no_ext + '.pdf')
            plot_alignment(plot_file, cost_matrix, path_x, path_y)
