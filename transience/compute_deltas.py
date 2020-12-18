#!/usr/bin/env python3

import optparse
import numpy as np
import sys
import os.path
from glob import iglob
from python_speech_features import delta

_DEFAULT_DELTA_WIN = 3
_DEFAULT_ACC_WIN = 2


def print_usage():
    print("Usage: {} [OPTIONS] input_dir output_dir\n".format(sys.argv[0]))
    print("OPTIONS are:")
    print("\t--delta_win: size of the window used for computing the derivatives [Default={}]".format(_DEFAULT_DELTA_WIN))
    print("\t--acc_win: size of the window used for computing the accelerations [Default={}]\n".format(_DEFAULT_ACC_WIN))


def parse_args():
    # Parse the program args
    p = optparse.OptionParser()
    p.add_option("--delta_win", type="int", default=_DEFAULT_DELTA_WIN)
    p.add_option("--acc_win", type="int", default=_DEFAULT_ACC_WIN)
    opt, args = p.parse_args()
    return opt, args[0], args[1]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    # Parse the command line arguments
    (opt, in_dir, out_dir) = parse_args()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filepath in iglob(os.path.join(in_dir, "*.npy")):
        fname = os.path.basename(filepath)
        print("Processing file {}...".format(fname))
        mfcc = np.load(filepath)
        delta = delta(mfcc, opt.delta_win)
        acc = delta(delta, opt.acc_win)
        mfcc_a_d = np.hstack([mfcc, delta, acc])
        np.save(os.path.join(out_dir, fname))
