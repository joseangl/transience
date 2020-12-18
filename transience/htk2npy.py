#!/usr/bin/env python3

import htk_io as htk
import numpy as np
import sys
import os
import os.path
from glob import iglob


def htk2npy(htk_file, numpy_file):
    x, _, _ = htk.readhtk(htk_file)
    np.save(numpy_file, x)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: {} input_dir ext [output_dir]\n".format(sys.argv[0]))
        sys.exit(1)

    in_dir = sys.argv[1]
    ext = sys.argv[2]
    if len(sys.argv) > 3:
        out_dir = sys.argv[3]
    else:
        out_dir = in_dir

    if ext[0] == '.':
        ext = ext[1:]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filepath in iglob(os.path.join(in_dir, "*." + ext)):
        print("Processing file {}...".format(os.path.basename(filepath)))
        filename = os.path.splitext(os.path.basename(filepath))[0]
        htk2npy(filepath, os.path.join(out_dir, filename + ".npy"))
