#!/usr/bin/env python3

import numpy as np
import os
from glob import iglob


def load_numpy_parallel_data(dir1, dir2):
    view1_data = []
    view2_data = []

    for f in iglob(os.path.join(dir1, '*.npy')):
        filename = os.path.basename(f)
        f2 = os.path.join(dir2, filename)

        # Ensure that the file also exists in dir2
        if not os.path.isfile(f2):
            continue

        view1_data.append(np.load(f))
        view2_data.append(np.load(f2))

    return view1_data, view2_data


def load_numpy_dataset(directory, files):
    return [np.load(os.path.join(directory, f)) for f in files]
