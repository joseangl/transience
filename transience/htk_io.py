# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:46:10 2015

@author: josean
"""

import sys
import numpy as np
from struct import pack, unpack


# ------------------------------------------------------------------------
def readhtk(file_name, endian='little'):
    unpack_fmt = '<iihh' if endian == 'little' else '>iihh'

    with open(file_name, 'rb') as fh:
        # Read the 12-bytes header: int + int + short + short
        data = fh.read(12)
        nframes, frm_period, frm_size, samp_kind = unpack(unpack_fmt, data)
        frm_size //= 4  # frm_size is originally expressed in bytes

        if sys.byteorder == endian:
            features = np.fromfile(fh, dtype=np.float32)
        else:
            features = np.fromfile(fh, dtype=np.float32).byteswap()

    return features.reshape((nframes, frm_size)), frm_period, samp_kind


# ------------------------------------------------------------------------
def writehtk(file_name, data, frm_period=100000, samp_kind=9, endian='little'):
    pack_fmt = '<iihh' if endian == 'little' else '>iihh'

    with open(file_name, 'wb') as fh:
        nframes, frm_size = data.shape
        frm_size *= 4
        # Write the 12-bytes header: int + int + short + short
        fh.write(pack(pack_fmt, nframes, frm_period, frm_size, samp_kind))
        # Write the data as float32
        if sys.byteorder == endian:
            data.astype(np.float32).tofile(fh)
        else:
            data.astype(np.float32).byteswap().tofile(fh)
