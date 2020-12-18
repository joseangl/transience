#!/usr/bin/env python3

import optparse
import sys
import os.path
from glob import glob
import re
from shutil import copyfile


def print_usage():
    print("Usage: {} in_sensor_dir in_mfcc_dir out_sensor_dir out_mfcc_dir\n".format(sys.argv[0]))


def file_basename(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]
 

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    isensor = sys.argv[1]
    imfcc = sys.argv[2]
    osensor = sys.argv[3]
    omfcc = sys.argv[4]

    if not os.path.exists(osensor):
        os.makedirs(osensor)

    if not os.path.exists(omfcc):
        os.makedirs(omfcc)        

    regex_seq_number = re.compile('-(\w+)\D$')
    list_sensor = glob(os.path.join(isensor, "*.npy"))
    list_mfcc = glob(os.path.join(imfcc, "*.npy"))
    for f1 in list_sensor:
        f1_name = file_basename(f1)
        m = regex_seq_number.search(f1_name)
        # Scan the list of MFCC files with the same digit sequence
        if m:
            digit_seq = m.group(1)
            re2 = re.compile('-{}\D.npy$'.format(digit_seq))
            matched_files = [f for f in list_mfcc if re2.search(f)]
            for f2 in matched_files:
                f2_name = file_basename(f2)
                copyfile(f1, os.path.join(osensor, '{}_{}.npy'.format(f1_name, f2_name)))
                copyfile(f2, os.path.join(omfcc, '{}_{}.npy'.format(f1_name, f2_name)))
