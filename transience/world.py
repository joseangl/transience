#!/usr/bin/env python3

import optparse
import numpy as np
import pyworld as pw
import sys
import os.path
from glob import iglob
import soundfile as sf
from nnmnkwii.preprocessing.f0 import interp1d
from librosa import resample


_DEFAULT_FRAME_PERIOD = 5
_DEFAULT_MFCC_ORDER = 25
_TARGET_FS = 16000


def print_usage():
    print("Usage: {} [OPTIONS] wav_dir feature_dir\n".format(sys.argv[0]))
    print("OPTIONS are:")
    print("\t--frame_period: frame period used for computing the speech features [Default={}]".format(_DEFAULT_FRAME_PERIOD))
    print("\t--num_mfcc: number of MFCCs used to represent the spectral envelope [Default={}]\n".format(_DEFAULT_MFCC_ORDER))


def parse_args():
    # Parse the program args
    p = optparse.OptionParser()
    p.add_option("--frame_period", type="int", default=_DEFAULT_FRAME_PERIOD)
    p.add_option("--num_mfcc", type="int", default=_DEFAULT_MFCC_ORDER)
    opt, args = p.parse_args()
    return opt, args[0], args[1]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    # Parse the command line arguments
    (opt, wav_dir, feat_dir) = parse_args()

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    for filepath in iglob(os.path.join(wav_dir, "*.wav")):
        print("Processing file {}...".format(os.path.basename(filepath)))
        x, fs = sf.read(filepath)
        if fs != _TARGET_FS:
            x = resample(x, fs, _TARGET_FS)
            fs = _TARGET_FS

        f0, sp, ap = pw.wav2world(x, fs, frame_period=opt.frame_period)
        mfcc = pw.code_spectral_envelope(sp, fs, opt.num_mfcc)
        bap = pw.code_aperiodicity(ap, fs)
        vuv = (f0 > 0).astype(np.float32)
        # Interpolate the F0 for the unvoiced segments
        continuous_f0 = np.log(interp1d(f0, kind='slinear'))

        # Save the features
        filename = os.path.splitext(os.path.basename(filepath))[0]
        # np.save(os.path.join(feat_dir, filename + ".sp.npy"), sp)
        # np.save(os.path.join(feat_dir, filename + ".ap.npy"), ap)
        # np.save(os.path.join(feat_dir, filename + ".mfcc.npy"), mfcc)
        # np.save(os.path.join(feat_dir, filename + ".bap.npy"), bap)
        # np.save(os.path.join(feat_dir, filename + ".vuv.npy"), vuv)
        # np.save(os.path.join(feat_dir, filename + ".f0.npy"), continuous_f0)
        np.save(os.path.join(feat_dir, filename + ".npy"), np.hstack([mfcc, bap, continuous_f0[:, np.newaxis], vuv[:, np.newaxis]]))
