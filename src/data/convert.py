#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from glob import glob

for source_path in sorted(glob("trueBelief_*.mat")):
    dest_path = source_path.split(".")[0] + ".npy"
    print "converting", source_path, "into", dest_path
    mat = loadmat(source_path)
    true_belief_key = [key for key in mat.keys() if key.startswith("trueBelief_")][0]
    true_belief = mat[true_belief_key]

    # XXX this is insane, but who cares
    np_true_belief = np.zeros((true_belief.shape[1],) * 2)
    for i in range(true_belief.shape[0]):
        for j in range(true_belief.shape[1]):
            np_true_belief[j,:] = true_belief[i,j].ravel()

    np.save(dest_path, np_true_belief)
