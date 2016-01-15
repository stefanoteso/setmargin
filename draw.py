#!/usr/bin/env python2

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import itertools as it
from glob import glob

if len(sys.argv) != 3:
    print "Usage: {} <results directory> <output image>".format(sys.argv[0])
    sys.exit(1)

fontP = FontProperties()
fontP.set_size('small')

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Number of queries")
ax.set_ylabel("Average loss over trials")

paths = sorted(glob(os.path.join(sys.argv[1], "results_*_loss_matrix.txt")))

for path in paths:
    parts = os.path.basename(path).split("__")

    loss_matrix = np.loadtxt(path)
    num_trials, max_queries = loss_matrix.shape

    loss_means = np.mean(loss_matrix, axis=0)
    loss_stds = np.std(loss_matrix, axis=0).reshape(-1, 1)

    ax.set_ylim([0.0, max(0.5, max(loss_means) + max(loss_stds) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), loss_means, yerr=loss_stds, fmt="--o", label=os.path.basename(path))

#lgd = ax.legend(prop=fontP, loc="upper center", bbox_to_anchor=(0.5, -0.1))
#fig.savefig(sys.argv[2], bbox_extra_artists=(lgd,), bbox_inches="tight")
fig.savefig(sys.argv[2], bbox_inches="tight")
