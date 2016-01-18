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

figs, axs = {}, {}

figs[True], axs[True] = plt.subplots(1, 1)
axs[True].set_xlabel("Number of queries")
axs[True].set_ylabel("Average loss per answer")

figs[False], axs[False] = plt.subplots(1, 1)
axs[False].set_xlabel("Number of queries")
axs[False].set_ylabel("Average time per answer (in seconds)")

paths = sorted(glob(os.path.join(sys.argv[1], "results_*_matrix.txt")))

# Colors taken from http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines
COLOR_OF = {
    2: ("#CE5C00", "#F57900"),
    3: ("#204A87", "#3465A4"),
    4: ("#5C3566", "#75507B"),
}

max_max = {}
for path in paths:
    parts = os.path.basename(path).split("__")
    set_size = int(parts[1].split("=")[1])
    if not set_size in (2, 3, 4):
        continue
    is_loss = parts[-1].split("_")[1] == "loss"

    matrix = np.loadtxt(path)
    if matrix.shape[1] > 100:
        matrix = matrix[:,:100]
    num_trials, max_queries = matrix.shape

    xs = np.arange(max_queries)
    ys = np.median(matrix, axis=0)
    yerrs = np.std(matrix, axis=0)
    max_y = max(ys + yerrs)

    key = (is_loss, "x")
    if not key in max_max or max_queries > max_max[key]:
        max_max[key] = max_queries

    key = (is_loss, "y")
    if not key in max_max or max_y > max_max[key]:
        max_max[key] = max_y

    fg, bg = COLOR_OF[set_size]

    ax = axs[is_loss]
    ax.plot(xs, ys, "k-", linewidth=2.0, color=fg)
    ax.fill_between(xs, ys - yerrs, ys + yerrs, color=bg, alpha=0.35, linewidth=0)

for is_loss in (True, False):
    try:
        max_max_y = max(100 if is_loss else 1.0, max_max[(is_loss, "y")] + 0.1)

        ax = axs[is_loss]
        ax.set_xlim([0.0, max_max[(is_loss, "x")]])
        ax.set_xticks(np.arange(0, max_max[(is_loss, "x")], 10))
        ax.set_ylim([0.0, max_max_y])
        ax.set_yticks(np.arange(0, max_max_y, 10))
    except:
        pass

    path = sys.argv[2] + ("_loss" if is_loss else "_time") + ".png"
    figs[is_loss].savefig(path, bbox_inches="tight")
