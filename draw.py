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

# Colors taken from http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines
COLOR_OF = {
#    ("#EDD400", "#FCE94F"), # yellow
    2: ("#CE5C00", "#F57900"), # orange
#    ("#4E9A06", "#73D216"), # green
    3: ("#204A87", "#3465A4"), # blue
#    ("#5C3566", "#75507B"), # violet
#    2: ("#A40000", "#CC0000"), # red
}

max_max_x = 0
max_max_y = 0
for path in paths:
    parts = os.path.basename(path).split("__")
    set_size = int(parts[1].split("=")[1])
    if not set_size in (2, 3):
        continue

    loss_matrix = np.loadtxt(path)
    if loss_matrix.shape[1] > 100:
        loss_matrix = loss_matrix[:,:100]
    num_trials, max_queries = loss_matrix.shape

    xs = np.arange(max_queries)
    ys = np.median(loss_matrix, axis=0)
    yerrs = np.std(loss_matrix, axis=0)

    if max_queries > max_max_x:
        max_max_x = max_queries
    max_y = max(ys + yerrs)
    if max_y > max_max_y:
        max_max_y = max_y

    fg, bg = COLOR_OF[set_size]

    ax.plot(xs, ys, "k-", linewidth=2.0, color=fg)
    ax.fill_between(xs, ys - yerrs, ys + yerrs, color=bg, alpha=0.35, linewidth=0)


max_max_y = max(0.5, max_max_y + 0.1)
ax.set_xticks(np.arange(0, max_max_x, 10))
ax.set_yticks(np.arange(0, max(0.5, max_max_y + 0.1), 0.1))
ax.set_ylim([0.0, max_max_y])
fig.savefig(sys.argv[2], bbox_inches="tight")
