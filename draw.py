#!/usr/bin/env python2

import sys, os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from glob import glob

# Colors taken from http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines
COLOR_OF = {
    2: ("#CE5C00", "#F57900"),
    3: ("#204A87", "#3465A4"),
#    4: ("#5C3566", "#75507B"),
    4: ("#4E9A06", "#73D216"),
}

def infos_to_matrices(infos, elongate=True):
    num_trials = len(infos)
    max_queries = max([sum(n for n, _, _ in info) for info in infos])
    max_iterations = max(len(info) for info in infos)

    if elongate:
        loss_matrix = np.zeros((num_trials, max_queries))
        time_matrix = np.zeros((num_trials, max_queries))
    else:
        loss_matrix = np.zeros((num_trials, max_iterations))
        time_matrix = np.zeros((num_trials, max_iterations))

    for trial, info in enumerate(infos):
        base = 0
        prev_loss = max(max(l for _, l, _ in info) for info in infos)
        for iteration, (num_queries, loss, time) in enumerate(info):
            if elongate:
                for j in range(num_queries):
                    alpha = 1 - (j + 1) / float(num_queries)
                    interpolated_loss = alpha*prev_loss + (1 - alpha)*loss
                    loss_matrix[trial, base+j] = interpolated_loss
                    time_matrix[trial, base+j] = time / num_queries
                base += num_queries
                prev_loss = loss
            else:
                loss_matrix[trial,iteration] = loss
                time_matrix[trial,iteration] = time

    return max_queries if elongate else max_iterations, loss_matrix, time_matrix

def draw(paths, dest_path):
    loss_fig, loss_ax = plt.subplots(1, 1)
    time_fig, time_ax = plt.subplots(1, 1)

    max_max_x = None
    max_loss_max_y = None
    max_time_max_y = None
    for path in paths:
        with open(path, "rb") as fp:
            infos = pickle.load(fp)

        max_x, loss_matrix, time_matrix = infos_to_matrices(infos, elongate=False)

        xs = np.arange(max_x)
        if max_max_x is None or max_max_x < max_x:
            max_max_x = max_x

        loss_ys = np.median(loss_matrix, axis=0)
        loss_yerrs = np.std(loss_matrix, axis=0) / np.sqrt(loss_matrix.shape[0])
        loss_max_y = max(loss_ys + loss_yerrs)
        if max_loss_max_y is None or max_loss_max_y < loss_max_y:
            max_loss_max_y = loss_max_y

        time_ys = np.median(time_matrix.cumsum(axis=1), axis=0)
        time_yerrs = np.std(time_matrix.cumsum(axis=1), axis=0) / np.sqrt(loss_matrix.shape[0])
        time_max_y = max(time_ys + time_yerrs)
        if max_time_max_y is None or max_time_max_y < time_max_y:
            max_time_max_y = time_max_y

        parts = os.path.basename(path).split("__")
        set_size = int(parts[1].split("=")[1])
        fg, bg = COLOR_OF[set_size]

        loss_ax.plot(xs, loss_ys, "o-", linewidth=2.0, color=fg)
        loss_ax.fill_between(xs, loss_ys - loss_yerrs, loss_ys + loss_yerrs,
                             color=bg, alpha=0.35, linewidth=0)

        time_ax.plot(xs, time_ys, "o-", linewidth=2.0, color=fg)
        time_ax.fill_between(xs, time_ys - time_yerrs, time_ys + time_yerrs,
                             color=bg, alpha=0.35, linewidth=0)

    loss_ax.set_xlabel("Number of queries")
    loss_ax.set_ylabel("Median average utility loss")
    try:
        loss_ax.set_xlim([0.0, max_max_x])
        loss_ax.set_xticks(np.arange(0, max_max_x, 10))
        loss_ax.set_ylim([0.0, max(100.0, max_loss_max_y + 0.1)])
        loss_ax.set_yticks(np.arange(0, max_loss_max_y, 10))
    except:
        pass
    loss_fig.savefig(dest_path + "_loss.png", bbox_inches="tight")

    time_ax.set_xlabel("Number of queries")
    time_ax.set_ylabel("Cumulative average time (in seconds)")
    try:
        time_ax.set_xlim([0.0, max_max_x])
        time_ax.set_xticks(np.arange(0, max_max_x, 10))
        time_ax.set_ylim([0.0, max(1.0, max_time_max_y + 0.1)])
        time_ax.set_yticks(np.arange(0, max_time_max_y, 10))
    except:
        pass
    time_fig.savefig(dest_path + "_time.png", bbox_inches="tight")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: {} <results directory> <output image>".format(sys.argv[0])
        sys.exit(1)
    draw(sorted(glob(os.path.join(sys.argv[1], "results_*.pickle"))), sys.argv[2])
