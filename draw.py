#!/usr/bin/env python2

import sys, os
import cPickle as pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import defaultdict
from glob import glob

# Colors taken from http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines
COLORS = [
    ("#CE5C00", "#F57900"), # orange
    ("#204A87", "#3465A4"), # sky blue
    ("#4E9A06", "#73D216"), # chameleon
    ("#2E3436", "#555753"), # aluminium
]

def infos_to_matrices(infos, per_query):
    num_trials = len(infos)
    max_queries = max([sum(n for n, _, _ in info) for info in infos])
    max_iterations = max(len(info) for info in infos)

    if per_query:
        loss_matrix = np.zeros((num_trials, max_queries))
        time_matrix = np.zeros((num_trials, max_queries))
    else:
        loss_matrix = np.zeros((num_trials, max_iterations))
        time_matrix = np.zeros((num_trials, max_iterations))

    for trial, info in enumerate(infos):
        base = 0
        prev_loss = max(max(l for _, l, _ in info) for info in infos)
        for iteration, (num_queries, loss, time) in enumerate(info):
            if per_query:
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

    return loss_matrix, time_matrix

def load_setmargin(path):
    with open(path, "rb") as fp:
        infos = pickle.load(fp)
    return infos

def load_viappiani(path):
    with open(path, "rb") as fp:
        lines = map(str.strip, fp.readlines())
    infos = defaultdict(dict)
    for line in lines[1:]:
        trial, iteration, loss, time = line.split(", ")
        infos[int(trial) - 1][int(iteration)] = (1, float(loss), float(time))
    info_list = []
    assert infos.keys() == range(20)
    for trial in range(20):
        assert infos[trial].keys() == range(101)
        info_list.append([infos[trial][iteration] for iteration in range(100)])
    return info_list

def load_guo(paths):
    infos = []
    assert len(paths) == 20
    for trial, path in enumerate(sorted(paths)):
        mat = loadmat(path, squeeze_me=True)
        losses = map(float, mat["expectedLossMatrix"].ravel())
        times = map(float, mat["timeUsed"].ravel())
        assert len(losses) == len(times) == 100
        infos.append(zip([1] * 100, losses, times))
    return infos

def draw_matrices(ax, matrices, cumulative):
    max_x, max_y = None, None
    for i, matrix in enumerate(matrices):
        cur_max_x = matrix.shape[1]
        if max_x is None or cur_max_x > max_x:
            max_x = cur_max_x

        assert matrix.shape[0] == 20

        xs = np.arange(cur_max_x)
        if cumulative:
            matrix = matrix.cumsum(axis=1)
        ys = np.median(matrix, axis=0)
        yerrs = np.std(matrix, axis=0) / np.sqrt(matrix.shape[0])

        cur_max_y = max(ys + yerrs)
        if max_y is None or cur_max_y > max_y:
            max_y = cur_max_y

        fg, bg = COLORS[i]

        ax.plot(xs, ys, "o-", linewidth=2.5, color=fg)
        ax.fill_between(xs, ys - yerrs, ys + yerrs, color=bg, alpha=0.35, linewidth=0)

    return max_x, max_y

def draw_groups(basename, groups, upper_max_x, per_query=False):
    loss_matrices, time_matrices = [], []
    for group in groups:
        paths = group.split()
        if len(paths) == 1 and paths[0].endswith(".pickle"):
            infos = load_setmargin(paths[0])
        elif len(paths) == 1 and paths[0].endswith(".txt"):
            infos = load_viappiani(paths[0])
        elif len(paths) >= 1 and paths[0].endswith(".mat"):
            infos = load_guo(paths)
        else:
            raise ValueError()
        loss_matrix, time_matrix = infos_to_matrices(infos, per_query)
        loss_matrices.append(loss_matrix[:,:upper_max_x])
        time_matrices.append(time_matrix[:,:upper_max_x])

    xlabel = "Number of queries" if per_query else "Number of iterations"

    # Loss
    fig, ax = plt.subplots(1, 1)
    max_x, max_y = draw_matrices(ax, loss_matrices, False)

    ax.set_xlabel(xlabel)
    ax.set_xlim([0.0, max_x])
    ax.set_xticks(np.arange(0, max_x, 10))

    ax.set_ylabel("Median utility loss")
    ax.set_ylim([0.0, max_y + 0.1])
    ax.set_yticks(np.arange(0, max_y, 10))

    fig.savefig(basename + "_loss.png", bbox_inches="tight")

    # Time
    fig, ax = plt.subplots(1, 1)
    max_x, max_y = draw_matrices(ax, time_matrices, True)

    ax.set_xlabel(xlabel)
    ax.set_xlim([0.0, max_x])
    ax.set_xticks(np.arange(0, max_x, 10))

    ax.set_ylabel("Cumulative average time (in seconds)")
    ax.set_ylim([0.0, max_y + 0.1])
    ax.set_yticks(np.arange(0, max_y, 50))

    fig.savefig(basename + "_time.png", bbox_inches="tight")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "Usage: {} <per query?> <max x> <output basename> (<group>)+".format(sys.argv[0])
        quit()
    draw_groups(sys.argv[2], sys.argv[4:], int(sys.argv[3]), per_query=bool(int(sys.argv[1])))
