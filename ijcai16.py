#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from os.path import join
from textwrap import dedent
from pprint import pformat
from multiprocessing import cpu_count

import setmargin

class Grid(object):
    def __init__(self, d):
        self.__dict__.update(d)
    def asdict(self):
        return self.__dict__
    def iterate(self):
        values = []
        for value in self.__dict__.values():
            try:
                len(value)
            except TypeError:
                value = (value,)
            values.append(value)
        for configuration in product(*values):
            yield Grid(dict(zip(self.__dict__.keys(), configuration)))

def _load_utilities(num_attrs, sampling_mode):
    basename = "utilityParams_synthetic_{num_attrs}_{sampling_mode}.txt".format(**locals())
    utilities = []
    with open(join("data", "randomUtility", basename), "rb") as fp:
        for line in fp:
            utilities.append(map(float, line.split(",")))
    # rescale the range from [0,100) to [0,1)
    utilities = np.array(utilities) / 100.0
    # XXX turn negative values into zeros
    utilities[utilities < 0] = 0
    return utilities

def dump_and_draw(dataset_name, config, infos):
    basename = "__".join(map(str, [
        dataset_name,
        "k={}".format(config.set_size),
        config.sampling_mode,
        config.ranking_mode,
        "deterministic={}".format(config.is_deterministic),
        "indifferent={}".format(config.is_indifferent),
        "theta={},{},{}".format(config.alpha, config.beta, config.gamma),
        "multimargin={}".format(config.multimargin),
        "threads={}".format(config.threads),
        config.num_trials,
        config.num_iterations,
    ]))

    num_trials = len(infos)
    max_queries = max([sum(n for n, _, _ in info) for info in infos])

    loss_matrix = np.zeros((num_trials, max_queries))
    time_matrix = np.zeros((num_trials, max_queries))
    for i, info in enumerate(infos):
        base = 0
        prev_loss = max(max(l for _, l, _ in info) for info in infos)
        for num_queries, loss, time in info:
            for j in range(num_queries):
                alpha = 1 - (j + 1) / float(num_queries)
                interpolated_loss = alpha*prev_loss + (1 - alpha)*loss
                loss_matrix[i, base+j] = interpolated_loss
                time_matrix[i, base+j] = time / num_queries
            base += num_queries
            prev_loss = loss

    np.savetxt("results_{}_loss_matrix.txt".format(basename), loss_matrix)
    np.savetxt("results_{}_time_matrix.txt".format(basename), time_matrix)

    def ms(x):
        return np.mean(x, axis=0), np.std(x, ddof=1, axis=0).reshape(-1, 1)

    loss_means, loss_stddevs = ms(loss_matrix)
    time_means, time_stddevs = ms(time_matrix)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Average loss over {} trials".format(num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average loss over trials")
    ax.set_ylim([0.0, max(0.5, max(loss_means) + max(loss_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), loss_means, yerr=loss_stddevs)
    fig.savefig("results_{}_avgloss.svg".format(basename), bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Average time over {} trials".format(num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average time over trials")
    ax.set_ylim([0.0, max(0.5, max(time_means) + max(time_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), time_means, yerr=time_stddevs)
    fig.savefig("results_{}_avgtime.svg".format(basename), bbox_inches="tight")

def run_synthetic():

    GRID = Grid({
        "num_trials": 10,
        "num_iterations": 10,
        "sampling_mode": ("uniform", "uniform_sparse", "normal", "normal_sparse"),
        "ranking_mode": ("all_pairs", "sorted_pairs"),
        "is_deterministic": False,
        "is_indifferent": True,
        "set_size": range(1, 4+1),
        "alpha": 10.0,
        "beta": 0.1,
        "gamma": 0.1,
        "multimargin": False,
        "threads": cpu_count(),
    })

    utilities_for = {}

    for num_attrs in range(2, 8):
        domain_sizes = [num_attrs] * num_attrs
        dataset = setmargin.SyntheticDataset(domain_sizes)

        for config in GRID.iterate():

            print dedent("""\
                =====================
                RUNNING CONFIGURATION
                {}
                """).format(pformat(config.asdict()))

            key = (num_attrs, config.sampling_mode)
            if not key in utilities_for:
                utilities_for[key] = _load_utilities(*key)
            utilities = utilities_for[key]

            solver = setmargin.Solver((config.alpha, config.beta, config.gamma),
                                      multimargin=config.multimargin,
                                      threads=config.threads)

            rng = np.random.RandomState(0)

            infos = []
            for trial in range(config.num_trials):
                print dedent("""\
                    ===========
                    TRIAL {}/{}
                    ===========
                    """).format(trial, config.num_trials)

                user = setmargin.User(domain_sizes,
                                      sampling_mode=config.sampling_mode,
                                      is_deterministic=config.is_deterministic,
                                      is_indifferent=config.is_indifferent,
                                      w=utilities[trial].reshape(1,-1),
                                      rng=rng)

                info = setmargin.run(dataset, user, solver, config.num_iterations,
                                     config.set_size, rng,
                                     ranking_mode=config.ranking_mode)
                infos.append(info)

            dump_and_draw("synthetic_{}".format(num_attrs), config, infos)

if __name__ == "__main__":
    run_synthetic()
