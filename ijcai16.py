#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from os.path import join
from textwrap import dedent
from pprint import pformat
from multiprocessing import cpu_count

import setmargin

class Grid(object):
    def __init__(self, d):
        self.__dict__.update(d)
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
    return np.array(utilities) / 100.0

def run_synthetic():

    GRID = Grid({
        "num_trials": 10,
        "num_iterations": 10,
        "sampling_mode": ("uniform", "uniform_sparse", "normal", "normal_sparse"),
        "ranking_mode": ("all_pairs", "sorted_pairs"),
        "is_deterministic": False,
        "is_indifferent": True,
        "set_size": range(1, 4+1),
        "alpha": (0.1, 1.0, 10.0),
        "beta": (0.1, 1.0, 10.0),
        "gamma": (0.1, 1.0, 10.0),
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
                """).format(pformat(config.__dict__))

            key = (num_attrs, config.sampling_mode)
            if not key in utilities_for:
                utilities_for[key] = _load_utilities(*key)
            utilities = utilities_for[key]

            solver = setmargin.Solver((config.alpha, config.beta, config.gamma),
                                      multimargin=config.multimargin,
                                      threads=config.threads,
                                      debug=True)

            rng = np.random.RandomState(0)

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
                                      w=utilities[trial], rng=rng)

                losses_for_trial, times_for_trial = \
                    setmargin.run(dataset, user, solver, config.num_iterations,
                                  config.set_size, rng,
                                  ranking_mode=config.ranking_mode,
                                  debug=True)

if __name__ == "__main__":
    run_synthetic()
