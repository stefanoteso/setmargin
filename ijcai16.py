#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
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

def get_result_paths(dataset_name, config):
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
    path1 = "results_{}_loss_matrix.txt".format(basename)
    path2 = "results_{}_time_matrix.txt".format(basename)
    path3 = "results_{}_avgloss.svg".format(basename)
    path4 = "results_{}_avgtime.svg".format(basename)
    return path1, path2, path3, path4

def dump_and_draw(dataset_name, config, infos):
    loss_matrix_path, time_matrix_path, loss_svg_path, time_svg_path = \
        get_result_paths(dataset_name, config)

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

    np.savetxt(loss_matrix_path, loss_matrix)
    np.savetxt(time_matrix_path, time_matrix)

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
    fig.savefig(loss_svg_path, bbox_inches="tight")
    del fig
    del ax

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Average time over {} trials".format(num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average time over trials")
    ax.set_ylim([0.0, max(0.5, max(time_means) + max(time_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), time_means, yerr=time_stddevs)
    fig.savefig(time_svg_path, bbox_inches="tight")
    del fig
    del ax

def solve(dataset, config, ws=None):
    rng = np.random.RandomState(config.seed)

    solver = setmargin.Solver((config.alpha, config.beta, config.gamma),
                              multimargin=config.multimargin,
                              threads=config.threads, debug=config.debug)
    infos = []
    for trial in range(config.num_trials):
        print dedent("""\
            ===========
            TRIAL {}/{}
            ===========
            """).format(trial, config.num_trials)

        w = None if ws is None else ws[trial].reshape(1,-1)

        user = setmargin.User(dataset.domain_sizes,
                              sampling_mode=config.sampling_mode,
                              ranking_mode=config.ranking_mode,
                              is_deterministic=config.is_deterministic,
                              is_indifferent=config.is_indifferent,
                              w=w,
                              rng=rng)

        info = setmargin.run(dataset, user, solver, config.num_iterations,
                             config.set_size, debug=config.debug)
        infos.append(info)
    return infos

def run_synthetic():
    CONFIGS = Grid({
        "num_trials": 10,
        "num_iterations": 10,
        "sampling_mode": ("uniform", "uniform_sparse", "normal", "normal_sparse"),
        "ranking_mode": ("all_pairs",),
        "is_deterministic": False,
        "is_indifferent": True,
        "set_size": range(1, 4+1),
        "alpha": 1.0,
        "beta": (10.0, 1.0, 0.1, 0.0),
        "gamma": (10.0, 1.0, 0.1, 0.0),
        "multimargin": False,
        "threads": cpu_count(),
        "debug": False,
        "seed": 0,
    })

    utilities = {}
    for num_attrs in range(2, 8):
        domain_sizes = [num_attrs] * num_attrs
        dataset = setmargin.SyntheticDataset(domain_sizes)

        for config in CONFIGS.iterate():

            print dedent("""\
                =====================
                RUNNING CONFIGURATION
                {}
                """).format(pformat(config.asdict()))

            key = (num_attrs, config.sampling_mode)
            if not key in utilities:
                utilities[key] = _load_utilities(*key)
            ws = utilities[key]

            infos = solve(dataset, config, ws=ws)
            dump_and_draw("synthetic_{}".format(num_attrs), config, infos)

def run_pc_nocost():
    pass

def run_pc():
    pass

def run_from_command_line():
    import argparse as ap

    parser = ap.ArgumentParser(description="setmargin experiment")
    parser.add_argument("dataset", type=str,
                        help="dataset")
    parser.add_argument("-N", "--num_trials", type=int, default=20,
                        help="number of trials (default: 20)")
    parser.add_argument("-n", "--num_iterations", type=int, default=20,
                        help="number of iterations (default: 20)")
    parser.add_argument("-m", "--set-size", type=int, default=3,
                        help="number of hyperplanes/items to solve for (default: 3)")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
                        help="hyperparameter controlling the importance of slacks (default: 0.1)")
    parser.add_argument("-b", "--beta", type=float, default=0.1,
                        help="hyperparameter controlling the importance of regularization (default: 0.1)")
    parser.add_argument("-c", "--gamma", type=float, default=0.1,
                        help="hyperparameter controlling the score of the output items (default: 0.1)")
    parser.add_argument("-r", "--ranking-mode", type=str, default="all_pairs",
                        help="ranking mode, any of ('all_pairs', 'sorted_pairs') (default: 'all_pairs')")
    parser.add_argument("-M", "--multimargin", action="store_true",
                        help="whether the example and generated object margins should be independent (default: False)")
    parser.add_argument("-u", "--sampling-mode", type=str, default="uniform",
                        help="utility sampling mode, any of ('uniform', 'normal') (default: 'uniform')")
    parser.add_argument("-d", "--is-deterministic", action="store_true",
                        help="whether the user answers should be deterministic rather than stochastic (default: False)")
    parser.add_argument("-i", "--is-indifferent", action="store_true",
                        help="whether the user can (not) be indifferent (default: False)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed (default: None)")
    parser.add_argument("--domain-sizes", type=str, default="2,2,5",
                        help="domain sizes for the synthetic dataset only (default: 2,2,5)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Max number of threads to user (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug spew")
    args = parser.parse_args()

    argsdict = vars(args)
    argsdict["dataset"] = args.dataset

    config = Grid(argsdict)

    domain_sizes = map(int, [ds for ds in args.domain_sizes.split(",") if len(ds)])
    if args.dataset == "synthetic":
        dataset = setmargin.SyntheticDataset(domain_sizes)
    elif args.dataset == "random_constraints":
        dataset = setmargin.RandomDataset(domain_sizes, rng=rng)
    elif args.dataset == "pc":
        dataset = setmargin.PCDataset()
    elif args.dataset == "liftedpc":
        dataset = setmargin.LiftedPCDataset()
    else:
        raise ValueError("invalid dataset.")
    if args.debug:
        print dataset

    infos = solve(dataset, config)
    dump_and_draw("{}_{}".format(args.dataset, ",".join(map(str, args.domain_sizes))),
                  config, infos)

if __name__ == "__main__":
    np.seterr(all="raise")
    if len(sys.argv) == 1:
        run_synthetic()
        run_pc_nocost()
        run_pc()
    else:
        run_from_command_line()
