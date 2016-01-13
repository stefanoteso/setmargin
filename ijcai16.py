#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from itertools import product
from os.path import join
from textwrap import dedent
from pprint import pformat
from multiprocessing import cpu_count

import setmargin

ALL_ALPHAS = list(product(
         [100.0, 20.0, 15.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.5, 5.0, 1.0],
         [20.0, 10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01],
         [20.0, 10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01],
))

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
    if config.crossval:
        alphas = "auto"
    else:
        alphas = ",".join(map(str, [config.alpha, config.beta, config.gamma]))
    basename = "__".join(map(str, [
        dataset_name,
        "k={}".format(config.set_size),
        config.sampling_mode,
        config.ranking_mode,
        "deterministic={}".format(config.is_deterministic),
        "indifferent={}".format(config.is_indifferent),
        "alphas={}".format(alphas),
        "crossval_set_size={}".format(config.crossval_set_size),
        "multimargin={}".format(config.multimargin),
        "threads={}".format(config.threads),
        config.num_trials,
        config.num_iterations,
    ]))
    path0 = "results_{}_infos.pickle".format(basename)
    path1 = "results_{}_loss_matrix.txt".format(basename)
    path2 = "results_{}_time_matrix.txt".format(basename)
    path3 = "results_{}_avgloss.svg".format(basename)
    path4 = "results_{}_avgtime.svg".format(basename)
    return path0, path1, path2, path3, path4

def infos_to_matrices(infos):
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

    return loss_matrix, time_matrix

# info = [
#     (2, 1.0, 1.0),
#     (2, 0.5, 1.0),
#     (2, 0.0, 1.0),
# ]
# infos = [info, info]
# print infos
# lm, tm = infos_to_matrices(infos)
# print lm.shape
# print lm
# print tm.shape
# print tm
# quit()

def dump_and_draw(dataset_name, config, infos):
    infos_path, loss_matrix_path, time_matrix_path, loss_svg_path, time_svg_path = \
        get_result_paths(dataset_name, config)

    with open(infos_path, "wb") as fp:
        pickle.dump(infos, fp)

    loss_matrix, time_matrix = infos_to_matrices(infos)
    np.savetxt(loss_matrix_path, loss_matrix)
    np.savetxt(time_matrix_path, time_matrix)

    def ms(x):
        return np.mean(x, axis=0), np.std(x, axis=0).reshape(-1, 1)

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

def precrossvalidate(dataset, config, solver, users):
    """Use the last 20 users for parameter selection."""
    target_users = users[len(users) - 20:]
    assert len(target_users) == 20

    loss_alphas = []
    for alphas in ALL_ALPHAS:

        print "pre-crossvalidating", alphas

        kfold = KFold(len(target_users), n_folds=5)

        losses = []
        for train_set, test_set in kfold:

            for i in train_set:
                infos = setmargin.run(dataset, target_users[i], solver,
                                      config.num_iterations, config.set_size,
                                      alphas, tol=config.tol, debug=config.debug)
                # XXX we only consider the true utility loss at the last
                # iteration
                losses.append(infos[-1][1])

        loss_alphas.append((sum(losses) / len(losses), alphas))

    loss_alphas = sorted(loss_alphas)
    best_alphas = loss_alphas[0][1]

    print "pre-crossvalidation: best alphas = ", alphas
    for loss, alphas in loss_alphas:
        print alphas, ":", loss

    return best_alphas

def solve(dataset, config, ws=None):
    rng = np.random.RandomState(config.seed)

    solver = setmargin.Solver(multimargin=config.multimargin,
                              threads=config.threads, debug=config.debug)

    num_users = config.num_trials if ws is None else ws.shape[0]

    users = []
    for i in range(num_users):
        w = None if ws is None else ws[i].reshape(1, -1)
        user = setmargin.User(dataset.domain_sizes,
                              sampling_mode=config.sampling_mode,
                              ranking_mode=config.ranking_mode,
                              is_deterministic=config.is_deterministic,
                              is_indifferent=config.is_indifferent,
                              w=w,
                              rng=rng)
        print "created user ="
        print user
        users.append(user)

    if config.precrossval:
        alphas = precrossvalidate(dataset, config, solver, users)
    if config.crossval:
        alphas = "auto"
    else:
        alphas = (config.alpha, config.beta, config.gamma)

    infos = []
    for trial in range(config.num_trials):
        print dedent("""\
            ===========
            TRIAL {}/{}
            ===========
            """).format(trial, config.num_trials)

        info = setmargin.run(dataset, users[trial], solver,
                             config.num_iterations, config.set_size, alphas,
                             tol=config.tol,
                             crossval_set_size=config.crossval_set_size,
                             debug=config.debug)
        infos.append(info)
    return infos

def run_synthetic():
    CONFIGS = Grid({
        "num_trials": 20,
        "num_iterations": 100,
        "sampling_mode": ("uniform", "uniform_sparse", "normal", "normal_sparse"),
        "ranking_mode": ("all_pairs",),
        "is_deterministic": False,
        "is_indifferent": True,
        "set_size": range(1, 4+1),
        "precrossval": False,
        "crossval": True,
        "crossval_set_size": 1,
        "multimargin": False,
        "tol": 1e-4,
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
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=str,
                        help="dataset")

    parser.add_argument("-N", "--num_trials", type=int, default=20,
                        help="number of trials")
    parser.add_argument("--domain-sizes", type=str, default="2,2,5",
                        help="domain sizes for the synthetic dataset only")

    parser.add_argument("-n", "--num_iterations", type=int, default=20,
                        help="number of iterations")
    parser.add_argument("-m", "--set-size", type=int, default=3,
                        help="number of hyperplanes/items to solve for")
    parser.add_argument("-t", "--tol", type=float, default=1e-4,
                        help="tolerance used for termination")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
                        help="hyperparameter controlling the importance of slacks")
    parser.add_argument("-b", "--beta", type=float, default=0.1,
                        help="hyperparameter controlling the importance of regularization")
    parser.add_argument("-c", "--gamma", type=float, default=0.1,
                        help="hyperparameter controlling the score of the output items")
    parser.add_argument("-x", "--crossval", action="store_true",
                        help="whether to perform automatic hyperparameter crossvalidation. If enabled, -a -b -c are ignored.")
    parser.add_argument("-X", "--crossval-set-size", type=int, default=None,
                        help="set_size for the hyperparameter crossvalidation.")
    parser.add_argument("-y", "--precrossval", action="store_true",
                        help="do parameter selection using crossvalidation prior to learning")
    parser.add_argument("-M", "--multimargin", action="store_true",
                        help="whether the example and generated object margins should be independent")

    parser.add_argument("-u", "--sampling-mode", type=str, default="uniform",
                        help="utility sampling mode, any of ('uniform', 'normal')")
    parser.add_argument("-r", "--ranking-mode", type=str, default="all_pairs",
                        help="ranking mode for set-wide queries, any of ('all_pairs', 'sorted_pairs')")
    parser.add_argument("-d", "--is-deterministic", action="store_true",
                        help="whether the user answers should be deterministic rather than stochastic")
    parser.add_argument("-i", "--is-indifferent", action="store_true",
                        help="whether the user can (not) be indifferent")

    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    parser.add_argument("--threads", type=int, default=1,
                        help="Max number of threads to user")
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

    dataset_name = args.dataset
    if args.dataset == "synthetic":
        dataset_name += "_" + args.domain_sizes

    infos = solve(dataset, config)
    dump_and_draw(dataset_name, config, infos)

if __name__ == "__main__":
    np.seterr(all="raise")
    if len(sys.argv) == 1:
        run_synthetic()
        run_pc_nocost()
        run_pc()
    else:
        run_from_command_line()
