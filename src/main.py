#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, time
import itertools as it
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

from solver import Solver
from user import User
from util import *
from datasets import *

def print_queries(queries, hidden_w):
    for xi, xj, sign in queries:
        relation = {-1:"<", 0:"~", 1:">"}[sign]
        score_xi = np.dot(hidden_w, xi.T)[0]
        score_xj = np.dot(hidden_w, xj.T)[0]
        print "  {} ({:6.3f}) {} ({:6.3f}) {} -- diff {:6.3f}".format(xi, score_xi, relation, score_xj, xj, score_xi - score_xj)
    print

def quicksort(user, xs, answers={}):
    lt, eq, gt = [], [], []
    if len(xs) > 1:
        pivot = xs[0]
        eq.append(pivot)
        for x in xs[1:]:
            try:
                ans = answers[(tuple(x), tuple(pivot))]
            except KeyError:
                ans = user.query(x, pivot)
                answers[(tuple(x), tuple(pivot))] = ans
            if ans < 0:
                lt.append(x)
            elif ans == 0:
                eq.append(x)
            else:
                gt.append(x)
        assert len(lt) < len(xs)
        assert len(gt) < len(xs)

        sorted_lt, _ = quicksort(user, lt, answers=answers)
        sorted_gt, _ = quicksort(user, gt, answers=answers)
        return [l for l in sorted_lt + [eq] + sorted_gt if len(l)], answers
    else:
        return [xs], answers

def update_queries(user, ws, xs, old_best_item, rng, ranking_mode="all_pairs"):
    """Computes the queries to ask the user for the given inputs.

    If there is only one candidate best item, then only one query is returned,
    namely a query comparing the current best item with the best item at the
    previous iteration.

    If there are multiple candidate best items, then multiple queries are
    returned, one for each pair of candidate best items.

    :param user: the user.
    :param ws: the estimated user preference(s) at the current iteration.
    :param xs: the estimated best item(s) at the current iteration.
    :param old_best_item: the estimated best item at the previous iteration.
    :param ranking_mode: either ``"all_pairs"`` or ``"sorted_pairs"``.
    :returns: a pair (list of new queries, 
    """
    num_items, num_features = xs.shape
    if num_items == 1:
        if old_best_item is None:
            old_best_item = rng.random_integers(0, 1, size=(num_features,))
        queries = [(xs[0], old_best_item, user.query(xs[0], old_best_item))]
        num_queries = 1
    elif ranking_mode == "all_pairs":
        # requires 1/2 * n * (n - 1) queries
        # XXX note that in the non-deterministic setting we may actually lose
        # information by only querying for ~half the pairs!
        queries = [(xi, xj, user.query(xi, xj))
                   for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs)) if i < j]
        num_queries = len(queries)
    elif ranking_mode == "sorted_pairs":
        sorted_sets, answers = quicksort(user, xs)
        num_queries = len(answers)
        assert num_queries > 0

        queries = []
        for (k, set_k), (l, set_l) in it.product(enumerate(sorted_sets), enumerate(sorted_sets)):
            if k > l:
                continue
            for xi, xj in it.product(set_k, set_l):
                queries.append((xi, xj, 0 if k == l else -1))
    else:
        raise ValueError("invalid ranking_mode '{}'".format(ranking_mode))
    return queries, num_queries

def run(dataset, user, solver, num_iterations, set_size, rng,
        ranking_mode="all_pairs", multimargin=False, threads=1, debug=False):

    if not num_iterations > 0:
        raise ValueError("invalid num_iterations '{}'".format(num_iterations))

    rng = check_random_state(rng)

    # Find the dataset item with the highest score wrt the hidden hyperlpane
    best_hidden_score, _ = solver.compute_best_score(dataset, user)

    # Iterate
    queries, old_best_item = [], None
    losses, times = [], []
    for t in range(num_iterations):

        if debug:
            print "\n\n\n==== ITERATION {} ====\n".format(t)
            print "input queries ="
            print_queries(queries, user.w)

        old_time = time.time()

        # Solve the utility/item learning problem for the current iteration
        ws, xs, scores, slacks, margin = \
            solver.compute_setmargin(dataset, queries, set_size)
        debug_scores = np.dot(ws, xs.T)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the m-item case:\n{}".format(ws)

        if debug:
            print "set_size=n ws =\n", ws
            print "set_size=n xs =\n", xs
            print "set_size=n scores =\n", scores
            print "set_size=n slacks =\n", slacks
            print "set_size=n margin =", margin

        assert all(dataset.is_item_valid(x) for x in xs)

        if (np.abs(scores - debug_scores) >= 1e-10).any():
            print "Warning: solver and debug scores mismatch:\n" \
                  "scores =\n{}\n" \
                  "debug scores =\n{}\n".format(scores, debug_scores)

        # Ask the user about the retrieved items
        new_queries, num_queries = \
            update_queries(user, ws, xs, old_best_item, rng,
                           ranking_mode=ranking_mode)
        assert num_queries > 0

        queries.extend(new_queries)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        if debug:
            print "\nupdated queries ="
            print_queries(queries, user.w)

        elapsed = time.time() - old_time
        times.extend([elapsed / num_queries] * num_queries)

        # Compute the utility loss between the best item that we would
        # recommend given the queries collected so far and the best
        # recommendation according to the hidden user hyperplane
        ws, xs, scores, slacks, margin = \
            solver.compute_setmargin(dataset, queries, 1)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the 1-item case:\n{}".format(ws)

        utility_loss = best_hidden_score - np.dot(user.w, xs[0])
        losses.extend([utility_loss / np.linalg.norm(user.w)] * num_queries)

        if debug:
            print "set_size=1 ws =", ws, "user's w =", user.w
            print "set_size=1 xs =", xs
            print "set_size=1 scores =", scores
            print "set_size=1 slacks =", slacks
            print "set_size=1 margin =", margin

            print "u(x) =", np.dot(user.w, xs[0])
            print "utility_loss =", utility_loss

        assert dataset.is_item_valid(xs[0])

    return losses, times

def dump_performance(basename, num_trials, losses, times):
    # Since distinct trials may have incurred a different number of queries
    # each, here we resize the performance data to be uniformly shaped
    max_queries = max(len(ls) for ls in losses)

    loss_matrix = np.zeros((num_trials, max_queries))
    time_matrix = np.zeros((num_trials, max_queries))
    for i, (ls, ts) in enumerate(zip(losses, times)):
        assert ls.shape == ts.shape
        loss_matrix[i,:ls.shape[0]] = ls
        time_matrix[i,:ts.shape[0]] = ts

    np.savetxt("results_{}_time_matrix.txt".format(basename), time_matrix)
    np.savetxt("results_{}_loss_matrix.txt".format(basename), loss_matrix)

    loss_means = np.mean(loss_matrix, axis=0)
    loss_stddevs = np.std(loss_matrix, ddof=1, axis=0).reshape(-1, 1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Avgerage loss over {} trials".format(num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average loss")
    ax.set_ylim([0.0, max(0.5, max(loss_means) + max(loss_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), loss_means, yerr=loss_stddevs)
    fig.savefig("results_{}_avgloss.svg".format(basename), bbox_inches="tight")

    time_means = np.mean(time_matrix, axis=0)
    time_stddevs = np.std(time_matrix, ddof=1, axis=0).reshape(-1, 1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Average time over {} trials".format(num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average time")
    ax.set_ylim([0.0, max(1.0, max(time_means) + max(time_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), time_means, yerr=time_stddevs)
    fig.savefig("results_{}_avgtime.svg".format(basename), bbox_inches="tight")

def main():
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

    print "running {num_trials} x {num_iterations} iterations on {dataset}\
           [threads={threads} seed={seed}]".format(**argsdict)

    rng = np.random.RandomState(args.seed)

    domain_sizes = map(int, [ds for ds in args.domain_sizes.split(",") if len(ds)])
    if args.dataset == "synthetic":
        dataset = SyntheticDataset(domain_sizes)
    elif args.dataset == "constsynthetic":
        dataset = RandomDataset(domain_sizes, rng=rng)
    elif args.dataset == "pc":
        dataset = PCDataset()
    else:
        raise ValueError("invalid dataset.")
    if args.debug:
        print dataset

    solver = Solver((args.alpha, args.beta, args.gamma),
                    multimargin=args.multimargin, threads=args.threads,
                    debug=args.debug)

    losses, times = [], []
    for i in range(args.num_trials):
        print "==== TRIAL {} ====".format(i)

        user = User(dataset.domain_sizes, sampling_mode=args.sampling_mode,
                    is_deterministic=args.is_deterministic,
                    is_indifferent=args.is_indifferent,
                    rng=rng)
        if args.debug:
            print "user =\n", user

        losses_for_trial, times_for_trial = \
            run(dataset, user, solver, args.num_iterations, args.set_size, rng,
                ranking_mode=args.ranking_mode,
                multimargin=args.multimargin,
                threads=args.threads, debug=args.debug)

        losses.append(np.array(losses_for_trial).ravel())
        times.append(np.array(times_for_trial))

    hyperparams = [
        "dataset", "num_trials", "num_iterations", "set_size", "alpha", "beta",
        "gamma", "ranking_mode", "multimargin", "sampling_mode",
        "is_deterministic", "is_indifferent", "seed"
    ]
    if args.dataset in ("synthetic", "constsynthetic"):
        hyperparams.insert(1, "domain_sizes")
    dump_performance("_".join(str(argsdict[h]) for h in hyperparams),
                     args.num_trials, losses, times)

if __name__ == "__main__":
    main()
