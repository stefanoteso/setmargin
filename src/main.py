#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import itertools as it
from datasets import *
from util import *
from user import User
import solver_omt
import solver_gurobi

def print_queries(queries, hidden_w):
    for xi, xj, sign in queries:
        relation = {-1:"<", 0:"~", 1:">"}[sign]
        score_xi = np.dot(hidden_w, xi.T)[0]
        score_xj = np.dot(hidden_w, xj.T)[0]
        print "  {} ({:6.3f}) {} ({:6.3f}) {} -- diff {:6.3f}".format(xi, score_xi, relation, score_xj, xj, score_xi - score_xj)
    print

def is_onehot(domain_sizes, set_size, xs):
    zs_in_domains = get_zs_in_domains(domain_sizes)
    for i in range(set_size):
        for zs_in_domain in zs_in_domains:
            if sum(xs[i,z] for z in zs_in_domain) != 1:
                return False
    return True

def quicksort(user, xs, answers={}):
    lt, eq, gt = [], [], []
    if len(xs) > 1:
        pivot = xs[0], 0
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

        sorted_lt, _ = quicksort(user, lt, rng, answers=answers)
        sorted_gt, _ = quicksort(user, gt, rng, answers=answers)
        return sorted_lt + eq + sorted_gt, answers
    else:
        return xs, answers

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
        # requires n*log(n) queries
        # XXX note that in the non-deterministic setting we may actually lose
        # information by only querying for a subset of the pairs!
        sorted_xs, answers = \
            quicksort(user, xs, rng, deterministic, no_indifference)
        num_queries = len(answers)
        sorted_xs = np.array(sorted_xs)
        assert xs.shape == sorted_xs.shape
        assert num_queries > 0
        queries = [(xi, xj, 1) for (i, xi), (j, xj)
                   in it.product(enumerate(sorted_xs), enumerate(sorted_xs)) if i < j]
    else:
        raise ValueError("invalid ranking_mode '{}'".format(ranking_mode))
    return queries, num_queries

def run(domain_sizes, items, w_constraints, x_constraints, num_iterations,
        set_size, alphas, user, rng,
        ranking_mode="all_pairs", multimargin=False, solver_name="optimathsat",
        debug=False):

    if not num_iterations > 0:
        raise ValueError("invalid num_iterations '{}'".format(num_iterations))
    if not len(alphas) == 3 or not all([alpha >= 0 for alpha in alphas]):
        raise ValueError("invalid hyperparameters '{}'".format(alphas))

    if solver_name == "optimathsat":
        solver = solver_omt
    elif solver_name == "gurobi":
        solver = solver_gurobi
    else:
        raise ValueError("invalid solver '{}'".format(solver_name))

    rng = check_random_state(rng)

    # Find the dataset item with the highest score wrt the hidden hyperlpane
    # TODO use the optimizer to find the highest scoring configuration
    best_hidden_score = np.max(np.dot(user.w, items.T), axis=1)

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
            solver.solve(domain_sizes, queries,
                         w_constraints, x_constraints,
                         set_size, alphas, multimargin=multimargin,
                         debug=debug)
        debug_scores = np.dot(ws, xs.T)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the m-item case:\n{}".format(ws)

        if debug:
            print "set_size=n ws =\n", ws
            print "set_size=n xs =\n", xs
            print "set_size=n scores =\n", scores
            print "set_size=n slacks =\n", slacks
            print "set_size=n margin =", margin

        assert is_onehot(domain_sizes, set_size, xs), "xs are not in onehot format"
        if solver_name != "gurobi":
            # XXX somehow gurobi fails to satisfy this assertion...
            assert (np.abs(scores - debug_scores) < 1e-6).all(), "solver scores and debug scores mismatch:\n{}\n{}".format(scores, debug_scores)

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
            solver.solve(domain_sizes, queries,
                         w_constraints, x_constraints,
                         1, alphas, multimargin=multimargin,
                         debug=debug)
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

        assert is_onehot(domain_sizes, 1, xs), "xs are not in onehot format"

        # If the user is fully satisfied, we are done
        if utility_loss < 1e-6:
            break

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
    parser.add_argument("--is-indifferent", action="store_true",
                        help="whether the user can (not) be indifferent (default: False)")
    parser.add_argument("-S", "--solver", type=str, default="gurobi",
                        help="solver to use (default: 'gurobi')")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed (default: None)")
    parser.add_argument("--domain-sizes", type=str, default="2,2,5",
                        help="domain sizes for the synthetic dataset only (default: 2,2,5)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug spew")
    args = parser.parse_args()

    domain_sizes = map(int, [ds for ds in args.domain_sizes.split(",") if len(ds)])
    datasets = {
        "synthetic": lambda: get_synthetic_dataset(domain_sizes=domain_sizes),
        "pc": get_pc_dataset,
        "housing": get_housing_dataset,
    }
    if not args.dataset in datasets:
        raise ValueError("invalid dataset '{}'".format(args.dataset))

    argsdict = vars(args)
    argsdict["dataset"] = args.dataset

    print "running {num_trials} trials on {dataset}, " \
          "{num_iterations} iterations per trial, seed is {seed}" \
            .format(**argsdict)

    rng = np.random.RandomState(args.seed)

    # Retrieve the dataset
    domain_sizes, items, w_constraints, x_constraints = datasets[args.dataset]()
    assert sum(domain_sizes) == items.shape[1]

    if args.debug:
        print "domain_sizes =", domain_sizes, "num_features =", sum(domain_sizes)
        print "# of items =", len(items)
        print items

    losses, times = [], []
    for i in range(args.num_trials):
        print "==== TRIAL {} ====".format(i)

        user = User(domain_sizes, sampling_mode=args.sampling_mode,
                    is_deterministic=args.is_deterministic,
                    is_indifferent=args.is_indifferent,
                    rng=rng)
        if args.debug:
            print "user =\n", user

        losses_for_trial, times_for_trial = \
            run(domain_sizes, items, w_constraints, x_constraints,
                args.num_iterations, args.set_size,
                (args.alpha, args.beta, args.gamma), user, rng,
                ranking_mode=args.ranking_mode,
                multimargin=args.multimargin,
                solver_name=args.solver,
                debug=args.debug)

        losses.append(np.array(losses_for_trial).ravel())
        times.append(np.array(times_for_trial))

    hyperparams = [
        "dataset", "num_trials", "num_iterations", "set_size", "alpha", "beta",
        "gamma", "multimargin", "sampling_mode", "is_deterministic",
        "is_indifferent", "seed"
    ]
    if args.dataset == "synthetic":
        hyperparams.insert(1, "domain_sizes")
    dump_performance("_".join(str(argsdict[h]) for h in hyperparams),
                     args.num_trials, losses, times)

if __name__ == "__main__":
    main()
