#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import itertools as it
from datasets import *
from util import *
import solver_omt
import solver_gurobi

def sample_utility(domain_sizes, rng, mode="uniform"):
    """Samples a utility weight vector.

    .. note::

        The computation is taken from p. 293 of the Guo & Sanner paper.

    :param domains: list of attribute domains (that is, integer intervals).
    :param mode: either ``"uniform"`` or ``"normal"``.
    :returns: a row vector with as many components as attributes.
    """
    assert mode in ("uniform", "normal")
    rng = check_random_state(rng)
    if mode == "uniform":
        return rng.uniform(0, 1, size=(sum(domain_sizes), 1)).reshape(1,-1)
    else:
        return rng.normal(0.25, 0.25 / 3, size=(sum(domain_sizes), 1)).reshape(1,-1)

def query_utility(w, xi, xj, rng, deterministic=False, no_indifference=False):
    """Use the indifference-augmented Bradley-Terry model to compute the
    preferences of a user between two items.

    :param w: the utility vector.
    :param xi: attribute vector of object i.
    :param xj: attribute vector of object j.
    :param deterministic: WRITEME.
    :param no_indifference: WRITEME.
    :returns: 0 (indifferent), 1 (i wins over j) or -1 (j wins over i).
    """
    # The original problem has weights sampled in the range [0, 100], and
    # uses ALPHA=1, BETA=1; here however we have weights in the range [0, 1]
    # so we must rescale ALPHA and BETA to obtain the same probabilities. 
    ALPHA, BETA = 100, 100

    rng = check_random_state(rng)

    diff = np.dot(w, xi.T - xj.T)

    if deterministic:
        ans = int(np.sign(diff))
    else:
        eq = np.exp(-BETA * np.abs(diff))
        if no_indifference:
            eq = 0.0
        gt = np.exp(ALPHA * diff) / (1 + np.exp(ALPHA * diff))
        lt = np.exp(-ALPHA * diff) / (1 + np.exp(-ALPHA * diff))

        z = rng.uniform(0, eq + gt + lt)

        if z < eq:
            ans = 0
        elif z < (eq + gt):
            ans = 1
        else:
            ans = -1

    return (xi, xj, ans)

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

def quicksort(w, xs, rng, deterministic, no_indifference, answers={}):
    lt, eq, gt = [], [], []
    if len(xs) > 1:
        pivot, num_queries = xs[0], 0
        eq.append(pivot)
        for x in xs[1:]:
            try:
                ans = answers[(tuple(x), tuple(pivot))]
            except KeyError:
                _, _, ans = query_utility(w, x, pivot, rng,
                                          deterministic=deterministic,
                                          no_indifference=no_indifference)
                num_queries += 1
                answers[(tuple(x), tuple(pivot))] = ans
            if ans < 0:
                lt.append(x)
            elif ans == 0:
                eq.append(x)
            else:
                gt.append(x)
        assert len(lt) < len(xs)
        assert len(gt) < len(xs)
        sorted_lt, num_queries_lt, _ = \
            quicksort(w, lt, rng, deterministic, no_indifference, answers=answers)
        sorted_gt, num_queries_gt, _ = \
            quicksort(w, gt, rng, deterministic, no_indifference, answers=answers)

        return sorted_lt + eq + sorted_gt, \
               num_queries_lt + num_queries + num_queries_gt, \
               answers
    else:
        return xs, 0, answers

def update_queries(hidden_w, ws, xs, old_best_item, rng, deterministic=False,
                   no_indifference=False, ranking_mode="all_pairs"):
    """Computes the queries to ask the user for the given inputs.

    If there is only one candidate best item, then only one query is returned,
    namely a query comparing the current best item with the best item at the
    previous iteration.

    If there are multiple candidate best items, then multiple queries are
    returned, one for each pair of candidate best items.

    :param hidden_w: the hidden user preferences.
    :param ws: the estimated user preference(s) at the current iteration.
    :param xs: the estimated best item(s) at the current iteration.
    :param old_best_item: the estimated best item at the previous iteration.
    :param rng: an RNG object.
    :param deterministic: whether the user answers should be deterministic.
    :param no_indifference: disable 'whatever' answers.
    :param ranking_mode: either ``"all_pairs"`` or ``"sorted_pairs"``.
    :returns: a pair (list of new queries, 
    """
    num_items, num_features = xs.shape
    if num_items == 1:
        if old_best_item is None:
            old_best_item = rng.random_integers(0, 1, size=(num_features,))
        queries = [query_utility(hidden_w, xs[0], old_best_item, rng,
                                 deterministic=deterministic,
                                 no_indifference=no_indifference)]
        num_queries = 1
    elif ranking_mode == "all_pairs":
        # requires 1/2 * n * (n - 1) queries
        # XXX note that in the non-deterministic setting we may actually lose
        # information by only querying for ~half the pairs!
        queries = [query_utility(hidden_w, xi, xj, rng,
                                 deterministic=deterministic,
                                 no_indifference=no_indifference)
                   for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs)) if i < j]
        num_queries = len(queries)
    elif ranking_mode == "sorted_pairs":
        # requires n*log(n) queries
        # XXX note that in the non-deterministic setting we may actually lose
        # information by only querying for a subset of the pairs!
        sorted_xs, num_queries, answers = \
            quicksort(hidden_w, xs, rng, deterministic, no_indifference)
        sorted_xs = np.array(sorted_xs)
        assert xs.shape == sorted_xs.shape
        assert num_queries > 0
        queries = [(xi, xj, 1) for (i, xi), (j, xj)
                   in it.product(enumerate(sorted_xs), enumerate(sorted_xs)) if i < j]
    else:
        raise ValueError("invalid ranking_mode '{}'".format(ranking_mode))
    return queries, num_queries

def run(get_dataset, num_iterations, set_size, alphas, utility_sampling_mode,
        rng, ranking_mode="all_pairs", deterministic=False,
        no_indifference=False, multimargin=False, solver_name="optimathsat",
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

    # Retrieve the dataset
    domain_sizes, items, w_constraints, x_constraints = get_dataset()
    assert sum(domain_sizes) == items.shape[1]

    if debug:
        print "domain_sizes =", domain_sizes, "num_features =", sum(domain_sizes)
        print "# of items =", len(items)
        print items

    # Sample the hidden utility function
    hidden_w = sample_utility(domain_sizes, rng, mode=utility_sampling_mode)

    if debug:
        print "hidden_w ="
        print hidden_w

    # Find the dataset item with the highest score wrt the hidden hyperlpane
    # TODO use the optimizer to find the highest scoring configuration
    best_hidden_score = np.max(np.dot(hidden_w, items.T), axis=1)

    # Iterate
    queries, old_best_item = [], None
    avg_losses, times = [], []
    for t in range(num_iterations):

        if debug:
            print "\n\n\n==== ITERATION {} ====\n".format(t)
            print "input queries ="
            print_queries(queries, hidden_w)

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
            update_queries(hidden_w, ws, xs, old_best_item, rng,
                           ranking_mode=ranking_mode,
                           deterministic=deterministic,
                           no_indifference=no_indifference)
        assert num_queries > 0

        queries.extend(new_queries)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        if debug:
            print "\nupdated queries ="
            print_queries(queries, hidden_w)

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

        utility_loss = best_hidden_score - np.dot(hidden_w, xs[0])
        avg_losses.extend([utility_loss / np.linalg.norm(hidden_w)] * num_queries)

        if debug:
            print "set_size=1 ws =", ws, "hidden_w =", hidden_w
            print "set_size=1 xs =", xs
            print "set_size=1 scores =", scores
            print "set_size=1 slacks =", slacks
            print "set_size=1 margin =", margin

            print "u(x) =", np.dot(hidden_w, xs[0])
            print "utility_loss =", utility_loss

        assert is_onehot(domain_sizes, 1, xs), "xs are not in onehot format"

        # If the user is fully satisfied, we are done
        if utility_loss < 1e-6:
            break

    return avg_losses, times

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
    parser.add_argument("-u", "--utility_sampling_mode", type=str, default="uniform",
                        help="utility sampling mode, any of ('uniform', 'normal') (default: 'uniform')")
    parser.add_argument("-r", "--ranking-mode", type=str, default="all_pairs",
                        help="ranking mode, any of ('all_pairs', 'sorted_pairs') (default: 'all_pairs')")
    parser.add_argument("-d", "--deterministic", action="store_true",
                        help="whether the user answers should be deterministic rather than stochastic (default: False)")
    parser.add_argument("--no-indifference", action="store_true",
                        help="whether the user can (not) be indifferent (default: False)")
    parser.add_argument("-M", "--multimargin", action="store_true",
                        help="whether the example and generated object margins should be independent (default: False)")
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

    hyperparams = [
        "dataset", "num_trials", "num_iterations", "set_size", "alpha", "beta",
        "gamma", "utility_sampling_mode", "deterministic", "no_indifference",
        "multimargin", "seed"
    ]
    if args.dataset == "synthetic":
        hyperparams.insert(1, "domain_sizes")

    basename = "_".join(str(argsdict[h]) for h in hyperparams)

    print "running {num_trials} trials on {dataset}, {num_iterations} iterations per trial, seed is {seed}" \
            .format(**argsdict)

    rng = np.random.RandomState(args.seed)

    losses, times = [], []
    for i in range(args.num_trials):
        print "==== TRIAL {} ====".format(i)

        losses_for_trial, times_for_trial = \
            run(datasets[args.dataset], args.num_iterations,
                args.set_size, (args.alpha, args.beta, args.gamma),
                args.utility_sampling_mode, rng,
                ranking_mode=args.ranking_mode,
                deterministic=args.deterministic,
                no_indifference=args.no_indifference,
                multimargin=args.multimargin,
                solver_name=args.solver,
                debug=args.debug)

        losses.append(np.array(losses_for_trial).ravel())
        times.append(np.array(times_for_trial))

    # Since distinct trials may have incurred a different number of queries
    # each, here we resize the performance data to be uniformly shaped
    max_queries = max(len(ls) for ls in losses)
    loss_matrix = np.zeros((args.num_trials, max_queries))
    time_matrix = np.zeros((args.num_trials, max_queries))
    for i, (ls, ts) in enumerate(zip(losses, times)):
        assert ls.shape == ts.shape
        loss_matrix[i,:ls.shape[0]] = ls
        time_matrix[i,:ts.shape[0]] = ts

    np.savetxt("results_{}_time_matrix.txt".format(basename), time_matrix)
    np.savetxt("results_{}_loss_matrix.txt".format(basename), loss_matrix)

    loss_means = np.mean(loss_matrix, axis=0)
    loss_stddevs = np.std(loss_matrix, ddof=1, axis=0).reshape(-1, 1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Avgerage loss over {} trials".format(args.num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average loss")
    ax.set_ylim([0.0, max(0.5, max(loss_means) + max(loss_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), loss_means, yerr=loss_stddevs)
    fig.savefig("results_{}_avgloss.svg".format(basename), bbox_inches="tight")

    time_means = np.mean(time_matrix, axis=0)
    time_stddevs = np.std(time_matrix, ddof=1, axis=0).reshape(-1, 1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Average time over {} trials".format(args.num_trials))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Average time")
    ax.set_ylim([0.0, max(1.0, max(time_means) + max(time_stddevs) + 0.1)])
    ax.errorbar(np.arange(1, max_queries + 1), time_means, yerr=time_stddevs)
    fig.savefig("results_{}_avgtime.svg".format(basename), bbox_inches="tight")

if __name__ == "__main__":
    main()
