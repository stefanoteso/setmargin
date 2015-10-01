#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import itertools as it
from datasets import *
import solver_omt
import solver_gurobi

def sample_utility(domain_sizes, rng, mode="uniform"):
    """Samples a utility weight vector.

    .. note::

        The computation is taken from p. 293 of the Guo & Sanner paper.

    .. warning:::

        I am not sure if this is what Guo & Sanner actually do!

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

def query_utility(w, xi, xj, rng, deterministic=False):
    """Use the indifference-augmented Bradley-Terry model to compute the
    preferences of a user between two items.

    :param w: the utility vector.
    :param xi: attribute vector of object i.
    :param xj: attribute vector of object j.
    :returns: 0 (indifferent), 1 (i wins over j) or -1 (j wins over i).
    """
    rng = check_random_state(rng)

    diff = np.dot(w, xi.T - xj.T)

    if deterministic:
        result = (xi, xj, int(np.sign(diff)))
    else:
        eq = np.exp(-np.abs(diff))
        gt = np.exp(diff) / (1 + np.exp(diff))
        lt = np.exp(-diff) / (1 + np.exp(-diff))

        z = rng.uniform(0, eq + gt + lt)

        if z < eq:
            ans = 0
        elif z < (eq + gt):
            ans = 1
        else:
            ans = -1
        result = (xi, xj, ans)

    return result

def print_queries(queries, hidden_w):
    for xi, xj, sign in queries:
        relation = {-1:"<", 0:"~", 1:">"}[sign]
        score_xi = np.dot(hidden_w, xi.T)[0]
        score_xj = np.dot(hidden_w, xj.T)[0]
        print "  {} ({:6.3f}) {} ({:6.3f}) {} -- diff {:6.3f}".format(xi, score_xi, relation, score_xj, xj, score_xi - score_xj)
    print

def update_queries(hidden_w, ws, xs, old_best_item, rng, deterministic=False):
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
    :returns: a pair (list of new queries, 
    """
    num_items, num_features = xs.shape
    if num_items == 1:
        if old_best_item is None:
            old_best_item = rng.random_integers(0, 1, size=(num_features,))
        queries = [query_utility(hidden_w, xs[0], old_best_item, rng,
                                 deterministic=deterministic)]
    else:
        queries = [query_utility(hidden_w, xi, xj, rng, deterministic=deterministic)
                   for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs)) if i < j]
    return queries

def run(get_dataset, num_iterations, set_size, alphas, utility_sampling_mode,
        rng, deterministic=False, solver_name="optimathsat", debug=False):

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
                         set_size, alphas, debug=debug)
        debug_scores = np.dot(ws, xs.T)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the m-item case:\n{}".format(ws)

        times.append(time.time() - old_time)

        if debug:
            print "set_size=n ws =\n", ws
            print "set_size=n xs =\n", xs
            print "set_size=n scores =\n", scores
            print "set_size=n slacks =\n", slacks
            print "set_size=n margin =", margin

        if solver_name != "gurobi":
            # XXX somehow gurobi fails to satisfy this assertion...
            assert (np.abs(scores - debug_scores) < 1e-6).all(), "solver scores and debug scores mismatch:\n{}\n{}".format(scores, debug_scores)

        # Ask the user about the retrieved items
        new_queries = update_queries(hidden_w, ws, xs,
                                     old_best_item, rng,
                                     deterministic=deterministic)
        queries.extend(new_queries)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        if debug:
            print "\nupdated queries ="
            print_queries(queries, hidden_w)

        # Compute the utility loss between the best item that we would
        # recommend given the queries collected so far and the best
        # recommendation according to the hidden user hyperplane
        ws, xs, scores, slacks, margin = \
            solver.solve(domain_sizes, queries,
                         w_constraints, x_constraints,
                         1, alphas, debug=debug)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the 1-item case:\n{}".format(ws)

        utility_loss = best_hidden_score - np.dot(hidden_w, xs[0])
        avg_losses.append(utility_loss / np.linalg.norm(hidden_w))

        if debug:
            print "set_size=1 ws =", ws, "hidden_w =", hidden_w
            print "set_size=1 xs =", xs
            print "set_size=1 scores =", scores
            print "set_size=1 slacks =", slacks
            print "set_size=1 margin =", margin

            print "u(x) =", np.dot(hidden_w, xs[0])
            print "utility_loss =", utility_loss

    return avg_losses, times

def main():
    import argparse as ap

    DATASETS = {
        "synthetic": get_synthetic_dataset,
        "pc": get_pc_dataset,
        "housing": get_housing_dataset,
    }

    parser = ap.ArgumentParser(description="setmargin experiment")
    parser.add_argument("dataset", type=str,
                        help="dataset, any of {}".format(DATASETS.keys()))
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
    parser.add_argument("-d", "--deterministic", action="store_true",
                        help="whether the user answers should be deterministic rather than stochastic (default: False)")
    parser.add_argument("-S", "--solver", type=str, default="optimathsat",
                        help="solver to use (default: 'optimathsat')")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed (default: None)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug spew")
    args = parser.parse_args()

    argsdict = vars(args)
    argsdict["dataset"] = args.dataset

    basename = "{dataset}_{num_trials}_{num_iterations}_{set_size}_{alpha}_{beta}_{gamma}_{utility_sampling_mode}_{deterministic}_{seed}".format(**argsdict)

    if not args.dataset in DATASETS:
        raise ValueError("invalid dataset '{}'".format(args.dataset))

    print "running {} trials on {}, {} iterations per trial, seed is {}".format(args.num_trials, args.dataset, args.num_iterations, args.seed)

    rng = np.random.RandomState(args.seed)

    avg_losses, times = [], []
    for i in range(args.num_trials):

        print "==== TRIAL {} ====".format(i)

        ls, ts = run(DATASETS[args.dataset], args.num_iterations,
                     args.set_size, (args.alpha, args.beta, args.gamma),
                     args.utility_sampling_mode, rng,
                     deterministic=args.deterministic,
                     solver_name=args.solver,
                     debug=args.debug)

        avg_losses.extend(ls)
        times.extend(ts)

    avg_losses, times = np.array(avg_losses), np.array(times)

    if args.debug:
        print "average losses:"
        print avg_losses

    print "results for {} trials:".format(args.num_trials)
    print "maximum likelihood mean/std loss per iteration =", np.mean(avg_losses), "±", np.std(avg_losses, ddof=1)
    print "maximum likelihood mean/std of time per iteration =", np.mean(times), "±", np.std(times, ddof=1)

    data = avg_losses.reshape(args.num_trials, -1)
    means, stddevs = np.mean(data, axis=0), np.std(data, ddof=1, axis=0)

    np.savetxt("results_{}_times.txt".format(basename), times)
    np.savetxt("results_{}_losses.txt".format(basename), avg_losses)
    np.savetxt("results_{}_avgloss_means.txt".format(basename), means)
    np.savetxt("results_{}_avgloss_stddevs.txt".format(basename), stddevs)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(np.arange(1, args.num_iterations + 1), means, yerr=stddevs.reshape(-1,1))
    ax.set_title("Avg. loss over {} trials".format(args.num_trials))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Average loss")
    fig.savefig("results_{}_avgloss.svg".format(basename), bbox_inches="tight")

if __name__ == "__main__":
    main()
