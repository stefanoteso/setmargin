#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import itertools as it
import solver

# TODO: guo synthetic experiment with more features
# TODO: continuous case
# TODO: hybrid case

def onehot(domain_size, value):
    assert 0 <= value < domain_size, "out-of-bounds value: 0/{}/{}".format(value, domain_size)
    value_onehot = np.zeros(domain_size, dtype=np.int8)
    value_onehot[value] = 1
    return value_onehot

def get_synthetic_dataset():
    """Builds the synthetic dataset of Guo & Sanner 2010.

    The dataset involves three attributes, with fixed domains sizes; items
    cover all value combinations in the given attributes, for a total of 20
    items.
    """
    domain_sizes = [2, 2, 5]
    items_onehot = None
    for item in it.product(*map(range, domain_sizes)):
        item_onehot = np.hstack((onehot(domain_sizes[i], attribute_value)
                                 for i, attribute_value in enumerate(item)))
        if items_onehot is None:
            items_onehot = item_onehot
        else:
            items_onehot = np.vstack((items_onehot, item_onehot))
    assert items_onehot.shape == (20, 2+2+5)
    return domain_sizes, items_onehot, np.array([]), np.array([])

def get_pc_dataset():
    raise NotImplementedError

def get_housing_dataset():
    from scipy.io import loadmat

    # See AISTATS_Housing_Uniform/readXML.m
    CATEGORICAL_FEATURES = [i-1 for i in [4, 9]]
    SCALAR_FEATURES = [i-1 for i in [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14]]
    INTERVAL = 10.0

    items = loadmat("houseMatlab.mat")["housing"]
    assert items.shape == (506, 14)

    num_items, num_features = items.shape

    attribute_info = [None for _ in range(num_features)]
    for z in CATEGORICAL_FEATURES:
        attribute_info[z] = sorted(set(items[:,z]))
    for z in SCALAR_FEATURES:
        l = min(items[:,z])
        u = max(items[:,z])
        # XXX the 1e-6 is there because MATLAB's arange notation is inclusive
        # wrt the upper bound
        attribute_info[z] = np.arange(l-(u-l)/INTERVAL, u+(u-l)/INTERVAL+1e-6, (u-l)/i)

    discretized_items = np.zeros(items.shape, dtype=np.int32)
    for i, item in enumerate(items):
        for z, value in enumerate(item):
            if z in CATEGORICAL_FEATURES:
                discretized_items[i,z] = attribute_info[z].index(value)
            else:
                temp = value - attribute_info[z]
                discretized_items[i,z] = np.where(temp <= 0)[0][0]

    domain_sizes = []
    for z in range(num_features):
        domain_sizes.append(max(discretized_items[:,z]) - min(discretized_items[:,z]) + 1)
        print max(discretized_items[:,z]), min(discretized_items[:,z]), "-->", domain_sizes[z]

    items_onehot = None
    for item in discretized_items:
        item_onehot = np.hstack((onehot(domain_sizes[z], attribute_value - min(discretized_items[:,z]))
                                 for z, attribute_value in enumerate(item)))
        if items_onehot is None:
            items_onehot = item_onehot
        else:
            items_onehot = np.vstack((items_onehot, item_onehot))
    assert items_onehot.shape == (506, sum(domain_sizes))

    return domain_sizes, items_onehot, np.array([]), np.array([])

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
        return rng.uniform(1, 100, size=(sum(domain_sizes), 1)).reshape(1,-1)
    else:
        return rng.normal(50.0, 50.0 / 3, size=(sum(domain_sizes), 1)).reshape(1,-1)

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
        return (xi, xj, int(np.sign(diff)))
    else:
        eq = np.exp(-np.abs(diff))
        gt = np.exp(diff) / (1 + np.exp(diff))
        lt = np.exp(-diff) / (1 + np.exp(-diff))

        z = rng.uniform(eq + gt + lt)
        if z < eq:
            ans = 0
        elif z < (eq + gt):
            ans = 1
        else:
            ans = -1
        return (xi, xj, ans)

def run(get_dataset, num_iterations, set_size, alphas, utility_sampling_mode,
        rng, deterministic_answers=False, debug=False):

    if not num_iterations > 0:
        raise ValueError("invalid num_iterations '{}'".format(num_iterations))
    if not len(alphas) == 3 or not all([alpha >= 0 for alpha in alphas]):
        raise ValueError("invalid hyperparameters '{}'".format(alphas))

    rng = check_random_state(rng)

    # Retrieve the dataset
    domain_sizes, items, w_constraints, x_constraints = get_dataset()

    if debug:
        print "domain_sizes =", domain_sizes
        print "# of items =", len(items)
        print items

    # Sample the hidden utility function
    hidden_w = sample_utility(domain_sizes, rng, mode=utility_sampling_mode)

    if debug:
        print "hidden_w ="
        print hidden_w

    # Find the dataset item with the highest score wrt the latent hyperlpane
    best_latent_score = np.max(np.dot(hidden_w, items.T), axis=1)

    # Iterate
    queries, avg_losses, times = [], [], []
    for t in range(num_iterations):

        if debug:
            print "\n\n\n==== ITERATION {} ====\n".format(t)
            print "queries =\n"
            for query in queries:
                print query

        old_time = time.time()

        # Solve the utility/item learning problem for the current iteration
        ws, xs, scores, margin = \
            solver.solve(domain_sizes, items, queries, w_constraints, x_constraints,
                         set_size, alphas, debug=debug)

        if debug:
            print "ws =\n", ws
            print "xs =\n", xs
            print "scores =\n", scores
            print "margin =\n", margin

        # Find the dataset items with highest score wrt each hyperplanes
        # XXX double check if this is the intended approach
        scores = np.dot(ws, items.T)
        best_scores, best_is = np.max(scores, axis=1), np.argmax(scores, axis=1)
        assert best_is.shape == (set_size,)
        best_items = items[best_is]

        if debug:
            print "best_is, best_items =\n", zip(best_is, best_items)

        times.append(time.time() - old_time)

        # Compute the loss between the best item and the picked items (in the
        # dataset, not the ones generated by the solver)
        avg_losses.append(best_latent_score - np.mean(best_scores))

        # Ask the user about the retrieved items
        for item1, item2 in it.product(best_items, best_items):
            if (item1 == item2).all():
                continue
            queries.append(query_utility(hidden_w, item1, item2, rng,
                           deterministic=deterministic_answers))

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
                        help="number of trials")
    parser.add_argument("-n", "--num_iterations", type=int, default=10,
                        help="number of iterations")
    parser.add_argument("-m", "--set-size", type=int, default=3,
                        help="number of hyperplanes/items to solve for [default: 3]")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
                        help="hyperparameter controlling the importance of slacks [default: 0.1]")
    parser.add_argument("-b", "--beta", type=float, default=0.1,
                        help="hyperparameter controlling the importance of regularization [default: 0.1]")
    parser.add_argument("-c", "--gamma", type=float, default=0.1,
                        help="hyperparameter controlling the score of the output items [default: 0.1]")
    parser.add_argument("-u", "--utility_sampling_mode", type=str, default="uniform",
                        help="utility sampling mode, any of ('uniform', 'normal') [default: 'uniform']")
    parser.add_argument("-d", "--deterministic", action="store_true",
                        help="whether the user answers should be deterministic rather than stochastic [default: False]")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug spew")
    args = parser.parse_args()

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
                     deterministic_answers=args.deterministic,
                     debug=args.debug)

        avg_losses.extend(ls)
        times.extend(ts)

    avg_losses, times = np.array(avg_losses), np.array(times)

    print avg_losses

    print "results for {} trials:".format(args.num_trials)
    print "maximum likelihood mean/std loss per iteration =", np.mean(avg_losses), "±", np.std(avg_losses, ddof=1)
    print "maximum likelihood mean/std of time per iteration =", np.mean(times), "±", np.std(times, ddof=1)

    data = avg_losses.reshape(args.num_trials, -1)
    means, stddevs = np.mean(data, axis=0), np.std(data, ddof=1, axis=0)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(np.arange(1, args.num_iterations + 1), means)
    ax.set_title("Avg. loss over {} trials".format(args.num_trials))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Average loss")
    fig.savefig("avgloss.svg", bbox_inches="tight")

if __name__ == "__main__":
    main()
