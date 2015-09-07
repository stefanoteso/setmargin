#!/usr/bin/env python2

import os
import numpy as np
from scipy.io import loadmat
from sklearn.utils import check_random_state
import itertools as it
import solver

# TODO: guo synthetic experiment with more features
# TODO: continuous case
# TODO: hybrid case

def get_synthetic_dataset():
    """Builds the synthetic dataset of Guo & Sanner 2010.

    The dataset involves three attributes, with fixed domains sizes; items
    cover all value combinations in the given attributes, for a total of 20
    items.
    """
    domain_sizes = [2, 2, 5]
    items = np.vstack(map(np.array, it.product(*map(range, domain_sizes))))
    assert len(items) == 20
    return domain_sizes, items, np.array([]), np.array([])

def get_pc_dataset():
    raise NotImplementedError

def get_housing_dataset():
    raise NotImplementedError

def query_utility(w, xi, xj, rng=None):
    """Use the indifference-augmented Bradley-Terry model to compute the
    preferences of a user between two items.

    :param w: the utility vector.
    :param xi: attribute vector of object i.
    :param xj: attribute vector of object j.
    :returns: 0 (indifferent), 1 (i wins over j) or -1 (j wins over i).
    """
    rng = check_random_state(rng)

    diff = np.dot(w, xi.T) - np.dot(w, xj.T)

    eq = np.exp(-np.abs(diff))
    gt = np.exp(diff) / (1 + exp(diff))
    lt = np.exp(-diff) / (1 + exp(-diff))

    z = rng.uniform(eq + gt + lt)
    if z < eq:
        return 0
    elif z < (eq + gt):
        return 1
    else:
        return -1

def sample_utility(domain_sizes, mode="uniform", rng=None):
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
        return rng.uniform(1, 100, size=(len(domain_sizes), 1))
    else:
        return rng.normal(50.0, 50.0 / 3, size=(len(domain_sizes), 1))

def run(get_dataset, num_iterations, set_size, alphas, utility_sampling_mode,
        rng=None):

    if not num_iterations > 0:
        raise ValueError("invalid num_iterations '{}'".format(num_iterations))
    if not len(alphas) == 3 or not all([alpha >= 0 for alpha in alphas]):
        raise ValueError("invalid hyperparameters '{}'".format(alphas))

    rng = check_random_state(rng)

    # Retrieve the dataset
    domain_sizes, items, w_constraints, x_constraints = get_dataset()

    print "domain_sizes =", domain_sizes
    print "# of items =", len(items)
    print items

    # Sample the hidden utility function
    hidden_w = sample_utility(domain_sizes, mode=utility_sampling_mode, rng=rng)

    print "hidden_w ="
    print hidden_w

    # Iterate
    queries = None
    for it in range(num_iterations):

        print "==== ITERATION {} ====".format(it)

        # Solve the utility/item learning problem for the current iteration
        ws, xs, scores, margin = \
            solver.solve(items, queries, w_constraints, x_constraints, set_size, alphas)

        print "ws =\n", ws
        print "xs =\n", xs
        print "scores =\n", scores
        print "margin =\n", margin

        # Find the dataset items that are closest to one of the generated items
        # XXX double check if this is the intended approach
        print items

        print xs

        # Ask the user about the retrieved items
        comparison = query_utility(hidden_w, x1, x2)



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
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    args = parser.parse_args()

    if not args.dataset in DATASETS:
        raise ValueError("invalid dataset '{}'".format(args.dataset))

    rng = np.random.RandomState(args.seed)

    run(DATASETS[args.dataset], args.num_iterations, args.set_size,
        (args.alpha, args.beta, args.gamma), args.utility_sampling_mode,
        rng=rng)

if __name__ == "__main__":
    main()
