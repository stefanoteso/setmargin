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

def onehot(domain_size, value):
    assert 0 <= value < domain_size
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
    raise NotImplementedError

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
        rng, deterministic_answers=False):

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
    hidden_w = sample_utility(domain_sizes, rng, mode=utility_sampling_mode)

    print "hidden_w ="
    print hidden_w

    # Iterate
    queries = []
    for it in range(num_iterations):

        print "==== ITERATION {} ====".format(it)

        # Solve the utility/item learning problem for the current iteration
        ws, xs, scores, margin = \
            solver.solve(domain_sizes, items, queries, w_constraints, x_constraints,
                         set_size, alphas)

        print "ws =\n", ws
        print "xs =\n", xs
        print "scores =\n", scores
        print "margin =\n", margin

        # Find the dataset items with highest score wrt each hyperplanes
        # XXX double check if this is the intended approach
        best_is = np.argmax(np.dot(ws, items.T), axis=1)
        assert best_is.shape == (set_size,)
        best_items = items[best_is]

        print "best_is, best_items =\n", zip(best_is, best_items)

        # Ask the user about the retrieved items
        queries.extend(query_utility(hidden_w, item1, item2, rng, deterministic=deterministic_answers)
                       for item2 in best_items
                       for item1 in best_items
                       if not (item1 == item2).all())

        print "queries =\n", "\n".join(map(str, queries))

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
    parser.add_argument("-d", "--deterministic", action="store_true",
                        help="whether the user answers should be deterministic rather than stochastic [default: False]")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    args = parser.parse_args()

    if not args.dataset in DATASETS:
        raise ValueError("invalid dataset '{}'".format(args.dataset))

    rng = np.random.RandomState(args.seed)

    run(DATASETS[args.dataset], args.num_iterations, args.set_size,
        (args.alpha, args.beta, args.gamma), args.utility_sampling_mode,
        rng, deterministic_answers=args.deterministic)

if __name__ == "__main__":
    main()