#!/usr/bin/env python2

import os
import numpy as np
from scipy.io import loadmat
from sklearn.utils import check_random_state
import itertools as it
from util import *

# TODO: guo synthetic experiment with more features
# TODO: continuous case
# TODO: hybrid case

C = 10000
MAX_W_Z = 1

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

def declare_variables(items, queries, set_size):
    num_examples = 0 if queries is None else queries.shape[0]
    num_features = items.shape[1]

    decls = []
    decls.append("(declare-fun objective () Real)")
    decls.append("(declare-fun margin () Real)")
    for i in range(set_size):
        for z in range(num_features):
            decls.append("(declare-fun w_{i}_{z} () Real)".format(i=i, z=z))
            decls.append("(declare-fun x_{i}_{z} () Bool)".format(i=i, z=z))
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                decls.append("(declare-fun a_{i}_{j}_{z} () Real)".format(i=i, j=j, z=z))
    for k in range(num_examples):
        decls.append("(declare-fun slack_{k} () Real)".format(k=k))
    return decls

def define_objective(items, queries, set_size, alphas):
    num_examples = queries.shape[0] if not queries is None else 0
    num_features = items.shape[1]

    slacks = ["slack_{k}".format(k=k) for k in range(num_examples)]
    sum_slacks = "(+ {})".format("\n\t\t".join(slacks))
    if num_examples:
        obj_slacks = "(- 0 (* {alpha} {sum_slacks}))".format(alpha=float2libsmt(alphas[0]),
                                                             sum_slacks=sum_slacks)
    else:
        obj_slacks = ""

    weights = ["w_{i}_{z}".format(i=i, z=z) for i in range(set_size) for z in range(num_features)]
    sum_weights = "(+ {})".format("\n\n\t".join(weights))
    obj_weights = "(- 0 (* {beta} {sum_weights}))".format(beta=float2libsmt(alphas[1]),
                                                          sum_weights=sum_weights)

    scores = ["a_{i}_{i}_{z}".format(i=i, z=z) for i in range(set_size) for z in range(num_features)]
    sum_scores = "(+ {})".format("\n\n\t".join(scores))
    obj_scores = "(* {gamma} {sum_scores})".format(gamma=float2libsmt(alphas[2]),
                                                   sum_scores=sum_scores)

    objective = """
;; Eq. 8
(= objective
    (+
        margin
        {obj_slacks}
        {obj_weights}
        {obj_scores}
    )
)
""".format(obj_slacks=obj_slacks, obj_weights=obj_weights, obj_scores=obj_scores)

    return objective

def define_constraints(items, queries, x_constraints, w_constraints, set_size):
    num_examples = queries.shape[0] if not queries is None else 0
    num_features = items.shape[1]

    constraints = []

    constraints.append("\n;; Eq. 9")
    if not queries is None:
        for i in range(set_size):
            for k in range(num_examples):
                dot = "(+ {})".format(" ".join("(* w_{i}_{z} {diff})".format(i=i, z=z, diff=float2libsmt(queries[k,z,0] - queries[k,z,1]))
                                               for z in range(num_features)))
                constraints.append("(>= {dot} (- margin slack_{k}))".format(dot=dot, k=k))

    constraints.append("\n;; Eq. 10")
    for i in range(set_size):
        for j in range(i) + range(i+1, set_size):
            sum_ = "(+ {})".format(" ".join("(- a_{i}_{i}_{z} a_{i}_{j}_{z})".format(i=i, j=j, z=z)
                                            for z in range(num_features)))
            constraints.append("(>= {sum_} margin)".format(sum_=sum_))

    constraints.append("\n;; Eq. 11")
    for i in range(set_size):
        for z in range(num_features):
            constraints.append("(<= a_{i}_{i}_{z} (* {max_w_z} (ite x_{i}_{z} 1 0)))".format(i=i, z=z, max_w_z=MAX_W_Z))

    constraints.append("\n;; Eq. 12")
    for i in range(set_size):
        for z in range(num_features):
            constraints.append("(<= a_{i}_{i}_{z} w_{i}_{z})".format(i=i, z=z))

    constraints.append("\n;; Eq. 13")
    for i in range(set_size):
        for j in range(i) + range(i+1, set_size):
            for z in range(num_features):
                constraints.append("(>= a_{i}_{j}_{z} (- w_{i}_{z} (* {c} (ite x_{j}_{z} 0 1))))".format(i=i, j=j, z=z, c=C))

    constraints.append("\n;; Eq. 15")
    for i in range(set_size):
        for z in range(num_features):
            constraints.append("(<= w_{i}_{z} {max_w_z})".format(i=i, z=z, max_w_z=MAX_W_Z))

    constraints.append("\n;; Eq. 16")
    constraints.append(";; TODO: constraints on x")

    constraints.append("\n;; Eq. 18")
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                constraints.append("(>= a_{i}_{j}_{z} 0)".format(i=i, j=j, z=z))

    constraints.append("\n;; Eq. 19")
    for k in range(num_examples):
        constraints.append("(>= slack_{k} 0)".format(k=k))

    constraints.append("\n;; Eq. 20")
    constraints.append("(>= margin 0)")

    return constraints

def solve(items, queries, w_constraints, x_constraints, set_size, alphas):
    PROBLEM_PATH = "problem.smt2"

    assert set_size > 0
    assert len(alphas) == 3 and all(alpha >= 0 for alpha in alphas)

    num_features = items.shape[1]

    print "building problem..."
    problem = []
    problem.append("(set-logic QF_LRA)")
    problem.append("(set-option :produce-models true)")
    problem.extend(declare_variables(items, queries, set_size))
    problem.append("(assert (and")
    problem.append(define_objective(items, queries, set_size, alphas))
    problem.extend(define_constraints(items, queries, x_constraints,
                                      w_constraints, set_size))
    problem.append("))")
    problem.append("(maximize objective)")
    problem.append("(check-sat)")
    problem.append("(set-model -1)")
    problem.append("(get-model)")
    problem.append("(exit)")

    with open(PROBLEM_PATH, "wb") as fp:
        fp.write("\n".join(problem))

    solver = OptiMathSAT5(debug=True)

    print "solving..."
    model = solver.optimize(PROBLEM_PATH)

    ws = np.zeros((set_size, num_features))
    xs = np.zeros((set_size, num_features))
    scores = np.zeros((set_size, set_size))
    for variable, assignment in sorted(model.iteritems()):
        print variable, "=", assignment
        if variable.startswith("w_"):
            _, i, j = variable.split("_")
            ws[i,j] = assignment
        elif variable.startswith("x_"):
            _, i, j = variable.split("_")
            xs[i,j] = {"true":1, "false":0}[assignment] # they are Booleans
        elif variable.startswith("a_"):
            _, i, j, z = variable.split("_")
            scores[i,j] += assignment
        elif variable == "margin":
            margin = assignment

    return ws, xs, scores, margin

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
            solve(items, queries, w_constraints, x_constraints, set_size, alphas)

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
