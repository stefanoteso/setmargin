#!/usr/bin/env python2

import numpy as np
from util import *

# TODO: debug against matlab version (requires gurobi)
# TODO: iterative version with user feedback
# TODO: guo synthetic experiment with more features
# TODO: continuous case
# TODO: hybrid case

SET_SIZE = 3
ALPHAS = (0.1, 0.1, 0.1)

C = 10000
MAX_W_Z = 1



def get_simple_dataset():
    pos_ys = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
    ])
    neg_ys = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
    ])
    return pos_ys, neg_ys

def get_feasibility_constraints():
    """Returns the constraints on the x's."""
    return np.array([])

def get_background_constraints():
    """Returns the constraints on the w's."""
    return np.array([])



def declare_variables(pos_ys, neg_ys, set_size):
    num_examples, num_features = pos_ys.shape

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
    for k in range(pos_ys.shape[0]):
        decls.append("(declare-fun slack_{k} () Real)".format(k=k))
    return decls

def define_objective(pos_ys, neg_ys, set_size, alphas):
    num_examples, num_features = pos_ys.shape

    slacks = [""]
    slacks.extend("slack_{k}".format(k=k) for k in range(num_examples))

    weights = [""]
    weights.extend("w_{i}_{z}".format(i=i, z=z)
                   for i in range(set_size) for z in range(num_features))

    scores = [""]
    scores.extend("a_{i}_{i}_{z}".format(i=i, z=z)
                  for i in range(set_size) for z in range(num_features))

    d = {
        "alpha_0": float2libsmt(alphas[0]),
        "alpha_1": float2libsmt(alphas[1]),
        "alpha_2": float2libsmt(alphas[2]),
        "sum_slacks": "(+ {})".format("\n\t\t".join(slacks)),
        "sum_weights": "(+ {})".format("\n\t\t".join(weights)),
        "sum_scores": "(+ {})".format("\n\t\t".join(scores)),
    }
    objective = """
;; Eq. 8
(= objective
    (+
        margin
        (- 0 (* {alpha_0} {sum_slacks}))
        (- 0 (* {alpha_1} {sum_weights}))
        (* {alpha_2} {sum_scores})
    )
)
""".format(**d)

    return objective

def define_constraints(pos_ys, neg_ys, x_constraints, w_constraints, set_size):
    num_examples, num_features = pos_ys.shape

    constraints = []

    constraints.append("\n;; Eq. 9")
    diff_ys = pos_ys - neg_ys
    for i in range(set_size):
        for k in range(num_examples):
            dot = "(+ {})".format(" ".join("(* w_{i}_{z} {diff})".format(i=i, z=z, diff=float2libsmt(diff_ys[k,z]))
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

def solve(pos_ys, neg_ys, x_constraints, w_constraints, set_size, alphas):
    PROBLEM_PATH = "problem.smt2"

    assert pos_ys.shape == neg_ys.shape
    assert set_size > 0
    assert len(alphas) == 3 and all(alpha >= 0 for alpha in alphas)

    print "building problem..."
    problem = []
    problem.append("(set-logic QF_LRA)")
    problem.append("(set-option :produce-models true)")
    problem.extend(declare_variables(pos_ys, neg_ys, set_size))
    problem.append("(assert (and")
    problem.append(define_objective(pos_ys, neg_ys, set_size, alphas))
    problem.extend(define_constraints(pos_ys, neg_ys, x_constraints, w_constraints, set_size))
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

    ws = np.zeros((set_size, pos_ys.shape[1]))
    xs = np.zeros((set_size, pos_ys.shape[1]))
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

def main():
    np.random.seed(0)

    pos_ys, neg_ys = get_simple_dataset()
    assert pos_ys.shape == neg_ys.shape

    x_constraints = get_feasibility_constraints()
    w_constraints = get_background_constraints()

    ws, xs, scores, margin = \
        solve(pos_ys, neg_ys, x_constraints, w_constraints, SET_SIZE, ALPHAS)

    print "ws ="
    print ws
    print "xs ="
    print xs
    print "scores ="
    print scores
    print "margin ="
    print margin

if __name__ == "__main__":
    main()
