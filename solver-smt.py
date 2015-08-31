#!/usr/bin/env python2

import numpy as np
from util import *

NUM_EXAMPLES = 10
NUM_FEATURES = 5
NUM_PLANES = 2
ALPHAS = (1.0, 1.0, 1.0)

C = 10000
MAX_W_Z = 10000



def get_dataset(num_examples, num_features):
    """Returns the dataset.

    The dataset is composed of a pair of ``np.ndarray``'s of identical shape,
    with one example per row.
    """
    pos_ys = [np.random.randint(0, 2, num_features).astype(np.float64)
              for _ in range(num_examples)]
    neg_ys = [-np.random.randint(0, 2, num_features).astype(np.float64)
              for _ in range(num_examples)]
    return np.array(pos_ys), np.array(neg_ys)

def get_feasibility_constraints():
    """Returns the constraints on the x's."""
    return np.array([])

def get_background_constraints():
    """Returns the constraints on the w's."""
    return np.array([])



def declare_variables(pos_ys, neg_ys, num_planes):
    num_features = pos_ys.shape[1]

    decls = []
    decls.append("(declare-fun cost () Real)")
    decls.append("(declare-fun margin () Real)")
    for i in range(num_planes):
        for z in range(num_features):
            decls.append("(declare-fun w_{i}_{z} () Real)".format(i=i, z=z))
    for i in range(num_planes):
        for z in range(num_features):
            decls.append("(declare-fun x_{i}_{z} () Bool)".format(i=i, z=z))
    for i in range(num_planes):
        for j in range(num_planes):
            for z in range(num_features):
                decls.append("(declare-fun a_{i}_{j}_{z} () Real)".format(i=i, j=j, z=z))
    for k in range(pos_ys.shape[0]):
        decls.append("(declare-fun slack_{k} () Real)".format(k=k))
    return decls

def define_cost(pos_ys, neg_ys, num_planes, alphas):
    num_features = pos_ys.shape[1]

    slacks = [""]
    slacks.extend("slack_{k}".format(k=k) for k in range(pos_ys.shape[0]))

    norms = [""]
    for i in range(num_planes):
        for z in range(num_features):
            w_i_z = "w_{i}_{z}".format(i=i, z=z)
            norms.append("(ite (> {w_i_z} 0) {w_i_z} (- 0 {w_i_z}))".format(w_i_z=w_i_z))

    scores = [""]
    for i in range(num_planes):
        for z in range(num_features):
            scores.append("a_{i}_{i}_{z}".format(i=i, z=z))

    d = {
        "alpha_0": alphas[0],
        "alpha_1": alphas[1],
        "alpha_2": alphas[2],
        "sum_slacks": "(+ {})".format("\n\t\t".join(slacks)),
        "sum_norms": "(+ {})".format("\n\t\t".join(norms)),
        "sum_scores": "(+ {})".format("\n\t\t".join(scores)),
    }
    cost = """
;; Eq. 8
(= cost (+
        margin
        (- 0 (* {alpha_0} {sum_slacks}))
        (- 0 (* {alpha_1} {sum_norms}))
        (* {alpha_2} {sum_scores})
))
""".format(**d)

    return cost

def define_constraints(pos_ys, neg_ys, x_constraints, w_constraints, num_planes):
    num_examples, num_features = pos_ys.shape
    diff_ys = pos_ys - neg_ys

    constraints = []

    constraints.append("\n;; Eq. 9")
    for i in range(num_planes):
        for k in range(num_examples):
            dot = "(+ {})".format(" ".join("(* w_{i}_{z} {diff})".format(i=i, z=z, diff=diff_ys[k,z]) for z in range(num_features)))
            constraints.append("(<= (- margin slack_{k}) {dot})".format(dot=dot, k=k))

    constraints.append("\n;; Eq. 10")
    for i in range(num_planes):
        for j in (range(i) + range(i+1, num_planes)):
            sum_ = "(+ {})".format(" ".join("(- a_{i}_{i}_{z} a_{i}_{j}_{z})".format(i=i, j=j, z=z) for z in range(num_features)))
            constraints.append("(<= margin {sum_})".format(sum_=sum_))

    constraints.append("\n;; Eq. 11")
    for i in range(num_planes):
        for z in range(num_features):
            constraints.append("(<= a_{i}_{i}_{z} (* {max_w_z} (ite x_{i}_{z} 1 0)))".format(i=i, z=z, max_w_z=MAX_W_Z))

    constraints.append("\n;; Eq. 12")
    for i in range(num_planes):
        for z in range(num_features):
            constraints.append("(<= a_{i}_{i}_{z} w_{i}_{z})".format(i=i, z=z))

    constraints.append("\n;; Eq. 13")
    for i in range(num_planes):
        for j in (range(i) + range(i+1, num_planes)):
            for z in range(num_features):
                constraints.append("(<= (- w_{i}_{z} (* {c} (ite x_{i}_{z} (- 0 1) 0))) a_{i}_{i}_{z})".format(i=i, j=j, z=z, c=C))

    constraints.append("\n;; Eq. 14")
    # XXX (same as eq 18)

    constraints.append("\n;; Eq. 15")
    constraints.append(";; TODO: constraints on w")

    constraints.append("\n;; Eq. 16")
    constraints.append(";; TODO: constraints on x")

    constraints.append("\n;; Eq. 18")
    for i in range(num_planes):
        for j in range(num_planes):
            for z in range(num_features):
                constraints.append("(>= a_{i}_{j}_{z} 0)".format(i=i, j=j, z=z))

    constraints.append("\n;; Eq. 19")
    for k in range(num_examples):
        constraints.append("(>= slack_{k} 0)".format(k=k))

    constraints.append("\n;; Eq. 20")
    constraints.append("(>= margin 0)")

    return constraints

def solve(pos_ys, neg_ys, x_constraints, w_constraints, num_planes, alphas):
    PROBLEM_PATH = "problem.smt2"

    assert pos_ys.shape == neg_ys.shape
    assert num_planes > 0
    assert len(alphas) == 3 and all(alpha > 0 for alpha in alphas)

    print "building problem..."
    problem = []
    problem.append("(set-logic QF_LRA)")
    problem.append("(set-option :produce-models true)")
    problem.extend(declare_variables(pos_ys, neg_ys, num_planes))
    problem.append("(assert (and")
    problem.append(define_cost(pos_ys, neg_ys, num_planes, alphas))
    problem.extend(define_constraints(pos_ys, neg_ys, x_constraints, w_constraints, num_planes))
    problem.append("))")
    problem.append("(minimize cost)")
    problem.append("(check-sat)")
    problem.append("(set-model -1)")
    problem.append("(get-model)")
    problem.append("(exit)")

    with open(PROBLEM_PATH, "wb") as fp:
        fp.write("\n".join(problem))

    solver = OptiMathSAT5(debug=True)

    print "solving..."
    model = solver.optimize(PROBLEM_PATH)

    ws = np.zeros((num_planes, pos_ys.shape[1]))
    xs = np.zeros((num_planes, pos_ys.shape[1]))
    for variable, assignment in model.iteritems():
        if variable[0] == "w":
            _, i, j = variable.split("_")
            ws[i,j] = assignment
        elif variable[0] == "x":
            _, i, j = variable.split("_")
            xs[i,j] = {"true":1, "false":0}[assignment] # they are Booleans

    return ws, xs

def main():
    np.random.seed(0)

    pos_ys, neg_ys = get_dataset(NUM_EXAMPLES, NUM_FEATURES)
    assert pos_ys.shape == neg_ys.shape

    x_constraints = get_feasibility_constraints()
    w_constraints = get_background_constraints()

    ws, xs = solve(pos_ys, neg_ys, x_constraints, w_constraints,
                   NUM_PLANES, ALPHAS)

    print "ws ="
    print ws
    print "xs ="
    print xs

if __name__ == "__main__":
    main()
