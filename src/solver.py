# -*- coding: utf-8 -*-

import numpy as np
from util import *

MAX_W_Z = 1

def declare_variables(num_features, queries, set_size):
    num_examples = len(queries)

    decls = []
    decls.append("(declare-fun objective () Real)")
    decls.append("(declare-fun margin () Real)")
    for i in range(set_size):
        for z in range(num_features):
            decls.append("(declare-fun w_{i}_{z} () Real)".format(i=i, z=z))
    for i in range(set_size):
        for z in range(num_features):
            decls.append("(declare-fun x_{i}_{z} () Bool)".format(i=i, z=z))
    for i in range(set_size):
        for k in range(num_examples):
            decls.append("(declare-fun slack_{i}_{k} () Real)".format(i=i, k=k))
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                decls.append("(declare-fun a_{i}_{j}_{z} () Real)".format(i=i, j=j, z=z))
    return decls

def define_objective(num_features, queries, set_size, alphas):

    slacks = ["slack_{i}_{k}".format(i=i, k=k)
              for i in range(set_size) for k in range(len(queries))]
    if len(slacks) == 0:
        obj_slacks = ""
    else:
        if len(slacks) == 1:
            sum_slacks = slacks[0]
        else:
            sum_slacks = "(+ {})".format("\n\t\t".join(slacks))
        obj_slacks = "(- 0 (* {alpha} {sum_slacks}))".format(alpha=float2libsmt(alphas[0] / (set_size * len(queries))),
                                                             sum_slacks=sum_slacks)

    weights = ["w_{i}_{z}".format(i=i, z=z) for i in range(set_size) for z in range(num_features)]
    sum_weights = "(+ {})".format("\n\t\t".join(weights))
    obj_weights = "(- 0 (* {beta} {sum_weights}))".format(beta=float2libsmt(alphas[1] / set_size),
                                                          sum_weights=sum_weights)

    scores = ["a_{i}_{i}_{z}".format(i=i, z=z) for i in range(set_size) for z in range(num_features)]
    sum_scores = "(+ {})".format("\n\t\t".join(scores))
    obj_scores = "(* {gamma} {sum_scores})".format(gamma=float2libsmt(alphas[2] / set_size),
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

def define_constraints(domain_sizes, num_features, queries,
                       x_constraints, w_constraints, set_size):
    num_examples = len(queries)

    constraints = []

    constraints.append("\n;; Eq. 9")
    for i in range(set_size):
        for k in range(num_examples):
            x1, x2, ans = queries[k]
            assert ans in (-1, 0, 1)

            diff = x1 - x2 if ans >= 0 else x2 - x1
            summands = ["(* w_{i}_{z} {diff})".format(i=i, z=z, diff=float2libsmt(diff[z]))
                        for z in range(num_features)]
            dot = "(+ {})".format(" ".join(summands))

            if ans == 0:
                constraint = "(<= (ite (>= {dot} 0) {dot} (- 0 {dot})) slack_{i}_{k})".format(dot=dot, i=i, k=k)
            else:
                constraint = "(>= {dot} (- margin slack_{i}_{k}))".format(dot=dot, i=i, k=k)

            constraints.append(";; -- plane {i}, example {k}".format(i=i, k=k))
            constraints.append(constraint)

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
                constraints.append("(>= a_{i}_{j}_{z} (- w_{i}_{z} (* {c} (ite x_{j}_{z} 0 1))))".format(i=i, j=j, z=z, c=MAX_W_Z))

    constraints.append("\n;; Eq. 15")
    for i in range(set_size):
        for z in range(num_features):
            constraints.append("(<= w_{i}_{z} {max_w_z})".format(i=i, z=z, max_w_z=MAX_W_Z))

    constraints.append("\n;; Eq. 16")
    constraints.append(";; TODO: constraints on x")

    constraints.append("\n;; Eq. 18a")
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                constraints.append("(>= a_{i}_{j}_{z} 0)".format(i=i, j=j, z=z))

    constraints.append("\n;; Eq. 18b")
    for i in range(set_size):
        for z in range(num_features):
            constraints.append("(>= w_{i}_{z} 0)".format(i=i, z=z))

    constraints.append("\n;; Eq. 19")
    for i in range(set_size):
        for k in range(num_examples):
            constraints.append("(>= slack_{i}_{k} 0)".format(i=i, k=k))

    constraints.append("\n;; Eq. 20")
    constraints.append("(>= margin 0)")

    constraints.append("\n;; one-hot constraints")
    for i in range(set_size):
        last_z = 0
        for domain_size in domain_sizes:
            assert domain_size > 1
            zs_in_domain = range(last_z, last_z + domain_size)
            for z1 in zs_in_domain:
                for z2 in zs_in_domain:
                    if z1 != z2:
                        constraints.append("(=> x_{i}_{z1} (not x_{i}_{z2}))".format(i=i, z1=z1, z2=z2))
            xs_in_domain = ["x_{i}_{z}".format(i=i, z=z)
                            for z in zs_in_domain]
            constraints.append("(or {})".format(" ".join(xs_in_domain)))
            last_z += domain_size

    return constraints

def solve(domain_sizes, queries, w_constraints, x_constraints,
          set_size, alphas, debug=False):

    num_features = sum(domain_sizes)

    if any(query[0].shape[0] != num_features for query in queries) or \
       any(query[1].shape[0] != num_features for query in queries):
        raise ValueError("domain_sizes and query items shape mismatch")
    if not set_size > 0:
        raise ValueError("set_size must be positive")
    if len(alphas) != 3:
        raise ValueError("len(alphas) must be 3")
    if any(alpha < 0 for alpha in alphas):
        raise ValueError("all alphas must be non-negative")

    if debug:
        print "building problem..."

    problem = []
    problem.append("(set-logic QF_LRA)")
    problem.append("(set-option :produce-models true)")
    problem.extend(declare_variables(num_features, queries, set_size))
    problem.append("(assert (and")
    problem.append(define_objective(num_features, queries, set_size, alphas))
    problem.extend(define_constraints(domain_sizes, num_features, queries,
                                      x_constraints, w_constraints, set_size))
    problem.append("))")
    problem.append("(maximize objective)")
    problem.append("(check-sat)")
    problem.append("(set-model -1)")
    problem.append("(get-model)")
    problem.append("(exit)")

    if debug:
        print "solving..."

    solver = OptiMathSAT5(debug=debug)
    model, _, _, _ = solver.optimize("\n".join(problem))

    ws = np.zeros((set_size, num_features))
    xs = np.zeros((set_size, num_features))
    scores = np.zeros((set_size, set_size))
    for variable, assignment in sorted(model.iteritems()):
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

class _TestSolver(object):

    def _solve(self, domain_sizes, queries, set_size, alphas):
        return solve(domain_sizes, queries,
                     np.array([]), np.array([]),
                     set_size, alphas, debug=True)

    def onehot(self, queries):
        from util import onehot
    
        def vonehot(vector):
            return np.hstack([onehot(2, x) for x in vector])

        return [(vonehot(query[0]), vonehot(query[1]), 1) for query in queries]

    def test_sanity(self):

        QUERIES = [
            (np.array([1, 1, 0, 0, 0]), np.array([0, 0, 1, 0, 0])),
            (np.array([0, 0, 0, 0, 1]), np.array([0, 1, 0, 1, 0])),
            (np.array([1, 0, 0, 0, 1]), np.array([0, 1, 0, 0, 0])),
        ]

        EXPECTED_WS = np.array([
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
        ])

        EXPECTED_XS = np.array([
            [0, 1,  0, 1,  1, 0,  0, 1,  0, 1], # [1, 1, 0, 1, 1]
            [0, 1,  0, 1,  0, 1,  1, 0,  0, 1], # [1, 1, 1, 0, 1]
            [0, 1,  1, 0,  1, 0,  1, 0,  1, 0], # [1, 0, 0, 0, 0]
        ])

        SET_SIZE, ALPHAS = 3, (0.1, 0.1, 0.1)

        onehot_queries = self.onehot(QUERIES)
        ws, xs, scores, margin = self._solve([2] * 5, onehot_queries, SET_SIZE, ALPHAS)

        assert (ws == EXPECTED_WS).all()
        assert (xs == EXPECTED_XS).all()