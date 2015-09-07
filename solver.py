import numpy as np
from util import *

C = 10000
MAX_W_Z = 1

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

def define_constraints(domain_sizes, items, queries,
                       x_constraints, w_constraints, set_size):
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

    constraints.append("\n;; one-hot constraints")
    for i in range(set_size):
        last_z = 0
        for domain_size in domain_sizes:
            zs_in_domain = range(last_z, last_z + domain_size)
            for z1 in zs_in_domain:
                for z2 in zs_in_domain:
                    if z1 != z2:
                        constraints.append("(=> x_{i}_{z1} (not x_{i}_{z2}))".format(i=i, z1=z1, z2=z2))
            last_z += domain_size

    return constraints

def solve(domain_sizes, items, queries, w_constraints, x_constraints,
          set_size, alphas):
    PROBLEM_PATH = "problem.smt2"

    if not set_size > 0:
        raise ValueError("set_size must be positive")
    if len(alphas) != 3:
        raise ValueError("len(alphas) must be 3")
    if any(alpha < 0 for alpha in alphas):
        raise ValueError("all alphas must be non-negative")

    num_features = items.shape[1]

    print "building problem..."

    problem = []
    problem.append("(set-logic QF_LRA)")
    problem.append("(set-option :produce-models true)")
    problem.extend(declare_variables(items, queries, set_size))
    problem.append("(assert (and")
    problem.append(define_objective(items, queries, set_size, alphas))
    problem.extend(define_constraints(domain_sizes, items, queries,
                                      x_constraints, w_constraints, set_size))
    problem.append("))")
    problem.append("(maximize objective)")
    problem.append("(check-sat)")
    problem.append("(set-model -1)")
    problem.append("(get-model)")
    problem.append("(exit)")

    with open(PROBLEM_PATH, "wb") as fp:
        fp.write("\n".join(problem))

    print "solving..."

    solver = OptiMathSAT5(debug=True)
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
