# -*- coding: utf-8 -*-

import numpy as np
import tempfile
import gurobipy as grb
from gurobipy import GRB
from textwrap import dedent
from util import *

MAX_W_Z = 1

def solve_best_score(dataset, user, debug=False):
    """Returns the highest-scoring item in the dataset accordin to the user.

    The solution is given by:

    .. math::

        \\max_x \\langle w, x \\rangle

    :param dataset: the dataset.
    :param user: the user.
    :returns: a scalar.
    """
    num_features = sum(dataset.domain_sizes)
    w = user.w.ravel()

    model = grb.Model("setmargin_dot")
    xs = [model.addVar(vtype=GRB.BINARY, name="x_{}".format(z)) for z in range(num_features)]
    model.modelSense = GRB.MAXIMIZE
    model.update()
    model.setObjective(grb.quicksum([w[i] * xs[i] for i in range(num_features)]))

    # One-hot constraints
    zs_in_domains = get_zs_in_domains(dataset.domain_sizes)
    for zs_in_domain in zs_in_domains:
        model.addConstr(grb.quicksum([xs[z] for z in zs_in_domain]) == 1)

    # Horn constraints
    if dataset.horn_constraints is not None:
        for head, body in dataset.horn_constraints:
            model.addConstr((1 - xs[head]) + grb.quicksum([xs[atom] for atom in body]) >= 1)

    model.optimize()
    best_score = model.objVal

    if dataset.items is not None and dataset.horn_constraints is None:
        # We have the grounded items
        scores = np.dot(user.w, dataset.items.T)
        debug_best_score = np.max(scores, axis=1)[0]
        if np.abs(best_score - debug_best_score) > 1e-10:
            best_item = np.array([x.x for x in xs])
            debug_best_item = dataset.items[np.argmax(scores, axis=1)].ravel()
            raise RuntimeError(dedent("""\
                best_score sanity check failed!
                best_score = {}
                best_item = {}
                debug best_score = {}
                debug best_item = {}
                """).format(best_score, best_item, debug_best_score,
                            debug_best_item))

    return best_score

def solve(dataset, queries, set_size, alphas,
          multimargin=False, threads=1, debug=False):
    assert dataset.w_constraints is None
    assert dataset.x_constraints is None

    num_examples = len(queries)
    num_features = sum(dataset.domain_sizes)

    model = grb.Model("facility")
    model.params.Threads = threads

    # Declare the variables
    ws, xs = {}, {}
    for i in range(set_size):
        for z in range(num_features):
            ws[i,z] = model.addVar(vtype=GRB.CONTINUOUS, name="w_{}_{}".format(i, z))
            xs[i,z] = model.addVar(vtype=GRB.BINARY, name="x_{}_{}".format(i, z))

    slacks = {}
    for i in range(set_size):
        for k in range(num_examples):
            slacks[i,k] = model.addVar(vtype=GRB.CONTINUOUS, name="slack_{}_{}".format(i, k))

    ps = {}
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                ps[i,j,z] = model.addVar(vtype=GRB.CONTINUOUS, name="p_{}_{}_{}".format(i, j, z))

    margins = [model.addVar(vtype=GRB.CONTINUOUS, name="margin_on_examples")]
    if multimargin:
        margins.append(model.addVar(vtype=GRB.CONTINUOUS, name="margin_on_generated_objects"))

    model.modelSense = GRB.MAXIMIZE
    model.update()

    # Define the objective function
    obj_margins = grb.quicksum(margins)

    obj_slacks = 0
    if len(slacks) > 0:
        alpha = alphas[0] / (set_size * num_examples)
        obj_slacks = alpha * grb.quicksum(slacks.values())

    alpha = alphas[1] / set_size
    obj_weights = alpha * grb.quicksum(ws.values())

    alpha = alphas[2] / set_size
    obj_scores = grb.quicksum([ps[i,i,z]
                               for i in range(set_size)
                               for z in range(num_features)])

    model.setObjective(obj_margins - obj_slacks - obj_weights + obj_scores)

    # Add the various constraints

    # Eq. 9
    for i in range(set_size):
        for k in range(num_examples):
            x1, x2, ans = queries[k]
            assert ans in (-1, 0, 1)

            diff = x1 - x2 if ans >= 0 else x2 - x1
            dot = grb.quicksum([ws[i,z] * diff[z] for z in range(num_features)])

            if ans == 0:
                # Only one of dot and -dot is positive, and the slacks are
                # always positive, so this should work fine as a replacement
                # for abs(dot) <= slacks[i,j]
                model.addConstr(dot <= slacks[i,k])
                model.addConstr(-dot <= slacks[i,k])
            else:
                model.addConstr(dot >= (margins[0] - slacks[i,k]))

    # Eq. 10
    for i in range(set_size):
        for j in range(i) + range(i+1, set_size):
            score_diff = grb.quicksum([ps[i,i,z] - ps[i,j,z] for z in range(num_features)])
            model.addConstr(score_diff >= margins[-1])

    # Eq. 11
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(ps[i,i,z] <= (MAX_W_Z * xs[i,z]))

    # Eq. 12
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(ps[i,i,z] <= ws[i,z])

    # Eq. 13
    for i in range(set_size):
        for j in range(i) + range(i+1, set_size):
            for z in range(num_features):
                model.addConstr(ps[i,j,z] >= (ws[i,z] - 2 * MAX_W_Z * (1 - xs[j,z])))

    # Eq. 15
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(ws[i,z] <= MAX_W_Z)

    # Eq. 16
    # TODO constraints on x

    # Eq. 18a
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                model.addConstr(ps[i,j,z] >= 0)

    # Eq. 18b
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(ws[i,z] >= 0)

    # Eq. 19
    for i in range(set_size):
        for k in range(num_examples):
            model.addConstr(slacks[i,k] >= 0)

    # Eq. 20
    for margin in margins:
        model.addConstr(margin >= 0)
        if set_size == 1 and all(ans == 0 for _, _, ans in queries):
            # XXX work around the fact that if we only have one hyperplane and
            # the user is indifferent to everything we throwed at her, the margin
            # will not appear in any constraint and thus the problem will be
            # unbounded.
            model.addConstr(margin == 0)
    if multimargin:
        if all(ans == 0 for _, _, ans in queries):
            model.addConstr(margins[0] == 0)
        if set_size == 1:
            model.addConstr(margins[1] == 0)

    # One-hot constraints
    zs_in_domains = get_zs_in_domains(dataset.domain_sizes)
    for i in range(set_size):
        for zs_in_domain in zs_in_domains:
            model.addConstr(grb.quicksum([xs[i,z] for z in zs_in_domain]) == 1)

    # Horn constraints
    if dataset.horn_constraints is not None:
        for i in range(set_size):
            for head, body in dataset.horn_constraints:
                model.addConstr((1 - xs[i,head]) + grb.quicksum([xs[i,atom] for atom in body]) >= 1)

    # Dump the problem for later inspection
    if debug:
        fp = tempfile.NamedTemporaryFile(prefix="setmargin_gurobi_", suffix=".lp", delete=False)
        fp.close()
        model.update()
        model.write(fp.name)
        print "dumped gurobi model to '{}'".format(fp.name)

    # Solve
    model.optimize()

    try:
        _ = model.objVal
    except:
        print "the optimization failed"
        raise RuntimeError

    output_ws = np.zeros((set_size, num_features))
    output_xs = np.zeros((set_size, num_features))
    output_scores = np.zeros((set_size, set_size))
    if len(queries):
        output_slacks = np.zeros((set_size, len(queries)))
    else:
        output_slacks = []

    for i in range(set_size):
        for z in range(num_features):
            output_ws[i,z] = ws[i,z].x
            output_xs[i,z] = xs[i,z].x

    for i in range(set_size):
        for k in range(num_examples):
            output_slacks[i,k] = slacks[i,k].x

    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                output_scores[i,j] += ps[i,j,z].x

    return output_ws, output_xs, output_scores, output_slacks, margin.x
