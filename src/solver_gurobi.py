# -*- coding: utf-8 -*-

import numpy as np
import gurobipy as grb
from gurobipy import GRB
from util import *

MAX_W_Z = 1

def solve(domain_sizes, queries, w_constraints, x_constraints,
          set_size, alphas, debug=False):

    num_examples = len(queries)
    num_features = sum(domain_sizes)

    model = grb.Model("facility")

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

    margin = model.addVar(vtype=GRB.CONTINUOUS, name="margin")

    model.modelSense = GRB.MAXIMIZE
    model.update()

    # Define the objective function
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

    model.setObjective(margin - obj_slacks - obj_weights + obj_scores)

    # Add the various constraints

    # Eq. 9
    for i in range(set_size):
        for k in range(num_examples):
            x1, x2, ans = queries[k]
            assert ans in (-1, 0, 1)

            diff = x1 - x2 if ans >= 0 else x2 - x1
            dot = grb.quicksum([w[i,z] * diff[z] for z in range(num_features)])

            if ans == 0:
                model.addConstr(abs(dot) <= slacks[i,j])
            else:
                model.addConstr(dot >= (margin - slacks[i,k]))

    # Eq. 10
    for i in range(set_size):
        for j in range(i) + range(i+1, set_size):
            score_diff = grb.quicksum([ps[i,i,z] - ps[i,j,z] for z in range(num_features)])
            model.addConstr(score_diff >= margin)

    # Eq. 11
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(a[i,i,z] <= (MAX_W_Z * xs[i,z]))

    # Eq. 12
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(a[i,i,z] <= ws[i,z])

    # Eq. 13
    for i in range(set_size):
        for j in range(i) + range(i+1, set_size):
            for z in range(num_features):
                model.addConstr(a[i,j,z] >= (w[i,z] - MAX_W_Z * (1 - x[j,z])))

    # Eq. 15
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(w[i,z] <= MAX_W_Z)

    # Eq. 16
    # TODO constraints on x

    # Eq. 18a
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_features):
                model.addConstr(a[i,j,z] >= 0)

    # Eq. 18b
    for i in range(set_size):
        for z in range(num_features):
            model.addConstr(w[i,z] >= 0)

    # Eq. 19
    for i in range(set_size):
        for k in range(num_examples):
            model.addConstr(slack[i,k] >= 0)

    # One-hot constraints
    for i in range(set_size):
        last_z = 0
        for domain_size in domain_sizes:
            assert domain_size > 1
            zs_in_domain = range(last_z, last_z + domain_size)
            model.addConstr(grb.quicksum([x[i,z] for z in zs_in_domains]) == 1)
            last_z += domain_size

    # Solve
    model.optimize()

    print "cost =", model.objVal

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
                output_scores[i,j,z] = scores[i,j,z].x

    return output_ws, output_xs, output_scores, output_slacks, margin.x
