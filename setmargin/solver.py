# -*- coding: utf-8 -*-

import numpy as np
import tempfile
import gurobipy as grb
from gurobipy import GRB
from textwrap import dedent
from util import *

MAX_W_Z = 1
ENABLE_LP_DUMP = False

status_to_reason = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
}

def gudot(x, z):
    """
    :param x: a list of scalars or Gurobi expressions.
    :param z: a list of scalars or Gurobi expressions.
    :returns: a Gurobi expression.
    """
    assert len(x) == len(z)
    return grb.quicksum([x[i] * z[i] for i in range(len(x))])

def gubilinear(x, a, z):
    """
    :param x: a list of scalars or Gurobi expressions.
    :param a: a numpy array.
    :param z: a list of scalars or Gurobi expressions.
    :returns: a Gurobi expression.
    """
    assert len(x) == a.shape[0]
    assert len(z) == a.shape[1]
    q = [gudot(a[i], z) for i in range(len(x))]
    return gudot(x, q)

class Solver(object):
    """Set-margin solver based on the Gurobi MILP solver.

    .. todo::

        Cache the model and update the constraints on demand before solving
        again.

    :param multimargin: whether to use two distinct margins. (default: ``False``)
    :param threads: how many threads Gurobi is allowed to use. (default: ``None``)
    :param debug: whether to spew debug output. (default: ``False``)
    """
    def __init__(self, multimargin=False, threads=None, debug=False):
        self._multimargin = multimargin
        self._threads = threads
        self._debug = debug

    def _dump_model(self, model, prefix):
        if self._debug and ENABLE_LP_DUMP:
            fp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".lp",
                                             delete=False)
            fp.close()
            print "dumping model to", fp.name
            model.write(fp.name)

    def _add_item_constraints(self, model, dataset, x):
        """Add one-hot and other item constraints."""
        zs_in_domains = get_zs_in_domains(dataset.domain_sizes)
        for zs_in_domain in zs_in_domains:
            model.addConstr(grb.quicksum([x[z] for z in zs_in_domain]) == 1)

        if dataset.x_constraints is not None:
            for body, head in dataset.x_constraints:
                model.addConstr((1 - x[body]) + grb.quicksum([x[atom] for atom in head]) >= 1)

    def _compose_item(self, dataset, x):
        """
        :param dataset:
        :param x: the Boolean part of an item as a list of Gurobi variables.
        :returns:
        """
        num_reals, num_bools = dataset.num_reals(), dataset.num_bools()
        item = np.array([x[z].x for z in range(num_bools)])
        assert item.shape == (num_bools,)
        if num_reals > 0:
            item = np.hstack((item, np.dot(dataset.costs, item)))
        assert item.shape == (num_bools + num_reals,)
        return item


    def compute_best_score(self, dataset, user):
        """Finds the highest-scoring item in the dataset according to the user.

        The score of an item (including Boolean and real parts) is:

        .. math::
            score(w,x) = \\langle w_B + C^\\top w_C, x \\rangle

        :param dataset: the dataset.
        :param user: the user.
        :returns: the best score and the corresponding best item.
        """
        num_bools, num_reals = dataset.num_bools(), dataset.num_reals()

        model = grb.Model("setmargin_dot")
        model.params.Seed = 0
        model.params.OutputFlag = 0

        x = [model.addVar(vtype=GRB.BINARY, name="x_{}".format(z))
             for z in range(num_bools)]

        model.modelSense = GRB.MAXIMIZE
        model.update()

        w = user.w.ravel()
        assert w.shape == (num_bools + num_reals,)

        obj = gudot(w[:num_bools], x)
        if num_reals > 0:
            assert dataset.costs.shape == (num_reals, num_bools)
            obj += gubilinear(w[-num_reals:], dataset.costs, x)
        model.setObjective(obj)

        self._add_item_constraints(model, dataset, x)

        model.update()
        self._dump_model(model, "setmargin_dot")

        try:
            model.optimize()
            best_score = model.objVal
        except:
            raise RuntimeError("optimization failed! {}".format(status_to_reason[model.status]))

        best_item = self._compose_item(dataset, x)
        assert dataset.is_item_valid(best_item)

        return best_score, best_item

    def compute_setmargin(self, dataset, answers, set_size, alphas):
        """Solves the set-margin problem.

        :param dataset: the dataset.
        :param answers: the known user pairwise preferences.
        :param set_size: how many distinct solutions to look for.
        :param alphas: hyperparameters.
        :returns: the value of the optimal ws, xs, scores, slacks and margin.
        """
        if not len(alphas) == 3 or not all([alpha >= 0 for alpha in alphas]):
            raise ValueError("invalid hyperparameters '{}'".format(alphas))

        num_examples = len(answers)
        num_features = sum(dataset.domain_sizes)

        model = grb.Model("facility")
        model.params.Seed = 0
        if self._threads is not None:
            model.params.Threads = self._threads
        model.params.OutputFlag = 0

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
        if self._multimargin:
            margins.append(model.addVar(vtype=GRB.CONTINUOUS, name="margin_on_generated_objects"))

        model.modelSense = GRB.MAXIMIZE
        model.update()

        # Define the objective function
        if False:
            # XXX getting temp[0] below 1 makes the set_size=1 problem
            # become unbounded; avoid normalizing the hyperparameters
            # for now
            temp = (
                0.0 if len(slacks) == 0 else alphas[0] / (set_size * num_examples),
                alphas[1] / set_size,
                alphas[2] / set_size,
            )
        else:
            temp = alphas

        obj_margins = grb.quicksum(margins)

        obj_slacks = 0
        if len(slacks) > 0:
            obj_slacks = temp[0] * grb.quicksum(slacks.values())

        obj_weights = temp[1] * grb.quicksum(ws.values())

        obj_scores = temp[2] * grb.quicksum([ps[i,i,z]
                                               for i in range(set_size)
                                               for z in range(num_features)])

        model.setObjective(obj_margins - obj_slacks - obj_weights + obj_scores)

        # Add the various constraints

        # Eq. 9
        for i in range(set_size):
            for k in range(num_examples):
                x1, x2, ans = answers[k]
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
            if set_size == 1 and all(ans == 0 for _, _, ans in answers):
                # XXX work around the fact that if we only have one hyperplane and
                # the user is indifferent to everything we throwed at her, the margin
                # will not appear in any constraint and thus the problem will be
                # unbounded.
                model.addConstr(margin == 0)
        if self._multimargin:
            if all(ans == 0 for _, _, ans in answers):
                model.addConstr(margins[0] == 0)
            if set_size == 1:
                model.addConstr(margins[1] == 0)

        for i in range(set_size):
            x = [xs[(i,z)] for z in range(num_features)]
            self._add_item_constraints(model, dataset, x)

        model.update()
        self._dump_model(model, "setmargin_full")
        try:
            model.optimize()
            _ = model.objVal
        except:
            message = dedent("""\
                optimization failed!

                answers =
                {}

                set_size = {}
                status = {}
                alphas = {}
                """).format(answers, set_size, status_to_reason[model.status], temp)
            raise RuntimeError(message)

        output_ws = np.zeros((set_size, num_features))
        output_xs = np.zeros((set_size, num_features))
        for i in range(set_size):
            for z in range(num_features):
                output_ws[i,z] = ws[i,z].x
                output_xs[i,z] = xs[i,z].x

        output_ps = np.zeros((set_size, set_size, num_features))
        output_scores = np.zeros((set_size, set_size))
        for i in range(set_size):
            for j in range(set_size):
                for z in range(num_features):
                    output_ps[i,j,z] = ps[i,j,z].x
                    output_scores[i,j] += ps[i,j,z].x

        if len(answers):
            output_slacks = np.zeros((set_size, len(answers)))
        else:
            output_slacks = []
        for i in range(set_size):
            for k in range(num_examples):
                output_slacks[i,k] = slacks[i,k].x

        output_margins = [margin.x for margin in margins]

        if self._debug:
            print dedent("""\
            set-margin solution (set_size = {})
            -----------------------------------
            ws =
            {}
            xs =
            {}
            ps =
            {}
            scores =
            {}
            slacks =
            {}
            margins = {}
            """).format(set_size, output_ws, output_xs, output_ps,
                        output_scores, output_slacks, output_margins)

        if any(np.linalg.norm(w) == 0 for w in output_ws):
            print "Warning: null weight vector found!"

        assert all(dataset.is_item_valid(x) for x in output_xs)

        debug_scores = np.dot(output_ws, output_xs.T)
        if (np.abs(output_scores - debug_scores) >= 1e-10).any():
            print dedent("""\
                Warning: solver and debug scores mismatch:
                scores =
                {}
                debug scores =
                {}
                """).format(output_scores, debug_scores)

        return output_ws, output_xs
