# -*- coding: utf-8 -*-

import numpy as np
import tempfile
import gurobipy as grb
from gurobipy import GRB
from textwrap import dedent
from util import *

MAX_W_Z = 1

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

class Solver(object):
    """Set-margin solver based on the Gurobi MILP solver.

    .. todo::

        Cache the model and update the constraints on demand before solving
        again.

    :param multimargin: whether to use two distinct margins. (default: ``False``)
    :param threads: how many threads Gurobi is allowed to use. (default: 1)
    :param debug: whether to spew debug output. (default: ``False``)
    """
    def __init__(self, multimargin=False, threads=1, debug=False):
        self._multimargin = multimargin
        self._threads = threads
        self._debug = debug

    def _dump_model(self, model, prefix):
        if self._debug:
            fp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".lp",
                                             delete=False)
            fp.close()
            print "dumping model to", fp.name
            model.write(fp.name)

    def _check_best_score(self, dataset, user, best_score, best_item):
        if dataset.items is None:
            return
        if dataset.x_constraints is not None:
            return

        in_dataset = False
        for item in dataset.items:
            in_dataset = in_dataset or (item == best_item).all()
        assert in_dataset, "best_item not in dataset.items"

        scores = np.dot(user.w.ravel(), dataset.items.T)
        debug_best_score = np.max(scores)
        if np.abs(best_score - debug_best_score) > 1e-10:
            debug_best_item = dataset.items[np.argmax(scores)].ravel()
            raise RuntimeError(dedent("""\
                best_score sanity check failed!

                best_score = {}
                best_item =
                {}

                debug best_score = {}
                debug best_item =
                {}

                all scores =
                {}
                """).format(best_score, best_item, debug_best_score,
                            debug_best_item, scores))

    def _add_item_constraints(self, model, dataset, x):
        """Add one-hot and other item constraints."""
        zs_in_domains = get_zs_in_domains(dataset.domain_sizes)
        for zs_in_domain in zs_in_domains:
            model.addConstr(grb.quicksum([x[z] for z in zs_in_domain]) == 1)

        if dataset.x_constraints is not None:
            for body, head in dataset.x_constraints:
                model.addConstr((1 - x[body]) + grb.quicksum([x[atom] for atom in head]) >= 1)

    def compute_best_score(self, dataset, user):
        """Returns the highest score for all items the dataset.

        :param dataset: the dataset.
        :param user: the user.
        :returns: the best score and the corresponding best item.
        """
        num_features = sum(dataset.domain_sizes)

        model = grb.Model("setmargin_dot")
        model.params.Seed = 0
        model.params.OutputFlag = 0

        x = [model.addVar(vtype=GRB.BINARY, name="x_{}".format(z))
             for z in range(num_features)]

        model.modelSense = GRB.MAXIMIZE
        model.update()

        w = user.w.ravel()
        model.setObjective(grb.quicksum([w[z] * x[z]
                                        for z in range(num_features)]))

        self._add_item_constraints(model, dataset, x)

        model.update()
        self._dump_model(model, "setmargin_dot")

        try:
            model.optimize()
            best_score = model.objVal
        except:
            raise RuntimeError("optimization failed! {}".format(status_to_reason[model.status]))
        best_item = np.array([var.x for var in x])

        self._check_best_score(dataset, user, best_score, best_item)
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
        temp = (
            0.0 if len(slacks) == 0 else alphas[0] / (set_size * num_examples),
            alphas[1] / set_size,
            alphas[2] / set_size,
        )

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

        output_scores = np.zeros((set_size, set_size))
        for i in range(set_size):
            for j in range(set_size):
                for z in range(num_features):
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
            scores =
            {}
            slacks =
            {}
            margins = {}
            """).format(set_size, output_ws, output_xs, output_scores,
                        output_slacks, output_margins)

        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found:\n{}".format(ws)

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
            assert (np.diag(output_scores) - np.diag(debug_scores) < 1e-10).all()

        return output_ws, output_xs
