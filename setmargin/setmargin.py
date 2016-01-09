# -*- coding: utf-8 -*-

import time
import itertools as it
import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent
from pprint import pformat

from util import *

def print_answers(queries, hidden_w):
    message = ["updated answers ="]
    for xi, xj, sign in queries:
        relation = {-1:"<", 0:"~", 1:">"}[sign]
        score_xi = np.dot(hidden_w, xi.T)[0]
        score_xj = np.dot(hidden_w, xj.T)[0]
        message.append("  {} ({:6.3f}) {} ({:6.3f}) {} -- diff {:6.3f}" \
                           .format(xi, score_xi, relation, score_xj, xj,
                                   score_xi - score_xj))
    print "\n".join(message)

def run(dataset, user, solver, num_iterations, set_size, alphas, debug=False):
    if not num_iterations > 0:
        raise ValueError("num_iterations must be positive")

    _, best_item = solver.compute_best_score(dataset, user)

    answers, info, old_best_item = [], [], None
    for t in range(num_iterations):
        if debug:
            print dedent("""\
            ===============
            ITERATION {}/{}
            ===============

            answers =
            {}
            """).format(t, num_iterations, pformat(answers))

        old_time = time.time()

        # Solve the set_size=k case
        ws, xs = solver.compute_setmargin(dataset, answers, set_size, alphas)

        # Update the user answers
        new_answers, num_queries = user.query_set(ws, xs, old_best_item)
        answers.extend(new_answers)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        elapsed = time.time() - old_time

        if debug:
            print_answers(answers, user.w)

        # Solve the set_size=1 case
        ws, xs = solver.compute_setmargin(dataset, answers, 1, alphas)

        # Compute the utility loss
        norm = np.linalg.norm(user.w.ravel())
        utility_loss = np.dot(user.w.ravel(), best_item - xs[0]) / norm

        info.append((num_queries, utility_loss, elapsed))

    return info
