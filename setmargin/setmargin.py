# -*- coding: utf-8 -*-

import time
import itertools as it
import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent
from pprint import pformat

from util import *

ALPHAS = [100.0, 10.0, 1.0]
BETAS  = [10.0, 1.0, 0.1, 0.0]
GAMMAS = [10.0, 1.0, 0.1, 0.0]

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

def run(dataset, user, solver, num_iterations, set_size, alphas="auto",
        tol=1e-6, crossval_interval=5, crossval_set_size=None, debug=False):
    """Runs the setmargin algorithm.

    :param dataset: the dataset.
    :param user: the user.
    :param solver: the setmargin solver.
    :param num_iterations: number of iterations to run for.
    :param set_size: set size.
    :param alphas: either a triple of non-negative floats, or ``"auto"``,
        in which case the hyperparameters are determined automatically through
        a periodic cross-validation procedure. (default: ``"auto"``)
    :param tol: user tolerance, used for termination. (default: ``1e-6``)
    :param crossval_interval: number of iterations between cross-validation
        calls. (default: ``5``)
    :param crossval_set_size: number of items. (default: ``set_size``)
    :param debug: whether to spew debug info. (default: ``False``)
    :return: the number of queries, utility loss and elapsed time for
        each iteration.
    """
    if not num_iterations > 0:
        raise ValueError("num_iterations must be positive")
    if not crossval_interval > 0:
        raise ValueError("crossval_interval must be positive")

    _, best_item = solver.compute_best_score(dataset, user)
    user_w_norm = np.linalg.norm(user.w.ravel())

    do_crossval = alphas == "auto"
    if crossval_set_size is None:
        crossval_set_size = set_size

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

        # Crossvalidate the hyperparameters if required
        if do_crossval and t % crossval_interval == 0:
            loss_alphas = []
            for alphas in it.product(ALPHAS, BETAS, GAMMAS):
                try:
                    ws, xs = solver.compute_setmargin(dataset, answers,
                                                      crossval_set_size,
                                                      alphas)
                except RuntimeError:
                    continue
                mean_x = xs.mean(axis=0)
                loss = np.dot(user.w.ravel(), best_item - mean_x) / user_w_norm
                loss_alphas.append((loss, alphas))
            assert len(loss_alphas) > 0
            alphas = sorted(loss_alphas)[0][1]

            if debug:
                print "CROSS VALIDATION --> new alphas =", alphas
                for loss, alpha in loss_alphas:
                    print alpha, ": loss =", loss

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
        loss = np.dot(user.w.ravel(), best_item - xs[0]) / user_w_norm
        info.append((num_queries, loss, elapsed))

        # If the user is satisfied (clicks the 'add to cart' button),
        # we are done
        if loss < tol:
            break

    return info
