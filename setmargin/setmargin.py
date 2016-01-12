# -*- coding: utf-8 -*-

import time
import itertools as it
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state
from textwrap import dedent
from pprint import pformat

from util import *

ALL_ALPHAS = list(it.product(
         [12.0, 11.0, 10.0, 9.0, 8.0],
         [10.0, 1.0, 0.1, 0.01],
         [10.0, 1.0, 0.1, 0.01],
))

def crossvalidate(dataset, solver, answers, set_size, debug):
    """Performs a 5-fold cross-validation.

    :param dataset:
    :param answers:
    :param set_size:
    :param debug:
    :returns:
    """
    loss_alphas = []
    for alphas in ALL_ALPHAS:

        if debug:
            print "crossvalidating", alphas

        kfold = KFold(len(answers), n_folds=5)

        losses = []
        for train_set, test_set in kfold:
            train_answers = [answers[i] for i in train_set]

            ws, _ = solver.compute_setmargin(dataset, train_answers, set_size,
                                             alphas)

            test_answers = [answers[i] for i in test_set]
            xis = np.array([x for x, _, _ in test_answers])
            xjs = np.array([x for _, x, _ in test_answers])
            ys  = np.array([s for _, _, s in test_answers])

            ys_hat = np.sign(np.dot(ws, (xis - xjs).T))
            diff = 0.5 * np.abs(ys - ys_hat) # the difference is broadcast
            losses.append(diff.sum(axis=1).mean())

        loss_alphas.append((sum(losses) / len(losses), alphas))

    loss_alphas = sorted(loss_alphas)
    alphas = loss_alphas[0][1]

    if debug:
        print "crossvalidation: best alphas = ", alphas
        for loss, alpha in loss_alphas:
            print alpha, ":", loss

    return alphas

def print_answers(queries, hidden_w):
    message = ["answers ="]
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

    best_score, best_item = solver.compute_best_score(dataset, user)
    user_w_norm = np.linalg.norm(user.w.ravel())

    if debug:
        print dedent("""
            best_score = {}
            best_item =
            {}

            user.w = {}
            user_w_norm = {}
            """).format(best_score, best_item, user.w, user_w_norm)

    do_crossval = alphas == "auto"
    if do_crossval:
        # XXX far from ideal
        alphas = (1.0, 1.0, 1.0)
        if crossval_set_size is None:
            crossval_set_size = set_size

    answers, info, old_best_item = [], [], None
    for t in range(num_iterations):
        if debug:
            print dedent("""\
            ===============
            ITERATION {}/{}
            ===============
            """).format(t, num_iterations)

            print_answers(answers, user.w)

        # Crossvalidate the hyperparameters if required
        if do_crossval and t % crossval_interval == 0 and t > 0:
            alphas = crossvalidate(dataset, solver, answers, crossval_set_size,
                                   debug)

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
        best_score = np.dot(user.w.ravel(), best_item)
        pred_score = np.dot(user.w.ravel(), xs[0])
        loss = (best_score - pred_score) / user_w_norm
        if debug:
            print dedent("""\
                    best_item = (true score = {})
                    {}
                    generated item = (true score = {})
                    {}
                    loss = {}
                    """).format(best_score, best_item, pred_score, xs[0], loss)
        info.append((num_queries, loss, elapsed))

        # If the user is satisfied (clicks the 'add to cart' button),
        # we are done
        #if loss < tol:
        #    break

    return info
