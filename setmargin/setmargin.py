# -*- coding: utf-8 -*-

import time
import itertools as it
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state
from textwrap import dedent
from pprint import pformat

NUM_FOLDS = 5
ALL_ALPHAS = list(it.product(
         [20.0, 10.0, 5.0, 1.0],
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
    if debug:
        print "crossvalidating..."
    loss_alphas = []
    for alphas in ALL_ALPHAS:
        kfold = KFold(len(answers), n_folds=NUM_FOLDS)

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

def print_answers(user, answers):
    message = ["answers ="]
    for xi, xj, sign in answers:
        relation = {-1:"<", 0:"~", 1:">"}[sign]
        utility_xi = user.utility(xi)
        utility_xj = user.utility(xj)
        message.append("  {} ({:6.3f}) {} ({:6.3f}) {} -- diff {:6.3f}" \
                           .format(xi, utility_xi, relation, utility_xj, xj,
                                   utility_xi - utility_xj))
    print "\n".join(message)

def run(dataset, user, solver, set_size, max_iterations=100, max_answers=100,
        tol="auto", alphas="auto", crossval_interval=5, crossval_set_size=None,
        debug=False):
    """Runs the setmargin algorithm.

    :param dataset: the dataset.
    :param user: the user.
    :param solver: the setmargin solver.
    :param set_size: set size.
    :param max_iterations: maximum number of iterations to run for.
        (default: ``100``)
    :param max_answers: maximum number of answers to elicit in total.
        (default: ``100``)
    :param tol: user tolerance, used for termination, or ``"auto"``, in
        which case user indifference is used instead. (default: ``"auto"``)
    :param alphas: either a triple of non-negative floats, or ``"auto"``,
        in which case the hyperparameters are determined automatically through
        a periodic cross-validation procedure. (default: ``"auto"``)
    :param crossval_interval: number of iterations between cross-validation
        calls. (default: ``5``)
    :param crossval_set_size: number of items. (default: ``set_size``)
    :param debug: whether to spew debug info. (default: ``False``)
    :return: the number of queries, utility loss and elapsed time for
        each iteration.
    """
    if not max_iterations > 0:
        raise ValueError("max_iterations must be positive")
    if not max_answers > 0:
        raise ValueError("max_answers must be positive")
    if not crossval_interval > 0:
        raise ValueError("crossval_interval must be positive")

    best_score, best_item = solver.compute_best_score(dataset, user)
    assert best_item.shape == (dataset.num_bools(),)
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

    answers, info, old_best_item, t = [], [], None, 0
    while True:

        if debug:
            print dedent("""\
            ============
            ITERATION {}
            ============
            """).format(t)

            print_answers(user, answers)

        old_time = time.time()

        # Crossvalidate the hyperparameters if required
        if do_crossval and t % crossval_interval == 0 and t >= NUM_FOLDS:
            alphas = crossvalidate(dataset, solver, answers, crossval_set_size,
                                   debug)

        # Solve the set_size=k case
        _, xs = solver.compute_setmargin(dataset, answers, set_size, alphas)
        assert xs.shape == (set_size, dataset.num_bools())

        # Update the user answers
        new_answers, num_queries = user.query_set(xs, old_best_item)
        num_identical_answers = 0
        for xi, xj, sx in new_answers:
            for zi, zj, sz in answers:
                if (xi == zi).all() and (xj == zj).all():
                    num_identical_answers += 1
        if num_identical_answers > 0:
            print "Warning: {} identical (up to sign) answers added!" \
                      .format(num_identical_answers)
        answers.extend(new_answers)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        elapsed = time.time() - old_time

        if debug:
            print_answers(user, answers)

        # Solve the set_size=1 case
        _, xs = solver.compute_setmargin(dataset, answers, 1, alphas)
        assert xs.shape == (1, dataset.num_bools())

        # Compute the utility loss
        best_score = np.dot(user.w.ravel(), dataset.compose_item(best_item))
        pred_score = np.dot(user.w.ravel(), dataset.compose_item(xs[0]))
        loss = best_score - pred_score
        if debug:
            print dedent("""\
                    best_item = (true score = {})
                    {}
                    generated item = (true score = {})
                    {}
                    loss = {}
                    """).format(best_score, best_item, pred_score, xs[0], loss)
        info.append((num_queries, loss, elapsed))

        t += 1
        if t >= max_iterations:
            if debug:
                print "maximum iterations reached, terminating"
            break

        if len(answers) >= max_answers:
            if debug:
                print "maximum number of answers reached, terminating"
            break

        # If the user is satisfied (clicks the 'add to cart' button),
        # we are done
        if tol == "auto" and user.query_diff(loss) == 0:
            if debug:
                print "user indifference reached, terminating"
            break
        elif tol != "auto" and loss < tol:
            if debug:
                print "minimum tolerance reached, terminating"
            break

    return info
