# -*- coding: utf-8 -*-

import time
import itertools as it
import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent
from pprint import pformat

from util import *

def quicksort(user, xs, answers):
    raise NotImplementedError("very roughly tested")
    lt, eq, gt = [], [], []
    if len(xs) > 1:
        pivot = xs[0]
        eq.append(pivot)
        for x in xs[1:]:
            try:
                ans = answers[(tuple(x), tuple(pivot))]
            except KeyError:
                ans = user.query(x, pivot)
                answers[(tuple(x), tuple(pivot))] = ans
            if ans < 0:
                lt.append(x)
            elif ans == 0:
                eq.append(x)
            else:
                gt.append(x)
        assert len(lt) < len(xs)
        assert len(gt) < len(xs)

        sorted_lt = quicksort(user, lt, answers)
        sorted_gt = quicksort(user, gt, answers)
        return [l for l in sorted_lt + [eq] + sorted_gt if len(l)]
    else:
        return [xs]

def update_answers(user, ws, xs, old_best_item, rng, ranking_mode="all_pairs"):
    """Queries the user about the provided set of items.

    If there is only one candidate best item, then only one query is returned,
    namely a query comparing the current best item with the best item at the
    previous iteration.

    If there are multiple candidate best items, then multiple answers are
    returned, one for each pair of candidate best items.

    :param user: the user.
    :param ws: the estimated user preference(s) at the current iteration.
    :param xs: the estimated best item(s) at the current iteration.
    :param old_best_item: the estimated best item at the previous iteration.
    :param ranking_mode: either ``"all_pairs"`` or ``"sorted_pairs"``.
    :returns: WRITEME
    """
    num_items, num_features = xs.shape
    if num_items == 1:
        if old_best_item is None:
            old_best_item = rng.random_integers(0, 1, size=(num_features,))
        answers = [(xs[0], old_best_item, user.query(xs[0], old_best_item))]
        num_queries = 1
    elif ranking_mode == "all_pairs":
        # requires 1/2 * n * (n - 1) queries
        # XXX note that in the non-deterministic setting we may actually lose
        # information by only querying for ~half the pairs!
        answers = [(xi, xj, user.query(xi, xj))
                   for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs)) if i < j]
        num_queries = len(answers)
    elif ranking_mode == "sorted_pairs":
        answers = {}
        sorted_sets = quicksort(user, xs, answers)
        num_queries = len(answers)
        assert num_queries > 0

        answers = []
        for (k, set_k), (l, set_l) in it.product(enumerate(sorted_sets), enumerate(sorted_sets)):
            if k > l:
                continue
            for xi, xj in it.product(set_k, set_l):
                if (xi != xj).any():
                    answers.append((xi, xj, 0 if k == l else -1))
    else:
        raise ValueError("invalid ranking_mode '{}'".format(ranking_mode))
    assert len(answers) > 0
    assert num_queries > 0
    return answers, num_queries

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

def run(dataset, user, solver, num_iterations, set_size, rng,
        ranking_mode="all_pairs", debug=False):
    if not num_iterations > 0:
        raise ValueError("num_iterations must be positive")

    rng = check_random_state(rng)

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
        ws, xs = solver.compute_setmargin(dataset, answers, set_size)

        # Update the user answers
        new_answers, num_queries = \
            update_answers(user, ws, xs, old_best_item, rng,
                           ranking_mode=ranking_mode)
        answers.extend(new_answers)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        elapsed = time.time() - old_time

        if debug:
            print_answers(answers, user.w)

        # Solve the set_size=1 case
        ws, xs = solver.compute_setmargin(dataset, answers, 1)

        # Compute the utility loss
        norm = np.linalg.norm(user.w.ravel())
        utility_loss = np.dot(user.w.ravel(), best_item - xs[0]) / norm

        info.append((num_queries, utility_loss, elapsed))

    return info
