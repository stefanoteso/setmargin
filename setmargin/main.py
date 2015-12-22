# -*- coding: utf-8 -*-

import time
import itertools as it
import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent
from pprint import pformat

from util import *

def print_queries(queries, hidden_w):
    for xi, xj, sign in queries:
        relation = {-1:"<", 0:"~", 1:">"}[sign]
        score_xi = np.dot(hidden_w, xi.T)[0]
        score_xj = np.dot(hidden_w, xj.T)[0]
        print "  {} ({:6.3f}) {} ({:6.3f}) {} -- diff {:6.3f}".format(xi, score_xi, relation, score_xj, xj, score_xi - score_xj)
    print

def quicksort(user, xs, answers):
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

def update_queries(user, ws, xs, old_best_item, rng, ranking_mode="all_pairs"):
    """Computes the queries to ask the user for the given inputs.

    If there is only one candidate best item, then only one query is returned,
    namely a query comparing the current best item with the best item at the
    previous iteration.

    If there are multiple candidate best items, then multiple queries are
    returned, one for each pair of candidate best items.

    :param user: the user.
    :param ws: the estimated user preference(s) at the current iteration.
    :param xs: the estimated best item(s) at the current iteration.
    :param old_best_item: the estimated best item at the previous iteration.
    :param ranking_mode: either ``"all_pairs"`` or ``"sorted_pairs"``.
    :returns: a pair (list of new queries, 
    """
    num_items, num_features = xs.shape
    if num_items == 1:
        if old_best_item is None:
            old_best_item = rng.random_integers(0, 1, size=(num_features,))
        queries = [(xs[0], old_best_item, user.query(xs[0], old_best_item))]
        num_queries = 1
    elif ranking_mode == "all_pairs":
        # requires 1/2 * n * (n - 1) queries
        # XXX note that in the non-deterministic setting we may actually lose
        # information by only querying for ~half the pairs!
        queries = [(xi, xj, user.query(xi, xj))
                   for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs)) if i < j]
        num_queries = len(queries)
    elif ranking_mode == "sorted_pairs":
        answers = {}
        sorted_sets = quicksort(user, xs, answers)
        num_queries = len(answers)
        assert num_queries > 0

        queries = []
        for (k, set_k), (l, set_l) in it.product(enumerate(sorted_sets), enumerate(sorted_sets)):
            if k > l:
                continue
            for xi, xj in it.product(set_k, set_l):
                if (xi != xj).any():
                    queries.append((xi, xj, 0 if k == l else -1))
    else:
        raise ValueError("invalid ranking_mode '{}'".format(ranking_mode))
    return queries, num_queries

def run(dataset, user, solver, num_iterations, set_size, rng,
        ranking_mode="all_pairs", debug=False):

    if not num_iterations > 0:
        raise ValueError("invalid num_iterations '{}'".format(num_iterations))

    rng = check_random_state(rng)

    # Find the dataset item with the highest score wrt the hidden hyperlpane
    best_hidden_score, _ = solver.compute_best_score(dataset, user)

    # Iterate
    queries, old_best_item = [], None
    losses, times = [], []
    for t in range(num_iterations):

        if debug:
            print dedent("""\
            ===============
            ITERATION {}/{}
            ===============
            answers =
            {}

            """).format(t, num_iterations, pformat(queries))

        old_time = time.time()

        # Solve the utility/item learning problem for the current iteration
        ws, xs, scores, slacks, margin = \
            solver.compute_setmargin(dataset, queries, set_size)
        debug_scores = np.dot(ws, xs.T)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the m-item case:\n{}".format(ws)

        if debug:
            print dedent("""\
            set-margin solution
            -------------------
            ws =
            {}
            xs =
            {}
            scores =
            {}
            slacks =
            {}
            margin = {}
            """).format(ws, xs, scores, slacks, margin)

        assert all(dataset.is_item_valid(x) for x in xs)

        if (np.abs(scores - debug_scores) >= 1e-10).any():
            print "Warning: solver and debug scores mismatch:\n" \
                  "scores =\n{}\n" \
                  "debug scores =\n{}\n".format(scores, debug_scores)
            assert (np.diag(scores) - np.diag(debug_scores) < 1e-10).all()

        # Ask the user about the retrieved items
        new_queries, num_queries = \
            update_queries(user, ws, xs, old_best_item, rng,
                           ranking_mode=ranking_mode)
        assert len(new_queries) > 0
        assert num_queries > 0

        queries.extend(new_queries)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        if debug:
            print "\nupdated queries ="
            print_queries(queries, user.w)

        elapsed = time.time() - old_time
        times.extend([elapsed / num_queries] * num_queries)

        # Compute the utility loss between the best item that we would
        # recommend given the queries collected so far and the best
        # recommendation according to the hidden user hyperplane
        ws, xs, scores, slacks, margin = \
            solver.compute_setmargin(dataset, queries, 1)
        if any(np.linalg.norm(w) == 0 for w in ws):
            print "Warning: null weight vector found in the 1-item case:\n{}".format(ws)

        utility_loss = best_hidden_score - np.dot(user.w, xs[0])
        losses.extend([utility_loss / np.linalg.norm(user.w)] * num_queries)

        if debug:
            real_score = np.dot(user.w, xs[0])
            print dedent("""\
            one-margin solution
            -------------------
            ws =
            {}
            xs =
            {}
            scores =
            {}
            slacks =
            {}
            margin = {}

            real score = {}
            utility_loss = {}
            """).format(ws, xs, scores, slacks, margin, real_score, utility_loss)

        assert dataset.is_item_valid(xs[0])

    return losses, times
