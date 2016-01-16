import numpy as np
import itertools as it
from sklearn.utils import check_random_state

class User(object):
    """A randomly sampled user that can answer ternary queries.

    Implemented according to the tree-structured Bradley-Torr model of
    indifference (see [1]_).

    The sampling ranges are adapted from those used in [1]_.

    :param domain_sizes: list of domain sizes.
    :param sampling_mode: sampling mode. (default: ``"uniform"``)
    :param ranking_mode: ranking mode for set queries. (default: ``"all_pairs"``)
    :param is_deterministic: whether the user is deterministic. (default: ``False``)
    :param is_indifferent: whether the user can be indifferent. (default: ``False``)
    :param w: user-provided preference vector. (default: ``None``)
    :param rng: random stream. (default: ``None``)

    .. [1] Shengbo Guo and Scott Sanner, *Real-time Multiattribute Bayesian
           Preference Elicitation with Pairwise Comparison Queries*, AISTATS
           2010.
    """
    def __init__(self, domain_sizes, sampling_mode="uniform", ranking_mode="all_pairs",
                 is_deterministic=False, is_indifferent=False, w=None,
                 rng=None):
        self._domain_sizes = domain_sizes
        self.ranking_mode = ranking_mode
        self.is_deterministic = is_deterministic
        self.is_indifferent = is_indifferent
        self._rng = check_random_state(rng)
        self.w = self._sample(sampling_mode, domain_sizes) if w is None else w

    def __str__(self):
        return "User({} D={} I={} RM={})".format(self.w, self.is_deterministic,
                                                 self.is_indifferent,
                                                 self.ranking_mode)

    def _sample(self, sampling_mode, domain_sizes, sparsity=0.2):
        num_attrs = sum(domain_sizes)
        if sampling_mode in ("uniform", "uniform_sparse"):
            w = self._rng.uniform(1, 100, size=num_attrs)
        elif sampling_mode in ("normal", "normal_sparse"):
            w = self._rng.normal(25.0, 25.0 / 3, size=num_attrs)
        else:
            raise ValueError("invalid sampling_mode")
        if sampling_mode.endswith("sparse"):
            mask, base = np.zeros(w.shape), 0
            for domain_size in domain_sizes:
                domain_mask = np.zeros(domain_size)
                num_ones = max(1, int(domain_size * sparsity))
                domain_mask[:num_ones] = 1
                mask[base:base+domain_size] = self._rng.permutation(domain_mask)
                base += domain_size
            w[mask == 0] = 0
        return w.reshape(1, -1)

    def query_diff(self, diff):
        """Queries the user about a single pairwise choice.

        :param diff: different in utility loss: u(xi) - u(xj).
        :returns: 0 if the user is indifferent, 1 if xi is better than xj,
            and -1 if xi is worse than xj.
        """
        ALPHA, BETA = 1.0, 1.0

        if self.is_deterministic:
            return int(np.sign(diff))

        eq = np.exp(-BETA * np.abs(diff)) if self.is_indifferent else 0.0
        gt = np.exp(ALPHA * diff) / (1 + np.exp(ALPHA * diff))
        lt = np.exp(-ALPHA * diff) / (1 + np.exp(-ALPHA * diff))

        z = self._rng.uniform(0, eq + gt + lt)
        if z < eq:
            return 0
        elif z < (eq + gt):
            return 1
        return -1

    def query(self, xi, xj):
        """Queries the user about a single pairwise choice.

        :param xi: an item.
        :param xj: the other item.
        :returns: 0 if the user is indifferent, 1 if xi is better than xj,
            and -1 if xi is worse than xj.
        """
        return self.query_diff(np.dot(self.w, xi.T - xj.T))

    def query_set(self, ws, xs, old_best_item):
        """Queries the user about the provided set of items.

        :param user: the user.
        :param ws: the estimated user preference(s) at the current iteration.
        :param xs: the estimated best item(s) at the current iteration.
        :param old_best_item: the estimated best item at the previous iteration.
        :returns: WRITEME
        """
        num_items, num_features = xs.shape

        if num_items == 1:
            if old_best_item is None:
                old_best_item = self._rng.random_integers(0, 1, size=(num_features,))
            answers = [(xs[0], old_best_item, self.query(xs[0], old_best_item))]
            num_queries = 1

        elif self.ranking_mode == "all_pairs":
            # requires 1/2 * n * (n - 1) queries
            # XXX note that in the non-deterministic setting we may actually lose
            # information by only querying for ~half the pairs!
            answers = [(xi, xj, self.query(xi, xj))
                       for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs)) if i < j]
            num_queries = len(answers)

        elif self.ranking_mode == "sorted_pairs":
            answers = {}
            sorted_sets = self.quicksort(xs, answers)
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
            raise ValueError("invalid ranking_mode {}".format(self.ranking_mode))

        assert len(answers) > 0
        assert num_queries > 0
        return answers, num_queries

    def quicksort(self, xs, answers):
        raise NotImplementedError("very roughly tested")
        lt, eq, gt = [], [], []
        if len(xs) > 1:
            pivot = xs[0]
            eq.append(pivot)
            for x in xs[1:]:
                try:
                    ans = answers[(tuple(x), tuple(pivot))]
                except KeyError:
                    ans = self.query(x, pivot)
                    answers[(tuple(x), tuple(pivot))] = ans
                if ans < 0:
                    lt.append(x)
                elif ans == 0:
                    eq.append(x)
                else:
                    gt.append(x)
            assert len(lt) < len(xs)
            assert len(gt) < len(xs)

            sorted_lt = quicksort(lt, answers)
            sorted_gt = quicksort(gt, answers)
            return [l for l in sorted_lt + [eq] + sorted_gt if len(l)]
        else:
            return [xs]
