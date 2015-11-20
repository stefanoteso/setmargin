import numpy as np
from sklearn.utils import check_random_state

class User(object):
    """A randomly sampled user that can answer ternary queries.

    Implemented according to the tree-structured Bradley-Torr model of
    indifference (see [1]_).

    The sampling ranges are adapted from those used in [1]_.

    :param domain_sizes: list of domain sizes.
    :param sampling_mode: sampling mode. (default: ``"uniform"``)
    :param is_deterministic: whether the user is deterministic. (default: ``False``)
    :param is_indifferent: whether the user can be indifferent. (default: ``False``)
    :param w: user-provided preference vector. (default: ``None``)
    :param rng: random stream. (default: ``None``)

    .. [1] Shengbo Guo and Scott Sanner, *Real-time Multiattribute Bayesian
           Preference Elicitation with Pairwise Comparison Queries*, AISTATS
           2010.
    """
    def __init__(self, domain_sizes, sampling_mode="uniform",
                 is_deterministic=False, is_indifferent=False, w=None,
                 rng=None):
        self._domain_sizes = domain_sizes
        self.is_deterministic = is_deterministic
        self.is_indifferent = is_indifferent
        self._rng = check_random_state(rng)
        self.w = self._sample(sampling_mode, domain_sizes) if w is None else w

    def _sample(self, sampling_mode, domain_sizes):
        if sampling_mode == "uniform":
            return self._rng.uniform(0, 1, size=(sum(domain_sizes), 1)).reshape(1,-1)
        elif sampling_mode == "normal":
            return self._rng.normal(0.25, 0.25 / 3, size=(sum(domain_sizes), 1)).reshape(1,-1)
        else:
            raise ValueError("invalid sampling_mode")

    def query(self, xi, xj):
        """Queries the user about a single pairwise choice.

        :param xi: an item.
        :param xj: the other item.
        :returns: 0 if the user is indifferent, 1 if xi is better than xj,
            and -1 if xi is worse than xj.
        """
        # The original problem has weights sampled in the range [0, 100], and
        # uses ALPHA=1, BETA=1; here however we have weights in the range [0, 1]
        # so we must rescale ALPHA and BETA to obtain the same probabilities.
        ALPHA, BETA = 100, 100

        diff = np.dot(self.w, xi.T - xj.T)

        if self.is_deterministic:
            return int(np.sign(diff))

        eq = 0.0 if self.is_indifferent else np.exp(-BETA * np.abs(diff))
        gt = np.exp(ALPHA * diff) / (1 + np.exp(ALPHA * diff))
        lt = np.exp(-ALPHA * diff) / (1 + np.exp(-ALPHA * diff))

        z = self._rng.uniform(0, eq + gt + lt)
        if z < eq:
            return 0
        elif z < (eq + gt):
            return 1
        return -1
