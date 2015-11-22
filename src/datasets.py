# -*- coding: utf-8 -*-

import numpy as np
import itertools as it
from sklearn.utils import check_random_state
from util import *

class Dataset(object):
    def __str__(self):
        return "Dataset(domain_sizes={} len(items)={} constrs={}" \
                   .format(self.domain_sizes, len(self.items),
                           self.x_constraints)

    def _feat_to_var(domain_sizes, dom_i, feature_i):
        assert 0 <= dom_i < len(domain_sizes)
        assert 0 <= feature_i < domain_sizes[dom_i]
        return sum(domain_sizes[:dom_i]) + feature_i

    def _atom_to_var(domain_sizes, atom_i):
        assert 0 <= atom_i <= sum(domain_sizes)
        base = 0
        for dom_i, domain_size in enumerate(domain_sizes):
            if base <= atom_i < base + domain_size:
                break
            base += domain_size
        feature_i = atom_i - base
        return self._feat_to_var(domain_sizes, dom_i, feature_i)

    def is_item_valid(self, x):
        for zs_in_domain in get_zs_in_domains(self.domain_sizes):
            if sum(x[z] for z in zs_in_domain) != 1:
                return False
        if self.x_constraints is not None:
            for head, body in self.x_constraints:
                if x[self._atom_to_var(self.domain_sizes, head)] and \
                   not any(x[self._atom_to_var(self.domain_sizes, atom)] == 1 for atom in body):
                    return False
        return True

class SyntheticDataset(Dataset):
    def __init__(self, domain_sizes):
        self.domain_sizes = domain_sizes
        self.items = self._ground_onehot(domain_sizes)
        self.x_constraints = None

    def _ground(self, domain_sizes):
        return [item for item in it.product(*map(range, domain_sizes))]

    def _ground_onehot(self, domain_sizes):
        items = np.array([vonehot(domain_sizes, item)
                          for item in self._ground(domain_sizes)])
        assert items.shape == (prod(domain_sizes), sum(domain_sizes))
        return items

class RandomlyConstrainedSyntheticDataset(SyntheticDataset):
    def __init__(self, domain_sizes, rng=None):
        super(RandomlyConstrainedSyntheticDataset, self).__init__(domain_sizes)
        self.x_constraints = self._sample_constraints(domain_sizes,
                                                         check_random_state(rng))

    def _sample_constraints(self, domain_sizes, rng):
        constraints = []
        print "sampling constraints"
        print "--------------------"
        print "domain_sizes =", domain_sizes
        for (i, dsi), (j, dsj) in it.product(enumerate(domain_sizes), enumerate(domain_sizes)):
            if i >= j:
                continue
            # XXX come up with something smarter
            head = rng.random_integers(0, dsi - 1)
            body = rng.random_integers(0, dsj - 1)
            print "{}:{} -> {}:{}".format(i, head, j, body)
            index_head = self._feat_to_var(domain_sizes, i, head)
            index_body = self._feat_to_var(domain_sizes, j, body)
            constraints.append((index_head, [index_body]))
        print "constraints =\n", constraints
        print "--------------------"
        return constraints

class PCDataset(Dataset):
    def __init__(self):
        ATTRIBUTES = ("Manufacturer", "CPUType", "CPUSpeed", "Monitor", "Type",
                      "Memory", "HDSize", "Price")

        categories = {}
        categories[0] = ("Apple", "Compaq", "Dell", "Fujitsu", "Gateway", "HP",
                         "Sony", "Toshiba")
        categories[1] = ("PowerPC G3", "PowerPC G4", "Intel Pentium",
                         "AMD Athlon", "Intel Celeron", "Crusoe", "AMD Duron")
        categories[4] = ("Laptop", "Desktop", "Tower")

        FIXED_ATTRIBUTES = categories.keys()

        PRICE_INTERVALS = np.hstack((np.arange(600, 2800+1e-6, 150), [3649]))

        items = []
        with open("pc_dataset.xml", "rb") as fp:
            for words in map(str.split, map(str.strip, fp)):
                if words[0] == "<item":
                    items.append({})
                elif words[0] == "<attribute":
                    assert words[1].startswith("name=")
                    assert len(words) in (4, 5)
                    k = words[1].split("=")[1].strip('"')
                    v = words[2].split("=")[1].strip('"')
                    if len(words) == 5:
                        v += " " + words[3].strip('"')
                    assert k in ATTRIBUTES
                    assert (not k in FIXED_ATTRIBUTES) or v in categories[ATTRIBUTES.index(k)]
                    items[-1][k] = v
        assert len(items) == 120
        assert all(len(item) == 8 for item in items)

        for z, attribute in enumerate(ATTRIBUTES):
            if z in FIXED_ATTRIBUTES or z == 8:
                continue
            categories[z] = sorted(set(item[attribute] for item in items))
        categories[7] = range(PRICE_INTERVALS.shape[0])

        discretized_items = []
        for item in items:
            discretized_items.append([])
            for z, attribute in enumerate(ATTRIBUTES):
                if z != 7:
                    discretized_items[-1].append(categories[z].index(item[attribute]))
                else:
                    if int(item[attribute]) == PRICE_INTERVALS[-1]:
                        ans = PRICE_INTERVALS.shape[0] - 1
                    else:
                        temp = float(item[attribute]) - PRICE_INTERVALS
                        ans = np.where(temp <= 0)[0][0]
                    discretized_items[-1].append(ans)
        discretized_items = np.array(discretized_items)

        domain_sizes = []
        for z, category in sorted(categories.iteritems()):
            domain_sizes.append(len(category))

        for row in discretized_items:
            for z in range(len(ATTRIBUTES)):
                assert 0 <= row[z] < domain_sizes[z], \
                       "invalid value {}/{} in attribute {}" \
                           .format(row[z], domain_sizes[z], z)

        items_onehot = None
        for item in discretized_items:
            item_onehot = np.hstack((onehot(domain_sizes[z], attribute_value)
                                     for z, attribute_value in enumerate(item)))
            if items_onehot is None:
                items_onehot = item_onehot
            else:
                items_onehot = np.vstack((items_onehot, item_onehot))
        assert items_onehot.shape == (120, sum(domain_sizes))

        self.domain_sizes = domain_sizes
        self.items = items_onehot
        self.x_constraints = None
