# -*- coding: utf-8 -*-

import numpy as np
import itertools as it
from sklearn.utils import check_random_state
from util import vonehot, prod

def _attr_index(domain_sizes, dom_i, feature_i):
    assert 0 <= dom_i < len(domain_sizes)
    assert 0 <= feature_i < len(domain_sizes[dom_i])
    return sum(domain_sizes[:dom_i]) + feature_i

class Dataset(object):
    def __str__(self):
        return "Dataset(domain_sizes={} len(items)={} wc={} xc={} hc={}" \
                   .format(self.domain_sizes, len(self.items),
                           self.w_constraints, self.x_constraints,
                           self.horn_constraints)

class SyntheticDataset(Dataset):
    def __init__(self, domain_sizes):
        self.domain_sizes = domain_sizes
        self.items = self._ground_onehot(domain_sizes)
        self.w_constraints = None
        self.x_constraints = None
        self.horn_constraints = None

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
        self.horn_constraints = self._sample_constraints(domain_sizes,
                                                         check_random_state(rng))

    def _sample_constraints(self, domain_sizes, rng):
        constraints = []
        for (i, dsi), (j, dsj) in enumerate(it.product(domain_sizes, domain_sizes)):
            # XXX come up with something smarter
            head = np.random.random_integer(0, dsi - 1)
            body = np.random.random_integer(0, dsj - 1)
            index_head = attr_index(domain_sizes, dsi, head)
            index_body = attr_index(domain_sizes, dsj, body)
            constraints.append((index_head, [index_body]))
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
        self.w_constraints = None
        self.x_constraints = None
        self.horn_constraints = None
