# -*- coding: utf-8 -*-

import numpy as np
import itertools as it
from sklearn.utils import check_random_state
from util import *

class Dataset(object):
    """A dataset over the Cartesian product of all attribute domains.

    .. warning::

        In lifted settings, i.e. where the dataset is only specificed
        by constraints, items is ``None``.

    :param domain_sizes: list of domain sizes.
    :param items: array of items as one-hot row vectors.
    :param x_constraints: constraints on the item configurations.
    """
    def __init__(self, domain_sizes, items, x_constraints):
        self.domain_sizes = domain_sizes
        self.items = items
        self.x_constraints = x_constraints

    def __str__(self):
        return "Dataset(domain_sizes={} len(items)={} constrs={}" \
                   .format(self.domain_sizes,
                           len(self.items) if self.items is not None else 0,
                           self.x_constraints)

    def _dom_var_to_bit(self, j, z):
        assert 0 <= j <= len(self.domain_sizes)
        assert 0 <= z <= self.domain_sizes[j]
        return sum(self.domain_sizes[:j]) + z

    def _ground(self, domain_sizes, x_constraints):
        items = np.array([vonehot(domain_sizes, item)
                          for item in it.product(*map(range, domain_sizes))])
        assert items.shape == (prod(domain_sizes), sum(domain_sizes))
        # XXX filter out invalid configurations
        return items

    def is_item_valid(self, x):
        for zs_in_domain in get_zs_in_domains(self.domain_sizes):
            if sum(x[z] for z in zs_in_domain) != 1:
                return False
        if self.x_constraints is not None:
            for head, body in self.x_constraints:
                if x[head] and not any(x[atom] == 1 for atom in body):
                    return False
        return True

class SyntheticDataset(Dataset):
    def __init__(self, domain_sizes, x_constraints=None):
        items = self._ground(domain_sizes, x_constraints)
        super(SyntheticDataset, self).__init__(domain_sizes, items, x_constraints)

class RandomDataset(Dataset):
    def __init__(self, domain_sizes, rng=None):
        x_constraints = self._sample_constraints(domain_sizes,
                                                 check_random_state(rng))
        super(RandomDataset, self).__init__(domain_sizes, x_constraints)

    def _sample_constraints(self, domain_sizes, rng):
        print "sampling constraints"
        print "--------------------"
        print "domain_sizes =", domain_sizes
        constraints = []
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

        super(PCDataset, self).__init__(domain_sizes, items_onehot, None)

class LiftedPCDataset(Dataset):
    def __init__(self):
        domain_of = {
            "Manufacturer": [
                "Apple", "Compaq", "Dell", "Fujitsu", "Gateway", "HP", "Sony",
                "Toshiba"
            ],
            "CPU": [
                "AMD Athlon @1000", "AMD Athlon @1330",
                "AMD Duron @700", "AMD Duron @900",
                "Crusoe @800",
                "Intel Celeron @500", "Intel Celeron @600",
                "Intel Celeron @800", "Intel Celeron @900",
                "Intel Celeron @1000", "Intel Celeron @1100",
                "Intel Celeron @1200", "Intel Celeron @1300",
                "Intel Celeron @1400", "Intel Celeron @1700",
                "Intel Pentium @500", "Intel Pentium @600",
                "Intel Pentium @800", "Intel Pentium @900",
                "Intel Pentium @1000", "Intel Pentium @1100",
                "Intel Pentium @1300", "Intel Pentium @1500",
                "Intel Pentium @1600", "Intel Pentium @1700",
                "Intel Pentium @1800", "Intel Pentium @2200",
                "PowerPC G3 @266", "PowerPC G3 @300", "PowerPC G3 @400",
                "PowerPC G3 @450", "PowerPC G3 @500", "PowerPC G3 @550",
                "PowerPC G3 @600", "PowerPC G3 @700",
                "PowerPC G4 @700", "PowerPC G4 @733",
            ],
            "Monitor": [10, 10.4, 12, 13.3, 14, 15, 17, 21],
            "Type": ["Laptop", "Desktop", "Tower"],
            "Memory": [64, 128, 160, 192, 256, 320, 384, 512, 1024, 2048],
            "HDSize": [8, 10, 12, 15, 20, 30, 40, 60, 80, 120],
        }
        self.domain_of = domain_of

        price_of = {
            "Manufacturer": [100, 0, 100, 50, 0, 0, 50, 50],
            "CPU": map(int, [
                # AMD Athlon
                1.4*100, 1.4*130,
                # AMD Duron
                1.1*70, 1.1*90,
                # Crusoe
                1.2*80,
                # Intel Celeron
                1.2*50, 1.2*60, 1.2*80, 1.2*90, 1.2*100, 1.2*110, 1.2*120,
                1.2*130, 1.2*140, 1.2*170,
                # Intel Pentium
                1.5*50, 1.5*60, 1.5*80, 1.5*90, 1.5*100, 1.5*110, 1.5*130,
                1.5*150, 1.5*160, 1.5*170, 1.5*180, 1.5*220,
                # PowerPC G3
                1.4*27, 1.4*30, 1.4*40, 1.4*45, 1.4*50, 1.4*55, 1.4*60, 1.4*70,
                # PowerPC G4
                1.6*70, 1.6*73
            ]),
            "Monitor": [
                0.6*100, 0.6*104, 0.6*120, 0.6*133, 0.6*140, 0.6*150, 0.6*170,
                0.6*210
            ],
            "Type": [80, 0, 120],
            "Memory": [
                0.8*64, 0.8*128, 0.8*160, 0.8*192, 0.8*256, 0.8*320, 0.8*384,
                0.8*512, 0.8*1024, 0.8*2048
            ],
            "HDSize": [
                4*8, 4*10, 4*12, 4*15, 4*20, 4*30, 4*40, 4*60, 4*80, 4*120
            ],
        }

        assert len(domain_of) == len(price_of)
        assert all(len(domain_of[attr]) == len(price_of[attr]) for attr in domain_of)

        self.attributes = sorted(domain_of.keys())

        domain_sizes = [len(domain_of[attr]) for attr in self.attributes]

        x_constraints = []

        # Manufacturer->Type
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Compaq"]),
            ("Type", ["Laptop", "Desktop"])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Fujitsu"]),
            ("Type", ["Laptop"])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["HP"]),
            ("Type", ["Desktop"])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Sony"]),
            ("Type", ["Laptop", "Tower"])))

        # Manufacturer->CPU
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Apple"]),
            ("CPU", [
                "PowerPC G3 @266", "PowerPC G3 @300", "PowerPC G3 @400",
                "PowerPC G3 @450", "PowerPC G3 @500", "PowerPC G3 @550",
                "PowerPC G3 @600", "PowerPC G3 @700",
                "PowerPC G4 @700", "PowerPC G4 @733",
             ])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Compaq", "Sony"]),
            ("CPU", [
                "AMD Athlon @1000", "AMD Athlon @1330",
                "AMD Duron @700", "AMD Duron @900",
                "Intel Celeron @500", "Intel Celeron @600",
                "Intel Celeron @800", "Intel Celeron @900",
                "Intel Celeron @1000", "Intel Celeron @1100",
                "Intel Celeron @1200", "Intel Celeron @1300",
                "Intel Celeron @1400", "Intel Celeron @1700",
                "Intel Pentium @500", "Intel Pentium @600",
                "Intel Pentium @800", "Intel Pentium @900",
                "Intel Pentium @1000", "Intel Pentium @1100",
                "Intel Pentium @1300", "Intel Pentium @1500",
                "Intel Pentium @1600", "Intel Pentium @1700",
                "Intel Pentium @1800", "Intel Pentium @2200",
             ])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Fujitsu"]),
            ("CPU", [
                "Crusoe @800",
                "Intel Celeron @500", "Intel Celeron @600",
                "Intel Celeron @800", "Intel Celeron @900",
                "Intel Celeron @1000", "Intel Celeron @1100",
                "Intel Celeron @1200", "Intel Celeron @1300",
                "Intel Celeron @1400", "Intel Celeron @1700",
                "Intel Pentium @500", "Intel Pentium @600",
                "Intel Pentium @800", "Intel Pentium @900",
                "Intel Pentium @1000", "Intel Pentium @1100",
                "Intel Pentium @1300", "Intel Pentium @1500",
                "Intel Pentium @1600", "Intel Pentium @1700",
                "Intel Pentium @1800", "Intel Pentium @2200",
             ])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["Dell", "Gateway", "Toshiba"]),
            ("CPU", [
                "Intel Celeron @500", "Intel Celeron @600",
                "Intel Celeron @800", "Intel Celeron @900",
                "Intel Celeron @1000", "Intel Celeron @1100",
                "Intel Celeron @1200", "Intel Celeron @1300",
                "Intel Celeron @1400", "Intel Celeron @1700",
                "Intel Pentium @500", "Intel Pentium @600",
                "Intel Pentium @800", "Intel Pentium @900",
                "Intel Pentium @1000", "Intel Pentium @1100",
                "Intel Pentium @1300", "Intel Pentium @1500",
                "Intel Pentium @1600", "Intel Pentium @1700",
                "Intel Pentium @1800", "Intel Pentium @2200",
             ])))
        x_constraints.extend(self._to_constraints(
            ("Manufacturer", ["HP"]),
            ("CPU", [
                "Intel Pentium @500", "Intel Pentium @600",
                "Intel Pentium @800", "Intel Pentium @900",
                "Intel Pentium @1000", "Intel Pentium @1100",
                "Intel Pentium @1300", "Intel Pentium @1500",
                "Intel Pentium @1600", "Intel Pentium @1700",
                "Intel Pentium @1800", "Intel Pentium @2200",
             ])))

        # Type->Memory
        x_constraints.extend(self._to_constraints(
            ("Type", ["Laptop"]),
            ("Memory", [64, 128, 160, 192, 256, 320, 384, 512, 1024])))
        x_constraints.extend(self._to_constraints(
            ("Type", ["Desktop"]),
            ("Memory", [128, 256, 512, 1024])))
        x_constraints.extend(self._to_constraints(
            ("Type", ["Tower"]),
            ("Memory", [256, 512, 1024, 2048])))

        # Type->HDSize
        x_constraints.extend(self._to_constraints(
            ("Type", ["Desktop", "Tower"]),
            ("HDSize", [20, 30, 40, 60, 80, 120])))
        x_constraints.extend(self._to_constraints(
            ("Type", ["Laptop"]),
            ("HDSize", [8, 10, 12, 15, 20, 30])))

        # Type->Monitor
        x_constraints.extend(self._to_constraints(
            ("Type", ["Desktop", "Tower"]),
            ("Monitor", [15, 17, 21])))
        x_constraints.extend(self._to_constraints(
            ("Type", ["Laptop"]),
            ("Monitor", [10, 10.4, 12, 13.3, 14, 15])))

        super(LiftedPCDataset, self).__init__(domain_sizes, None, x_constraints)

    def _attr_value_to_bit(self, attr, value):
        base, i = 0, None
        for attr_j in self.attributes:
            if attr == attr_j:
                assert value in self.domain_of[attr]
                i = self.domain_of[attr].index(value)
                break
            base += len(self.domain_of[attr])
        assert i is not None
        return base + i

    def _to_constraints(self, body_vars, head_vars):
        constraints = []
        body_attr, body_vals = body_vars
        head_attr, head_vals = head_vars
        for body_val in body_vals:
            constraints.append((self._attr_value_to_bit(body_attr, body_val),
                                [self._attr_value_to_bit(head_attr, head_val)
                                 for head_val in head_vals]))
        return constraints
