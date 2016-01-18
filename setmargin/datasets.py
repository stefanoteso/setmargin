# -*- coding: utf-8 -*-

import numpy as np
import itertools as it
from sklearn.utils import check_random_state

class Dataset(object):
    """A dataset over the Cartesian product of all attribute domains.

    :param domain_sizes: list of domain sizes.
    :param x_constraints: constraints on the item configurations.
    """
    def __init__(self, domain_sizes, costs, x_constraints):
        self.domain_sizes = domain_sizes
        self.costs = costs
        if costs is not None:
            assert costs.shape == (self.num_reals(), self.num_bools())
        self.x_constraints = x_constraints

    def __str__(self):
        return "Dataset(domain_sizes={} bools={} reals={} constrs={}" \
                   .format(self.domain_sizes, self.num_bools(),
                           self.num_reals(), self.x_constraints)

    def num_bools(self):
        return sum(self.domain_sizes)

    def num_reals(self):
        return 0 if self.costs is None else self.costs.shape[0]

    def num_vars(self):
        return self.num_bools() + self.num_reals()

    def get_domain_ranges(self):
        base = 0
        for size in self.domain_sizes:
            yield base, base + size
            base += size

    def get_zs_in_domains(self):
        zs_in_domains, last_z = [], 0
        for domain_size in self.domain_sizes:
            assert domain_size > 1
            zs_in_domains.append(range(last_z, last_z + domain_size))
            last_z += domain_size
        return zs_in_domains

    def is_item_valid(self, x):
        if (x < -1e-10).any():
            return False
        for zs_in_domain in self.get_zs_in_domains():
            if sum(x[zs_in_domain]) != 1:
                return False
        if self.costs is not None:
            x, c = x[:self.num_bools()], x[self.num_bools():]
            if not (np.dot(self.costs, x) == c).all():
                return False
        if self.x_constraints is not None:
            for head, body in self.x_constraints:
                if x[head] and not any(x[atom] == 1 for atom in body):
                    return False
        return True

    def compose_item(self, x):
        assert x.shape == (self.num_bools(),)
        if self.num_reals() > 0:
            x = np.hstack((x, np.dot(self.costs, x)))
        assert x.shape == (self.num_bools() + self.num_reals(),)
        return x

class SyntheticDataset(Dataset):
    def __init__(self, domain_sizes):
        super(SyntheticDataset, self).__init__(domain_sizes, None, None)

class DebugConstraintDataset(Dataset):
    def __init__(self, domain_sizes, rng=None):
        x_constraints = self._sample_constraints(domain_sizes,
                                                 check_random_state(rng))
        super(DebugConstraintDataset, self).__init__(domain_sizes, None, x_constraints)

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
            index_head = sum(domain_sizes[:i]) + head
            index_body = sum(domain_sizes[:j]) + body
            constraints.append((index_head, [index_body]))
        print "constraints =\n", constraints
        print "--------------------"
        return constraints

class DebugCostDataset(Dataset):
    def __init__(self, domain_sizes, num_costs=2, rng=None):
        rng = check_random_state(rng)
        costs = rng.uniform(0, 1, size=(num_costs, sum(domain_sizes)))
        super(DebugCostDataset, self).__init__(domain_sizes, costs, None)

class PCDataset(Dataset):
    def __init__(self, has_costs=False):
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

        costs_of = {
            "Manufacturer": map(float, [100, 0, 100, 50, 0, 0, 50, 50]),
            "CPU": [
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
            ],
            "Monitor": map(float, [
                0.6*100, 0.6*104, 0.6*120, 0.6*133, 0.6*140, 0.6*150, 0.6*170,
                0.6*210
            ]),
            "Type": map(float, [50, 0, 80]),
            "Memory": [
                0.8*64, 0.8*128, 0.8*160, 0.8*192, 0.8*256, 0.8*320, 0.8*384,
                0.8*512, 0.8*1024, 0.8*2048
            ],
            "HDSize": map(float, [
                4*8, 4*10, 4*12, 4*15, 4*20, 4*30, 4*40, 4*60, 4*80, 4*120
            ]),
        }

        self.attributes = sorted(domain_of.keys())

        domain_sizes = [len(domain_of[attr]) for attr in self.attributes]

        cost, max_cost = [], 0.0
        assert len(domain_of) == len(costs_of)
        for attr in self.attributes:
            assert len(domain_of[attr]) == len(costs_of[attr])
            costs_for_attr = np.array(costs_of[attr])
            max_cost += max(costs_for_attr)
            cost.extend(costs_for_attr)
        costs = np.array([cost]) / max_cost

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

        super(PCDataset, self).__init__(domain_sizes, costs if has_costs else None, x_constraints)

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
