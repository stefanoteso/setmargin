# -*- coding: utf-8 -*-

import numpy as np
import itertools as it

def onehot(domain_size, value):
    assert 0 <= value < domain_size, "out-of-bounds value: 0/{}/{}".format(value, domain_size)
    value_onehot = np.zeros(domain_size, dtype=np.int8)
    value_onehot[value] = 1
    return value_onehot

def get_synthetic_dataset(domain_sizes=[2, 2, 5]):
    """Builds the synthetic dataset of Guo & Sanner 2010.

    The dataset involves three attributes, with fixed domains sizes; items
    cover all value combinations in the given attributes, for a total of 20
    items.
    """
    items_onehot = None
    for item in it.product(*map(range, domain_sizes)):
        item_onehot = np.hstack((onehot(domain_sizes[i], attribute_value)
                                 for i, attribute_value in enumerate(item)))
        if items_onehot is None:
            items_onehot = item_onehot
        else:
            items_onehot = np.vstack((items_onehot, item_onehot))
    def prod(l):
        return l[0] if len(l) == 1 else l[0]*prod(l[1:])
    assert items_onehot.shape == (prod(domain_sizes), sum(domain_sizes))
    return domain_sizes, items_onehot, np.array([]), np.array([])

def get_pc_dataset():

    ATTRIBUTES = ["Manufacturer", "CPUType", "CPUSpeed", "Monitor", "Type",
                  "Memory", "HDSize", "Price"]

    categories = {}
    categories[0] = ["Apple", "Compaq", "Dell", "Fujitsu", "Gateway", "HP", "Sony", "Toshiba"]
    categories[1] = ["PowerPC G3", "PowerPC G4", "Intel Pentium", "ADM Athlon", "Intel Celeron", "Crusoe", "AMD Duron"]
    categories[4] = ["Laptop", "Desktop", "Tower"]

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
            assert 0 <= row[z] < domain_sizes[z], "invalid value {}/{} in attribute {}".format(row[z], domain_sizes[z], z)

    items_onehot = None
    for item in discretized_items:
        item_onehot = np.hstack((onehot(domain_sizes[z], attribute_value)
                                 for z, attribute_value in enumerate(item)))
        if items_onehot is None:
            items_onehot = item_onehot
        else:
            items_onehot = np.vstack((items_onehot, item_onehot))
    assert items_onehot.shape == (120, sum(domain_sizes))

    return domain_sizes, items_onehot, np.array([]), np.array([])

def get_housing_dataset():
    from scipy.io import loadmat

    # See AISTATS_Housing_Uniform/readXML.m
    CATEGORICAL_FEATURES = [i-1 for i in [4, 9]]
    SCALAR_FEATURES = [i-1 for i in [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14]]
    INTERVAL = 10.0

    items = loadmat("houseMatlab.mat")["housing"]
    assert items.shape == (506, 14)

    num_items, num_features = items.shape

    attribute_info = [None for _ in range(num_features)]
    for z in CATEGORICAL_FEATURES:
        attribute_info[z] = sorted(set(items[:,z]))
    for z in SCALAR_FEATURES:
        l = min(items[:,z])
        u = max(items[:,z])
        # XXX the 1e-6 is there because MATLAB's arange notation is inclusive
        # wrt the upper bound
        attribute_info[z] = np.arange(l-(u-l)/INTERVAL, u+(u-l)/INTERVAL+1e-6, (u-l)/i)

    discretized_items = np.zeros(items.shape, dtype=np.int32)
    for i, item in enumerate(items):
        for z, value in enumerate(item):
            if z in CATEGORICAL_FEATURES:
                discretized_items[i,z] = attribute_info[z].index(value)
            else:
                temp = value - attribute_info[z]
                discretized_items[i,z] = np.where(temp <= 0)[0][0]

    domain_sizes = []
    for z in range(num_features):
        domain_sizes.append(max(discretized_items[:,z]) - min(discretized_items[:,z]) + 1)
        print max(discretized_items[:,z]), min(discretized_items[:,z]), "-->", domain_sizes[z]

    items_onehot = None
    for item in discretized_items:
        item_onehot = np.hstack((onehot(domain_sizes[z], attribute_value - min(discretized_items[:,z]))
                                 for z, attribute_value in enumerate(item)))
        if items_onehot is None:
            items_onehot = item_onehot
        else:
            items_onehot = np.vstack((items_onehot, item_onehot))
    assert items_onehot.shape == (506, sum(domain_sizes))

    return domain_sizes, items_onehot, np.array([]), np.array([])
