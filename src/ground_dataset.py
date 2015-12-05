#!/usr/bin/env python2

import itertools as it
from datasets import SyntheticDataset

for i in range(2, 8+1):
    path = "synthetic_dataset_{}.txt".format(i)
    print "writing dataset for {} attributes in {}".format(i, path)
    with open(path, "wb") as fp:
        items = SyntheticDataset([i] * i).items
        assert items.shape == (i**i, i*i)
        for item in items:
            fp.write("{}\n".format(" ".join(map(str, item))))
