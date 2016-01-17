import numpy as np

def _cls(obj):
    return type(obj).__name__

def prod(l):
    return l[0] if len(l) == 1 else l[0]*prod(l[1:])

def onehot(domain_size, value):
    assert 0 <= value < domain_size, "out-of-bounds value: 0/{}/{}".format(value, domain_size)
    value_onehot = np.zeros(domain_size, dtype=np.int8)
    value_onehot[value] = 1
    return value_onehot

def vonehot(domain_sizes, item):
    return np.hstack((onehot(domain_sizes[i], v) for i, v in enumerate(item)))

def get_zs_in_domains(domain_sizes):
    zs_in_domains, last_z = [], 0
    for domain_size in domain_sizes:
        assert domain_size > 1
        zs_in_domains.append(range(last_z, last_z + domain_size))
        last_z += domain_size
    return zs_in_domains
