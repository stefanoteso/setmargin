import os
import subprocess
import tempfile
import fractions
import numpy as np

def _cls(obj):
    return type(obj).__name__

def onehot(domain_size, value):
    assert 0 <= value < domain_size, "out-of-bounds value: 0/{}/{}".format(value, domain_size)
    value_onehot = np.zeros(domain_size, dtype=np.int8)
    value_onehot[value] = 1
    return value_onehot

def get_zs_in_domains(domain_sizes):
    zs_in_domains, last_z = [], 0
    for domain_size in domain_sizes:
        assert domain_size > 1
        zs_in_domains.append(range(last_z, last_z + domain_size))
        last_z += domain_size
    return zs_in_domains

def float2libsmt(x):
    z = fractions.Fraction(x)
    p, q = z.numerator, z.denominator
    if q == 1:
        ret = str(abs(p))
    else:
        ret = "(/ {} {})".format(abs(p), q)
    if p < 0:
        ret = "(- 0 {})".format(ret)
    return ret

class Binary(object):
    """A simple wrapper around binary executables."""
    def __init__(self, path):
        self.path = path

    def run(self, args, shell=True):
        pipe = subprocess.Popen(self.path + " " + " ".join(args),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=shell)
        out, err = pipe.communicate()
        ret = pipe.wait()
        return ret, out, err

class OptiMathSAT5(object):
    """A dumb wrapper around optimathsat5."""
    def __init__(self, debug=False):
        self._omt5 = Binary("optimathsat")
        self._debug = debug

    def _d(self, msg):
        if self._debug:
            print _cls(self), msg

    def _read_assignment(self, line):
        parts = [part.strip("() ") for part in line.split()]
        parts = [part for part in parts if len(part)]
        if len(parts) == 2 and parts[1] in ("true", "false"):
            return parts[0], parts[1]
        elif len(parts) == 2:
            return parts[0], float(parts[-1])
        elif len(parts) == 3 and parts[1] == '-':
            return parts[0], -float(parts[-1])
        elif len(parts) == 4 and parts[1] == '/':
            return parts[0], float(parts[2]) / float(parts[3])
        elif len(parts) == 5 and parts[1] == '-' and parts[2] == '/':
            return parts[0], -float(parts[3]) / float(parts[4])
        raise NotImplementedError("unhandled assignment string '{}'".format(line))

    def _read_assignments(self, lines):
        lines = [line for line in lines if not line.startswith("#")]
        assert lines[0] == "sat"
        assignments = {}
        for line in lines[1:]:
            if len(line):
                k, v = self._read_assignment(line[2:])
                assignments[k] = v
        return assignments

    def optimize(self, problem):
        fp = tempfile.NamedTemporaryFile(prefix="setmargin_", delete=False)
        try:
            fp.write(problem)
            fp.close()
            args = ["< {}".format(fp.name)]
            self._d("running '{}'".format(" ".join(args)))
            ret, out, err = self._omt5.run(args)
            assert ret == 0, "ERROR: OptiMathSat5: {}@{}:\n{}".format(cost_var, problem_path, err)
            self._d("done")
            assignments = self._read_assignments(out.split("\n"))
        finally:
            if not self._debug:
                os.remove(fp.name)
        return assignments, ret, out, err
