#!/bin/bash

# these WORK
python ijcai16.py synthetic -N 1 -n 10 -m 2 -a 1.0000 -b 0.0 -c 0.0 -i -s 0 --domain-sizes 2,2 --debug
python ijcai16.py synthetic -N 1 -n 10 -m 2 -a 1.0000 -b 100.0 -c 100.0 -i -s 0 --domain-sizes 2,2 --debug

# these FAIL (the problem is unbounded)
python ijcai16.py synthetic -N 1 -n 10 -m 2 -a 0.9999 -b 0.0 -c 0.0 -i -s 0 --domain-sizes 2,2 --debug
python ijcai16.py synthetic -N 1 -n 10 -m 2 -a 0.9999 -b 100.0 -c 100.0 -i -s 0 --domain-sizes 2,2 --debug
