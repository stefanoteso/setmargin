Preference Elicitation via Set-wise Max-margin
==============================================

**TODO**: add description.

## Requirements

This package requires the following packages:

- [numpy](http://www.numpy.org/)
- [gurobi](http://www.gurobi.com/)
- [matplotlib](http://matplotlib.org/)

## Usage

The experiments and source code can be found in the `src/` directory.

From a shell, you can run an experiment with the following command:
```
$ ./main.py $dataset
```
To get a full description of the arguments accepted by the script, run:
```
$ ./main.py --help
```
A more complex usage example:
```
$ ./main.py synthetic -N 2 -n 10 -m 2 -d --seed 0 --debug
```
