Preference Elicitation via Set-wise Max-margin
==============================================

**TODO**: add description.

## Requirements

This package requires the following packages:

- [numpy](http://www.numpy.org/)
- [gurobi](http://www.gurobi.com/)
- [matplotlib](http://matplotlib.org/)

## Usage

To run the IJCAI-16 experiments, simply type:
```
    python ijcai16.py
```
To perform preference elicitation on a specific dataset with given parameters,
type:
```
    python ijcai16.py $dataset $parameters
```
For instance, to run 20 trials of length 10 with set size 3 on the synthetic
dataset, write:
```
    python ijcai16.py synthetic -N 20 -n 10 -m 3
```
See:
```
    python ijcai16.py --help
```
for a full list of the accepted arguments.
