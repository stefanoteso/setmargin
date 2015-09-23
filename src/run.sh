#!/bin/bash

N=5
n=10

for set_size in 2 3; do
    for alpha in 1 10 100 1000; do
        for beta in 0 1 10 100; do
            for gamma in 1 10 100 1000; do
                ./main.py synthetic -N $N -n $n -m $set_size -a $alpha -b $beta -c $gamma -d -s 0 --debug > log_${N}_${n}_${set_size}_${alpha}_${beta}_${gamma} &
            done
            wait
        done
    done
done
