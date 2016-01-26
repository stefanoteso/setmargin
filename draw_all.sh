#!/bin/bash

for sampling_mode in uniform uniform_sparse normal normal_sparse; do
    for domain_size in `seq 3 6`; do
        ipath="results/ijcai16/synthetic_${domain_size}/${sampling_mode}"
        opath="paper/figures/synthetic_${domain_size}_${sampling_mode}"
        echo "drawing $ipath -> $opath"
        ./draw.py $ipath ${opath}_per_iter 0
        ./draw.py $ipath ${opath}_per_query 1
    done
done

for sampling_mode in uniform_sparse normal_sparse; do
    ipath="results/ijcai16/pc_with_costs/${sampling_mode}"
    opath="paper/figures/pc_with_costs_${sampling_mode}"
    echo "drawing $ipath -> $opath"
    ./draw.py $ipath ${opath}_per_iter 0
    ./draw.py $ipath ${opath}_per_query 1
done
