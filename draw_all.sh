#!/bin/bash

for sampling_mode in uniform uniform_sparse normal normal_sparse; do
    for domain_size in `seq 3 6`; do
        ipath="results/final_cv@1/synthetic_${domain_size}/${sampling_mode}"
        opath="paper/figures/synthetic_${domain_size}_${sampling_mode}_cv1"
        echo "drawing $ipath -> $opath"
        ./draw.py $ipath $opath

        ipath="results/final_cv@5/synthetic_${domain_size}/${sampling_mode}"
        opath="paper/figures/synthetic_${domain_size}_${sampling_mode}_cv5"
        echo "drawing $ipath -> $opath"
        ./draw.py $ipath $opath
    done
done

for sampling_mode in uniform_sparse normal_sparse; do
    ipath="results/final_cv@5/pc_no_costs/${sampling_mode}"
    opath="paper/figures/pc_no_costs_${sampling_mode}_cv5"
    echo "drawing $ipath -> $opath"
    ./draw.py $ipath $opath

    ipath="results/final_cv@5/pc_with_costs/${sampling_mode}"
    opath="paper/figures/pc_with_costs_${sampling_mode}_cv5"
    echo "drawing $ipath -> $opath"
    ./draw.py $ipath $opath
done
