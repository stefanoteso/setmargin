#!/bin/bash

for domain_size in `seq 2 5`; do
    for sampling_mode in uniform uniform_sparse normal normal_sparse; do
        echo "drawing $domain_size $sampling_mode"
        ./draw.py results/synthetic_${domain_size}/${sampling_mode} paper/figures/synthetic_${domain_size}_${sampling_mode}.png
    done
done
