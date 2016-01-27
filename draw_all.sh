#!/bin/bash

# setmargin vs guo vs viappiani

for domain_size in `seq 3 4`; do
    if [ $domain_size -eq 3 ]; then
        x_max=21
    else
        x_max=31
    fi
    for sampling_mode in uniform uniform_sparse normal normal_sparse; do
        echo "setmargin vs guo vs viappiani, per iteration, $domain_size $sampling_mode"
        setmargin_files=`ls results/ijcai16/threads\=1/synthetic_${domain_size}/${sampling_mode}/*k=2*.pickle`
        guo_files=`ls results/ijcai16/guo/synthetic_${domain_size}/${sampling_mode}/*.mat`
        viappiani_qi_files=`ls results/ijcai16/viappiani/synthetic_${domain_size}/${sampling_mode}/*QI*.txt`
        viappiani_eus_files=`ls results/ijcai16/viappiani/synthetic_${domain_size}/${sampling_mode}/*EUS*.txt`
        output="paper/figures/synthetic_vs_others_${domain_size}_${sampling_mode}_per_iter"
        ./draw.py 0 $output $x_max "$setmargin_files" "$guo_files" "$viappiani_qi_files" "$viappiani_eus_files"
    done
done

for domain_size in `seq 3 4`; do
    if [ $domain_size -eq 3 ]; then
        x_max=21
    else
        x_max=31
    fi
    for sampling_mode in uniform uniform_sparse normal normal_sparse; do
        echo "setmargin vs guo vs viappiani, per query, $domain_size $sampling_mode"
        setmargin_files=`ls results/ijcai16/threads\=1/synthetic_${domain_size}/${sampling_mode}/*k=2*.pickle`
        guo_files=`ls results/ijcai16/guo/synthetic_${domain_size}/${sampling_mode}/*.mat`
        viappiani_qi_files=`ls results/ijcai16/viappiani/synthetic_${domain_size}/${sampling_mode}/*QI*.txt`
        viappiani_eus_files=`ls results/ijcai16/viappiani/synthetic_${domain_size}/${sampling_mode}/*EUS*.txt`
        output="paper/figures/synthetic_vs_others_${domain_size}_${sampling_mode}_per_query"
        ./draw.py 1 $output $x_max "$setmargin_files" "$guo_files" "$viappiani_qi_files" "$viappiani_eus_files"
    done
done

# setmargin k=2 vs k=3 vs k=4

for domain_size in `seq 3 5`; do
    for sampling_mode in uniform uniform_sparse normal normal_sparse; do
        echo "setmargin vs self, per iteration, $domain_size $sampling_mode"
        k2_files=`ls results/ijcai16/synthetic_${domain_size}/${sampling_mode}/*k=2*50__300*.pickle`
        k3_files=`ls results/ijcai16/synthetic_${domain_size}/${sampling_mode}/*k=3*50__300*.pickle`
        k4_files=`ls results/ijcai16/synthetic_${domain_size}/${sampling_mode}/*k=4*50__300*.pickle`
        output="paper/figures/synthetic_vs_self_${domain_size}_${sampling_mode}_per_iter"
        ./draw.py 0 $output 50 "$k2_files" "$k3_files" "$k4_files"
    done
done

for domain_size in `seq 3 5`; do
    for sampling_mode in uniform uniform_sparse normal normal_sparse; do
        echo "setmargin vs self, per query, $domain_size $sampling_mode"
        k2_files=`ls results/ijcai16/synthetic_${domain_size}/${sampling_mode}/*k=2*100__100*.pickle`
        k3_files=`ls results/ijcai16/synthetic_${domain_size}/${sampling_mode}/*k=3*100__100*.pickle`
        k4_files=`ls results/ijcai16/synthetic_${domain_size}/${sampling_mode}/*k=4*100__100*.pickle`
        output="paper/figures/synthetic_vs_self_${domain_size}_${sampling_mode}_per_query"
        ./draw.py 1 $output 100 "$k2_files" "$k3_files" "$k4_files"
    done
done

# PC

for sampling_mode in uniform_sparse normal_sparse; do
    echo "PC, per iteration, $sampling_mode"
    k2_files=`ls results/ijcai16/pc_with_costs/${sampling_mode}/*k=2*50__300*.pickle`
    k3_files=`ls results/ijcai16/pc_with_costs/${sampling_mode}/*k=3*50__300*.pickle`
    k4_files=`ls results/ijcai16/pc_with_costs/${sampling_mode}/*k=4*50__300*.pickle`
    output="paper/figures/pc_with_costs_${sampling_mode}_per_iter"
    ./draw.py 0 $output 50 "$k2_files" "$k3_files" "$k4_files"
done

for sampling_mode in uniform_sparse normal_sparse; do
    echo "PC, per query, $sampling_mode"
    k2_files=`ls results/ijcai16/pc_with_costs/${sampling_mode}/*k=2*100__100*.pickle`
    k3_files=`ls results/ijcai16/pc_with_costs/${sampling_mode}/*k=3*100__100*.pickle`
    k4_files=`ls results/ijcai16/pc_with_costs/${sampling_mode}/*k=4*100__100*.pickle`
    output="paper/figures/pc_with_costs_${sampling_mode}_per_query"
    ./draw.py 1 $output 100 "$k2_files" "$k3_files" "$k4_files"
done
