#!/bin/bash

make_pngs() {
    which inkscape
    if [ $? -eq 0 ]; then
        for f in `ls results*.svg`; do
            inkscape -z -e ${f%%.svg}.png -w 512 -h 384 $f
        done
    fi
}

test_constraints() {
    for set_size in 1 2 3 4; do
        for sampling_mode in uniform normal; do
            command="./main.py constsynthetic -N 10 -m $set_size -a 10 --is-indifferent -u $sampling_mode -s 0 --debug"
            $command
            if [ $? -ne 0 ]; then
                echo "failed: '$command'"
                exit 1
            fi
        done
    done
}

compare_to_self() {
    for domain_sizes in "2,2" "3,3,3" "4,4,4,4" "5,5,5,5" "6,6,6,6,6,6" "7,7,7,7,7,7,7" "8,8,8,8,8,8,8,8" "9,9,9,9,9,9,9,9,9", "10,10,10,10,10,10,10,10,10,10"; do
        for set_size in 1 2 3 4; do
            for sampling_mode in uniform normal; do
                command="./main.py synthetic --domain-sizes $domain_sizes -N 10 -m $set_size -a 10 --is-indifferent -u $sampling_mode -s 0 --debug"
                $command
                if [ $? -ne 0 ]; then
                    echo "failed: '$command'"
                    exit 1
                fi
            done
        done
    done
}

test_pc() {
    for set_size in 1 2 3 4; do
        for sampling_mode in uniform normal; do
            command="./main.py pc -N 10 -m $set_size -a 10 --is-indifferent -u $sampling_mode -s 0 --debug"
            $command
            if [ $? -ne 0 ]; then
                echo "failed: '$command'"
                exit 1
            fi
        done
    done
}

if [ $# -ne 1 ]; then
    echo "Usage: $0 <what to run>"
    exit 1
fi

case $1 in
    "test_constraints")
        test_constraints
        ;;
    "compare_to_self")
        compare_to_self
        ;;
    "test_pc")
        test_pc
        ;;
esac
make_pngs
