#!/bin/bash

for set_size in 1 2 3 4; do
    for multimargin in "" "-M"; do
        for usermode in "" "-d" "--no-indifference"; do
            for samplingmode in uniform normal; do
                ./main.py synthetic -N 10 -m $set_size -a 10 $multimargin $usermode -u $samplingmode -s 0 -S gurobi --debug
            done
        done
    done
done

which inkscape
if [ $? -eq 0 ]; then
    for f in `ls results*.svg`; do
        inkscape -z -e ${f%%.svg}.png -w 512 -h 384 $f
    done
fi
