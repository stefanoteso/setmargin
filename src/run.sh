#!/bin/bash

for dataset in synthetic constsynthetic; do
    for set_size in 1 2 3 4; do
        for multimargin in "" "-M"; do
            for usermode in "" "--is-deterministic" "--is-indifferent"; do
                for samplingmode in uniform normal; do
                    command="./main.py synthetic -N 10 -m $set_size -a 10 $multimargin $usermode -u $samplingmode -s 0 --debug"
                    $command
                    if [ $? -ne 0 ]; then
                        echo "$command failed."
                        exit 1
                    fi
                done
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
