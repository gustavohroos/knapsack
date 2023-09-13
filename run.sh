#!/bin/bash

files="
kp_10_269
kp_10_60
kp_20_878
kp_20_879
kp_23_10000
kp_24_6404180
kp_4_11
kp_4_20
kp_5_80
kp_7_50
"

algorithms="
bruteforce
greedy 
dynamic
"

for i in {1..100}; do
    for file in $files; do
        for algorithm in $algorithms; do
                python3 main.py --file $file --algorithm $algorithm
        done
    done
done