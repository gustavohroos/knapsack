#!/bin/bash

files="
knapPI_1_10000_1000_1 
knapPI_1_1000_1000_1 
knapPI_1_100_1000_1 
knapPI_1_2000_1000_1 
knapPI_1_200_1000_1 
knapPI_1_5000_1000_1 
knapPI_1_500_1000_1 
knapPI_2_10000_1000_1 
knapPI_2_1000_1000_1 
knapPI_2_100_1000_1 
knapPI_2_2000_1000_1 
knapPI_2_200_1000_1 
knapPI_2_5000_1000_1 
knapPI_2_500_1000_1 
knapPI_3_10000_1000_1 
knapPI_3_1000_1000_1 
knapPI_3_100_1000_1 
knapPI_3_2000_1000_1 
knapPI_3_200_1000_1 
knapPI_3_5000_1000_1 
knapPI_3_500_1000_1 
"

algorithms="
greedy 
dynamic 
branch_and_bound 
"

for i in {1..100}; do
    for file in $files; do
        for algorithm in $algorithms; do
                python3 main.py --file $file --algorithm $algorithm
        done
    done
done