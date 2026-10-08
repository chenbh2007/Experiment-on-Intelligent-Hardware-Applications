#!/bin/bash
for TYPE in blobs circles moons bars
do
    mkdir ./$2_result_$TYPE
    gcc generate_data_$TYPE.c -o generate_data.out -lm
    nvcc -O3 -arch=native $1 -o K_means.out -lcublas
    gcc generate_graph.c -o generate_graph.out -lm
    for i in 1 2 3 4 5
    do
        ./generate_data.out && ./K_means.out && ./generate_graph.out $2_result_$TYPE/result_$i.ppm
        echo "test $i complete"
    done
done
