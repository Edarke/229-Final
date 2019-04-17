#!/bin/bash

for b in 32 #64
do
    for d in 0 # 1 2 3
    do
        echo "Batch: "
        echo $b
        echo "Device: "
        echo $d

        time CUDA_VISIBLE_DEVICES=$d python3 main.py --epochs 5 --no-early-stop --batch-size $b

    done
done
