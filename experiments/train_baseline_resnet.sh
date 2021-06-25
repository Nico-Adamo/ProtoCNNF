#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python3 train.py --max-epoch 300 \
                 --save-epoch 20 \
                 --shot 5 \
                 --train-way 10 \
                 --test-way 5 \
                 --query 15 \
                 --save-path "models/proto-1" \
                 --gpu 0 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype"

wait
echo "All done"
