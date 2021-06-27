#!/bin/bash

python3 pretrain.py --max-epoch 100 \
                 --save-epoch 20 \
                 --query 15 \
                 --save-path "models-backbone/proto-2" \
                 --gpu 0 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-Pretrain"

export CUDA_VISIBLE_DEVICES=1
python3 train.py --max-epoch 300 \
                 --save-epoch 20 \
                 --shot 5 \
                 --train-way 10 \
                 --test-way 5 \
                 --lr 0.005 \
                 --query 15 \
                 --restore-from "models-backbone/proto-2/epoch-last.pth"
                 --save-path "models/proto-pre-2" \
                 --gpu 1 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype"
