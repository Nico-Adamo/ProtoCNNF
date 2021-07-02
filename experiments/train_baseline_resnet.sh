#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python3 train.py --max-epoch 300 \
                 --save-epoch 20 \
                 --shot 1 \
                 --train-way 5 \
                 --test-way 5 \
                 --lr 0.0005 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-12/epoch-last.pth" \
                 --save-path "models/proto-12-1shot" \
                 --gpu 0 \
                 --ind-block 1 \
                 --cycles 2 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-1shot"

