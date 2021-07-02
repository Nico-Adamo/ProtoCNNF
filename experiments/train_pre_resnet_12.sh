#!/bin/bash

python3 pretrain.py --max-epoch 150 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.01 \
                 --step-size 10 \
                 --gamma 0.5 \
                 --save-path "models-backbone-feedback/proto-feedback-12" \
                 --gpu 1 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-Pretrain" \
                 --ind-block 1 \
                 --cycles 2

python3 train.py --max-epoch 50 \
                 --save-epoch 20 \
                 --shot 5 \
                 --train-way 5 \
                 --test-way 5 \
                 --lr 0.001 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-feedback-12/epoch-last.pth" \
                 --save-path "models/proto-feedback-12-5shot" \
                 --gpu 1 \
                 --ind-block 1 \
                 --cycles 2 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot"

# try with max acc rather than last epoch
python3 train.py --max-epoch 50 \
                 --save-epoch 20 \
                 --shot 5 \
                 --train-way 5 \
                 --test-way 5 \
                 --lr 0.001 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-baseline/max-acc.pth" \
                 --save-path "models/proto-feedback-12-5shot-2" \
                 --gpu 1 \
                 --ind-block 1 \
                 --cycles 2 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot"
