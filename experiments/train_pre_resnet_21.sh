#!/bin/bash

python3 pretrain.py --max-epoch 150 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.001 \
                 --save-path "models-backbone-feedback/proto-feedback-21-3" \
                 --gpu 2 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-Pretrain" \
                 --ind-block 2 \
                 --cycles 1

# try with max acc rather than last epoch
python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0001 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-feedback-21-3/max-acc.pth" \
                 --save-path "models/proto-feedback-21-5shot-3" \
                 --gpu 2 \
                 --ind-block 2 \
                 --cycles 1 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot"
