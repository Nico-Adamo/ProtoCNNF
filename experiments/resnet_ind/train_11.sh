#!/bin/bash

python3 pretrain.py --max-epoch 100 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.002 \
                 --save-path "models-backbone-feedback/proto-feedback-11" \
                 --gpu 1 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-Pretrain" \
                 --ind-block 1 \
                 --cycles 1

# try with max acc rather than last epoch
python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0002 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-feedback-11/max-acc.pth" \
                 --save-path "models/proto-feedback-11-5shot" \
                 --gpu 1 \
                 --ind-block 1 \
                 --cycles 1 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot"
