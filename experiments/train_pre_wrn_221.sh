#!/bin/bash

python3 pretrain.py --max-epoch 100 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.001 \
                 --save-path "models-backbone-feedback/proto-feedback-21-wrn" \
                 --gpu 1 \
                 --model "WRN28" \
                 --project "CNNF-Prototype-Pretrain" \
                 --ind-block 3 \
                 --ind-layer 2 \
                 --cycles 1

# try with max acc rather than last epoch
python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0001 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-feedback-21-wrn/max-acc.pth" \
                 --save-path "models/proto-feedback-21-wrn-5shot" \
                 --gpu 1 \
                 --ind-block 3 \
                 --ind-layer 2 \
                 --cycles 1 \
                 --model "WRN28" \
                 --project "CNNF-Prototype-5shot"
