#!/bin/bash

python3 pretrain.py --max-epoch 100 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.001 \
                 --save-path "models-backbone-feedback/proto-feedback-22-wrn" \
                 --gpu 0 \
                 --model "WRN28" \
                 --project "CNNF-Prototype-Pretrain" \
                 --ind-block 2 \
                 --ind-layer 2 \
                 --bias-shift \
                 --cycles 1

# try with max acc rather than last epoch
python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0001 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/proto-feedback-22-wrn/max-acc-sim.pth" \
                 --save-path "models/proto-feedback-22-wrn-5shot" \
                 --gpu 0 \
                 --ind-block 2 \
                 --ind-layer 2 \
                 --cycles 1 \
                 --model "WRN28" \
                 --bias-shift \
                 --use-cosine-similarity \
                 --memory-size 0
                 --project "CNNF-Prototype-5shot"
