#!/bin/bash

python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0002 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/resnet-feedback-22/max-acc-sim.pth" \
                 --save-path "models/resnet-feedback-22-inter" \
                 --gpu 1 \
                 --ind-block 2 \
                 --cycles 2 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot" \
                 --use-cosine-similarity \
                 --inter-cycle-loss
