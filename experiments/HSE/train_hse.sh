#!/bin/bash

python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0002 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/resnet-feedback-21/max-acc-sim.pth" \
                 --save-path "models/resnet-feedback-21-inter" \
                 --gpu 0 \
                 --ind-block 2 \
                 --cycles 1 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot" \
                 --use-cosine-similarity \
                 --memory-start 1000 \
                 --test-query 15
