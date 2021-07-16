#!/bin/bash

python3 train.py --max-epoch 200 \
                 --save-epoch 20 \
                 --shot 5 \
                 --way 5 \
                 --lr 0.0002 \
                 --query 15 \
                 --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --save-path "models/resnet-21-dist" \
                 --gpu 1 \
                 --ind-block 0 \
                 --cycles 0 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-5shot" \
                 --use-cosine-similarity
