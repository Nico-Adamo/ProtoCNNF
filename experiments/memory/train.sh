#!/bin/bash

python3 train.py --max-epoch 200 \
                 --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-weight 0.5 \
                 --test-query 50 \
                 --test-augment-size 50 \
                 --transductive \
                 --test-transduction-steps 1
