#!/bin/bash

python3 pretrain.py --max-epoch 100 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.001 \
                 --save-path "models-backbone-feedback/wrn-baseline" \
                 --gpu 0 \
                 --model "WRN28" \
                 --project "CNNF-Prototype-Pretrain" \
                 --wandb
