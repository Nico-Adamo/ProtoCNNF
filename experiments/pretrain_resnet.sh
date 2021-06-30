#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python3 pretrain.py --max-epoch 100 \
                 --save-epoch 20 \
                 --query 15 \
                 --lr 0.1 \
                 --step-size 10 \
                 --gamma 0.5 \
                 --save-path "models-backbone-feedback/proto-1" \
                 --gpu 0 \
                 --model "ResNet12" \
                 --project "CNNF-Prototype-Pretrain"
                 --ind-block 0 \
                 --cycles 0 \
wait
echo "All done"
