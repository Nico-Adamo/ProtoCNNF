#!/bin/bash

# ALL: Test augment size 40

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-size 256
                 --test-memory-weight 0.5 \
                 --test-transduction-steps 1 \
                 --test-augment-size 40

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-size 256 \
                 --test-memory-weight 0.2 \
                 --test-transduction-steps 1 \
                 --test-augment-size 40

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-size 512 \
                 --test-memory-weight 0.5 \
                 --test-transduction-steps 1 \
                 --test-augment-size 40

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-size 512 \
                 --test-memory-weight 0.2 \
                 --test-transduction-steps 1 \
                 --test-augment-size 40

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-size 1024 \
                 --test-memory-weight 0.5 \
                 --test-transduction-steps 1 \
                 --test-augment-size 40

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256 \
                 --test-memory-size 1024 \
                 --test-memory-weight 0.2 \
                 --test-transduction-steps 1 \
                 --test-augment-size 40


# Ablation:
# Test augment size: 40, 50, 60
# Test memory Weight: 0.2, 0.5
# Test memory bank size: (already done 256), 512, 1024
