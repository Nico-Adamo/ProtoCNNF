#!/bin/bash

# ALL: Augment size 16

# Memory weight 0.2
python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.2 \
                 --memory-size 256

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.2 \
                 --memory-size 512

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.2 \
                 --memory-size 1024

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.2 \
                 --memory-size 2048

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.2 \
                 --memory-size 4096

# Memory weight 0.5
python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 256

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 512

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 1024

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 2048

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.5 \
                 --memory-size 4096

# Memory weight 0.9
python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.9 \
                 --memory-size 256

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.9 \
                 --memory-size 512

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.9 \
                 --memory-size 1024

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.9 \
                 --memory-size 2048

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 0 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 16 \
                 --memory-weight 0.9 \
                 --memory-size 4096



# Ablation:
# Augment size: 16, 32, 64, 128
# Memory Weight: 0.2, 0.5, 0.9
# Memory bank size: 256, 512, 1024, 2048, 4096
# Use train labels: True, False
