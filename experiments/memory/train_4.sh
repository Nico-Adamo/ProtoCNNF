#!/bin/bash

# ALL: Augment size 128

# Memory weight 0.2
python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.2 \
                 --memory-size 256

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.2 \
                 --memory-size 512

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.2 \
                 --memory-size 1024

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.2 \
                 --memory-size 2048

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.2 \
                 --memory-size 4096

# Memory weight 0.5
python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.5 \
                 --memory-size 256

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.5 \
                 --memory-size 512

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.5 \
                 --memory-size 1024

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.5 \
                 --memory-size 2048

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.5 \
                 --memory-size 4096

# Memory weight 0.9
python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.9 \
                 --memory-size 256

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.9 \
                 --memory-size 512

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.9 \
                 --memory-size 1024

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.9 \
                 --memory-size 2048

python3 train.py --restore-from "models-backbone-feedback/resnet-21/max-acc-sim.pth" \
                 --no-save \
                 --gpu 3 \
                 --project "CNNF-Prototype-5shot-memorybank" \
                 --use-cosine-similarity \
                 --bias-shift \
                 --augment-size 128 \
                 --memory-weight 0.9 \
                 --memory-size 4096
