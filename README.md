# ProtoCNNF

An implementation of [Prototypical Network](https://arxiv.org/abs/1703.05175) using generative feedback and memory replay on a variety of backbone models.

### Usage

The experiments folder contains template experiments - each one consists of pretraining using `pretrain.py` and training use `train.py`. 

Miniimagenet images are expected to be in miniimagenet/images, and train/test splits in materials/

Note that the experiments generally have sane defaults while the default args may not.

