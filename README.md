# Zero-Shot Dialogue Disentanglement by Self-Supervised Entangled Response Selection

- This repository is the official implementation of our EMNLP 2021 paper [Zero-Shot Dialogue Disentanglement by Self-Supervised Entangled Response Selection](https://arxiv.org/abs/2110.12646).

## DSTC 8 Subtask 2 Data

1. Download the data from the official competition [site](https://github.com/dstc8-track2/NOESIS-II). Put train.json, dev.json and test.json into dstc8_2/

2. cd dstc8_2/ then bash parse.sh

## Download BERT Checkpoints
Download the BERT checkpoint you want from Huggingface model hub. Set the path in three scripts below.

## Pretraining
bash pertrain.sh

## Finetuning
bash finetune.sh

## Zero-Shot Inference
bash zero_shot.sh
