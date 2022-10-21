#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 80\
                --batch_size 1\
                --lr 0.001 \
                --maxdisp 192 \
                --checkpoint deep360_train\
                --num_workers 4\
                --train \
                --dataset deep360\
                --dataset_directory /home/data/wangpinzhi/Omni_Transformer/Deep360\
                
