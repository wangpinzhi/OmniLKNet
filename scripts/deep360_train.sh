#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 80\
                --batch_size 1\
                --checkpoint deep360_train\
                --num_workers 4\
                --dataset deep360\
                --dataset_directory /home/data/wangpinzhi/Omni_Transformer/Deep360\
                
