#!/bin/bash

cd /projects/p32958/chengxuan/ProgressLM/eval/baseline/LIV/eval

CUDA_VISIBLE_DEVICES=0 python run_liv_visual_demo.py \
    --model-path /projects/p32958/chengxuan/models/LIV \
    --dataset-path /projects/p32958/chengxuan/ProgressLM/data/benchmark/visual/visual_eval_one_view.jsonl \
    --image-root /projects/p32958/chengxuan/data/images \
    --output-dir ./outputs \
    --batch-size 2991
