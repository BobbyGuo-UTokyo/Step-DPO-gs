#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L elapse=72:00:00
#PJM -L gpu=4
#PJM -N name=step1
#PJM -j


module load cuda/12.2.2
module load cudnn/8.9.7
module load gcc-toolset/12
module load nccl/2.22.3
source /home/pj24002027/ku40003401/python_env/step_dpo/bin/activate

cd /home/pj24002027/ku40003401/repos/Step-DPO-gs

export MODEL_PATH='./dataset/pretrained-models/Qwen2-VL-7B-Instruct'
export PRED_PATH='./data_pipeline/predictions/LLaVA-CoT-100k-train-Qwen2-VL-7B-Instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'
export EVAL_PROMPT='qwen2-vl-step'
export DATA_FILE='../../data/LLaVA-CoT-100k/train.jsonl'
export IMAGE_ROOT='../../data/LLaVA-CoT-100k'
export BATCH_SIZE=200

CUDA_VISIBLE_DEVICES=0 python eval_vllm.py --model $MODEL_PATH --remainder 0 --n_groups 4 --batch_size $BATCH_SIZE --save_path $PRED_PATH"0.json" --data_file $DATA_FILE --image_root $IMAGE_ROOT --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=1 python eval_vllm.py --model $MODEL_PATH --remainder 1 --n_groups 4 --batch_size $BATCH_SIZE --save_path $PRED_PATH"1.json" --data_file $DATA_FILE --image_root $IMAGE_ROOT --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --model $MODEL_PATH --remainder 2 --n_groups 4 --batch_size $BATCH_SIZE --save_path $PRED_PATH"2.json" --data_file $DATA_FILE --image_root $IMAGE_ROOT --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=3 python eval_vllm.py --model $MODEL_PATH --remainder 3 --n_groups 4 --batch_size $BATCH_SIZE --save_path $PRED_PATH"3.json" --data_file $DATA_FILE --image_root $IMAGE_ROOT --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1
