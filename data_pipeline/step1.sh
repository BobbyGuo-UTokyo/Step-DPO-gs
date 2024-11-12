export MODEL_PATH='./dataset/pretrained-models/Qwen2-VL-7B-Instruct'
export PRED_PATH='./data_pipeline/predictions/Share4oReasoning-train-Qwen2-VL-7B-Instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'
export EVAL_PROMPT='qwen2-vl-step'
export DATA_FILE='./dataset/Share4oReasoning/train.jsonl'
export IMAGE_ROOT='../../data/Share4oReasoning'

CUDA_VISIBLE_DEVICES=2 python eval_vllm.py --model $MODEL_PATH --remainder 0 --n_groups 1 --save_path $PRED_PATH"0.json" --data_file $DATA_FILE --image_root $IMAGE_ROOT --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1
