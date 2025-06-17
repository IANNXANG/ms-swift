export MODELSCOPE_CACHE="/c22940/zy/cache"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=2,3,4,5
export MAX_PIXELS=12845056
export MIN_PIXELS=3136
export IMAGE_FACTOR=28
# 4 * 50GiB
nproc_per_node=4


#'swift/RLAIF-V-Dataset#20000'    /c22940/zy/code/ms-swift/code2image.jsonl
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type rm \
    --model /c22940/zy/model/Qwen2.5-VL-7B-Instruct \
    --dataset /c22940/zy/code/ms-swift/code2image.jsonl \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 4 \
    --report_to wandb \
    --save_only_model true
