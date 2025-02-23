DATA_DIR=../rating/alpaca_filtered_data.json
# MODEL_DIR=../models/llama-7b
# MODEL_DIR="../models/open_llama_3b_v2/
MODEL_DIR=../models/alpagasus-7b
OUTPUT_DIR=../models/alpagasus-7b_mine

torchrun --nproc_per_node=2 --master_port=15929 train_alpaca.py \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config ./fsdp.json \
    --report_to "none" \
    # --tf32 True
