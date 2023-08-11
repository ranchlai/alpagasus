DATA_DIR=../rating/alpaca_filtered_data.json
MODEL_DIR=../models/open_llama_3b_v2
# MODEL_DIR=../models/alpagasus-7b

OUTPUT_DIR=./outputs/

python train_alpaca.py \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    # --fsdp_config ./fsdp.json \
    # --fsdp "full_shard auto_wrap" \
    #--report_to "none" \
    #--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
