#!/bin/bash
# Copyright 2024 the LlamaFactory team.
#
# World Model Training Demo Script
# This script demonstrates world model fine-tuning with embedding-based autoregression

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Configuration
MODEL_NAME="standardmodelbio/Qwen3-MM-0.6B"
DATASET="wm_demo"  # Changed from embedding_demo to wm_demo
OUTPUT_DIR="./saves/world_model_demo"

echo "🦙🏭 World Model Training Demo"
echo "=============================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET (Patient temporal progression)"
echo "Output: $OUTPUT_DIR"
echo "=============================="

# Run world model training
python -m llamafactory.train.tuner \
    --stage wm \
    --do_train True \
    --model_name $MODEL_NAME \
    --template qwen3_embedding \
    --dataset $DATASET \
    --cutoff_len 512 \
    --learning_rate 1e-4 \
    --num_train_epochs 2.0 \
    --max_samples 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --logging_steps 1 \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir True \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --embedding_loss_weight 1.0 \
    --lm_loss_weight 0.1 \
    --embedding_dim 4 \
    --world_model_loss_type l2 \
    --plot_loss True

# echo ""
# if [ $? -eq 0 ]; then
#     echo "✅ World Model Training completed successfully!"
#     echo "📊 Check $OUTPUT_DIR for results"
# else
#     echo "❌ Training failed!"
#     exit 1
# fi