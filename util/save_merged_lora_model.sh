#!/usr/bin/env bash

CHECKPOINT_PATH=$1
MODEL_TYPE=$2
OUTPUT_DIR=$3

python main_pretrain_cook.py \
--model ${MODEL_TYPE} \
--resume ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_DIR} \
--lora_rank 64 \
--save_merged_lora_model


# bash util/save_merged_lora_model.sh outputs/text_mlm_regloss_1e1/checkpoint-80.pth vlmo_base_patch16 pretrained_models/