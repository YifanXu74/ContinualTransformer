#!/usr/bin/env bash
######## multi machines multi gpu cards
MASTER_IP=$(bash get_master_ip.sh)
echo ${MASTER_IP}
RANK=$1
echo $1

#if [ ${RANK} -eq 0 ]
#then
 # export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_4,mlx5_5,mlx5_6
 # echo "export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_4,mlx5_5,mlx5_6"

#else
# echo "not master!"
#fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



# Set the path to save checkpoints
OUTPUT_DIR='./output/pretrain_test2'
#DATA_PATH='/userhome/cls_little/'
DATA_PATH='/dataset/'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 --node_rank ${RANK} --master_addr ${MASTER_IP} --master_port 4479 --use_env main_pretrain.py \
   --batch_size 64 \
   --data_path $DATA_PATH \
   --output_dir $OUTPUT_DIR \
   --epochs 800 \
   --model mae_vit_base_patch16 \
   --warmup_epochs 40 \
   --blr 5.e-4 \
   --input_size 224 \
   --norm_pix_loss \
   --mask_ratio 0.75 \
   --warmup_epochs 40 \
   --weight_decay 0.05 \
