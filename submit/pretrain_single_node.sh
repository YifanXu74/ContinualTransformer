DATA_PATH='/mnt/hdd/Datasets/increment_json/cc3m_captions.json'
PROJECT_DIR='/home/yfxu/git/ContinualTransformer'

cd $PROJECT_DIR

# # without self-regularization
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
# --batch_size 384 \
# --save_per_epochs 20 \
# --model vlmo_base_patch16 \
# --epochs 100 \
# --warmup_epochs 40 \
# --blr 1.5e-4 --weight_decay 0.05 \
# --data_path $DATA_PATH \
# --output_dir outputs/test_cc3m/ \
# --log_dir outputs/test_cc3m/ \
# --resume checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth \
# --lora_rank 64 \
# --exp_name text_mlm \
# > logs/test_cc3m.txt


# # without self-regularization
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
# --batch_size 384 \
# --save_per_epochs 20 \
# --model vlmo_base_patch16 \
# --epochs 100 \
# --warmup_epochs 40 \
# --blr 1.5e-4 --weight_decay 0.05 \
# --data_path $DATA_PATH \
# --output_dir outputs/test_cc3m/ \
# --log_dir outputs/test_cc3m/ \
# --resume /home/yfxu/git/ContinualTransformer/outputs/test_cc3m/checkpoint-20.pth \
# --lora_rank 16 \
# --exp_name text_mlm \
# &> logs/test_cc3m_resume.txt

# self-regularization
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
# --batch_size 384 \
# --save_per_epochs 20 \
# --model vlmo_base_patch16 \
# --epochs 100 \
# --warmup_epochs 40 \
# --blr 1.5e-4 --weight_decay 0.05 \
# --data_path $DATA_PATH \
# --output_dir outputs/text_mlm_regloss/ \
# --log_dir outputs/text_mlm_regloss/ \
# --resume checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth \
# --lora_rank 64 \
# --self_regularization \
# --reg_loss_weight 1. \
# --exp_name text_mlm \
# &> logs/test_cc3m_regloss.txt

# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
# --exp_name text_mlm \
# --model vlmo_base_patch16 \
# --data_path data/CC3M/cc3m_captions.json \
# --batch_size 384 \
# --output_dir outputs/text_mlm_regloss_1e1/ \
# --log_dir outputs/text_mlm_regloss_1e1/ \
# --resume checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth \
# --lora_rank 64 \
# --reg_loss_weight 1e1 \
# --self_regularization \
# --save_per_epochs 20 \
# --epochs 100 \
# --warmup_epochs 40 \
# --blr 1.5e-4 --weight_decay 0.05 \
# &> logs/test_cc3m_regloss_1e1.txt

# test opentext
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
# --exp_name text_mlm \
# --model vlmo_base_patch16 \
# --data_file_path /mnt/hdd/Datasets/increment_json/openwebtext_captions_a10000.json \
# --batch_size 64 \
# --output_dir outputs/debug/ \
# --log_dir outputs/debug/ \
# --resume outputs/text_mlm_regloss_1e4/checkpoint-80.pth \
# --lora_rank 64 \
# --reg_loss_weight 1e4 \
# --self_regularization \
# --save_per_epochs 20 \
# --epochs 100 \
# --warmup_epochs 40 \
# --blr 1.5e-4 --weight_decay 0.05 \
# --debug \

torchrun --nnodes=1 --nproc_per_node=2 --master_port=12345 main_pretrain_cook.py \
--exp_name image_text_itc \
--model vlmo_base_patch16 \
--data_file_path /mnt/hdd/Datasets/coco2017/annotations/cococaptions_train2017.json \
--data_path /mnt/hdd/Datasets/coco2017/ \
--batch_size 1 \
--output_dir outputs/debug/ \
--log_dir outputs/debug/ \
--resume checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth \
--lora_rank 64 \
--save_per_epochs 20 \
--epochs 100 \
--warmup_epochs 5 \
--blr 1.5e-4 --weight_decay 0.05

# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
# --exp_name text_mlm \
# --model vlmo_base_patch16 \
# --data_file_path data/CC3M/cc3m_captions.json \
# --batch_size 384 \
# --output_dir outputs/text_mlm_regloss_1e8/ \
# --log_dir outputs/text_mlm_regloss_1e8/ \
# --resume checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth \
# --lora_rank 64 \
# --reg_loss_weight 1e8 \
# --self_regularization \
# --save_per_epochs 20 \
# --epochs 100 \
# --warmup_epochs 40 \
# --blr 1.5e-4 --weight_decay 0.05 \
# &> logs/test_cc3m_regloss_1e8.txt



# torchrun --nnodes=1 --nproc_per_node=2 main_pretrain_cook.py \
# --exp_name image_text_itc \
# --model vlmo_large_patch16 \
# --data_file_path /userhome/datasets/downstream/cococaptions_train2017.json \
# --data_path /userhome/datasets/downstream/coco/ \
# --batch_size 32 \
# --output_dir outputs/debug/ \
# --log_dir outputs/debug/ \
# --resume checkpoints/beit_large_patch16_224_pt22k_ft22k_transfertovlmo.pth \
# --lora_rank 0 \
# --save_per_epochs 20 \
# --epochs 100 \
# --warmup_epochs 0 \
# --blr 1.5e-3 --weight_decay 0.05


torchrun --nnodes=1 --nproc_per_node=2 main_pretrain_cook.py \
--exp_name image_text_itc \
--model vlmo_large_patch16 \
--data_path "/userhome/datasets/pretrain_dataset/400M1/laion400m-data/{00000..19719}.tar" \
--batch_size 32 \
--output_dir outputs/debug/ \
--log_dir outputs/debug/ \
--resume checkpoints/beit_large_patch16_224_pt22k_ft22k_transfertovlmo.pth \
--lora_rank 64 \
--save_per_epochs 20 \
--epochs 100 \
--warmup_epochs 0 \
--blr 1.5e-3 --weight_decay 0.05 \
--webdataset \
--train_num_samples 197200000