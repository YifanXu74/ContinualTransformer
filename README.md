##  Preparation
**环境依赖**
目前运行环境：
```
cuda==11.7
python==3.10.11
torch==2.0.1
torchvision==0.15.2
timm==0.3.2
```
应该其他环境也可以运行，但是timm必须为0.3.2。其中timm可能和高版本torch存在冲突，解决方法见：[this issue](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842)


**数据格式**

文本json：[{'caption': xxx}, {'caption': xxx}, {'caption': xxx}, ...], 

图像imagenet

按以下方式存储：

```
data/
    ILSVRC2012/
        train/
            n01440764/
                n01440764_18.JPEG
                ...
            ...
        val/
            n01440712/
                n01440212_38.JPEG
                ...
            ...
    
    CC3M/
        cc3m_captions.json

```

**注意！相关必要模型已上传集群，无需再下载,按步骤建立软链接即可：**
```
cd $project_dir
mkdir checkpoints
cd checkpoints
ln -s /userhome/models/BEIT/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth .
ln -s /userhome/models/DALLE/dall_e_tokenizer_weight .
ln -s /userhome/models/ContinualTransformer/checkpoint-reg1e0-cc3m-100ep-merged.pth .
```

必要模型下载（仅用于其他客户端复现）：
```
mkdir checkpoints
cd checkpoints

wget -O checkpoints/beit_base_patch16_224_pt22k_ft22kto1k.pth https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D

mkdir checkpoints/dall_e_tokenizer_weight

wget -O checkpoints/dall_e_tokenizer_weight/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl

wget -O checkpoints/dall_e_tokenizer_weight/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl

```
转换beit权重：
```
python util/convert_beit_ckpt.py
```
最终得到checkponts目录如下：
```
checkpoints/
    beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth
    beit_base_patch16_224_pt22k_ft22kto1k.pth
    dall_e_tokenizer_weight/
        encoder.pkl
        decoder.pkl
```



## Pre-training

Text MLM pre-training:
```
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
--exp_name text_mlm \
--model vlmo_base_patch16 \
--data_path data/CC3M/cc3m_captions.json \
--batch_size 384 \
--output_dir outputs/text_mlm/ \
--log_dir outputs/text_mlm/ \
--resume checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth \
--lora_rank 64 \
--reg_loss_weight 1e1 \
--self_regularization \
--save_per_epochs 20 \
--epochs 100 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
```

Image MIM pre-training:
```
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 main_pretrain_cook.py \
--exp_name image_mim \
--model vlmo_base_patch16 \
--data_path data/ILSVRC2012/train/ \
--batch_size 128 \
--output_dir outputs/image_mim/ \
--log_dir outputs/image_mim/ \
--lora_rank 0 \
--save_per_epochs 20 \
--epochs 800 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
```

# Finetuning
各个下游任务需要自行编写框架，目前代码框架基于MAE代码修改。

**注意：**

1. 目前只修改了预训练相关代码，finetune部分（如main_finetune.py和engine_finetune.py）没有修改，需要自行适配
2. 目前数据加载均为自己实现，见`custom_datasets/`,没有用MAE代码，下游任务需要自行修改撰写相关数据加载
3. 目前模型的forward函数仅写了预训练相关代码，下游任务需要自行适配编写相关forward函数、后端head、训练损失、以及输出评测框架。
4. 目前模型forward输入参数为 samples, mode，目前mode仅支持三种预训练任务: "text_mlm", "image_mim", "image_text_itc"，下游任务需要定义新的mode来传入
5. 目前模型能支持的最大文本token数量为196，最大图像分辨率为224*224
6. 下游任务finetune时`--lora_rank`一律设置成0，不要加`--self_regularization`
7. 下游任务修改可以自行编写`main_finetune.py`和`engine_finetune.py`，可以参考`main_pretrain_cook.py`和`engine_pretrain.py`，可以仿照MAE的代码

目前数据集加载输出格式：
```
custom_datasets/text_dataset.py 文本数据集:
        {
        'raw_text': text_list, # ['caption1', 'caption2', ...]
        }

custom_datasets/image_dataset.py 图像数据集:
        {
        "images": torch.stack(images), # 训练用图像, torch.tensor, [B,3,H,W]
        "images_for_vae": torch.stack(images_for_vae), # 仅用于预训练，下游finetune不需要，torch.tensor, [B,3,H/2,W/2]
        }, 
        targets # 分类标签，torch.tensor
```

CC3M文本MLM预训练模型:
```
/userhome/models/ContinualTransformer/checkpoint-reg1e0-cc3m-100ep-merged.pth
```

文本下游任务可加载上面这个模型，并指定下面参数:
```
--lora_rank 0 --resume $PRETRAINED_CKPT 
```
