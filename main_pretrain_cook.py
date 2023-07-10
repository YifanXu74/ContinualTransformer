# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc

from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_cook

from engine_pretrain import train_one_epoch

import custom_datasets

import custom_loralib

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    pass

def get_args_parser():
    parser = argparse.ArgumentParser('ContinualTransformer pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_per_epochs', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='vlmo_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--image_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--second_image_size', default=112, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--drop_path_rate', default=0.1, type=float)
    parser.add_argument('--d_vae_type', default="dall-e", type=str,
                        help='type of discrete VAE')
    parser.add_argument('--d_vae_weight_path', default='checkpoints/dall_e_tokenizer_weight/', type=str,
                        help='weight path of discrete VAE')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', nargs='+', help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--disable_imagenet_default_mean_and_std', action='store_true')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # lora training
    parser.add_argument('--lora_rank', default=0, type=int)
    parser.add_argument('--exception', default=['norm.weight', 'norm.bias'], type=list,
                         help='Non-lora parameters kept trainng with lora')
    parser.add_argument('--self_regularization', action='store_true')
    parser.add_argument('--reg_loss_weight', default=1., type=float, 
                        help="weight of the self-regularization loss")

    # cook
    parser.add_argument('--exp_name', default='text_mlm', type=str, choices=['image_mim', 'text_mlm', 'image_text_itc'])

    # languge modeling 
    parser.add_argument('--max_text_len', default=196, type=int)
    parser.add_argument('--max_text_len_of_initckpt', default=196, type=int)
    parser.add_argument('--vocab_size', default=30522, type=int)
    parser.add_argument('--mlm_probability', default=0.15, type=float)

    # image modeling
    parser.add_argument('--mim_probability', default=0.55, type=float)
    parser.add_argument('--img_vocab_size', default=8192, type=int)

    # image_text_matching
    parser.add_argument('--aggregate_itc', action='store_true',
                        help='gather tensors from all gpus to get more negatives to contrast with.')
    parser.set_defaults(aggregate_itc=True)

    # debug 
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_merged_lora_model', action='store_true', 
                        help='if set True, this script only serves to convert a checkpoint to a merged version (for downstreams).')
    parser.add_argument('--convert_ckpt', action='store_true', 
                        help='if set True, this script only serves to convert a checkpoint of BEIT to suit the model.')
    parser.add_argument('--converted_ckpt_save_path', default='', type=str, help='path to save the converted checkpoint if set convert_ckpt to True.')
    return parser


def main(args):
    if len(args.data_path) == 1:
        args.data_path = args.data_path[0]

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ------------------------------------
    # dataset
    if args.exp_name == 'image_mim':
        dataset_train = custom_datasets.build_image_pretraining_dataset(args)
        collate_fn = custom_datasets.simple_image_collate_fn
    elif args.exp_name == 'text_mlm':
        dataset_train = custom_datasets.TextDataset(args.data_path, args.max_text_len)
        collate_fn = custom_datasets.simple_text_collate_fn
    elif args.exp_name == 'image_text_itc':
        dataset_train = custom_datasets.CaptionDataset(args.data_path, config=args)
        collate_fn = custom_datasets.simple_caption_collate_fn
    else:
        raise NotImplementedError
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True if not args.debug else False
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn = collate_fn,
    )


    
    # define the model
    model = models_cook.ContinualModel(args)

    if args.convert_ckpt:
        assert args.resume
        assert args.converted_ckpt_save_path != ''
        model.convert_pretrained_weight(args.resume, args.converted_ckpt_save_path)
        exit()


    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.save_merged_lora_model:
        assert args.lora_rank > 0
        assert args.resume != ''
        misc.save_merged_model(args=args, model_without_ddp=model_without_ddp)
        exit()

    # Set lora training
    if args.exp_name == 'image_mim':
        assert not args.self_regularization, 'MIM is the first pre-training step, no LoRA parameters are provided.'
        assert args.lora_rank == 0
    elif args.exp_name == 'text_mlm':
        assert args.lora_rank > 0
        custom_loralib.mark_only_lora_as_trainable(model_without_ddp.transformer, exception=args.exception)   
    elif args.exp_name == 'image_text_itc':
        assert args.lora_rank > 0
        custom_loralib.mark_only_loraB_as_trainable(model_without_ddp.transformer, exception=args.exception)
    # Debug
    print('Parameter status with lora training:')
    for n, p in model_without_ddp.named_parameters():
        print('{}: {}'.format(n, 'Not Frozen' if p.requires_grad else 'Frozen'))
            

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_per_epochs == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
