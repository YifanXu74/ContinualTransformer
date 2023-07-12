# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
from torch import nn 

from scipy import interpolate
import numpy as np
import random

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, webdataset=False):
        i = 0
        if webdataset:
            len_iter = iterable.num_batches
        else:
            len_iter = len(iterable)
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len_iter))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iter - 1:
                eta_seconds = iter_time.global_avg * (len_iter - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iter, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len_iter, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len_iter))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_merged_model(args, model_without_ddp):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ('checkpoint-merged-{}.pth'.format(str(random.randint(0, 99999)).zfill(5)))
    model_without_ddp.eval() # set eval to enable lora merging
    to_save = {
                'model': model_without_ddp.state_dict(),
            }
    print('save merged model to {}'.format(checkpoint_path))
    save_on_master(to_save, checkpoint_path)

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False) # debug
        # model_without_ddp.load_state_dict(checkpoint['model']) 
        print("missing_keys: {}".format(missing_keys))
        print("unexpected_keys: {}".format(unexpected_keys))

        if not (args.debug or args.save_merged_lora_model):
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint

def convert_init_ckpt(ckpt_path, module, config):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}

    # Merge relative_position_bias_table from all layer into one tensor, 
    # so we can use one op for gather the relative position bias for speed up
    relative_position_bias_tables = {}

    for key in state_dict:
        value = state_dict[key]

        if "relative_position_bias_table" in key:
            # transformer.blocks.0.attn.relative_position_bias_table
            layer_idx = int(key.split(".attn.")[0].split('.')[-1])
            relative_position_bias_tables[layer_idx] = value
            continue

        if "mlp" in key:
            key_imag = "transformer." + key
            new_state_dict[key_imag] = value
        elif "norm2" in key:
            key_imag = "transformer." + key
            new_state_dict[key_imag] = value
        else:
            new_key = "transformer." + key
            new_state_dict[new_key] = value
    
    if len(relative_position_bias_tables) > 0:
        tensor_list = []
        for layer_idx in sorted(relative_position_bias_tables.keys()):
            tensor_list.append(relative_position_bias_tables[layer_idx])
        relative_position_bias_table = torch.cat(tensor_list, dim=1)

        num_distence, _ = relative_position_bias_table.shape
        all_relative_position_bias_table = module.relative_position_bias_table.data.clone()
        all_relative_position_bias_table[:num_distence, :] = relative_position_bias_table

        new_state_dict["relative_position_bias_table"] = all_relative_position_bias_table
    

    state_dict = new_state_dict

    max_text_len = config.max_text_len

    if "text_embeddings.position_embeddings.weight" in state_dict and state_dict["text_embeddings.position_embeddings.weight"].size(0) != max_text_len:
        state_dict["text_embeddings.position_embeddings.weight"].data = state_dict["text_embeddings.position_embeddings.weight"].data[:max_text_len, :]
        state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:, :max_text_len]

        for check_key in ("relative_position_index", "text_relative_position_index", "text_imag_relative_position_index"):
            if check_key in state_dict:
                state_dict.pop(check_key)

    if "transformer.pos_embed" in state_dict:
        pos_embed_reshaped = interpolate_pos_embed(state_dict['transformer.pos_embed'], module.transformer)         
        state_dict['transformer.pos_embed'] = pos_embed_reshaped

    if "relative_position_bias_table" in state_dict:
        rel_pos_bias = state_dict["relative_position_bias_table"]
        src_num_pos, num_attn_heads = rel_pos_bias.size()
        dst_num_pos, _ = module.relative_position_bias_table.size()
        dst_patch_shape = module.transformer.patch_embed.patch_shape
        if dst_patch_shape[0] != dst_patch_shape[1]:
            raise NotImplementedError()
        num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
        if src_size != dst_size:
            state_dict.pop("relative_position_index")
            state_dict.pop("text_relative_position_index")
            state_dict.pop("text_imag_relative_position_index")
            
            extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
            rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

            def geometric_progression(a, r, n):
                return a * (1.0 - r ** n) / (1.0 - r)

            left, right = 1.01, 1.5
            while right - left > 1e-6:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src_size // 2)
                if gp > dst_size // 2:
                    right = q
                else:
                    left = q

            # if q > 1.090307:
            #     q = 1.090307

            dis = []
            cur = 1
            for i in range(src_size // 2):
                dis.append(cur)
                cur += q ** (i + 1)

            r_ids = [-_ for _ in reversed(dis)]

            x = r_ids + [0] + dis
            y = r_ids + [0] + dis

            t = dst_size // 2.0
            dx = np.arange(-t, t + 0.1, 1.0)
            dy = np.arange(-t, t + 0.1, 1.0)


            all_rel_pos_bias = []

            for i in range(num_attn_heads):
                z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                f = interpolate.interp2d(x, y, z, kind='cubic')
                all_rel_pos_bias.append(
                    torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

            new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
            state_dict["relative_position_bias_table"] = new_rel_pos_bias

        
    return state_dict