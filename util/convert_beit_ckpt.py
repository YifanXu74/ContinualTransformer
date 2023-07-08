import torch
from util.misc import convert_init_ckpt


ckpt_path = 'checkpoints/beit_base_patch16_224_pt22k_ft22kto1k.pth'
state_dict = convert_init_ckpt(ckpt_path)
new_ckpt = {'model': state_dict}
torch.save(new_ckpt, 'checkpoints/beit_base_patch16_224_pt22k_ft22kto1k_transfertovlmo.pth')
