import torch
import os
from torch import nn


from modules_cook.modeling_discrete_vae import Dalle_VAE, DiscreteVAE

def create_d_vae(weight_path, d_vae_type, image_size, device):
    if d_vae_type == "dall-e":
        return get_dalle_vae(weight_path, image_size, device)
    elif d_vae_type == "customized":
        return get_d_vae(weight_path, image_size, device)
    else:
        raise NotImplementedError()

def get_dalle_vae(weight_path, image_size, device):
    vae = Dalle_VAE(image_size)
    vae.load_model(model_dir=weight_path, device=device)
    return vae

def get_d_vae(weight_path, image_size, device):
    NUM_TOKENS = 8192
    NUM_LAYERS = 3
    EMB_DIM = 512
    HID_DIM = 256

    state_dict = torch.load(os.path.join(weight_path, "pytorch_model.bin"), map_location="cpu")["weights"]

    model = DiscreteVAE(
        image_size=image_size,
        num_layers=NUM_LAYERS,
        num_tokens=NUM_TOKENS,
        codebook_dim=EMB_DIM,
        hidden_dim=HID_DIM,
    ).to(device)

    model.load_state_dict(state_dict)
    return model

class MIMProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_vae = create_d_vae(weight_path=config.d_vae_weight_path, d_vae_type=config.d_vae_type, image_size=config.image_size, device='cpu')
        self.mim_probability = config.mim_probability

        for _, p in self.d_vae.named_parameters():
             p.requires_grad = False


    @torch.no_grad()
    def forward(self, images):
        input_ids = self.d_vae.get_codebook_indices(images).flatten(1)
        bool_mask = torch.bernoulli(torch.full(input_ids.shape, self.mim_probability, device=images.device, dtype=torch.float)) # N+1 for CLS token
        input_ids[bool_mask==0] = -100
        labels = input_ids

        return labels, bool_mask






