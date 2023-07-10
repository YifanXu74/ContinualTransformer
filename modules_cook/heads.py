import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from util import misc


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

class ITCHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc_img = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_txt = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, modality):
        if modality == 'image':
            x = self.fc_img(x)
        elif modality == 'text':
            x = self.fc_txt(x)
        else:
            raise NotImplementedError
        return x


class MIMHead(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, config.img_vocab_size)
    
    def forward(self, x):
        return self.head(x)


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class ITCScoreHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aggregate = config.aggregate_itc
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale

        world_size = misc.get_world_size()
        rank = misc.get_rank()
        if self.aggregate and (world_size > 1):
            # world_size = dist.get_world_size()
            # rank = dist.get_rank()

            # We gather tensors from all gpus to get more negatives to contrast with.
            gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
            )
            all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
            )

            # this is needed to send gradients back everywhere.
            logits_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_per_text = logits_per_image.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

        return {'logits_per_image': logits_per_image, 'logits_per_text': logits_per_text}, labels