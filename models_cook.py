import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from timm.models import create_model

from util.misc import init_weights

from modules_cook.text_modules import get_pretrained_tokenizer, ProcessorForWholeWordMask
from modules_cook.image_modules import MIMProcessor
from modules_cook import heads
import modules_cook.backbone_modules
from util.my_metrics import Accuracy, VQAScore, Scalar
from timm.models.layers import trunc_normal_


class ContinualModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_regularization = config.self_regularization
        # backbone & patch projection
        self.img_size = config.image_size
        self.transformer = create_model(
            config.model, # debug
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config.drop_path_rate,
            attn_drop_rate=0,
            drop_block_rate=None,
            config=self.config,
        )
        self.patch_size = self.transformer.patch_size
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)


        # image modeling
        self.mim_processor = MIMProcessor(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        trunc_normal_(self.mask_token, std=0.02)
        self.mim_score = heads.MIMHead(config, hidden_dim=self.transformer.num_features)
        self.mim_score.apply(init_weights)



        # language embedding
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=self.num_features,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_path_rate,
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute", 
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(init_weights)

        # task layers        
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(init_weights)

        # language modeling
        self.tokenizer = get_pretrained_tokenizer('bert-base-uncased')
        self.mlm_processor = ProcessorForWholeWordMask(self.tokenizer, mlm_probability=config.mlm_probability)
        self.mlm_score = heads.MLMHead(bert_config)
        self.mlm_score.apply(init_weights)
        self.mlm_accuracy = Accuracy()
        
        self.load_pretrained_weight()

    def load_pretrained_weight(self):
        pass

    def get_rel_pos_bias(self, relative_position_index):
        if self.relative_position_embed:
            relative_position_bias = F.embedding(relative_position_index.long().to(self.relative_position_bias_table.device),
                                                    self.relative_position_bias_table)
            all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
            relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)
            return relative_position_bias_list
        else:
            return [None] * self.num_layers

    def build_relative_position_embed(self, config):
        if not self.transformer.need_relative_position_embed:
            self.relative_position_embed = False
            self.text_imag_relative_position_index = None
            self.text_relative_position_index = None
            self.relative_position_index = None
            return
        self.relative_position_embed = True
        window_size = (int(self.img_size / self.patch_size), int(self.img_size / self.patch_size)) #(14, 14)
        num_heads = self.transformer.num_heads
        max_text_len_of_initckpt = config.max_text_len_of_initckpt #196
        max_text_len = config.max_text_len #40
        max_imag_len = window_size[0] * window_size[1] + 1 #197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers))
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index
        
        text_position_ids = torch.arange(max_text_len-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        # rank_zero_info("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index
        
        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        text_row_relative_position_index = torch.cat((text_relative_position_index, text2imag_relative_position_index), 1)
        imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, relative_position_index), 1)
        text_imag_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0)
        self.text_imag_relative_position_index = text_imag_relative_position_index

    def forward_text(self, samples, device=None, mask_text=False):
        assert device is not None
        samples = samples['raw_text']
        samples = self.tokenizer(
            samples,
            padding="longest",
            truncation=True,
            max_length=self.config.max_text_len,
            return_special_tokens_mask=True,
        )

        if mask_text:
            inputs, labels = self.mlm_processor(samples, device).values()
        else:
            inputs = torch.tensor(samples['input_ids'], device=device)
            labels = None
        attention_mask = samples['attention_mask']
        attention_mask = torch.tensor(attention_mask, device=device)
        text_embeds = self.text_embeddings(inputs)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(attention_mask))

        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        x = text_embeds
        if self.self_regularization:
            reg_loss = x.new_zeros(1)[0]
            for i, blk in enumerate(self.transformer.blocks):
                x, loss = blk(x, mask=attention_mask, relative_position_bias=relative_position_bias_list[i])
                reg_loss +=loss
        else:
            for i, blk in enumerate(self.transformer.blocks):
                x = blk(x, mask=attention_mask, relative_position_bias=relative_position_bias_list[i])
        lffn_hiddens = x
        text_feats = self.transformer.norm(lffn_hiddens)

        ret = {
            "feats": text_feats,
            "cls_feats": text_feats[:, 0],
            "raw_cls_feats": x[:, 0],
            "labels": labels,
            "masks": attention_mask,
            "reg_loss": reg_loss if self.self_regularization else None
        } 
        return ret
    
    def forward_image(self, samples, mask_image=False):
        img = samples['images']
        if mask_image:
            img_vae = samples['images_for_vae']
            labels, bool_mask = self.mim_processor(img_vae)
            labels = torch.cat([labels.new_full((labels.shape[0],1), -100), labels], dim=1) # add label for CLS token
            image_embeds, image_masks = self.transformer.visual_embed(img, bool_mask=bool_mask, mask_token=self.mask_token)
        else:
            image_embeds, image_masks = self.transformer.visual_embed(img)
            labels = None

        image_masks = image_masks.long().to(device=img.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, 1)
            )
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        x = image_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=image_masks, relative_position_bias=relative_position_bias_list[i])
        vffn_hiddens = x
        image_features = self.transformer.norm(vffn_hiddens)

        ret = {
            "feats": image_features,
            "cls_feats": image_features[:, 0],
            "raw_cls_feats": x[:, 0],
            "labels": labels,
            "masks": image_masks,
            "reg_loss": None
        } 
        return ret

    def forward(self, samples, mode):
        if mode == 'text_mlm':
            assert self.training
            losses = {}

            device = samples['target_device']
            ret = self.forward_text(samples, device=device, mask_text=True)
            mlm_logits = self.mlm_score(ret["feats"])
            mlm_labels = ret["labels"]
            losses['mlm_loss'] = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            acc =self.mlm_accuracy(mlm_logits, mlm_labels)     

            if self.self_regularization:
                losses['reg_loss'] = ret["reg_loss"] * self.config.reg_loss_weight
            
            return losses, acc
        elif mode == 'image_mim':
            assert self.training
            losses = {}

            ret = self.forward_image(samples, mask_image=True)
            mim_logits = self.mim_score(ret["feats"])
            mim_labels = ret["labels"]
            losses['mim_loss'] = F.cross_entropy(
                mim_logits.view(-1, self.config.img_vocab_size),
                mim_labels.view(-1),
                ignore_index=-100,
            )
            return losses
        elif mode == 'image_text_itc':
            pass
        elif mode == 'inference':
            pass
        else:
            raise NotImplementedError
           
