import torch
import json
import random

from transformers import (
    BertTokenizer,
)

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, max_text_len=40):
        self.ann = []
        if not isinstance(ann_file, (list, tuple)):
            ann_file = [ann_file]
        print('loading data files...')
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        # self.max_text_len = max_text_len
        # self.tokenizer = get_pretrained_tokenizer("bert-base-uncased")

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index): 
        ann = self.ann[index]['caption']
        if isinstance(ann, list):
            caption = random.choice(ann)
        else:
            caption = ann
        # encoding = self.tokenizer(
        #     caption,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_text_len,
        #     return_special_tokens_mask=True,
        # )

        return {
            "raw_text": caption,
            # "encoded_text": encoding,
        }

def simple_text_collate_fn(batch):
    return {'raw_text': [b['raw_text'] for b in batch]}, None


def text_collate_fn(batch):
    text_list = []
    text_ids_list = []
    special_tokens_mask_list = []
    attention_mask_list = []
    for b in batch:
        raw_text, encoded_text = b['raw_text'], b['encoded_text']
        text_list.append(raw_text)
        text_ids_list.append(encoded_text['input_ids'])
        special_tokens_mask_list.append(encoded_text['special_tokens_mask'])
        attention_mask_list.append(encoded_text['attention_mask'])
    return {'raw_text': text_list, 'input_ids':text_ids_list, 'special_tokens_mask':special_tokens_mask_list, 'attention_mask': attention_mask_list}, None




        



