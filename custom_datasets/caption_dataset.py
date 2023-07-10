import torch
import json
import random
from pathlib import Path

from custom_datasets.image_dataset import SimpleDataAugmentationForContinualTransformer
from custom_datasets.dataset_folder import default_loader



class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, config, data_path=None):
        self.data_path = data_path
        self.ann = []
        if not isinstance(ann_file, (list, tuple)):
            ann_file = [ann_file]
        print('loading data files...')
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        
        self.image_transform = SimpleDataAugmentationForContinualTransformer(config)
        self.image_loader = default_loader


    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index): 
        ann = self.ann[index]
        caption = ann['caption']
        if self.data_path is not None:
            image_path = Path(self.data_path, ann['filename'])
        else:
            image_path = Path(ann['filename'])

        # load image
        image = self.image_loader(image_path)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "caption": caption,
        }

def simple_caption_collate_fn(batch):
    images = []
    captions = []
    for b in batch:
        images.append(b["image"])
        captions.append(b["caption"])

    return {
        "images": torch.stack(images),
        "raw_text": captions
    }, None