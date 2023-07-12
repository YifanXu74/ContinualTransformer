import torch
import json
import random
from pathlib import Path

from custom_datasets.image_dataset import SimpleDataAugmentationForContinualTransformer, DataAugmentationForContinualTransformer
from custom_datasets.dataset_folder import default_loader

from random import choice



class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, config, data_path=None):
        self.data_path = data_path
        self.force_vae = config.force_vae
        self.ann = []
        if not isinstance(ann_file, (list, tuple)):
            ann_file = [ann_file]
        print('loading data files...')
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        if config.force_vae:
            self.image_transform = DataAugmentationForContinualTransformer(config)
        else:
            self.image_transform = SimpleDataAugmentationForContinualTransformer(config)
        self.image_loader = default_loader


    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index): 
        ann = self.ann[index]
        caption = ann['caption']
        if isinstance(caption, list):
            caption = choice(caption)
        if self.data_path is None or self.data_path == '':
            image_path = Path(ann['filename'])
        else:
            image_path = Path(self.data_path, ann['filename'])

        # load image
        image = self.image_loader(image_path)
        if self.image_transform is not None:
            if self.force_vae:
                image, image_for_vae = self.image_transform(image)
            else:
                image = self.image_transform(image)
                image_for_vae = None

        return {
            "image": image,
            "image_for_vae": image_for_vae,
            "caption": caption,
        }

def simple_caption_collate_fn(batch):
    images = []
    images_for_vae = []
    captions = []
    for b in batch:
        images.append(b["image"])
        images_for_vae.append(b["image_for_vae"])
        captions.append(b["caption"])

    return {
        "images": torch.stack(images),
        "images_for_vae": torch.stack(images_for_vae) if images_for_vae[0] is not None else None,
        "raw_text": captions
    }, None