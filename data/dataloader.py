import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')
    image_dir = path
    dataloader = DataLoader(
        GetDataset(image_dir,ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    image_dir = path

    dataloader = DataLoader(
        GetDataset(image_dir,is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    image_dir = path

    dataloader = DataLoader(
        GetDataset(image_dir,is_valid=True,ps=256),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random

class GetDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):

        if is_test or is_valid:
            self.image_dir = os.path.join(image_dir,'test')
            self.image_list = os.listdir(os.path.join(self.image_dir,'img'))
        else:
            self.image_dir = os.path.join(image_dir,'train')
            self.image_list = os.listdir(os.path.join(self.image_dir,'img'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'img',self.image_list[idx])).convert('RGB')
        if self.is_valid or self.is_test:      
            label = Image.open(os.path.join(self.image_dir,'gt', self.image_list[idx])).convert('RGB')
        else:
            label = Image.open(os.path.join(self.image_dir,'gt', self.image_list[idx])).convert('RGB')
        ps = self.ps

        if self.ps is not None:
            width,height = image.size
            if width < self.ps or height < self.ps:
                width = max(width,260)
                height = max(height,260)
                print(image.size,os.path.join(self.image_dir, 'img',self.image_list[idx]))
                image = image.resize((width, height), Image.BILINEAR)
                label = label.resize((width, height), Image.BILINEAR)

            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)
            
            image = image[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)
                label = label.flip(2)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label



    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
