import torch
import torch.utils.data as data
import random
import torchvision
import os
from pathlib import Path
from PIL import Image

class WaterMarkDataset(data.Dataset):
    def __init__(self, root, water_mark_path, transform=None):
        self.transform=transform

        root=Path(root)
        self.data_list=[str(root / x) for x in root.iterdir()]

        self.water_mark = Image.open(water_mark_path)
        self.w_mark, self.h_mark = self.water_mark.size

    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx]).convert('RGBA')
        img_clean = img.convert('RGB')

        w, h = img.size
        w_t = int(random.uniform(0.8, 1.0)*w)
        h_t = int(self.h_mark*(w_t/self.w_mark))
        water_mark = self.water_mark.resize((w_t, h_t))
        img.paste(water_mark, ((w-w_t)//2, (h-h_t)//2))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            img_clean = self.transform(img_clean)

        return img, img_clean

    def __len__(self):
        return len(self.data_list)