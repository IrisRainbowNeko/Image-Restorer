import torch
import torch.utils.data as data
import random
import torchvision
import os
from pathlib import Path
from PIL import Image
import numpy as np

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class WaterMarkDataset(data.Dataset):
    def __init__(self, root, water_mark, water_mark_mask, transform=None, noise_std=0.1):
        self.transform=transform
        self.noise_std=noise_std

        root=Path(root)
        self.data_list=[str(x) for x in root.iterdir()]

        self.water_mark = water_mark.copy()
        self.water_mark_mask = water_mark_mask.copy()
        self.w_mark, self.h_mark = self.water_mark.size

    def make_water_mark(self, img):
        w, h = img.size
        w_t = int(random.uniform(0.8, 1.0)*w)
        h_t = int(self.h_mark*(w_t/self.w_mark))
        water_mark = self.water_mark.resize((w_t, h_t))
        wm_mask = self.water_mark_mask.resize((w_t, h_t))

        img_cv = np.array(img.convert('RGB'))/255.
        water_mark_cv = np.asarray(water_mark)/255.
        alpha = (water_mark_cv[:, :, 3:]*random.uniform(0.9, 1.1)).clip(0, 1)
        water_mark_cv[:, :, :3] = water_mark_cv[:, :, :3]**random.uniform(0.76, 1.3)
        l, t, r, b = (w-w_t)//2, (h-h_t)//2, (w-w_t)//2+w_t, (h-h_t)//2+h_t
        img_cv[t:b, l:r, :] = alpha*water_mark_cv[:, :, :3]+(1.-alpha)*img_cv[t:b, l:r, :]
        if self.noise_std>0:
            img_cv[t:b, l:r, :] += np.random.randn(b-t, r-l, 3)*random.uniform(0, self.noise_std)
        img = Image.fromarray((img_cv*255.).astype(np.uint8))

        img_mask = np.zeros_like(img_cv)
        img_mask[t:b, l:r, :] = wm_mask
        img_mask = Image.fromarray((img_mask*255.).astype(np.uint8))

        return img, img_mask

    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx]).convert('RGBA')
        img_clean = img.convert('RGB')

        img, img_mask = self.make_water_mark(img)

        if self.transform is not None:
            img = self.transform(img)
            img_clean = self.transform(img_clean)
            img_mask = self.transform(img_mask)
            img_mask = img_mask*0.5+0.5

        return img, img_clean, img_mask

    def __len__(self):
        return len(self.data_list)

class PairDataset(data.Dataset):
    def __init__(self, root_clean, root_mark, transform=None):
        self.transform=transform

        root_clean=Path(root_clean)
        self.data_list_clean=[str(x) for x in root_clean.iterdir()]
        root_mark = Path(root_mark)
        self.data_list_mark=[str(x) for x in root_mark.iterdir()]

    def __getitem__(self, idx):
        img_mark = Image.open(self.data_list_mark[idx]).convert('RGB')
        img_clean = Image.open(random.choice(self.data_list_clean)).convert('RGB')

        if self.transform is not None:
            img_mark = self.transform(img_mark)
            img_clean = self.transform(img_clean)

        return img_mark, img_clean

    def __len__(self):
        return len(self.data_list_mark)

class PairDatasetMark(data.Dataset):
    def __init__(self, root_clean, root_mark, water_mark, water_mark_mask, transform=None):
        self.transform=transform

        root_clean=Path(root_clean)
        self.data_list_clean=[str(x) for x in root_clean.iterdir()]
        root_mark = Path(root_mark)
        self.data_list_mark=[str(x) for x in root_mark.iterdir()]

        self.water_mark = water_mark.copy()
        self.water_mark_mask = water_mark_mask.copy()
        self.w_mark, self.h_mark = self.water_mark.size

    def make_water_mark(self, img):
        w, h = img.size
        w_t = int(random.uniform(0.8, 1.0)*w)
        h_t = int(self.h_mark*(w_t/self.w_mark))
        water_mark = self.water_mark.resize((w_t, h_t))
        wm_mask = self.water_mark_mask.resize((w_t, h_t))

        img_cv = np.array(img.convert('RGB'))/255.
        water_mark_cv = np.asarray(water_mark)/255.
        alpha = (water_mark_cv[:, :, 3:]*random.uniform(0.9, 1.1)).clip(0, 1)
        water_mark_cv[:, :, :3] = water_mark_cv[:, :, :3]**random.uniform(0.76, 1.3)
        l, t, r, b = (w-w_t)//2, (h-h_t)//2, (w-w_t)//2+w_t, (h-h_t)//2+h_t
        img_cv[t:b, l:r, :] = alpha*water_mark_cv[:, :, :3]+(1.-alpha)*img_cv[t:b, l:r, :]
        if self.noise_std>0:
            img_cv[t:b, l:r, :] += np.random.randn(b-t, r-l, 3)*random.uniform(0, self.noise_std)
        img = Image.fromarray((img_cv*255.).astype(np.uint8))

        img_mask = np.zeros_like(img_cv)
        img_mask[t:b, l:r, :] = wm_mask
        img_mask = Image.fromarray((img_mask*255.).astype(np.uint8))

        return img, img_mask

    def __getitem__(self, idx):
        img_mark = Image.open(self.data_list_mark[idx]).convert('RGB')
        img_clean = Image.open(random.choice(self.data_list_clean)).convert('RGB')

        fake_mark, mark_mask = self.make_water_mark(img_clean.copy())

        if self.transform is not None:
            img_mark = self.transform(img_mark)
            img_clean = self.transform(img_clean)
            fake_mark = self.transform(fake_mark)

        return img_mark, img_clean, fake_mark

    def __len__(self):
        return len(self.data_list_mark)