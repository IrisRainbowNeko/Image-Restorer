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

    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx]).convert('RGBA')
        img_clean = img.convert('RGB')

        w, h = img.size
        w_t = int(random.uniform(0.8, 1.0)*w)
        h_t = int(self.h_mark*(w_t/self.w_mark))
        water_mark = self.water_mark.resize((w_t, h_t))
        wm_mask = self.water_mark_mask.resize((w_t, h_t))

        img_cv = np.array(img.convert('RGB'))/255.
        water_mark_cv = np.asarray(water_mark)/255.
        alpha = (water_mark_cv[:,:,3:]*random.uniform(0.9, 1.1)).clip(0,1)
        water_mark_cv[:, :, :3] = water_mark_cv[:,:,:3]**random.uniform(0.76, 1.3)
        l,t,r,b = (w-w_t)//2, (h-h_t)//2, (w-w_t)//2+w_t, (h-h_t)//2+h_t
        img_cv[t:b, l:r, :] = alpha*water_mark_cv[:,:,:3] + (1.-alpha)*img_cv[t:b, l:r, :]
        img_cv[t:b, l:r, :] += np.random.randn(b-t, r-l, 3)*random.uniform(0, self.noise_std)
        img = Image.fromarray((img_cv*255.).astype(np.uint8))

        img_mask = np.zeros_like(img_cv)
        img_mask[t:b, l:r, :] = wm_mask
        img_mask = Image.fromarray((img_mask*255.).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)
            img_clean = self.transform(img_clean)
            img_mask = self.transform(img_mask)
            img_mask = img_mask*0.5+0.5

        return img, img_clean, img_mask

    def __len__(self):
        return len(self.data_list)