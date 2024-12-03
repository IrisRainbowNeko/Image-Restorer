import os
from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data import PadResize
from models import get_NAFNet
from utils import get_ext

device = 'cuda'
types_support = ['bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'tiff', 'webp']

class Infer:
    def __init__(self, ckpt, arch, crop=True):
        self.net = get_NAFNet(arch)

        sd = torch.load(ckpt, map_location='cpu')
        if next(iter(sd.keys())).startswith('module'):
            sd = {k[7:]:v for k,v in sd.items()}
        self.net.load_state_dict(sd)
        self.net = self.net.to(device)

        self.resize = PadResize(800, make_pad=False)

        self.crop = crop
        if crop:
            self.trans = transforms.Compose([
                PadResize(800, make_pad=False),
                transforms.CenterCrop((400, 800)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.trans = transforms.Compose([
                #PadResize(800, make_pad=False),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        mean = torch.tensor([0.5]).to(device)
        std = torch.tensor([0.5]).to(device)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

    def load_image(self, path):
        img_raw = Image.open(path).convert('RGB')
        img = self.trans(img_raw)
        return img, img_raw
    
    def calculate_psnr(self, img1, img2):
        # 确保输入的图像数组是浮点型
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # 计算 MSE
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')  # 两个图像完全相同

        # 计算 PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    @torch.no_grad()
    def infer_one(self, path):
        img, img_raw = self.load_image(path)
        img = img.to(device).unsqueeze(0)
        pred = self.net(img)
        pred = pred*self.std+self.mean
        pred = pred.squeeze(0).clip(0, 1)

        pred = transforms.ToPILImage()(pred)
        if self.crop:
            img_new = self.resize(img_raw)
            wm, hm = pred.size
            w, h = img_new.size
            img_new.paste(pred, (0, round((h-hm)/2)))
        else:
            img_new = pred
        
        img_raw_np = np.array(img_raw)
        img_new_np = np.array(img_new)
        print(f'{path} psnr:{self.calculate_psnr(img_raw_np, img_new_np)}')

        return img_new

    def infer(self, path, out_dir):
        if os.path.isdir(path):
            files = [os.path.join(path, x) for x in os.listdir(path) if get_ext(x).lower() in types_support]
            for file in tqdm(files):
                img = self.infer_one(file)
                img.save(os.path.join(out_dir, os.path.basename(file))+'.png')
        else:
            img = self.infer_one(path)
            img.save(os.path.join(out_dir, os.path.basename(path))+'.png')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mark-s', type=str)
    parser.add_argument("--ckpt", default='', type=str)
    parser.add_argument("--img", default='', type=str)
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--out_dir", default='results', type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    infer = Infer(args.ckpt, args.arch, crop=args.crop)
    infer.infer(args.img, args.out_dir)
