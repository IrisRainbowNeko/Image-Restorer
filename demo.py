from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm

from models import get_NAFNet

device = 'cpu'
types_support = ['bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'tiff', 'webp']

class Infer:
    def __init__(self, ckpt, arch):
        self.net = get_NAFNet(arch)

        self.net.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.net = self.net.to(device)

        self.resize = transforms.Resize(800)

        self.trans = transforms.Compose([
            transforms.Resize(800),
            transforms.CenterCrop((400, 800)),
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

    @torch.no_grad()
    def infer_one(self, path):
        img, img_raw = self.load_image(path)
        img = img.to(device).unsqueeze(0)
        pred = self.net(img)
        pred = pred*self.std+self.mean
        pred = pred.squeeze(0).clip(0,1)

        pred = transforms.ToPILImage()(pred)
        img_raw = self.resize(img_raw)
        wm,hm=pred.size
        w,h=img_raw.size
        img_raw.paste(pred, (0, (h-hm)//2))

        return img_raw

    def infer(self, path, out_dir):
        if os.path.isdir(path):
            files = [os.path.join(path, x) for x in os.listdir(path)]
            for file in tqdm(files):
                img = self.infer_one(file)
                img.save(os.path.join(out_dir, os.path.basename(file)))
        else:
            img = self.infer_one(path)
            img.save(os.path.join(out_dir, os.path.basename(path)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mark-s', type=str)
    parser.add_argument("--ckpt", default='', type=str)
    parser.add_argument("--img", default='', type=str)
    parser.add_argument("--out_dir", default='results', type=str)
    args = parser.parse_args()

    infer = Infer(args.ckpt, args.arch)
    infer.infer(args.img, args.out_dir)