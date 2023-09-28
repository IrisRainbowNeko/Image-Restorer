from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms

from models import get_NAFNet

device = 'cpu'

class Infer:
    def __init__(self, ckpt, arch):
        self.net = get_NAFNet(arch)

        self.net.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.net = self.net.to(device)

        self.trans = transforms.Compose([
            transforms.Resize(800),
            transforms.CenterCrop(800),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        mean = torch.tensor([0.5]).to(device)
        std = torch.tensor([0.5]).to(device)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = self.trans(img)
        return img

    @torch.no_grad()
    def infer_one(self, path):
        img = self.load_image(path).to(device)
        img = img.unsqueeze(0)
        pred = self.net(img)
        pred = pred*self.std+self.mean
        pred = pred.squeeze(0).clip(0,1)

        pred = transforms.ToPILImage()(pred)
        return pred

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mark-s', type=str)
    parser.add_argument("--ckpt", default='', type=str)
    parser.add_argument("--img", default='', type=str)
    args = parser.parse_args()

    infer = Infer(args.ckpt, args.arch)
    pred = infer.infer_one(args.img)
    pred.save('test.png')