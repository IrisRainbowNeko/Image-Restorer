from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms

from models import NAFNet

device = 'cuda'

class Infer:
    def __init__(self, ckpt):
        self.net = NAFNet()
        self.net.load_state_dict(torch.load(ckpt))

        self.trans = transforms.Compose([
            transforms.Resize(800),
            # transforms.CenterCrop(800),
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

    def infer_one(self, path):
        img = self.load_image(path).to(device)
        img = img.unsqueeze(0)
        pred = self.net(img)
        pred = pred*self.std+self.mean

        pred = transforms.ToPILImage(pred)
        return pred

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt", default='', type=str)
    parser.add_argument("--img", default='', type=str)
    args = parser.parse_args()

    infer = Infer(args.ckpt)
    infer.infer_one(args.img)
