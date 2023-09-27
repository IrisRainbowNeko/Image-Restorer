import os

import torchvision
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet
from anime_data import WaterMarkDataset, PairDataset
import torchvision.datasets as datasets
from argparse import ArgumentParser
from loguru import logger
import datetime
from models import NAFNet
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import lr_scheduler
from utils import cal_psnr

class Trainer:
    def __init__(self, args):
        self.args=args
        self.alpha = args.alpha

        set_seed(42)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            step_scheduler_with_optimizer=False,
        )

        self.build_data()
        self.build_model()

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes

        if self.accelerator.is_local_main_process:
            logname = os.path.join(args.log_dir, datetime.datetime.now().isoformat() + '.log')
            logger.add(logname)
            logger.info(f'world size: {self.world_size}')
        else:
            logger.disable("__main__")

        self.net, self.optimizer, train_loader, val_loader, scheduler = \
            self.accelerator.prepare(self.net, self.optimizer, self.train_loader, self.test_loader, self.scheduler)

    def build_model(self):
        self.net = NAFNet(width=32, enc_blk_nums=[1,2,4,6], middle_blk_num=8, dec_blk_nums=[2,2,1,1])

        #summary(self.net, (3, 224, 224))

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr)
        self.criterion = nn.SmoothL1Loss()
        self.criterion_mask = nn.SmoothL1Loss(reduction='none')
        print(len(self.train_loader))

        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr,
                                            steps_per_epoch=len(self.train_loader), epochs=self.args.epochs,
                                            pct_start=0.2)

    def build_data(self):
        water_mark = Image.open(self.args.water_mark)
        water_mark_mask = Image.open(self.args.water_mark_mask).convert('RGB')
        # self.data_train = WaterMarkDataset(root=self.args.train_root, water_mark=water_mark, water_mark_mask=water_mark_mask,
        #                                    transform=transforms.Compose([
        #                                         transforms.Resize(800),
        #                                         transforms.CenterCrop(800),
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize([0.5], [0.5]),
        #                                    ]),)
        # self.data_test = WaterMarkDataset(root=self.args.test_root, water_mark=water_mark, water_mark_mask=water_mark_mask,
        #                                   noise_std=0,
        #                                   transform=transforms.Compose([
        #                                       transforms.Resize(800),
        #                                       transforms.CenterCrop(800),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize([0.5], [0.5]),
        #                                   ]),)

        self.data_train = PairDataset(root_clean=self.args.train_root_clean, root_mark=self.args.train_root_mark,
                                           transform=transforms.Compose([
                                                transforms.Resize(800),
                                                transforms.CenterCrop(800),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5], [0.5]),
                                           ]),)
        self.data_test = WaterMarkDataset(root=self.args.test_root, water_mark=water_mark, water_mark_mask=water_mark_mask,
                                          noise_std=0,
                                          transform=transforms.Compose([
                                              transforms.Resize(800),
                                              transforms.CenterCrop(800),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5]),
                                          ]),)


        self.train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=self.args.bs, shuffle=True,
                                                        num_workers=self.args.num_workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.data_test, batch_size=self.args.bs, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)

    def local_loss(self, img_clean, pred, img_mask):
        return (self.criterion_mask(img_clean, pred)*img_mask).mean()

    def train(self):
        loss_sum = 0
        for ep in range(self.args.epochs):
            self.net.train()
            for step, (img, img_clean, img_mask) in enumerate(self.train_loader):
                img = img.to(self.accelerator.device)
                img_clean = img_clean.to(self.accelerator.device)
                img_mask = img_mask.to(self.accelerator.device)

                pred = self.net(img)

                loss = self.alpha*self.criterion(pred, img_clean) + (1-self.alpha)*self.local_loss(img_clean, pred, img_mask)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                loss_sum += loss.item()

                if step % self.args.log_step == 0:
                    logger.info(f'[{ep+1}/{self.args.epochs}]<{step+1}/{len(self.train_loader)}>, '
                                f'loss:{loss_sum / self.args.log_step:.3e}, '
                                f'lr:{self.scheduler.get_lr()[0]:.3e}')
                    loss_sum = 0
            self.test()
            if self.accelerator.is_local_main_process:
                torch.save(self.net.state_dict(), f'output/ep_{ep}.pth')

    @torch.no_grad()
    def test(self):
        self.net.eval()
        mean = torch.tensor([0.5]).to(self.accelerator.device)
        std = torch.tensor([0.5]).to(self.accelerator.device)
        psnr=0
        for step, (img, img_clean, img_mask) in enumerate(self.test_loader):
            img = img.to(self.accelerator.device)
            img_clean = img_clean.to(self.accelerator.device)

            pred = self.net(img)

            psnr+=cal_psnr(pred, img_clean, mean, std).sum().item()

        psnr = torch.tensor(psnr).to(self.accelerator.device)
        psnr = self.accelerator.reduce(psnr, reduction="sum")

        logger.info(f'psnr: {psnr/len(self.data_test):.3f}')

def make_args():
    parser = ArgumentParser()
    # parser.add_argument("--train_root", default='../datas/anime_SR/train/HR', type=str)
    parser.add_argument("--train_root_clean", default='../datas/anime_SR/train/HR', type=str)
    parser.add_argument("--train_root_mark", default='../datas/anime_SR/train/HR', type=str)
    parser.add_argument("--test_root", default='../datas/anime_SR/test/HR', type=str)
    parser.add_argument("--water_mark", default='./water_mark2.png', type=str)
    parser.add_argument("--water_mark_mask", default='./water_mark2_mask.png', type=str)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--log_dir", default='logs/', type=str)
    parser.add_argument("--log_step", default=20, type=int)

    parser.add_argument("--alpha", default=0.3, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    os.makedirs('./output', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()