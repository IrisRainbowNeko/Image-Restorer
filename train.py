import datetime
import os
import random
from argparse import ArgumentParser

import torch
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms
from transformers.optimization import Adafactor

from data import WaterMarkDataset, PairDataset, PadResize, ShortResize
from models import get_NAFNet
from utils import cal_psnr
from loss import CharbonnierLoss, MSSSIMLoss, GWLoss

class Trainer:
    def __init__(self, args):
        self.args = args
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
            logname = os.path.join(args.log_dir, datetime.datetime.now().isoformat()+'.log')
            logger.add(logname)
            logger.info(f'world size: {self.world_size}')
        else:
            logger.disable("__main__")

        self.net, self.optimizer, train_loader, val_loader, scheduler = \
            self.accelerator.prepare(self.net, self.optimizer, self.train_loader, self.test_loader, self.scheduler)

    def build_model(self):
        self.net = get_NAFNet(self.args.arch)

        no_decay_layer = lambda name: 'norm' in name or name.endswith('bias') or name.endswith('beta') or name.endswith('gamma')
        groups = [
            {"params": [p for name, p in self.net.named_parameters() if no_decay_layer(name)],
             "weight_decay": 0},
            {"params": [p for name, p in self.net.named_parameters() if not no_decay_layer(name)],
             "weight_decay": 1e-3},
        ]
        if self.args.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(groups, lr=self.args.lr, betas=(0.9, 0.9))
        else:
            self.optimizer = Adafactor(groups, lr=self.args.lr, relative_step=False)

        self.cb_loss = CharbonnierLoss()
        self.ssim_loss = MSSSIMLoss()
        self.gw_loss = GWLoss()

        self.criterion_mask = nn.SmoothL1Loss(reduction='none')
        #self.loss_lbp = LBPLoss()
        print(len(self.train_loader))

        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr,
                                                 steps_per_epoch=len(self.train_loader), epochs=self.args.epochs,
                                                 pct_start=0.01)

    def collate_fn(self, batch):
        limit = 700 * 700
        w = random.randint(400, 1000)
        max_h = min(limit // w, 1000)
        h = random.randint(400, max_h)

        crop = transforms.RandomCrop((h,w), pad_if_needed=True, padding_mode='reflect')

        imgs, targets = [], []
        for item in batch:
            img_pair = torch.cat(item, dim=0)
            img, target = crop(img_pair).chunk(2)
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs, dim=0), torch.stack(targets, dim=0)

    def build_data(self):
        self.data_train = PairDataset(data_file=self.args.train_data,
                                      noise_std=0,
                                      transform=transforms.Compose([
                                          ShortResize(512),
                                          #transforms.CenterCrop((400, 800)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5]),
                                      ]), )
        self.data_test = PairDataset(data_file=self.args.test_data,
                                     noise_std=0,
                                     transform=transforms.Compose([
                                         PadResize(800),
                                         transforms.CenterCrop((400, 800)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5], [0.5]),
                                     ]), )

        self.train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=self.args.bs, shuffle=True,
                                                        num_workers=self.args.num_workers, pin_memory=True, collate_fn=self.collate_fn)
        self.test_loader = torch.utils.data.DataLoader(self.data_test, batch_size=self.args.bs, shuffle=False,
                                                       num_workers=self.args.num_workers, pin_memory=True)

    def local_loss(self, img_clean, pred, img_mask):
        return (self.criterion_mask(img_clean, pred)*img_mask).mean()

    def train(self):
        loss_sum = 0
        cb_loss_sum = 0
        gw_loss_sum = 0
        ssim_loss_sum = 0
        for ep in range(self.args.epochs):
            self.net.train()
            for step, (img, img_clean) in enumerate(self.train_loader):
                img = img.to(self.accelerator.device)
                img_clean = img_clean.to(self.accelerator.device)
                # img_mask = img_mask.to(self.accelerator.device)

                pred = self.net(img)

                # loss = self.alpha*self.criterion(pred, img_clean) + (1-self.alpha)*self.local_loss(img_clean, pred, img_mask)
                #loss = self.criterion(pred, img_clean) #+ 0.5*self.loss_lbp(pred*50, img_clean*50)

                cb_loss = self.cb_loss(pred, img_clean)
                gw_loss = self.gw_loss(pred, img_clean)
                ssim_loss = self.ssim_loss(pred, img_clean)
                loss = cb_loss + 0.5*gw_loss + ssim_loss

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:  # fine-tuning
                    self.accelerator.clip_grad_norm_(self.net.parameters(), 1.)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                loss_sum += loss.item()
                cb_loss_sum += cb_loss.item()
                gw_loss_sum += gw_loss.item()
                ssim_loss_sum += ssim_loss.item()

                if step%self.args.log_step == 0:
                    logger.info(f'[{ep+1}/{self.args.epochs}]<{step+1}/{len(self.train_loader)}>, '
                                f'loss:{loss_sum/self.args.log_step:.3e}, '
                                f'cb_loss:{cb_loss_sum/self.args.log_step:.3e}, '
                                f'gw_loss:{gw_loss_sum/self.args.log_step:.3e}, '
                                f'ssim_loss:{ssim_loss_sum/self.args.log_step:.3e}, '
                                f'lr:{self.scheduler.get_lr()[0]:.3e}')
                    loss_sum = 0
                    cb_loss_sum = 0
                    gw_loss_sum = 0
                    ssim_loss_sum = 0
            self.evaluate()
            if self.accelerator.is_local_main_process:
                torch.save(self.net.state_dict(), f'output/ep_{ep}.pth')

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()
        mean = torch.tensor([0.5]).to(self.accelerator.device)
        std = torch.tensor([0.5]).to(self.accelerator.device)
        psnr = 0
        for step, (img_mark, img_clean) in enumerate(self.test_loader):
            img_clean = img_clean.to(self.accelerator.device)
            img_mark = img_mark.to(self.accelerator.device)

            pred = self.net(img_mark)

            psnr += cal_psnr(pred, img_clean, mean, std).sum().item()

        psnr = torch.tensor(psnr).to(self.accelerator.device)
        psnr = self.accelerator.reduce(psnr, reduction="sum")
        psnr /= self.accelerator.num_processes

        logger.info(f'psnr: {psnr/len(self.data_test):.3f}')

def make_args():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mark-s', type=str)
    parser.add_argument("--optim", default='adamw', type=str)

    # parser.add_argument("--train_root", default='../datas/anime_SR/train/HR', type=str)
    parser.add_argument("--train_data", default='/data1/dzy/dataset_raw/skeb/train.json', type=str)
    parser.add_argument("--test_data", default='/data1/dzy/dataset_raw/skeb/test.json', type=str)
    # parser.add_argument("--water_mark", default='./water_mark4.png', type=str)
    # parser.add_argument("--water_mark_mask", default='./water_mark4_mask.png', type=str)
    parser.add_argument("--bs", default=32, type=int)
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
