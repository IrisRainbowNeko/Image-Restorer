from functools import partial

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_manager
from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import CfgWDModelParser
from rainbowneko.train.data import BaseBucket
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.handler import HandlerChain, ImageHandler, LoadImageHandler, HandlerGroup, DataHandler
from rainbowneko.train.data.source import UnLabelSource
from rainbowneko.train.loss import LossContainer, LossGroup
from rainbowneko.utils import neko_cfg, CosineLR
from rainbowneko.parser.model import NekoModelLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from cfgs.py.train import train_base, tuning_base
from data import PadToSize
from loss import CharbonnierLoss, MSSSIMLoss, GWLoss
from models import get_NAFNet
from bitsandbytes.optim import AdamW8bit
from PIL import Image
from io import BytesIO
import random


def make_cfg():
    dict(
        _base_=[train_base, tuning_base],
        exp_dir=f'exps/webp_fix-more',
        mixed_precision='fp16',

        model_part=CfgWDModelParser([
            dict(
                lr=5e-4,
                layers=[''],  # train all layers
            )
        ], weight_decay=1e-2),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_manager=[
            ckpt_manager(saved_model=({'model':'model', 'trainable':False},))
        ],

        train=dict(
            train_epochs=10,
            workers=8,
            max_grad_norm=None,
            save_step=4000,
            gradient_accumulation_steps=1,

            resume=dict(
                skeb=NekoModelLoader(
                    module_to_load='model',
                    #path='exps/webp_fix/ckpts/NAFormer-s-4828.ckpt',
                    path='ckpts/NAFormer-s-24000-webp.ckpt',
                ),
            ),

            loss=LossGroup([
                LossContainer(CharbonnierLoss()),
                LossContainer(GWLoss(), weight=0.5),
                LossContainer(MSSSIMLoss()),
            ]),

            optimizer=partial(AdamW8bit, betas=(0.9, 0.9)),

            scale_lr=False,
            scheduler=CosineLR(
                warmup_steps=1000,
            ),
            metrics=MetricGroup(
                psnr=MetricContainer(PeakSignalNoiseRatio(data_range=tuple([-1.0, 1.0]))),
                ssim=MetricContainer(StructuralSimilarityIndexMeasure(data_range=tuple([-1.0, 1.0]))),
            ),
        ),

        model=dict(
            name='NAFormer-s',
            wrapper=partial(SingleWrapper, model=get_NAFNet('mark-s'))
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

class RandDegenerate:
    def __init__(self, quality=(40,100)):
        self.quality = quality
        self.fmts = ['WEBP', 'JPEG', None]

    def compress_image(self, img: Image.Image, fmt='WEBP', quality=80) -> Image.Image:
        img_bytes = BytesIO()
        img.save(img_bytes, format=fmt, quality=quality)
        img_bytes.seek(0)
        compressed_img = Image.open(img_bytes)
        return compressed_img
    
    def __call__(self, img):
        quality = random.randint(*self.quality)
        fmt = random.choice(self.fmts)
        if fmt is None:
            return img
        else:
            compressed_img = self.compress_image(img, fmt, quality)
            return compressed_img

class BatchCropHandler(DataHandler):
    def __init__(self, crop_rate, key_map_in=('image -> image', 'label -> label'), key_map_out=('image -> image', 'label -> label')):
        super().__init__(key_map_in, key_map_out)
        self.crop_rate = crop_rate

    def handle(self, image, label):
        B,C,H,W = image.shape

        h_new = random.randint(int(H*self.crop_rate[0]), int(H*self.crop_rate[1]))
        w_new = random.randint(int(W*self.crop_rate[0]), int(W*self.crop_rate[1]))

        y = random.randint(0, H-h_new)
        x = random.randint(0, W-w_new)

        image = image[:,:,y:y+h_new,x:x+w_new]
        label = label[:,:,y:y+h_new,x:x+w_new]

        return {'image': image, 'label': label}

data_root = '/GPUFS/sysu_pxwei_1/dzy/datas/skeb'

@neko_cfg
def cfg_data():
    dict(
        dataset1=BaseDataset(_partial_=True, batch_size=16, loss_weight=1.0,
            source=dict(
                data_source1=UnLabelSource(
                    img_root=f'{data_root}/png_origin',
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                image=ImageHandler(transform=T.Compose([
                        PadToSize(512, 512),
                        T.RandomCrop(512),
                    ])),
                label=ImageHandler(transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ]), key_map_out=('image -> label',)),
                degenerate=ImageHandler(transform=T.Compose([
                        RandDegenerate(),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ])),
            ),
            #batch_handler=BatchCropHandler(crop_rate=(0.5,1)),
            bucket=BaseBucket(),
        )
    )

@neko_cfg
def cfg_evaluator():
    partial(Evaluator,
        interval=2000,
        metric=MetricGroup(
            psnr=MetricContainer(PeakSignalNoiseRatio(data_range=tuple([-1.0, 1.0]))),
            ssim=MetricContainer(StructuralSimilarityIndexMeasure(data_range=tuple([-1.0, 1.0]))),
        ),
        dataset=dict(
            dataset1=partial(BaseDataset, batch_size=32, loss_weight=1.0,
                source=dict(
                    data_source1=UnLabelSource(
                        img_root=f'{data_root}/png_origin_test',
                    ),
                ),
                handler=HandlerChain(
                    load=LoadImageHandler(),
                    image=ImageHandler(transform=T.Compose([
                            PadToSize(512, 512),
                            T.CenterCrop(512),
                        ])),
                    label=ImageHandler(transform=T.Compose([
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ]), key_map_out=('image -> label',)),
                    degenerate=ImageHandler(transform=T.Compose([
                            RandDegenerate(),
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ])),
                ),
                bucket=BaseBucket(),
            )
        )
    )