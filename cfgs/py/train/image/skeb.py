from functools import partial

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import CfgWDModelParser
from rainbowneko.train.data import BaseBucket
from rainbowneko.train.data import BaseDataset
from rainbowneko.train.data.handler import HandlerChain, ImageHandler, LoadImageHandler, HandlerGroup
from rainbowneko.train.data.source import ImagePairSource
from rainbowneko.train.loss import LossContainer, LossGroup
from rainbowneko.utils import neko_cfg, CosineLR
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from cfgs.py.train import train_base, tuning_base
from data import PadResize
from loss import CharbonnierLoss, MSSSIMLoss, GWLoss
from models import get_NAFNet
from bitsandbytes.optim import AdamW8bit


def make_cfg():
    dict(
        _base_=[train_base, tuning_base],
        exp_dir=f'exps/skeb-l-v2',
        mixed_precision='fp16',

        model_part=CfgWDModelParser([
            dict(
                lr=5e-4,
                layers=[''],  # train all layers
            )
        ], weight_decay=1e-2),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_manager=CkptManagerPKL(_partial_=True, saved_model=(
            {'model':'model', 'trainable':False},
        )),

        train=dict(
            train_epochs=10,
            workers=8,
            max_grad_norm=None,
            save_step=4000,
            gradient_accumulation_steps=1,

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
            metrics=MetricGroup(metric_dict=dict(
                psnr=MetricContainer(PeakSignalNoiseRatio(data_range=tuple([-1.0, 1.0]))),
                ssim=MetricContainer(StructuralSimilarityIndexMeasure(data_range=tuple([-1.0, 1.0]))),
            )),
        ),

        model=dict(
            name='NAF-de_watermark-l',
            wrapper=partial(SingleWrapper, model=get_NAFNet('mark-l'))
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

data_root = '/GPUFS/sysu_pxwei_1/dzy/datas/skeb'

@neko_cfg
def cfg_data():
    dict(
        dataset1=partial(BaseDataset, batch_size=8, loss_weight=1.0,
            source=dict(
                data_source1=ImagePairSource(
                    img_root=data_root,
                    label_file=f'{data_root}/train.json',
                ),
            ),
            handler=HandlerChain(handlers=dict(
                group_mark=HandlerChain(handlers=dict(
                    load=LoadImageHandler(),
                    image=ImageHandler(transform=T.Compose([
                        PadResize(800),
                        T.CenterCrop((400, 800)),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ])),
                )),
                group_clean=HandlerChain(handlers=dict(
                    load=LoadImageHandler(),
                    image=ImageHandler(transform=T.Compose([
                        PadResize(800),
                        T.CenterCrop((400, 800)),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ])),
                ), key_map_in=("label -> image",), key_map_out=("image -> label",)),
            )),
            bucket=BaseBucket(),
        )
    )

@neko_cfg
def cfg_evaluator():
    partial(Evaluator,
        interval=2000,
        metric=MetricGroup(metric_dict=dict(
            psnr=MetricContainer(PeakSignalNoiseRatio(data_range=tuple([-1.0, 1.0]))),
            ssim=MetricContainer(StructuralSimilarityIndexMeasure(data_range=tuple([-1.0, 1.0]))),
        )),
        dataset=dict(
            dataset1=partial(BaseDataset, batch_size=4, loss_weight=1.0,
                source=dict(
                    data_source1=ImagePairSource(
                        img_root=data_root,
                        label_file=f'{data_root}/test.json',
                    ),
                ),
                handler=HandlerChain(handlers=dict(
                    group_mark=HandlerChain(handlers=dict(
                        load=LoadImageHandler(),
                        image=ImageHandler(transform=T.Compose([
                            PadResize(800),
                            T.CenterCrop((400, 800)),
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ])),
                    )),
                    group_clean=HandlerChain(handlers=dict(
                        load=LoadImageHandler(),
                        image=ImageHandler(transform=T.Compose([
                            PadResize(800),
                            T.CenterCrop((400, 800)),
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ])),
                    ), key_map_in=("label -> image",), key_map_out=("image -> label",)),
                )),
                bucket=BaseBucket(),
            )
        )
    )