import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from layers import LayerNorm2d


def checkpoint(function):
    def wrapper(*args, **kwargs):
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

    if os.environ.get("GRAD_CKPT", "1") == "1":
        return wrapper
    else:
        return function


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GEGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * F.gelu(x2)


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=4, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = GEGLU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    @checkpoint
    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, N_block: int, group=32, FFN_Expand=4):
        super().__init__()
        self.norm = nn.GroupNorm(group, in_ch)
        self.proj_in = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv2d(in_ch, out_ch, 1)

        self.gamma = nn.Parameter(torch.zeros((1, out_ch, 1, 1)), requires_grad=True)

        self.blocks = nn.ModuleList([
            NAFBlock(in_ch, FFN_Expand=FFN_Expand) for _ in range(N_block)
        ])

    def forward(self, inp):
        x = inp

        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        return inp + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=(16, 32, 64, 128), middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width[0], kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width[0], out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, (num_stage, num_blk) in enumerate(enc_blk_nums):
            self.encoders.append(nn.ModuleList([NAFStage(width[i], width[i], num_blk) for _ in range(num_stage)]))
            self.downs.append(
                nn.Conv2d(width[i], width[i+1], 2, 2)
            )

        self.middle_blks = nn.Sequential(NAFStage(width[-1], width[-1], middle_blk_num))

        for i, (num_stage, num_blk) in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(width[-i-1], width[-i-2]*4, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.decoders.append(nn.ModuleList([NAFStage(width[-i-2], width[-i-2], num_blk) for _ in range(num_stage)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder_stages, down in zip(self.encoders, self.downs):
            for encoder in encoder_stages:
                x = encoder(x)
                encs.append(x)
                print(x.shape)
            x = down(x)

        x = self.middle_blks(x)

        skip_count=len(encs)
        for decoder_stages, up in zip(self.decoders, self.ups):
            x = up(x)
            for decoder in decoder_stages:
                skip_count -= 1
                print('up', x.shape)
                x = x + encs[skip_count]
                x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def get_NAFNet(arch):
    if arch == 'mark-s':
        return NAFNet(width=24, enc_blk_nums=[(1,1), (1,2), (1,4), (1,6)], middle_blk_num=8, dec_blk_nums=[(1,2), (1,2), (1,1), (1,1)])
    elif arch == 'mark-m':
        return NAFNet(width=24, enc_blk_nums=[(1,1), (1,2), (1,4), (1,6)], middle_blk_num=8, dec_blk_nums=[(2,2), (2,2), (2,2), (2,1)])
    elif arch == 'mark-l':
        return NAFNet(width=(32, 128, 256, 512, 768), enc_blk_nums=[(1,1), (2,2), (2,4), (2,8)], middle_blk_num=8, dec_blk_nums=[(2,12), (2,6), (2,3), (1,2)])
    elif arch == 'mark-xl':
        return NAFNet(width=(96, 320, 512, 832, 1280), enc_blk_nums=[(1,2), (2,2), (2,4), (2,8)],
                      middle_blk_num=8, dec_blk_nums=[(2,12), (2,6), (2,3), (1,3)])
    elif arch == 'mark-H':
        return NAFNet(width=(96, 320, 512, 832, 1280), enc_blk_nums=[(2,2), (3,2), (3,4), (3,10)],
                      middle_blk_num=10, dec_blk_nums=[(3,15), (3,6), (3,3), (2,3)])

if __name__ == '__main__':
    from torchanalyzer import TorchViser, ModelFlopsAnalyzer, ModelTimeMemAnalyzer

    model = get_NAFNet('mark-xl')
    inputs = torch.randn(1, 3, 400, 800)

    analyzer = ModelFlopsAnalyzer(model)
    info = analyzer.analyze(inputs)
    TorchViser().show(model, info)