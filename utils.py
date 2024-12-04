import torch
import math
from torch.optim.lr_scheduler import LambdaLR

def cal_psnr(x, y, mean, std):
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    x = x*std+mean
    y = y*std+mean
    mse = torch.mean((x-y)**2, dim=(1,2,3))
    psnr = 10*torch.log10(1.0/mse)
    return psnr

def get_ext(path:str):
    idx = path.rfind('.')
    return path[idx+1:]