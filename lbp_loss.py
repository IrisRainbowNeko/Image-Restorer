from torch import nn
from torch.nn import functional as F
import torch

class LBPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rec, real):
        return F.mse_loss(self.lbp(rec), self.lbp(real))

    @staticmethod
    def lbp(img):
        center_img = img[:, :, 1:-1, 1:-1]
        p1 = img[:, :, 0:-2, 0:-2]-center_img
        p2 = img[:, :, 1:-1, 0:-2]-center_img
        p3 = img[:, :, 2:, 0:-2]-center_img
        p4 = img[:, :, 2:, 1:-1]-center_img
        p5 = img[:, :, 2:, 2:]-center_img
        p6 = img[:, :, 1:-1, 2:]-center_img
        p7 = img[:, :, 0:-2, 2:]-center_img
        p8 = img[:, :, 0:-2, 1:-1]-center_img

        lbp = torch.cat([p1,p2,p3,p4,p5,p6,p7,p8], dim=1)
        lbp = torch.sigmoid(lbp)
        return lbp