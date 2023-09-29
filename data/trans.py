from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from PIL import Image

class PadResize:
    def __init__(self, w, interpolation=InterpolationMode.BILINEAR):
        self.w = w
        self.interpolation = interpolation

    def __call__(self, img:Image):
        w, h = img.size
        hs = (h/w)*self.w
        img = F.resize(img, [hs, self.w], self.interpolation, antialias=False)
        if hs<self.w:
            h_pad = (self.w-hs)//2
            img = F.pad(img, [0, h_pad, 0, (self.w-hs)-h_pad])
        return img