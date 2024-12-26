from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from PIL import Image

class PadResize:
    def __init__(self, w, interpolation=InterpolationMode.BILINEAR, make_pad=True):
        self.w = w
        self.interpolation = interpolation
        self.make_pad=make_pad

    def __call__(self, img:Image):
        w, h = img.size
        hs = int((h/w)*self.w)
        img = F.resize(img, [hs, self.w], self.interpolation)
        if self.make_pad and hs<self.w:
            h_pad = (self.w-hs)//2
            img = F.pad(img, [0, h_pad, 0, (self.w-hs)-h_pad])
        return img
    
class PadToSize(object):
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):
        width, height = image.size
        padding = (
            0, 0, max(0, self.target_width - width), max(0, self.target_height - height)
        )
        return F.pad(image, padding)

class ShortResize:
    def __init__(self, edge, interpolation=InterpolationMode.BILINEAR):
        self.edge = edge
        self.interpolation = interpolation

    def __call__(self, img:Image):
        w, h = img.size
        if min(w,h) < self.edge:
            if w < h:
                hs = int((h/w)*self.edge)
                img = F.resize(img, [hs, self.edge], self.interpolation)
            else:
                ws = int((w/h)*self.edge)
                img = F.resize(img, [self.edge, ws], self.interpolation)
        return img