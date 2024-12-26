import os
from argparse import ArgumentParser

from demo import Infer, types_support
from tqdm import tqdm
from utils import get_ext


def infer(infer_mark, infer_fix, path, out_dir):
    if os.path.isdir(path):
        files = [os.path.join(path, x) for x in os.listdir(path) if get_ext(x).lower() in types_support]
        for file in tqdm(files):
            img = infer_mark.infer_one(file)
            img = infer_fix.infer_one(img)
            img.save(os.path.join(out_dir, os.path.basename(file))+'.png')
    else:
        img = infer_mark.infer_one(path)
        img = infer_fix.infer_one(img)
        img.save(os.path.join(out_dir, os.path.basename(path))+'.png')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--arch_mark", default='mark-l', type=str)
    parser.add_argument("--arch_fix", default='mark-s', type=str)
    parser.add_argument("--ckpt_mark", default='', type=str)
    parser.add_argument("--ckpt_fix", default='', type=str)
    parser.add_argument("--img", default='', type=str)
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--out_dir", default='results', type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    infer_mark = Infer(args.ckpt_mark, args.arch_mark, crop=args.crop)
    infer_fix = Infer(args.ckpt_fix, args.arch_fix, crop=False)
    infer(infer_mark, infer_fix, args.img, args.out_dir)