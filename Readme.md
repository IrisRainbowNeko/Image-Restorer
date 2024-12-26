# Image Restoration Framework

## Prepare

Install RainbowNeko Engine
```bash
pip install rainbowneko
```

## Train

```bash
neko_train --cfg cfgs/py/train/image/webp_fix.py
```


## Inference

webp or jpeg fix
```bash
python demo.py --arch mark-s --ckpt "path to model" --img imgs/ --out_dir results --crop
```

Tow stage fix
```bash
python demo2stage.py --arch_mark mark-l --arch_fix mark-s --ckpt_mark "path to mark model" --ckpt_fix "path to webp model" --img 
imgs/ --out_dir results/ --crop
```