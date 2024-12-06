#CUDA_VISIBLE_DEVICES=7 accelerate launch train.py --arch mark-xl --bs 2
#CUDA_VISIBLE_DEVICES=7 neko_train_1gpu --cfg cfgs/py/train/image/skeb.py
neko_train --cfg cfgs/py/train/image/skeb.py