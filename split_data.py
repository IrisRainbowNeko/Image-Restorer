import os
import json

root_clean='/GPUFS/sysu_pxwei_1/dzy/datas/skeb/png_origin'
png_origin=sorted([os.path.join('png_origin', x) for x in os.listdir(root_clean)])
root_mark = '/GPUFS/sysu_pxwei_1/dzy/datas/skeb/webp_sample'
webp_sample=sorted([os.path.join('webp_sample', x) for x in os.listdir(root_mark)])

#png_origin = sorted(os.listdir('/data1/dzy/dataset_raw/skeb/png_origin'))
#webp_sample = sorted(os.listdir('/data1/dzy/dataset_raw/skeb/webp_sample'))

rate=0.9
N_train = int(len(png_origin)*rate)
N_test = len(png_origin)-N_train

train_list={webp: png  for png, webp in zip(png_origin[:N_train], webp_sample[:N_train])}
test_list={webp: png for png, webp in zip(png_origin[N_train:], webp_sample[N_train:])}

with open('/GPUFS/sysu_pxwei_1/dzy/datas/skeb/train.json', 'w') as file:
    json.dump(train_list, file, indent=2)

with open('/GPUFS/sysu_pxwei_1/dzy/datas/skeb/test.json', 'w') as file:
    json.dump(test_list, file, indent=2)

# png_origin = [x[:-4] for x in png_origin]
# webp_sample = [x[:-5] for x in webp_sample]

# png_origin = set(png_origin)
# webp_sample = set(webp_sample)

# print(png_origin-webp_sample)

# print(webp_sample-png_origin)
# /data1/dzy/dataset_raw/skeb/webp_sample/custom_347728.png.webp