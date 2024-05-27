import numpy as np
from numpy.linalg import lstsq
from PIL import Image
import os
from scipy.optimize import leastsq

# 图像加载函数
def load_image(path):
    with Image.open(path) as img:
        return np.array(img)

# 指定原始图像和水印图像的文件夹路径
original_images_path = '/mnt/others/dataset/water_mark/pure_color_samples/png_origin/'
watermarked_images_path = '/mnt/others/dataset/water_mark/pure_color_samples/png_sample/'

# 获取所有图像文件名
original_images_files = sorted(os.listdir(original_images_path))[:20]
watermarked_images_files = sorted(os.listdir(watermarked_images_path))[:20]

def mix_color(a, c1, c2):
    return a*c1 + (1-a)*c2

def error(param, c_ori, c_mark):
    res= mix_color(param[0], param[1:].reshape(1,3), c_ori) - c_mark
    print(res.shape)
    return res
    # a,r,g,b = param
    # return (
    #         mix_color(a, r, c_ori[0]) - c_mark[0]+
    #         mix_color(a, g, c_ori[1]) - c_mark[1]+
    #         mix_color(a, b, c_ori[2]) - c_mark[2]
    # )

def get_watermark(original_images_files, watermarked_images_files):
    original_images = []
    watermarked_images = []
    for original_image_file, watermarked_image_file in zip(original_images_files, watermarked_images_files):
        # 加载原始图像和水印图像
        original_image = load_image(os.path.join(original_images_path, original_image_file))
        watermarked_image = load_image(os.path.join(watermarked_images_path, watermarked_image_file))

        original_images.append(original_image[:,:, :3])
        watermarked_images.append(watermarked_image[:,:, :3])


    original_images = np.stack(original_images, axis=2)
    watermarked_images = np.stack(watermarked_images, axis=2)

    h, w = original_images.shape[:2]

    watermark_image = np.zeros((h,w,4), dtype=original_images.dtype)

    for y in range(h):
        for x in range(w):
            c0 = np.array([0.5, 255., 255., 255.])
            c_pred = leastsq(error, c0, args=(original_images[y, x, :, :], watermarked_images[y, x, :, :]))
            watermark_image[y, x, :3] = c_pred[0][1:]
            watermark_image[y, x, 3] = c_pred[0][0]


    return watermark_image

# 保存平均水印图像
watermark = get_watermark(original_images_files, watermarked_images_files)
average_watermark_image = Image.fromarray(np.uint8(watermark))
average_watermark_image.save('average_watermark.png')

print('The average watermark image has been saved as "average_watermark.png".')
