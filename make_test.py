import os
import random
import shutil

def copy_random_files(source_dir, target_dir, n):
    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # 确保文件夹中至少有N个文件
    if len(files) < n:
        print(f"文件夹中只有 {len(files)} 个文件，无法选择 {n} 个文件.")
        return
    
    # 随机选择N个文件
    selected_files = random.sample(files, n)
    
    # 如果目标文件夹不存在，则创建目标文件夹
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 复制文件到目标文件夹
    for file in selected_files:
        src_file = os.path.join(source_dir, file)
        dst_file = os.path.join(target_dir, file)
        shutil.copy(src_file, dst_file)
        print(f"已复制: {file}")

# 示例用法
source_directory = '/GPUFS/sysu_pxwei_1/dzy/datas/skeb/png_origin'
target_directory = '/GPUFS/sysu_pxwei_1/dzy/datas/skeb/png_origin_test'
number_of_files = 500

copy_random_files(source_directory, target_directory, number_of_files)
