import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import seaborn as sns

import tifffile
import cv2


RANDOM_SEED = 42
'''
修改路径！
'''
BASE_DIR = "E:/hubmap-organ-segmentation"
TRAIN_DIR = "E:/input/hubmap-organ-segmentation/train_images"
TEST_DIR = "E:/input/hubmap-organ-segmentation/test_images"
LABEL_DIR = "E:/input/hubmap-organ-segmentation/train_annotations"

def read_image(image_id, scale=1, verbose=1):
    #scale为缩小比例
    image = tifffile.imread(
        os.path.join(BASE_DIR, f"train_images/{image_id}.tiff")
    )
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)

    mask = rle2mask(
        train_df[train_df["id"] == image_id]["rle"].values[0],
        (image.shape[1], image.shape[0])
    )

    if verbose:
        print(f"[{image_id}] Image shape: {image.shape}")
        print(f"[{image_id}] Mask shape: {mask.shape}")

    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        mask = cv2.resize(mask, new_size)

        if verbose:
            print(f"[{image_id}] Resized Image shape: {image.shape}")
            print(f"[{image_id}] Resized Mask shape: {mask.shape}")

    return image, mask




#train_df.sample(5)
#   Column            Non-Null Count  Dtype
#---  ------            --------------  -----
# 0   id                351 non-null    int64
# 1   organ             351 non-null    object
# 2   data_source       351 non-null    object
# 3   img_height        351 non-null    int64
# 4   img_width         351 non-null    int64
# 5   pixel_size        351 non-null    float64
# 6   tissue_thickness  351 non-null    int64
# 7   rle               351 non-null    object
# 8   age               351 non-null    float64
# 9   sex               351 non-null    object


def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    # 图像size:3000x3000，即9000000个像素
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        #print(lo, hi)
        img[lo: hi] = 1
        #plt.imshow(img.reshape(shape).T)  #逐步绘制显示
        #plt.show()
        #先是横着赋值，return的时候转置为竖排列
    return img.reshape(shape).T




if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    os.makedirs('E:\hubmap-organ-segmentation\labelimg/')  #修改路径
    for i in range(len(train_df)):
        img, mask = read_image(train_df.iloc[i]['id'], scale=1, verbose=0)
        cv2.imwrite('E:\hubmap-organ-segmentation\labelimg/' + str(train_df.iloc[i]['id'])+'.jpg', mask)