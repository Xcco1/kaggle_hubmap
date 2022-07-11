import  os
import pandas as pd
import numpy as np
import cv2
def rle_decode(mask_rle, shape, color=1):
    """ TBD

    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return

    Returns:
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image image is actually flattened since RLE is a 1D "run"
    if len(shape) == 3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo: hi] = color

    # Don't forget to change the image back to the original shape
    return img.reshape(shape).T


# https://www.kaggle.com/namgalielei/which-reshape-is-used-in-rle
def rle_decode_top_to_bot_first(mask_rle, shape):
    """ TBD

    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return

    Returns:
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0]), order='F').T  # Reshape from top -> bottom first


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """ TBD

    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    img = img.T
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]



TEST_CSV = os.path.join('E://hubmap-organ-segmentation', "test.csv")
test_df = pd.read_csv(TEST_CSV)
SS_CSV   = os.path.join('E://hubmap-organ-segmentation', "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)

import tifffile as tiff
from torchvision.transforms import transforms
import torch
import albumentations as A
import numpy as np
from PIL import Image
def tile_image(p_img, folder, size: int = 1024) -> list:
    w = h = size
    im = np.array(Image.open(p_img))
    # https://stackoverflow.com/a/47581978/4521646
    tiles = [im[i:(i + h), j:(j + w), ...] for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    idxs = [(i, (i + h), j, (j + w)) for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    name, _ = os.path.splitext(os.path.basename(p_img))
    files = []
    for k, tile in enumerate(tiles):
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile = np.zeros_like(tiles[0])
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_

        p_img = os.path.join(folder, f"{name}_{k:03}.png")
        #Image.fromarray(tile).save(p_img)
        files.append(p_img)
    return files, idxs


def make_prediction(model, test_transforms):
    device = 'cuda'
    test_img =tiff.imread(r'E:\hubmap-organ-segmentation\test_images\10078.tiff')
    test_img=np.array(test_img)
    print(test_img.shape)
    test_img=np.pad(test_img, ((12, 13), (12, 13),(0,0)), 'constant', constant_values=0)
    print(test_img.shape)
    pred_patches = np.zeros((2048, 2048), dtype=np.uint8)
    window = 512
    with torch.no_grad():
        model.eval()
        for r in range(0, 2048, window):
            for c in range(0, 2048, window):
                img_patch = test_transforms(test_img[r: r+window, c: c+window]).unsqueeze(0).to(device)
                output = model(img_patch)
                #print(output[0][0])
                y_pred = (output[0][0] > 0.5).cpu()
                pred_patches[r: r+window, c: c+window] = y_pred
    return pred_patches
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
img = Image.open(r'E:\hubmap-organ-segmentation\test_images\10078.tiff')
img = np.array(img)
[w,h,c]=img.shape

#tiles_img, _ = tile_image(r'E:\hubmap-organ-segmentation\test_images\10078.tiff', r"/kaggle/temp/images", size=512)
#print(tiles_img[1].shape)
#Image._show(tiles_img[1])
pad=transforms.Pad([25,25],fill=(0,0,0),padding_mode='constant')
transform = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),

        ])

#img=torch.tensor(np.moveaxis(transform(image=img)['image'],2,0), dtype=torch.float32)
#img=torch.unsqueeze(img,dim=0)
from PIL import Image
from model import UneXt50
import matplotlib.pyplot as plt
import torch
rles=[]
model=UneXt50().cuda()
model.eval()
model.load_state_dict(torch.load("./best_metric_model_segmentation2d_array.pth"))
#prediction=make_prediction(model,transform)
test_img =tiff.imread(r'E:\hubmap-organ-segmentation\test_images\10078.tiff')
test_img=Image.fromarray(test_img)
print(test_img.shape)
#test_img=np.array(test_img)
img_patch =transform(test_img).unsqueeze(0).cuda()
output = model(img_patch).cpu().detach().numpy()
y_pred = (output[0][0] > 0.5)
#print(y_pred.shape)
mask = cv2.resize(output[0][0],(2023, 2023))
#ones=np.ones_like(mask)
#mask=ones-mask
mask[mask>0.5]=1
mask[mask<=0.5]=0


def rle_encode(img):
    """ TBD

    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    img = img.T
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_encode_less_memory(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    '''
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
circle = np.zeros((2023, 2023))
circle = cv2.circle(circle, (int(np.round(1024)), int(np.round(1024))), int(np.round((1024)*0.8)), 1, -1)
plt.imshow(mask)
plt.show()
plt.imshow(test_img)
plt.show()
rles = []
rles.append(rle_encode(circle))
SS_CSV   = os.path.join('E:\hubmap-organ-segmentation', "sample_submission.csv")
import pandas as pd
ss_df = pd.read_csv(SS_CSV)
ss_df["rle"] = rles
ss_df = ss_df[["id", "rle"]]
ss_df.to_csv("UneXt.csv", index=False)
