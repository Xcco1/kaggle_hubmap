import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
import cv2

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from monai.data import decollate_batch

from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
)

INPUT_PATH = '/home/leadmove/dataset/Kaggle/'

# IMG Settings

IMG_SHAPE = (512, 512)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape).T

def random_box(inputshape):
    x,y=np.random.randint(0,inputshape-512),np.random.randint(0,inputshape-512)
    return x,y


class HubMAP_Dataset(Dataset):
    def __init__(self, train):
        self.dataframe = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))

        # Stratified dataframe division by organ trait
        self.input = self.dataframe.groupby("organ", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(0.8 * len(x)))))

        if not train:
            self.input = self.dataframe.drop(self.input.index)
        self.train=train
        # Augmentations
        self.transform = A.Compose([
            A.Resize(1024,1024),
            A.ShiftScaleRotate(rotate_limit=25, scale_limit=0.15, shift_limit=0, p=0.75),
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, p=0.5),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, ndx):
        # Image properties
        img_path = str(self.input.iloc[ndx, 0]) + ".tiff"
        shape = (int(self.input.iloc[ndx, 3]), int(self.input.iloc[ndx, 4]))
        # Image/Mask Loading
        img = Image.open(os.path.join(INPUT_PATH, "train_images", img_path))
        img = np.array(img)
        img= cv2.resize(img,(1024,1024))
        mask = rle_decode(self.input.iloc[ndx, -3], shape)
        mask=cv2.resize(mask,(1024,1024))
        mask[mask>0.5]=1
        mask[mask<=0.5]=0
        '''if self.train:
            while 1:
                x, y = random_box(img.shape[0])
                maskroi = mask[x:x+512, y:y+512]
                if np.sum(maskroi) > 100: break
            img=img[x:x+512,y:y+512]'''

        if self.train:
            aug_data = self.transform(image=img, mask=mask)
            return torch.tensor(np.moveaxis(aug_data["image"], 2, 0), dtype=torch.float32), torch.tensor(
                np.expand_dims(aug_data["mask"], axis=0), dtype=torch.int16)
        else:
            return img,mask
from model import UneXt50

model=UneXt50().to(device)
#model.load_state_dict(torch.load("./best_metric_model_segmentation2d_array.pth"))
loss_function = monai.losses.DiceLoss(sigmoid=False)
from monai.losses import FocalLoss
focal=FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# Dataset
train_ds = HubMAP_Dataset(train=True)
train_loader = DataLoader(train_ds, batch_size=4,num_workers=0,shuffle=True)

val_ds = HubMAP_Dataset(train=False)
val_loader = DataLoader(val_ds, batch_size=4,shuffle=True)

# Training Loop
# Training Loop
minloss=10  #??????loss
recordmetric=0.5 #??????val?????????
val_interval = 1  #??????epoch val??????

num_epochs = 200
epoch_loss_values = list()

writer = SummaryWriter()
for epoch in range(num_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in tqdm(train_loader):
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)+focal(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    schedule.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if (epoch +1 ) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in tqdm(val_loader):
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)
                #val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print('metric',metric)
            if metric>recordmetric:
              print('saved new best metric model')
              torch.save(model.state_dict(), "bestmetric.pth")
            # reset the status for next validation round
            dice_metric.reset()


    if epoch_loss < minloss:
                minloss=epoch_loss
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best.pth")
                print("saved new best loss model")
