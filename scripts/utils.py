import cv2
import albumentations as A

import numpy as np
import pandas as pd
import os
from pathlib import Path
import random


DATASET_PATH = os.path.join("..", "datasets", "extracted_dataset")
DS_IMAGES = os.path.join(DATASET_PATH, "images")
DS_MASKS = os.path.join(DATASET_PATH, "masks")

MAX_VALUE = 255


def train_test_split(image_names, test_size, random_state=0, shuffle=True):
    if random_state:
        np.random.seed(random_state)
    
    train_names, test_names = [], []
    if shuffle:
        image_inds = np.random.permutation(len(image_names))
    else: 
        image_inds = np.arange(0, len(image_names))

    split_count = int(len(image_inds) * test_size)
    for i in image_inds[:split_count]:
        train_names.append(image_names[i])

    for i in image_inds[split_count:]:
        test_names.append(image_names[i])

    return train_names, test_names


def read_sample(image_name, resize_shape):
    base_name, ext = os.path.splitext(image_name)

    img = cv2.imread(os.path.join(DS_IMAGES, base_name + ext), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    if os.path.exists(os.path.join(DS_MASKS, base_name + "_mask" + ext)):
        mask = cv2.imread(os.path.join(DS_MASKS, base_name + "_mask" + ext), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    else:
        mask = np.zeros(resize_shape).astype(np.uint8)

    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_AREA)

    rgb_img = np.moveaxis(np.array([img, img, img]), 0, -1)
    rgb_mask = np.moveaxis(np.array([mask, mask, mask]), 0, -1)

    return rgb_img, rgb_mask


def image_batch_generator(image_names, batch_size, resize_shape, normalize=True, aug_transform=None):

    batch_X, batch_y = [], []
    for i, image_name in enumerate(image_names):
        img, mask = read_sample(image_name, resize_shape)
        if aug_transform:
            img, mask = apply_augmentation(aug_transform, img, mask)
        batch_X.append(img)
        batch_y.append(mask)

        if len(batch_X) == batch_size or i+1 >= len(image_names):
            return_batch = np.array(batch_X), np.array(batch_y)
            batch_X, batch_y = [], []
            if normalize:
                yield return_batch[0]/MAX_VALUE, return_batch[1]/MAX_VALUE
            else:
                yield return_batch


def apply_augmentation(transform, image, mask):
    transformed = transform(image=image, mask=mask)
        # trans = A.Compose([
        #     A.RandomCrop(height=image.shape[0], width=image.shape[1], always_apply=True),
        #     A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0)
        # ])
        # return trans(image=transformed["image"], mask=transformed["mask"])

    return transformed["image"], transformed["mask"]


def get_train_augmentation():
    train_transform = [
        # A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Rotate(limit=45, border_mode=0, p=1),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=45, border_mode=0, p=1, value=(255, 255, 255)),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5)
    ]
    return A.Compose(train_transform)