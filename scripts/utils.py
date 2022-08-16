import cv2
import albumentations as A

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random

DATASET_PATH = os.path.join("..", "datasets")
# DATASET_PATH = "/content/gdrive/MyDrive/analysed/table_extraction_dataset/table_extractor"
DS_IMAGES = os.path.join(DATASET_PATH, "images")
DS_MASKS = os.path.join(DATASET_PATH, "masks")
PAGE_IMAGES = "../datasets/Pages"

TABLE_NAMES = os.listdir("../datasets/tables")
PAGE_NAMES = os.listdir(PAGE_IMAGES)

MAX_VALUE = 255

MEAN = MAX_VALUE * (0.485 + 0.456 + 0.406) / 3
VARIANCE = MAX_VALUE * (0.229 + 0.224 + 0.225) / 3


def train_test_split(image_names, test_size, random_state=0, shuffle=True):
    if random_state:
        np.random.seed(random_state)
    
    train_names, test_names = [], []
    if shuffle:
        np.random.seed(2022)
        image_inds = np.random.permutation(len(image_names))
        np.random.seed(None)
    else: 
        image_inds = np.arange(0, len(image_names))

    split_count = int(len(image_inds) * test_size)
    for i in image_inds[split_count:]:
        train_names.append(image_names[i])

    for i in image_inds[:split_count]:
        test_names.append(image_names[i])

    return train_names, test_names


def read_inf_samples(image_names, resize_shape):
    batch_X, batch_y = [], []
    for name in image_names:
        img, mask = read_sample(name, resize_shape)
        edges = cv2.bitwise_not(cv2.Canny(img, 1, 10))
        img = np.moveaxis(np.array([img, edges]), 0, -1)

        img = img / MAX_VALUE
        mask = mask / MAX_VALUE

        batch_X.append(img)
        batch_y.append(mask)

    return tf.convert_to_tensor(batch_X, dtype=tf.float32), tf.convert_to_tensor(batch_y, dtype=tf.float32)


def read_sample(image_name, resize_shape):
    base_name, ext = os.path.splitext(image_name)

    if os.path.exists(os.path.join(DS_IMAGES, image_name)):
        img = cv2.imread(os.path.join(DS_IMAGES, image_name), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    else:
        img = cv2.imread(os.path.join(PAGE_IMAGES, image_name), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    
    if os.path.exists(os.path.join(DS_MASKS, base_name + "_mask" + ext)):
        mask = cv2.imread(os.path.join(DS_MASKS, base_name + "_mask" + ext), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    else:
        mask = np.zeros(resize_shape).astype(np.uint8)

    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_AREA)

    # rgb_img = np.moveaxis(np.array([img, img, img]), 0, -1)
    rgb_img = img
    rgb_mask = mask
    # print("Read image fn ", rgb_img.shape, rgb_mask.shape)
    return rgb_img, rgb_mask


def random_batch_generator(
            batch_size,
            resize_shape,
            train_names,
            max_tables_on_image=5,
            train_aug_transform=None, normalize=True,
            include_edges_as_band=False,
            table_aug_transform=None
        ):

    while True:

        batch_X, batch_y = [], []

        while True:

            sample_gen_method_ind = np.random.choice([1, 2])

            if sample_gen_method_ind == 1:
                random_name_ind = random.randint(0, len(train_names)-1)
                img, mask = read_sample(train_names[random_name_ind], resize_shape)

            # elif sample_gen_method_ind == 2:
            #     random_page_name = PAGE_NAMES[random.randint(0, len(PAGE_NAMES)-1)]
            #     img = cv2.imread("../datasets/Pages/" + random_page_name, cv2.IMREAD_GRAYSCALE)
            #     mask = np.zeros_like(img).astype(np.uint8)

            #     number_of_tables = random.randint(1, max_tables_on_image)

            #     for _ in range(number_of_tables):
            #         random_table_name = TABLE_NAMES[random.randint(0, len(TABLE_NAMES)-1)]
            #         table_img = cv2.imread("../datasets/tables/" + random_table_name, cv2.IMREAD_GRAYSCALE)

            #         if table_aug_transform:
            #             table_img = apply_table_augmentation(table_aug_transform, table_img)

            #         current_possible_pos = []
            #         if table_img.shape[0] > img.shape[0] or table_img.shape[1] > img.shape[1]:
            #             continue

            #         for i in range(0, img.shape[0] - table_img.shape[0], 10):
            #             for j in range(0, img.shape[1] - table_img.shape[1], 10):
            #                 if np.all(img[i:i+table_img.shape[0], j:j+table_img.shape[1]] == 255):
            #                     current_possible_pos.append((i, j))

            #         if len(current_possible_pos) > 0:
            #             rand_ind = np.random.choice(np.arange(len(current_possible_pos)))
            #             y, x = current_possible_pos[rand_ind]

            #             img[y:y+table_img.shape[0], x:x+table_img.shape[1]] = table_img
            #             mask[y:y+table_img.shape[0], x:x+table_img.shape[1]] = 255

            else:

                random_page_name = PAGE_NAMES[random.randint(0, len(PAGE_NAMES)-1)]
                img = cv2.imread("../datasets/Pages/" + random_page_name, cv2.IMREAD_GRAYSCALE)
                mask = np.zeros_like(img).astype(np.uint8)

                number_of_tables = random.randint(1, max_tables_on_image)

                min_y = 0

                for _ in range(number_of_tables):
                    random_table_name = TABLE_NAMES[random.randint(0, len(TABLE_NAMES)-1)]
                    table_img = cv2.imread("../datasets/tables/" + random_table_name, cv2.IMREAD_GRAYSCALE)

                    if table_aug_transform:
                        table_img = apply_table_augmentation(table_aug_transform, table_img)

                    if table_img.shape[0] > img.shape[0] or table_img.shape[1] > img.shape[1]:
                        continue

                    y = random.randint(0, img.shape[0]-table_img.shape[0])
                    x = random.randint(0, img.shape[1]-table_img.shape[1]+min_y)

                    if np.any(mask[y:y+table_img.shape[0], x:x+table_img.shape[1]] == 255):
                        continue
                    
                    img[y:y+table_img.shape[0], x:x+table_img.shape[1]] = table_img
                    mask[y:y+table_img.shape[0], x:x+table_img.shape[1]] = 255

                # img = np.moveaxis(np.array([img, img, img]), 0, -1)


            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_AREA)

            if include_edges_as_band:
                edges = cv2.bitwise_not(cv2.Canny(img, 5, 10))
                img = np.moveaxis(np.array([img, edges]), 0, -1)

            if train_aug_transform:
                img, mask = apply_train_augmentation(train_aug_transform, img, mask)
                # img[:, :, 0] = apply_augmentation(get_orig_image_transform(), img[:, :, 0])

            if normalize:
                img = img / MAX_VALUE
                # img[:, :, 1][img[:, :, 1] == 255] = 1
                mask = mask / MAX_VALUE

            batch_X.append(img)
            batch_y.append(mask)

            if len(batch_X) >= batch_size:
                break

        # print([X.shape for X in batch_X])
        yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)


def image_batch_generator(image_names, batch_size, resize_shape, normalize=True, aug_transform=None, include_edges_as_band=False):

    while True:
        batch_X, batch_y = [], []
        for i, image_name in enumerate(image_names):
            img, mask = read_sample(image_name, resize_shape)

            if include_edges_as_band:
                edges = cv2.bitwise_not(cv2.Canny(img, 5, 10))
                img = np.moveaxis(np.array([img, edges]), 0, -1)

            if aug_transform:
                img, mask = apply_train_augmentation(aug_transform, img, mask)
                # img[:, :, 0] = apply_augmentation(get_orig_image_transform(), img[:, :, 0])

            if normalize:
                img = img / MAX_VALUE
                # img[:, :, 1][img[:, :, 1] == 255] = 1
                mask = mask / MAX_VALUE

            batch_X.append(img)
            batch_y.append(mask)

            if len(batch_X) == batch_size or i+1 >= len(image_names):
                return_batch = np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)
                batch_X, batch_y = [], []
                yield return_batch


def apply_augmentation(transform, image):
    return transform(image=image)["image"]


def apply_table_augmentation(transform, image):
    transformed_table = transform(image=image)["image"]
    if random.random() > 0.5:
        transformed_table = transformed_table.T

    scale_factor = 0.2
    if random.random() > 0.5:
        new_h = image.shape[0] + int(image.shape[0] * (random.random() * 2 * scale_factor - scale_factor))
        new_w = image.shape[1] + int(image.shape[1] * (random.random() * 2 * scale_factor - scale_factor))
        transformed_table = cv2.resize(transformed_table, (new_h, new_w), interpolation=cv2.INTER_AREA)
    return transformed_table


def apply_train_augmentation(transform, image, mask):
    transformed = transform(image=image, mask=mask)
    return transformed["image"], transformed["mask"]


def get_table_augmentation():
    table_transform = [
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.4),
        A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5)
    ]
    return A.Compose(table_transform)


def get_orig_image_transform():
    trans = [
        # A.GaussNoise(var_limit=(10.0, 90.0), p=0.5),
        # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
        # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)
    ]
    return A.Compose(trans)


def get_train_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Rotate(limit=45, border_mode=0, p=1, value=(255, 255, 255)),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.0),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, p=1),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.3),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=1)
        # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)
    ]
    return A.Compose(train_transform)


class SaveValidSamplesCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, train_names, valid_names, resize_shape, date_str):
        super().__init__()
        self.model = model
        self.train_names = train_names
        self.valid_names = valid_names
        self.resize_shape = resize_shape
        self.date_str = date_str

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(f"./predicted_samples/{self.date_str}"):
            os.mkdir(f"./predicted_samples/{self.date_str}")

        save_pred_samples(
            self.model, self.train_names, self.resize_shape, epoch, 
            "train", directory=f"./predicted_samples/{self.date_str}"
        )
        save_pred_samples(
            self.model, self.valid_names, self.resize_shape, epoch, 
            "valid", directory=f"./predicted_samples/{self.date_str}"
        )


def save_pred_samples(model, sample_names, resize_shape, epoch, set_name, directory):

    for i, image_name in enumerate(sample_names):
        img, mask = read_sample(image_name, resize_shape)

        edges = cv2.bitwise_not(cv2.Canny(img, 1, 10))
        input_tensor = np.moveaxis(np.array([img, edges]), 0, -1)

        input_tensor = input_tensor / MAX_VALUE
        mask = mask / MAX_VALUE

        logits = model(tf.expand_dims(input_tensor, 0), training=False)
        logits = np.squeeze(logits)

        b_n, ext = os.path.splitext(image_name)

        rgb_img = np.moveaxis(
            np.array([img, img, img]), 
            0, -1
        )

        pred_3dim = np.moveaxis(
            (np.array([logits, logits, logits])*255).astype(np.uint8), 
            0, -1
        )

        target_3dim = np.moveaxis(
            (np.array([mask, mask, mask])*255).astype(np.uint8), 
            0, -1
        )

        final_img = cv2.hconcat([
            rgb_img, 
            target_3dim,
            pred_3dim
        ])

        cv2.imwrite(
            os.path.join(directory, f"{epoch}_{b_n}_{set_name}.png"),
            final_img
        )
        print("Saved predicted sample ", image_name, end='\r')