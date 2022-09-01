import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.join("..", "keras_unets"))

from utils import train_test_split, image_batch_generator, get_train_augmentation, random_batch_generator, get_table_augmentation
from utils import DATASET_PATH, DS_IMAGES, PAGE_IMAGES, DS_MASKS, TEST_IMAGES, TEST_MASKS
import utils
from metrics import dice_coef, iou, f1_score, jaccard_distance
import metrics
from vis import anshow, imshow
from models import TableNet, att_unet, load_unet_model
from keras_unet_collection.models import att_unet_2d

# SCRIPTS_PATH = "/content/gdrive/MyDrive/table_extraction_dataset/table_extractor/scripts/"

TR_CONFIG = {
    "epochs" : 100,
    "batch_size" : 7,
    # "val_batch_size" : 32,
    "lr" : 1e-4,
    "input_shape" : (512, 512),
    "band_size" : 2,
    "three_channel" : False
}


def print_progress(name, metrics, step, all_steps):
    str_prog = f"{all_steps}/{step}: "
    str_prog += "{} loss {:.4f}, tf_iou {:.4f}, iou {:.4f}, f1 {:.4f}, prec {:.4f}, rec {:.4f}".format(
        name,
        np.mean(metrics["loss"]),
        np.mean(metrics["tf_iou"]),
        np.mean(metrics["iou"]),
        np.mean(metrics["f1"]),
        np.mean(metrics["precision"]),
        np.mean(metrics["recall"])
    )

    print(str_prog, end='\r')


class CycleLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, step_size, change):
        self.current_learning_rate = initial_learning_rate
        self.dir = -1
        self.step_size = step_size
        self.steps = 0
        self.change = change

    def __call__(self, step):
        # print("Call")
        if self.steps % self.step_size == 0:
            self.dir *= -1
        self.current_learning_rate += self.dir * self.change
        self.steps += 1
        # print("\n", self.current_learning_rate)
        return self.current_learning_rate

def train():

    # model = TableNet.build(inputShape=(TR_CONFIG["input_shape"][0], TR_CONFIG["input_shape"][1], TR_CONFIG["band_size"]))
    down_scales = [32, 64, 128, 256]
    # down_scales = [16, 32, 64, 128]
    model = att_unet_2d((TR_CONFIG["input_shape"][0], TR_CONFIG["input_shape"][1], 2), down_scales, n_labels=1,
                stack_num_down=2, stack_num_up=2,
                activation='ReLU', atten_activation='ReLU', attention='add', output_activation="Sigmoid", 
                batch_norm=True, pool=False, unpool='bilinear', name='attunet'
            )
    # model = load_unet_model(TR_CONFIG["input_shape"], TR_CONFIG["band_size"], weight_decay=0.1, weight_scale=2)

    # lr_schedule = CycleLRSchedule(1e-7, 10, 1e-7)
    
    optim = tf.keras.optimizers.Adam(learning_rate=1e-8, beta_1=0.9, beta_2=0.999)
    # optim = tf.keras.optimizers.SGD(learning_rate=TR_CONFIG["lr"], momentum=0.0)
    loss_fn = jaccard_distance
    # loss_fn = metrics.jaccard_plus_cross_entropy(beta=0.2)
    # loss_fn = metrics.dice_coef_loss
    # loss_fn = metrics.dice_plus_cross_entropy(beta=0.3)
    # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    train_names = os.listdir(DS_IMAGES) + os.listdir(PAGE_IMAGES)
    valid_names = os.listdir(TEST_IMAGES)

    DATE_STR = str(datetime.now().strftime("%Y.%m.%d-%H"))
    LOG_DIR = "training_logs/" + DATE_STR + "/"

    tf_writers = {
        "train" : tf.summary.create_file_writer(LOG_DIR + "train/"),
        "valid" : tf.summary.create_file_writer(LOG_DIR + "valid/")
    }

    # setup checkpoints
    checkpoint_directory = "training_checkpoints/" + DATE_STR
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    ### keras checkpoints
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #                                     checkpoint_prefix, 
    #                                     save_weights_only=True, 
    #                                     save_best_only=True
    #                                 )

    # tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)

    # save_val_samples_callback = SaveValidSamplesCallback(
    #                                     model, 
    #                                     train_names[:10], 
    #                                     valid_names[:10], 
    #                                     TR_CONFIG["input_shape"], 
    #                                     date_str=DATE_STR
    #                                 )

    # model.load_weights("training_checkpoints/2022.07.29-22/ckpt-1")
    # print("successfully loaded checkpoint.")

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=model)
    print(f"loading checkpoint {'training_checkpoints/' + '2022.08.29-07/ckpt-192'}")
    status = checkpoint.restore("training_checkpoints/" + '2022.08.29-07/ckpt-192')
    # status.expect_partial()

    valid_batch_generator = image_batch_generator(
                                valid_names, 
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"],
                                aug_transform=None,
                                normalize=True, three_channel=TR_CONFIG["three_channel"],
                                ds_images=TEST_IMAGES, ds_masks=TEST_MASKS
                            )

    for epoch in range(1, TR_CONFIG["epochs"] + 1):

        print(f"\nEpoch {TR_CONFIG['epochs']}/{epoch}")
        print(f"Shuffling...")
        import time
        t = 1000 * time.time() # current time in milliseconds
        np.random.seed(int(t) % 2**32)
        random_inds = np.random.permutation(len(train_names))
        train_names = np.array(train_names)[random_inds]

        train_batch_generator = image_batch_generator(
                                    train_names, 
                                    batch_size=TR_CONFIG["batch_size"], 
                                    resize_shape=TR_CONFIG["input_shape"], 
                                    aug_transform=get_train_augmentation(),
                                    normalize=True, three_channel=TR_CONFIG["three_channel"]
                                )

        # train_batch_generator = random_batch_generator(
        #                             batch_size=TR_CONFIG["batch_size"], 
        #                             resize_shape=TR_CONFIG["input_shape"],
        #                             train_names=train_names,
        #                             train_aug_transform=get_train_augmentation(),
        #                             table_aug_transform=get_table_augmentation(), 
        #                             max_tables_on_image=6, normalize=True, three_channel=True
        #                         )

        tr_metrics = {n:[] for n in ("loss", "iou", "tf_iou", "f1", "precision", "recall")}
        val_metrics = {n:[] for n in ("loss", "iou", "tf_iou", "f1", "precision", "recall")}

        # train loop
        for i, (batch_X, batch_y) in enumerate(train_batch_generator):
            # break
            # print(batch_X.dtype, batch_y.dtype)
            # print(batch_y.min(), batch_y.max())
            batch_X = tf.convert_to_tensor(batch_X, dtype=tf.float32)
            batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)

            with tf.GradientTape() as tape:
                # print(X.shape)
                pred = model(batch_X, training=True)
                pred = tf.squeeze(pred, -1)

                loss_value = loss_fn(pred, batch_y)
                # print(loss_value)

                grads = tape.gradient(loss_value, model.trainable_weights)

            gradient_message = ""
            for g_i, g in enumerate(grads):
                if np.all(np.array(g) == 0):
                    gradient_message += f"{g_i} = 0, "
                elif np.all(np.abs(g) <= 10e-7):
                    gradient_message += f"{g_i} < 10e-7 ({np.sum(g)}), "
    
            if gradient_message:
                print(f"\nNote: {gradient_message}.")

            optim.apply_gradients(zip(grads, model.trainable_weights))

            (
                iou_value, tf_iou_value,
                f1_score_value, 
                presicion_value, 
                recall_value
            ) = metrics.calculate_metrics(batch_y, pred)

            tr_metrics["loss"].append(loss_value)
            tr_metrics["iou"].append(iou_value)
            tr_metrics["tf_iou"].append(tf_iou_value)
            tr_metrics["f1"].append(f1_score_value)
            tr_metrics["precision"].append(presicion_value)
            tr_metrics["recall"].append(recall_value)

            # print(i+1, len(train_names)//TR_CONFIG["batch_size"])
            print_progress("train", tr_metrics, i+1, len(train_names)//TR_CONFIG["batch_size"])
            # break
            if (i + 1) >= len(train_names)//TR_CONFIG["batch_size"]:
                break

        print("\n", end="")

        # valid loop
        for i, (batch_X, batch_y) in enumerate(valid_batch_generator):

            # print(batch_X.dtype, batch_y.dtype)
            # print(batch_y.min(), batch_y.max())
            batch_X = tf.convert_to_tensor(batch_X, dtype=tf.float32)
            batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)

            pred = model(batch_X, training=False)
            pred = tf.squeeze(pred, -1)

            loss_value = loss_fn(pred, batch_y)
  
            (
                iou_value, tf_iou_value,
                f1_score_value, 
                presicion_value, 
                recall_value
            ) = metrics.calculate_metrics(batch_y, pred)

            val_metrics["loss"].append(loss_value)
            val_metrics["iou"].append(iou_value)
            val_metrics["tf_iou"].append(tf_iou_value)
            val_metrics["f1"].append(f1_score_value)
            val_metrics["precision"].append(presicion_value)
            val_metrics["recall"].append(recall_value)

            print_progress("valid", val_metrics, i+1, len(valid_names)//TR_CONFIG["batch_size"])
            # break
            if (i + 1) >= len(valid_names)//TR_CONFIG["batch_size"]:
                break

        print("\n", end="")

        print("Saving predicted samples.")
        if not os.path.exists(f"./predicted_samples/{DATE_STR}"):
            os.mkdir(f"./predicted_samples/{DATE_STR}")

        print("Writing to the log...")
        with tf_writers["train"].as_default():
            for k, v in tr_metrics.items():
                tf.summary.scalar(k, np.mean(v), step=epoch)

        with tf_writers["valid"].as_default():
            for k, v in val_metrics.items():
                tf.summary.scalar(k, np.mean(v), step=epoch)

        path = checkpoint.save(file_prefix=checkpoint_prefix)
        print("Saved checkpoint for epoch {} to {}".format(epoch, path))

        utils.save_pred_samples(
            model, train_names[:20], TR_CONFIG["input_shape"], epoch,
            "train", directory=f"./predicted_samples/{DATE_STR}", three_channel=TR_CONFIG["three_channel"]
        )
        
        utils.save_pred_samples(
            model, valid_names[:40], TR_CONFIG["input_shape"], epoch,
            "valid", directory=f"./predicted_samples/{DATE_STR}", three_channel=TR_CONFIG["three_channel"],
            ds_images=TEST_IMAGES, ds_masks=TEST_MASKS
        )

    # model.compile(
    #     optimizer=optim,
    #     loss=loss_fn,
    #     metrics=[iou, f1_score, metrics.precision, metrics.recall]
    # )

    # model.fit(
    #     train_batch_generator,
    #     steps_per_epoch=(len(train_names) // TR_CONFIG["batch_size"]) + 1,
    #     # steps_per_epoch=100,
    #     validation_data=valid_batch_generator,
    #     validation_steps=(len(valid_names) // TR_CONFIG["batch_size"]) + 1,
    #     # validation_steps=5,
    #     batch_size=TR_CONFIG["batch_size"],
    #     epochs=TR_CONFIG["epochs"],
    #     callbacks=[model_checkpoint, tb_callback, save_val_samples_callback]
    # )


if __name__ == "__main__":
    train()