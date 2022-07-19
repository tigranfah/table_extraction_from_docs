from random import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

from utils import train_test_split, image_batch_generator, get_train_augmentation
from utils import DATASET_PATH, DS_IMAGES, DS_MASKS
from metrics import iou, f1_score, jaccard_distance
from vis import anshow, imshow
from models import TableNet

IMAGE_NAMES = os.listdir(DS_IMAGES)

TR_CONFIG = {
    "epochs" : 100,
    "batch_size" : 8,
    "lr" : 10e-4,
    "input_shape" : (512, 512),
    "band_size" : 3
}

def train():

    model = TableNet.build(inputShape=(TR_CONFIG["input_shape"][0], TR_CONFIG["input_shape"][1], TR_CONFIG["band_size"]))
    optim = tf.keras.optimizers.Adam(learning_rate=TR_CONFIG["lr"])
    loss_fn = jaccard_distance

    train_augmentation = get_train_augmentation()

    DATE_STR = str(datetime.now().strftime("%Y.%m.%d-%H"))
    LOG_DIR = "training_logs/" + DATE_STR + "/"
    tf_writers = {
        "train" : tf.summary.create_file_writer(LOG_DIR + "train/"),
        "valid" : tf.summary.create_file_writer(LOG_DIR + "valid/")
    }

    # setup checkpoints
    checkpoint_directory = "./training_checkpoints/" + DATE_STR
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=model)

    train_names, valid_names = train_test_split(IMAGE_NAMES, shuffle=True, test_size=0.2)

    for epoch in range(TR_CONFIG["epochs"]):

        tr_metrics = {
            "loss" : [],
            "iou" : [],
            "f1" : []
        }

        for batch_X, batch_y in image_batch_generator(
                                train_names, 
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"], 
                                aug_transform=train_augmentation
                            ):

            with tf.GradientTape() as tape:

                logits = model(batch_X, training=True)
                logits = tf.squeeze(logits)

                loss_value = loss_fn(batch_y, logits)
            
            tr_metrics["loss"].append(loss_value)
            tr_metrics["iou"].append(np.mean([iou(pr, gt) for pr, gt in zip(logits, batch_y)]))
            tr_metrics["f1"].append(np.mean([f1_score(pr, gt) for pr, gt in zip(logits, batch_y)]))

            grads = tape.gradient(loss_value, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))

            print("training ~ loss {:.4f}, IOU {:.4f}, f1 score {:.4f}.".format(
                np.mean(tr_metrics["loss"]), np.mean(tr_metrics["iou"]), tr_metrics["f1"]
            ), end="\r")
        
        print()

        val_metrics = {
            "loss" : [],
            "iou" : [],
            "f1" : []
        }

        for batch_X, batch_y in image_batch_generator(
                                valid_names, 
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"], 
                                aug_transform=train_augmentation
                            ):

            logits = model(batch_X, training=False)
            logits = tf.squeeze(logits)

            loss_value = loss_fn(batch_y, logits)
            
            val_metrics["loss"].append(loss_value)
            val_metrics["iou"].append(np.mean([iou(pr, gt) for pr, gt in zip(logits, batch_y)]))
            val_metrics["f1"].append(np.mean([f1_score(pr, gt) for pr, gt in zip(logits, batch_y)]))

            print("validation ~ loss {:.4f}, IOU {:.4f}, f1 score {:.4f}.".format(
                np.mean(val_metrics["loss"]), np.mean(val_metrics["iou"]), val_metrics["f1"]
            ), end="\r")
        
        print()

        print("Writing tensorboard logs...")
        with tf_writers["train"].as_default():
            tf.summary.scalar("loss", np.mean(tr_metrics["loss"]), step=epoch)
            tf.summary.scalar("IOU", np.mean(tr_metrics["iou"]), step=epoch)
            tf.summary.scalar("f1 score", np.mean(tr_metrics["f1"]), step=epoch)

        with tf_writers["valid"].as_default():
            tf.summary.scalar("loss", np.mean(val_metrics['loss']), step=epoch)
            tf.summary.scalar("IOU", np.mean(val_metrics['iou']), step=epoch)
            tf.summary.scalar("f1 score", np.mean(val_metrics['f1']), step=epoch)

        print("Saving checkpoints...")
        path = checkpoint.save(file_prefix=checkpoint_prefix)
        print("Metrics for epoch {}".format(epoch))
        print("Train loss : {:.4f}, Iou : {:.4f}, f1 score {:.4f}".format(np.mean(tr_metrics["loss"]), np.mean(tr_metrics["iou"]), np.mean(tr_metrics["f1"])))
        print("Valid loss : {:.4f}, Iou : {:.4f}, f1 score {:.4f}".format(np.mean(val_metrics['loss']), np.mean(val_metrics['iou']), np.mean(val_metrics['f1'])))


if __name__ == "__main__":
    train()