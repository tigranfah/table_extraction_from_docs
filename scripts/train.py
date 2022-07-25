from random import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

from utils import train_test_split, image_batch_generator, get_train_augmentation, random_batch_generator
from utils import DATASET_PATH, DS_IMAGES, DS_MASKS, SaveValidSamplesCallback
from metrics import iou, f1_score, jaccard_distance
from vis import anshow, imshow
from models import TableNet, load_unet_model

IMAGE_NAMES = os.listdir(DS_IMAGES)

# SCRIPTS_PATH = "/content/gdrive/MyDrive/table_extraction_dataset/table_extractor/scripts/"

TR_CONFIG = {
    "epochs" : 100,
    "batch_size" : 8,
    "lr" : 10e-5,
    "input_shape" : (512, 512),
    "band_size" : 2
}

def train():

    # model = TableNet.build(inputShape=(TR_CONFIG["input_shape"][0], TR_CONFIG["input_shape"][1], TR_CONFIG["band_size"]))
    model = load_unet_model(TR_CONFIG["input_shape"], TR_CONFIG["band_size"], weight_decay=0.05)
    optim = tf.keras.optimizers.Adam(learning_rate=TR_CONFIG["lr"])
    loss_fn = jaccard_distance
    # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_augmentation = get_train_augmentation()

    DATE_STR = str(datetime.now().strftime("%Y.%m.%d-%H"))
    LOG_DIR = "training_logs/" + DATE_STR

    # setup checkpoints
    checkpoint_directory = "training_checkpoints/" + DATE_STR
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        checkpoint_prefix, 
                                        save_weights_only=True, 
                                        save_best_only=True
                                    )

    tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)

    train_names, valid_names = train_test_split(IMAGE_NAMES, shuffle=True, random_state=2022, test_size=0.2)

    model.load_weights("training_checkpoints/2022.07.25-17/ckpt")
    print("successfully loaded checkpoints.")

    save_val_samples_callback = SaveValidSamplesCallback(
                                        model, 
                                        train_names[:10], 
                                        valid_names[:10], 
                                        TR_CONFIG["input_shape"],
                                        date_str=DATE_STR
                                    )

    train_batch_generator = image_batch_generator(
                                train_names, 
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"], 
                                aug_transform=train_augmentation,
                                normalize=True, include_edges_as_band=True
                            )

    # train_batch_generator = random_batch_generator(
    #                             TR_CONFIG["batch_size"], 
    #                             TR_CONFIG["input_shape"],
    #                             aug_transform=train_augmentation,
    #                             normalize=True
    #                         )

    valid_batch_generator = image_batch_generator(
                                valid_names, 
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"],
                                aug_transform=None,
                                normalize=True, include_edges_as_band=True
                            )

    # s = 0;
    # for X, y in train_batch_generator:
    #     s += 1
    #     print(f"train {s}.", end='\r')

    # print("train total - ", s)
    # print("expected total - ", len(train_names) // TR_CONFIG["batch_size"])

    # s = 0;
    # for X, y in valid_batch_generator:
    #     s += 1
    #     print(f"valid {s}.", end='\r')

    # print("valid total - ", s)
    # print("expected total - ", len(valid_names) // TR_CONFIG["batch_size"])

    model.compile(
        optimizer=optim,
        loss=loss_fn,
        metrics=[iou, f1_score]
    )

    model.fit(
        train_batch_generator,
        steps_per_epoch=((len(train_names)) // TR_CONFIG["batch_size"]) + 1,
        # steps_per_epoch=5,
        validation_data=valid_batch_generator,
        validation_steps=(len(valid_names) // TR_CONFIG["batch_size"]) + 1,
        # validation_steps=5,
        batch_size=TR_CONFIG["batch_size"],
        epochs=TR_CONFIG["epochs"],
        callbacks=[model_checkpoint, tb_callback, save_val_samples_callback]
    )

    # for epoch in range(TR_CONFIG["epochs"]):

    #     tr_metrics = {
    #         "loss" : [],
    #         "iou" : [],
    #         "f1" : []
    #     }

    #     NUM_SAMPLES = 0

    #     for batch_X, batch_y in image_batch_generator(
    #                             train_names, 
    #                             batch_size=TR_CONFIG["batch_size"], 
    #                             resize_shape=TR_CONFIG["input_shape"], 
    #                             aug_transform=train_augmentation
    #                         ):

    #         batch_X, batch_y = tf.convert_to_tensor(batch_X, dtype=tf.float32), tf.convert_to_tensor(batch_y, dtype=tf.float32)
            
    #         NUM_SAMPLES += len(batch_X)

    #         with tf.GradientTape() as tape:
                
    #             logits = model(batch_X, training=True)
    #             print("predicted on ", batch_X.shape)
    #             logits = tf.squeeze(logits)

    #             loss_value = loss_fn(batch_y, logits)
            
    #         tr_metrics["loss"].append(loss_value)
    #         tr_metrics["iou"].append(np.mean([iou(pr, gt) for pr, gt in zip(logits, batch_y)]))
    #         tr_metrics["f1"].append(np.mean([f1_score(pr, gt) for pr, gt in zip(logits, batch_y)]))

    #         grads = tape.gradient(loss_value, model.trainable_weights)
    #         optim.apply_gradients(zip(grads, model.trainable_weights))

    #         print("training {}/{} ~ loss {:.4f}, IOU {:.4f}, f1 score {:.4f}.".format(
    #             NUM_SAMPLES, len(train_names),
    #             np.mean(tr_metrics["loss"]), np.mean(tr_metrics["iou"]), np.mean(tr_metrics["f1"])
    #         ), end="\r")
        
    #     print()

    #     val_metrics = {
    #         "loss" : [],
    #         "iou" : [],
    #         "f1" : []
    #     }

    #     NUM_SAMPLES = 0

    #     for batch_X, batch_y in image_batch_generator(
    #                             valid_names, 
    #                             batch_size=TR_CONFIG["batch_size"], 
    #                             resize_shape=TR_CONFIG["input_shape"]
    #                         ):

    #         batch_X, batch_y = tf.convert_to_tensor(batch_X, dtype=tf.float32), tf.convert_to_tensor(batch_y, dtype=tf.float32)

    #         NUM_SAMPLES += len(batch_X)

    #         logits = model(batch_X, training=False)
    #         logits = tf.squeeze(logits)

    #         loss_value = loss_fn(batch_y, logits)
            
    #         val_metrics["loss"].append(loss_value)
    #         val_metrics["iou"].append(np.mean([iou(pr, gt) for pr, gt in zip(logits, batch_y)]))
    #         val_metrics["f1"].append(np.mean([f1_score(pr, gt) for pr, gt in zip(logits, batch_y)]))

    #         print("validation {}/{} ~ loss {:.4f}, IOU {:.4f}, f1 score {:.4f}.".format(
    #             NUM_SAMPLES, len(valid_names),
    #             np.mean(val_metrics["loss"]), np.mean(val_metrics["iou"]), np.mean(val_metrics["f1"])
    #         ), end="\r")
        
    #     print()

    #     print("Writing tensorboard logs...")
    #     with tf_writers["train"].as_default():
    #         tf.summary.scalar("loss", np.mean(tr_metrics["loss"]), step=epoch)
    #         tf.summary.scalar("IOU", np.mean(tr_metrics["iou"]), step=epoch)
    #         tf.summary.scalar("f1 score", np.mean(tr_metrics["f1"]), step=epoch)

    #     with tf_writers["valid"].as_default():
    #         tf.summary.scalar("loss", np.mean(val_metrics['loss']), step=epoch)
    #         tf.summary.scalar("IOU", np.mean(val_metrics['iou']), step=epoch)
    #         tf.summary.scalar("f1 score", np.mean(val_metrics['f1']), step=epoch)

    #     print("Saving checkpoints...")
    #     path = checkpoint.save(file_prefix=checkpoint_prefix)
    #     print("Metrics for epoch {}".format(epoch))
    #     print("Train loss : {:.4f}, Iou : {:.4f}, f1 score {:.4f}".format(np.mean(tr_metrics["loss"]), np.mean(tr_metrics["iou"]), np.mean(tr_metrics["f1"])))
    #     print("Valid loss : {:.4f}, Iou : {:.4f}, f1 score {:.4f}".format(np.mean(val_metrics['loss']), np.mean(val_metrics['iou']), np.mean(val_metrics['f1'])))


if __name__ == "__main__":
    train()