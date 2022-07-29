from random import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

from utils import train_test_split, image_batch_generator, get_train_augmentation, random_batch_generator, get_table_augmentation
from utils import DATASET_PATH, DS_IMAGES, DS_MASKS, SaveValidSamplesCallback
import utils
from metrics import iou, f1_score, jaccard_distance
import metrics
from vis import anshow, imshow
from models import TableNet, load_unet_model

IMAGE_NAMES = os.listdir(DS_IMAGES)

# SCRIPTS_PATH = "/content/gdrive/MyDrive/table_extraction_dataset/table_extractor/scripts/"

TR_CONFIG = {
    "epochs" : 100,
    "batch_size" : 2,
    "lr" : 10e-6,
    "input_shape" : (512, 512),
    "band_size" : 2
}


def print_progress(name, metrics, step, all_steps):
    str_prog = f"{step}/{all_steps}: "
    str_prog += "{} loss {:.4f}, IOU {:.4f}, f1 {:.4f}, prec {:.4f}, rec {:.4f}".format(
        name,
        np.mean(metrics["loss"]), 
        np.mean(metrics["iou"]), 
        np.mean(metrics["f1"]),
        np.mean(metrics["precision"]),
        np.mean(metrics["recall"])
    )

    print(str_prog, end='\r')


def train():

    # model = TableNet.build(inputShape=(TR_CONFIG["input_shape"][0], TR_CONFIG["input_shape"][1], TR_CONFIG["band_size"]))
    model = load_unet_model(TR_CONFIG["input_shape"], TR_CONFIG["band_size"], weight_decay=0.1)
    optim = tf.keras.optimizers.Adam(learning_rate=TR_CONFIG["lr"])
    # optim = tf.keras.optimizers.SGD(learning_rate=TR_CONFIG["lr"], momentum=0.7)
    loss_fn = jaccard_distance
    # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    train_names, valid_names = train_test_split(IMAGE_NAMES, shuffle=True, random_state=2022, test_size=0.2)

    DATE_STR = str(datetime.now().strftime("%Y.%m.%d-%H"))
    LOG_DIR = "training_logs/" + DATE_STR

    tf_writers = {
        "train" : tf.summary.create_file_writer(LOG_DIR + "train/"),
        "valid" : tf.summary.create_file_writer(LOG_DIR + "valid/")
    }

    # setup checkpoints
    checkpoint_directory = "training_checkpoints/" + DATE_STR
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    ### keras checkppints
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

    model.load_weights("training_checkpoints/2022.07.29-16/ckpt")
    print("successfully loaded checkpoint.")

    # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    # checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=model)
    # print(f"loading checkpoint {'../training_checkpoints/' + '2022.07.29-16/' + 'ckpt'}")
    # status = checkpoint.restore("../training_checkpoints/" + '2022.07.29-16/' + 'ckpt')

    # train_batch_generator = image_batch_generator(
    #                             train_names, 
    #                             batch_size=TR_CONFIG["batch_size"], 
    #                             resize_shape=TR_CONFIG["input_shape"], 
    #                             aug_transform=get_train_augmentation(),
    #                             normalize=True, include_edges_as_band=True
    #                         )

    train_batch_generator = random_batch_generator(
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"],
                                train_names=train_names,
                                train_aug_transform=get_train_augmentation(),
                                table_aug_transform=get_table_augmentation(), 
                                max_tables_on_image=6, normalize=True, include_edges_as_band=True
                            )

    valid_batch_generator = image_batch_generator(
                                valid_names, 
                                batch_size=TR_CONFIG["batch_size"], 
                                resize_shape=TR_CONFIG["input_shape"],
                                aug_transform=None,
                                normalize=True, include_edges_as_band=True
                            )

    for epoch in range(1, TR_CONFIG["epochs"] + 1):

        print(f"\nEpoch {epoch}/{TR_CONFIG['epochs']}")
        # print(f"Shuffling...")

        tr_metrics = {n:[] for n in ("loss", "iou", "f1", "precision", "recall")}
        val_metrics = {n:[] for n in ("loss", "iou", "f1", "precision", "recall")}

        # train loop
        for i, (batch_X, batch_y) in enumerate(train_batch_generator):

            # print(batch_X.dtype, batch_y.dtype)
            # print(batch_y.min(), batch_y.max())
            batch_X = tf.convert_to_tensor(batch_X, dtype=tf.float32)
            batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)

            # print("The size of the input: ", tf.expand_dims(image, 0).dtype, mask.dtype)
            with tf.GradientTape() as tape:
                logits = model(batch_X, training=True)
                logits = tf.squeeze(logits)

                loss_value = loss_fn(batch_y, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))
                
            (
                iou_value, f1_score_value, 
                presicion_value, 
                recall_value
            ) = metrics.calculate_metrics(logits, batch_y)
            tr_metrics["loss"].append(loss_value)
            tr_metrics["iou"].append(iou_value)
            tr_metrics["f1"].append(f1_score_value)
            tr_metrics["precision"].append(presicion_value)
            tr_metrics["recall"].append(recall_value)

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

            logits = model(batch_X, training=False)
            logits = tf.squeeze(logits)

            loss_value = loss_fn(batch_y, logits)
  
            (
                iou_value, f1_score_value, 
                presicion_value, 
                recall_value
            ) = metrics.calculate_metrics(logits, batch_y)
            val_metrics["loss"].append(loss_value)
            val_metrics["iou"].append(iou_value)
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

        utils.save_pred_samples(
            model, train_names, TR_CONFIG["input_shape"], epoch,
            "train", directory=f"./predicted_samples/{DATE_STR}"
        )
        utils.save_pred_samples(
            model, valid_names, TR_CONFIG["input_shape"], epoch,
            "valid", directory=f"./predicted_samples/{DATE_STR}"
        )


        print("Writing to the log...")
        with tf_writers["train"].as_default():
            for k, v in tr_metrics.items():
                tf.summary.scalar(k, np.mean(v), step=epoch)

        with tf_writers["valid"].as_default():
            for k, v in val_metrics.items():
                tf.summary.scalar(k, np.mean(v), step=epoch)

        path = checkpoint.save(file_prefix=checkpoint_prefix)
        print("Saved checkpoint for epoch {} to {}".format(epoch, path))

    # model.compile(
    #     optimizer=optim,
    #     loss=loss_fn,
    #     metrics=[iou, f1_score]
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