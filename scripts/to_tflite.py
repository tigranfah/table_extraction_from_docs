import tensorflow as tf
import sys
import os

sys.path.insert(0, os.path.join("..", "keras_unets"))

from keras_unet_collection.models import att_unet_2d

TR_CONFIG = {
    "epochs" : 100,
    "batch_size" : 8,
    # "val_batch_size" : 32,
    "lr" : 10e-5,
    "input_shape" : (512, 512),
    "band_size" : 2,
    "three_channel" : False
}

model = att_unet_2d((TR_CONFIG["input_shape"][0], TR_CONFIG["input_shape"][1], 2), [32, 64, 128, 256], n_labels=1, 
            stack_num_down=2, stack_num_up=2, 
            activation='ReLU', atten_activation='ReLU', attention='add', output_activation="Sigmoid", 
            batch_norm=True, pool=False, unpool='bilinear', name='attunet'
        )

checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
# print(os.path.exists('training_checkpoints/' + '2022.08.25-00/ckpt-203.index'))
print(f"loading checkpoint {'training_checkpoints/' + '2022.08.29-07/ckpt-192'}")
status = checkpoint.restore("training_checkpoints/" + '2022.08.29-07/ckpt-192')
status.expect_partial()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('../models/att_unet_table_detector_v7.tflite', 'wb') as f:
    f.write(tflite_model)