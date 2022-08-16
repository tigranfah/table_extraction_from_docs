import tensorflow as tf
import numpy as np
import cv2

from models import load_unet_model

INFERENCE_CONFIG = {
    "shape" : (512, 512),
    "band_size" : 2,
}

model = load_unet_model(INFERENCE_CONFIG["shape"], INFERENCE_CONFIG["band_size"])
model.training = False

checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
# print(f"loading checkpoint {'training_checkpoints/' + '2022.07.30-22/' + 'ckpt-238'}")
status = checkpoint.restore("../checkpoints/ckpt-643")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('table_detector.tflite', 'wb') as f:
    f.write(tflite_model)