import tensorflow as tf
import numpy as np
import cv2

from models import load_unet_model
from utils import convert_to_inf_samples, preprocess_raw_output
from pdf2image import convert_from_path

INFERENCE_CONFIG = {
    "shape" : (512, 512),
    "band_size" : 2,
}

model = load_unet_model(INFERENCE_CONFIG["shape"], INFERENCE_CONFIG["band_size"])
model.training = False

checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
# print(f"loading checkpoint {'training_checkpoints/' + '2022.07.30-22/' + 'ckpt-238'}")
status = checkpoint.restore("../checkpoints/ckpt-624")

print("Reading and converting pdf pages to images...")
images = convert_from_path("sample.pdf", dpi=200, grayscale=True)
# print(images))

print("Normalizing and preparing images for input to the net...")
# print([np.array(im).shape for im in images])
normalized_images = convert_to_inf_samples([np.array(im) for im in images], INFERENCE_CONFIG["shape"])

print("Predicing...")
for i, inp in enumerate(normalized_images):
    print(np.min(inp), np.max(inp), np.mean(inp))
    # print(inp.shape)
    raw_out = tf.squeeze(model(tf.expand_dims(inp, 0), training=False))

    pred1 = preprocess_raw_output(raw_out, 2, 3000)
    pred2 = preprocess_raw_output(pred1, 2, 0, max_seg_dist=100)
    # print(pred2.shape, inp.shape)
    # print(out.shape)
    pred_img = cv2.hconcat([
        np.array(inp[:, :, 0] * 255, dtype=np.uint8),
        np.array(pred2 * 255, dtype=np.uint8),
        np.array(pred2 * inp[:, :, 0] * 255, dtype=np.uint8)
    ])
    cv2.imwrite(f"masks/Page_{i}.png", pred_img)
    print(f"Saved Page_{i}.png.")
# print(out)