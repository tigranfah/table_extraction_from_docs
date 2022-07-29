import tensorflow as tf

from models import TableNet, load_unet_model

with tf.device("CPU:0"):
    model = load_unet_model((512, 512), 3)

    print(model.summary())