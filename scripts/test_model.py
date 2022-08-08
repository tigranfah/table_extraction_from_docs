import tensorflow as tf
import numpy as np
import time

from models import TableNet, load_unet_model, att_unet

with tf.device("CPU:0"):
    model = load_unet_model((512, 512), 2, weight_scale=3)
    # model = TableNet.build((1024, 1024, 3))
    # model = att_unet(512, 512, 3, n_label=1, depth=4, features=8, data_format="channels_last")
    model.summary()
    start = time.time()
    print(model(tf.convert_to_tensor(np.random.random((1, 512, 512, 2))), training=False).shape)
    print(time.time() - start)