import tensorflow as tf
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join("..", "keras_unets"))

from models import TableNet, load_unet_model, att_unet
from keras_unet_collection.models import att_unet_2d

with tf.device("CPU:0"):
    # model = load_unet_model((512, 512), 2, weight_scale=2)
    # model = TableNet.build((1024, 1024, 3))
    model = att_unet_2d((720, 720, 2), [32, 64, 128, 256], n_labels=1,
                stack_num_down=2, stack_num_up=2,
                activation='ReLU', atten_activation='ReLU', attention='add', output_activation=None, 
                batch_norm=True, pool=False, unpool='bilinear', name='attunet'
            )
    # model = att_unet(512, 512, 3, n_label=1, depth=4, features=8, data_format="channels_last")
    model.summary()
    start = time.time()
    print(model(tf.convert_to_tensor(np.random.random((1, 720, 720, 2))), training=False).shape)
    print(time.time() - start)