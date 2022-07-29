import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D


class TableNet:

  @staticmethod
  def build_table_decoder(inputs, pool3, pool4):
    x = Conv2D(512, (1, 1), activation = 'relu', name='conv7_table')(inputs)
    x = UpSampling2D(size=(2, 2))(x)

    concatenated = Concatenate()([x, pool4])

    # concatenated = concatenate([x, pool4])

    x = UpSampling2D(size=(2,2))(concatenated)
    
    concatenated = Concatenate()([x, pool3])

    x = UpSampling2D(size=(2,2))(concatenated)
    x = UpSampling2D(size=(2,2))(x)

    last = tf.keras.layers.Conv2DTranspose(
      3, 3, strides=2, # activation="sigmoid",
      padding='same', name='table_output') 
    
    x = last(x)

    return x

  @staticmethod
  def build_column_decoder(inputs, pool3, pool4):
    
    x = Conv2D(512, (1, 1), activation = 'relu', name='block7_conv1_column')(inputs)
    x = Dropout(0.8, name='block7_dropout_column')(x)

    x = Conv2D(512, (1, 1), activation = 'relu', name='block8_conv1_column')(x)
    x = UpSampling2D(size=(2, 2))(x)

    concatenated = Concatenate()([x, pool4])

    # concatenated = concatenate([x, pool4])

    x = UpSampling2D(size=(2,2))(concatenated)
    
    concatenated = Concatenate()([x, pool3])

    x = UpSampling2D(size=(2,2))(concatenated)
    x = UpSampling2D(size=(2,2))(x)

    last = tf.keras.layers.Conv2DTranspose(
      3, 3, strides=2,
      padding='same', name='column_output') 
    
    x = last(x)

    return x  

  @staticmethod
  def vgg_base(inputs, input_shape):
    base_model = tf.keras.applications.vgg19.VGG19(
        input_shape=input_shape,
        include_top=False, weights='imagenet')
    
    layer_names = ['block3_pool', 'block4_pool', 'block5_pool']
    layers = [base_model.get_layer(name).output for name in layer_names]

    pool_layers_model = Model(inputs=base_model.input, outputs=layers, name='VGG-19')
    pool_layers_model.trainable = False

    return pool_layers_model(inputs)
  
  @staticmethod
  def build(inputShape=(256, 256, 3)):

    inputs = Input(shape=inputShape, name='input')

    pool_layers = TableNet.vgg_base(inputs, inputShape)

    x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv1')(pool_layers[2])
    x = Dropout(0.8, name='block6_dropout1')(x)
    x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv2')(x)
    x = Dropout(0.8, name = 'block6_dropout2')(x)
    
    table_mask = TableNet.build_table_decoder(x, pool_layers[0], pool_layers[1])
    # column_mask = TableNet.build_column_decoder(x, pool_layers[0], pool_layers[1])

    model = Model(			
            inputs=inputs,
            outputs=table_mask,
            # outputs=[table_mask, column_mask],
            name="tablenet")
    
    return model


# model = TableNet.build()
# print(model.summary())

def conv_blocks(
    ip_,
    nfilters,
    axis_batch_norm,
    reg,
    name,
    batch_norm,
    remove_bias_if_batch_norm=False,
    dilation_rate=(1, 1),
):
    use_bias = not (remove_bias_if_batch_norm and batch_norm)

    conv = tf.keras.layers.SeparableConv2D(
        nfilters,
        (3, 3),
        padding="same",
        name=name + "_conv_1",
        depthwise_regularizer=reg,
        pointwise_regularizer=reg,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
    )(ip_)

    if batch_norm:
        conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm, name=name + "_bn_1")(conv)

    conv = tf.keras.layers.Activation("relu", name=name + "_act_1")(conv)

    conv = tf.keras.layers.SeparableConv2D(
        nfilters,
        (3, 3),
        padding="same",
        name=name + "_conv_2",
        use_bias=use_bias,
        dilation_rate=dilation_rate,
        depthwise_regularizer=reg,
        pointwise_regularizer=reg,
    )(conv)

    if batch_norm:
        conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm, name=name + "_bn_2")(conv)

    return tf.keras.layers.Activation("relu", name=name + "_act_2")(conv)


def build_unet_model_fun(
        x_init, weight_decay=0.05, 
        batch_norm=True, 
        final_activation="sigmoid",
        model_scale=3
    ):

    axis_batch_norm = 3

    reg = tf.keras.regularizers.l2(weight_decay)

    conv1 = conv_blocks(x_init, 32 * model_scale, axis_batch_norm, reg, name="input", batch_norm=batch_norm)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1")(conv1)

    conv2 = conv_blocks(pool1, 64 * model_scale, axis_batch_norm, reg, name="pool1", batch_norm=batch_norm)

    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2")(conv2)

    conv3 = conv_blocks(pool2, 128 * model_scale, axis_batch_norm, reg, name="pool2", batch_norm=batch_norm)

    up8 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                64 * model_scale, (2, 2), strides=(2, 2), padding="same", name="upconv1", kernel_regularizer=reg
            )(conv3),
            conv2,
        ],
        axis=axis_batch_norm,
        name="concatenate_up_1",
    )

    conv8 = conv_blocks(up8, 64 * model_scale, axis_batch_norm, reg, name="up1", batch_norm=batch_norm)

    up9 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                32 * model_scale, (2, 2), strides=(2, 2), padding="same", name="upconv2", kernel_regularizer=reg
            )(conv8),
            conv1,
        ],
        axis=axis_batch_norm,
        name="concatenate_up_2",
    )

    conv9 = conv_blocks(up9, 32 * model_scale, axis_batch_norm, reg, name="up2", batch_norm=batch_norm)

    conv10 = tf.keras.layers.Conv2D(
        1, (1, 1), kernel_regularizer=reg, name="linear_model", activation=final_activation
    )(conv9)

    return conv10


# NORM_OFF_PROBAV = np.array([0.43052389, 0.40560079, 0.46504526, 0.23876471])
# ID_KERNEL_INITIALIZER =np.eye(4)[None, None]
# c11.set_weights([ID_KERNEL_INITIALIZER, -NORM_OFF_PROBAV])


def load_unet_model(shape=(None, None), bands_input=4, weight_decay=0.0, final_activation="sigmoid"):
    ip = tf.keras.layers.Input(shape + (bands_input,), name="ip_cloud")
    c11 = tf.keras.layers.Conv2D(bands_input, (1, 1), name="normalization_cloud", trainable=False)
    x_init = c11(ip)
    conv2d10 = build_unet_model_fun(
        x_init, weight_decay=weight_decay, final_activation=final_activation, batch_norm=True
    )
    return tf.keras.models.Model(inputs=[ip], outputs=[conv2d10], name="UNet-Clouds")