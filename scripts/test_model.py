import tensorflow as tf

y = [1, 2]
# y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
y_pred = [0.1, 0.3]
cs = tf.keras.losses.BinaryCrossentropy(from_logits=False)
print(cs(y, y_pred))
# from models import TableNet

# with tf.device("CPU:0"):
#     model = TableNet().build()

#     print(model.summary())