"""Custom Layer to reshape the data BEFORE entering the LSTM"""

import tensorflow as tf

class TimeSpaceReshape(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def call(self, x):
        # x: (B, T, N, F)
        x = tf.keras.ops.transpose(x, (0, 2, 1, 3))   # (B, N, T, F)

        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        T = tf.shape(x)[2]
        F = tf.shape(x)[3]

        x = tf.keras.ops.reshape(x, (B*N, T, F))    # (B*N, T, F)
        return x