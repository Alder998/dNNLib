"""Custom Layer to Restore the Original Dimensions after the LSTM reshape"""

import tensorflow as tf

class TimeSpaceRestore(tf.keras.layers.Layer):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def call(self, x):
        # x: (B*N, T, units)
        BN = tf.shape(x)[0]
        T  = tf.shape(x)[1]
        U  = tf.shape(x)[2]

        B = BN // self.N

        x = tf.reshape(x, (B, self.N, T, U))   # (B, N, T, U)
        x = tf.transpose(x, (0, 2, 1, 3))      # (B, T, N, U)
        return x