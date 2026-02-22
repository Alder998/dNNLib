"""Class to handle Periodic Conv1D layers"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, GlobalAveragePooling1D, Dense, Multiply

class SeasonalGatedConv1D(Layer):

    def __init__(self, units, kernel_size, gate_bias=3.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.gate_bias = gate_bias
        self.conv = Conv1D(filters=units, kernel_size=kernel_size, padding="valid")
        self.pool = GlobalAveragePooling1D()
        self.gate_dense = Dense(units, activation="sigmoid",
                                bias_initializer=tf.keras.initializers.Constant(gate_bias))

    def circular_pad(self, x):
        w = tf.shape(x)[1]
        k_left = self.kernel_size // 2
        k_right = self.kernel_size - k_left - 1
        left = tf.gather(x, tf.range(w - k_left, w) % w, axis=1)
        right = tf.gather(x, tf.range(k_right) % w, axis=1)
        x_pad = tf.concat([left, x, right], axis=1)
        return x_pad

    def call(self, x):
        if x.shape.rank != 3:
            raise ValueError(f"Expected input rank 3 (batch, window, features), got {x.shape}")

        x_pad = self.circular_pad(x)
        x_conv = self.conv(x_pad)  # (batch, window, units)
        gate = self.gate_dense(x_conv)
        x_scaled = Multiply()([x_conv, gate])  # (batch, window, units)
        return x_scaled  # mantiene dimensione temporale

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, self.units)
