"""Implement the peak-aware MSE"""

import tensorflow as tf

class PeakAwareMSEWithSlope(tf.keras.losses.Loss):
    def __init__(self, alpha=3.0, beta=2.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _to_3d(self, y):
        # convert (B,T) -> (B,T,1)
        if y.shape.rank == 2:
            y = y[..., tf.newaxis]
        # convert (B,T,1,1) -> (B,T,1)
        if y.shape.rank == 4:
            y = tf.squeeze(y, axis=-1)
        return y

    def call(self, y_true, y_pred):
        y_true = self._to_3d(y_true)
        y_pred = self._to_3d(y_pred)

        # peak weight
        max_val = tf.reduce_max(y_true, axis=1, keepdims=True) + 1e-6
        norm = y_true / max_val
        peak_w = 1.0 + self.alpha * tf.pow(norm, self.beta)

        mse = tf.square(y_true - y_pred)

        # causal slope (aligned T)
        dy_true = y_true - tf.concat([y_true[:,0:1,:], y_true[:,:-1,:]], axis=1)
        dy_pred = y_pred - tf.concat([y_pred[:,0:1,:], y_pred[:,:-1,:]], axis=1)

        slope = tf.square(dy_true - dy_pred)

        loss = peak_w * mse + self.gamma * slope
        return tf.reduce_mean(loss)