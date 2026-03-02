"""Implement the peak-aware MSE"""

import tensorflow as tf

class PeakAwareMSEWithSlopeRecall(tf.keras.losses.Loss):
    def __init__(self, alpha=3.0, beta=2.0, gamma=1.0, delta=2.0, peak_threshold=0.8):
        """
        alpha, beta : peak-aware MSE weight
        gamma       : slope weight
        delta       : peak recall weight
        peak_threshold : % max to define the peak level (0.7-0.9)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.peak_threshold = peak_threshold

    def _to_3d(self, y):
        if y.shape.rank == 2:
            y = y[..., tf.newaxis]
        if y.shape.rank == 4:
            y = tf.squeeze(y, axis=-1)
        return y

    def call(self, y_true, y_pred):
        y_true = self._to_3d(y_true)
        y_pred = self._to_3d(y_pred)

        # ---------- peak-aware weight ----------
        max_val = tf.reduce_max(y_true, axis=1, keepdims=True) + 1e-6
        norm = y_true / max_val
        peak_w = 1.0 + self.alpha * tf.pow(norm, self.beta)

        mse = tf.square(y_true - y_pred)

        # ---------- slope ----------
        dy_true = y_true - tf.concat([y_true[:,0:1,:], y_true[:,:-1,:]], axis=1)
        dy_pred = y_pred - tf.concat([y_pred[:,0:1,:], y_pred[:,:-1,:]], axis=1)
        slope = tf.square(dy_true - dy_pred)

        # ---------- peak recall ----------
        peak_mask = tf.cast(y_true > self.peak_threshold * max_val, y_true.dtype)
        miss = tf.nn.relu(y_true - y_pred)
        peak_recall = peak_mask * tf.square(miss)

        # ---------- total ----------
        loss = peak_w * mse + self.gamma * slope + self.delta * peak_recall
        return tf.reduce_mean(loss)