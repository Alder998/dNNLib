"""Class to handle graphGRU layers, to manage space while keeping time dynamics"""

import tensorflow as tf

class graphGRU(tf.keras.layers.Layer):

    def __init__(self, units, adjacency_matrix, activation, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.A = tf.constant(adjacency_matrix, dtype=tf.float32)
        self.activation = tf.keras.activations.get(activation)