"""Graph Conv Layer Implementation"""

import tensorflow as tf

class GraphConvLayer (tf.keras.layers.Layer):

    def __init__(self, adjacency_matrix, units, **kwargs):
        super().__init__()
        self.adjacency_matrix = tf.constant(adjacency_matrix, dtype=tf.float32)
        self.W = self.add_weight(...)

    def call(self, X):
        X = tf.matmul(self.adjacency_matrix, X)
        return tf.matmul(X, self.W)


