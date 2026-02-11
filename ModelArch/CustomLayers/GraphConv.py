"""Graph Conv Layer Implementation"""

import tensorflow as tf

# Class initialization upon keras layer.Layers class
class GraphConv(tf.keras.layers.Layer):

    def __init__(self, units, adjacency_matrix, activation, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.adjacency_matrix_np = adjacency_matrix  # matrix has to be fixed with tf.constant
        self.activation = tf.keras.activations.get(activation)

    # Build function to add weights to the model
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="W"
        )

        # Adjacency Matrix needs to be a non-trainable Weight, and treated as such
        self.adjacency_matrix = self.add_weight(
            shape=self.adjacency_matrix_np.shape,
            initializer=tf.constant_initializer(self.adjacency_matrix_np),
            trainable=False,
            name="adjacency_matrix"
        )

    def call(self, X):
        # 0. X must be in shape (batch, space points, features)
        # 1. Then, you just need to multiply the two matrices, the features and the adjacency matrix (respectively: (batch, space points, features) and (space points, space points))
        # 1.1. tf.matmul acts an automatic broadcast from (N,N) to (batch,N,F)
        AX = tf.linalg.matmul(self.adjacency_matrix, X)
        # 2. Then, multiply by the weights to relate them with the feature matrix multiplied by adjacency Matrix
        AXW = tf.linalg.matmul(AX, self.W)
        # 3. add the activation function, otherwise it would remain a simple linear model
        if self.activation:
            AXW = self.activation(AXW)
        return AXW

    # This function is REQUIRED to make it run
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)



