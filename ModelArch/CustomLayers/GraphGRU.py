"""Class to handle graphGRU layers, to manage space while keeping time dynamics"""

import tensorflow as tf

class GraphGRU(tf.keras.layers.Layer):
    def __init__(self, units, adjacency_matrix, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.A = tf.constant(adjacency_matrix, dtype=tf.float32)
        self.N = adjacency_matrix.shape[0]
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        F = input_shape[-1]

        self.Wz = self.add_weight(shape=(F, self.units))
        self.Uz = self.add_weight(shape=(self.units, self.units))

        self.Wr = self.add_weight(shape=(F, self.units))
        self.Ur = self.add_weight(shape=(self.units, self.units))

        self.Wh = self.add_weight(shape=(F, self.units))
        self.Uh = self.add_weight(shape=(self.units, self.units))

    def call(self, x):
        # x: (B,T,N,F)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # initial hidden state per nodo
        h0 = tf.zeros((B, self.N, self.units), dtype=x.dtype)

        # scan expects time first
        x_t = tf.transpose(x, [1, 0, 2, 3])  # (T,B,N,F)

        def step(h, xt):
            # xt: (B,N,F)
            # h: (B,N,units)

            xt = tf.einsum("ij,bjf->bif", self.A, xt)
            h_prop = tf.einsum("ij,bjf->bif", self.A, h)

            z = tf.sigmoid(tf.matmul(xt, self.Wz) + tf.matmul(h_prop, self.Uz))
            r = tf.sigmoid(tf.matmul(xt, self.Wr) + tf.matmul(h_prop, self.Ur))

            h_tilde = self.activation(tf.matmul(xt, self.Wh) + tf.matmul(r * h_prop, self.Uh))

            h_new = (1 - z) * h + z * h_tilde
            return h_new

        h_seq = tf.scan(step, x_t, initializer=h0)  # (T,B,N,units)
        h_seq = tf.transpose(h_seq, [1, 0, 2, 3])  # (B,T,N,units)

        return h_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.units)