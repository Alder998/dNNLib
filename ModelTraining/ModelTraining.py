"""Class to train the model that has been built before, according to training params from user"""

import tensorflow as tf

class ModelTraining:

    def __init__(self, model):
        self.model = model

    def trainModel (self, train_set, train_labels, epochs, batch_size, validation_split):

        # 0. Compile the model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mse'])

        # 1. Train
        self.model.fit(train_set, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        # 2. Return for Evaluation
        return self.model