"""Class to create the Model Architecture according user params"""

import tensorflow as tf

class ModelArch:

    def __init__(self, modelStructure):
        self.modelStructure = modelStructure
        pass

    def createFeedForwardNN(self, model, dropout_FF=None):

        # 1. Iterate for the FF layers specified by the user
        for l in range(len(self.modelStructure['FF'])):
            # 1.1. extract the Feed-Forward units and the nodes for each one of the layer
            unitsFF = self.modelStructure['FF'][l]
            layerFF = tf.keras.layers.Dense(unitsFF, activation='relu')
            # 1.2. Add the Dropout layer for FF
            if dropout_FF is not None:
                model.add(tf.keras.layers.Dropout(dropout_FF))
            # 1.3. Finally, add the FF layer to the model
            model.add(layerFF)

        return model

    # Generalized method to create a Model with custom layers
    def createModelArchitecture(self, dropout_FF=None):

        # 0. Initialize tf model object
        model = tf.keras.Sequential()

        if self.modelStructure["FF"]:
            self.createFeedForwardNN(model, dropout_FF=dropout_FF)

        return model
