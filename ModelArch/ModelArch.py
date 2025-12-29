"""Class to create the Model Architecture according user params"""

import tensorflow as tf

class ModelArch:

    def __init__(self, model, modelStructure):
        self.model = model
        self.modelStructure = modelStructure
        pass

    def createFeedForwardNN(self, dropout_FF=None):

        # 1. Iterate for the FF layers specified by the user
        for l in range(len(self.modelStructure['FF'])):
            # 1.1. extract the Feed-Forward units and the nodes for each one of the layer
            unitsFF = self.modelStructure['FF'][l]
            layerFF = tf.keras.layers.Dense(unitsFF, activation='relu')
            # 1.2. Add the Dropout layer for FF
            if dropout_FF is not None:
                self.model.add(tf.keras.layers.Dropout(dropout_FF))
            # 1.3. Finally, add the FF layer to the model
            self.model.add(layerFF)

        return self.model
