"""Class to create the Model Architecture according user params"""
import tensorflow as tf

class ModelArch:

    def __init__(self, modelStructure):
        self.modelStructure = modelStructure
        pass

    # Function to create and add to model FF layers
    def createFeedForwardLayer(self, model, dropout_FF=None):

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

    # Function to create and add to model LSTM layers
    def createRecurrentLayer(self, model):

        # 1. Iterate for the FF layers specified by the user
        for l in range(len(self.modelStructure['LSTM'])):
            # 1.1. extract the LSTM units and the nodes for each one of the layer
            unitsLSTM = self.modelStructure['LSTM'][l]
            layerLSTM = tf.keras.layers.LSTM(unitsLSTM, activation='tanh', return_sequences=True)
            # 1.3. Finally, add the FF layer to the model
            model.add(layerLSTM)

        return model


    # Generalized method to create a Model with custom layers
    def createModelArchitecture(self, dropout_FF=None):

        # 0. Initialize tf model object
        model = tf.keras.Sequential()

        if "FF" in self.modelStructure.keys():
            self.createFeedForwardLayer(model, dropout_FF=dropout_FF)
        if "LSTM" in self.modelStructure.keys():
            self.createRecurrentLayer(model)

        return model

    # Super-generalized function to have a Regression Model
    def createRegressionModelArchitecture(self, dropout_FF=None):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for regression...")
        modelInfo = {}
        model = self.createModelArchitecture(dropout_FF=dropout_FF)
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        modelInfo["model"] = model

        # Add the typical loss used for regression problems (MSE)
        loss = tf.keras.losses.MeanSquaredError()
        modelInfo["loss"] = loss

        # Add the structure (functional to vectorize the dataset)
        modelInfo["modelStructure"] = self.modelStructure

        return modelInfo

    # Super-generalized function to have a Classification Model
    def create2ClassificationModelArchitecture(self, dropout_FF=None):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for 2-class classification...")
        modelInfo = {}
        model = self.createModelArchitecture(dropout_FF=dropout_FF)
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        modelInfo["model"] = model

        # Add the typical loss used for 2-class problems (binary cross-loss entropy)
        loss = tf.keras.losses.BinaryCrossentropy()
        modelInfo["loss"] = loss

        # Add the structure (functional to vectorize the dataset)
        modelInfo["modelStructure"] = self.modelStructure

        return modelInfo

    # Super-generalized function to have a multi-Classification Model
    def createMultiClassificationModelArchitecture(self, classes, dropout_FF=None):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for multi-class classification...")
        modelInfo = {}
        model = self.createModelArchitecture(dropout_FF=dropout_FF)
        model.add(tf.keras.layers.Dense(units=classes, activation='softmax'))
        modelInfo["model"] = model

        # Add the typical loss used for multi-class problems (sparse categorical loss entropy)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        modelInfo["loss"] = loss

        # Add the structure (functional to vectorize the dataset)
        modelInfo["modelStructure"] = self.modelStructure

        return modelInfo

