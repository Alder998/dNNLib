"""Class to create the Model Architecture according user params"""
import tensorflow as tf
from tensorflow.keras import backend as K

class ModelArch:

    def __init__(self, modelStructure):
        self.modelStructure = modelStructure
        pass

    # Function to create and add to model FF layers
    def createFeedForwardLayer(self, model=None, modelBuilder=None, mode="sequential", dropout_FF=None):

        # 1. Iterate for the FF layers specified by the user
        for l in range(len(self.modelStructure['FF']["layers"])):
            # 1.1. extract the Feed-Forward units and the nodes for each one of the layer
            unitsFF = self.modelStructure['FF']["layers"][l]
            layerFF = tf.keras.layers.Dense(unitsFF, activation=self.modelStructure['FF']["activation"])

            if mode=="functional":
                modelBuilder = layerFF(modelBuilder)
            elif mode=="sequential":
                model.add(layerFF)
            else:
                raise Exception("Mode " + str(mode) + " not recognised!")

            # 1.2. Add the Dropout layer for FF
            if dropout_FF is not None:
                if mode == "functional":
                    modelBuilder = tf.keras.layers.Dropout(dropout_FF)(modelBuilder)
                elif mode == "sequential":
                    model.add(tf.keras.layers.Dropout(dropout_FF))
                else:
                    raise Exception("Mode " + str(mode) + " not recognised!")

        if mode == "functional":
            return modelBuilder
        elif mode == "sequential":
            return model
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Function to create and add to model LSTM layers
    def createRecurrentLayer(self, model=None, modelBuilder=None, mode="sequential"):

        # 1. Iterate for the FF layers specified by the user
        if 'LSTM' in self.modelStructure.keys():
            for l in range(len(self.modelStructure['LSTM']["layers"])):
                # 1.1. extract the LSTM units and the nodes for each one of the layer
                unitsLSTM = self.modelStructure['LSTM']["layers"][l]
                layerLSTM = tf.keras.layers.LSTM(unitsLSTM, activation=self.modelStructure['LSTM']["activation"],
                                                 return_sequences=True, dropout=self.modelStructure['LSTM']["dropout"])
                # 1.3. Finally, add the FF layer to the model

                if mode == "functional":
                    modelBuilder = layerLSTM(modelBuilder)
                elif mode == "sequential":
                    model.add(layerLSTM)
                else:
                    raise Exception("Mode " + str(mode) + " not recognised!")

        # 2. Add option for Bidirectional Layer

        if mode == "functional":
            return modelBuilder
        elif mode == "sequential":
            return model
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Generalized method to create a Model with custom layers
    def createModelArchitecture(self, dropout_FF=None, mode="sequential", features=None):

        # 0. Initialize tf model object
        if mode=="sequential":
            # 0. Initialize tf model object
            model = tf.keras.Sequential()

            if ("LSTM" in self.modelStructure.keys()) | ("Bidirectional" in self.modelStructure.keys()):
                self.createRecurrentLayer(model)

            if "FF" in self.modelStructure.keys():
                self.createFeedForwardLayer(model, dropout_FF=dropout_FF)

            return model, None

        # Functional API implementation
        elif mode == "functional":
            inputs = tf.keras.Input(shape=(None, 716, features))
            modelBuilder = inputs

            if ("LSTM" in self.modelStructure.keys()) | ("Bidirectional" in self.modelStructure.keys()):
                modelBuilder = self.createRecurrentLayer(modelBuilder=modelBuilder, mode=mode)

            if "FF" in self.modelStructure.keys():
                modelBuilder = self.createFeedForwardLayer(modelBuilder=modelBuilder, mode=mode)

            return modelBuilder, inputs

        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Super-generalized function to have a Regression Model
    def createRegressionModelArchitecture(self, mode="sequential", dropout_FF=None, features=None):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for regression...")
        modelInfo = {}
        modelBuilder, inputs = self.createModelArchitecture(mode=mode, dropout_FF=dropout_FF)
        if mode=="sequential":
            modelBuilder.add(tf.keras.layers.Dense(1, activation='linear'))
        elif mode=="functional":
            outputs = tf.keras.layers.Dense(1, activation='linear')(modelBuilder)
            modelBuilder = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")
        modelInfo["model"] = modelBuilder

        # Add the typical loss used for regression problems (MSE)
        loss = tf.keras.losses.MeanSquaredError()
        modelInfo["loss"] = loss

        # Add the structure (functional to vectorize the dataset)
        modelInfo["modelStructure"] = self.modelStructure

        return modelInfo

    # Super-generalized function to have a Classification Model
    def create2ClassificationModelArchitecture(self, dropout_FF=None, mode="sequential"):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for 2-class classification...")
        modelInfo = {}
        model = self.createModelArchitecture(dropout_FF=dropout_FF, mode=mode)
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

