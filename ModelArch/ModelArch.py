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

    # Function to add a con2D layer
    def createConv2DLayer (self, modelBuilder):

        if "Conv2D" in self.modelStructure.keys():
            for l in range(len(self.modelStructure['Conv2D']["layers"])):
                # 1.1. extract the LSTM units and the nodes for each one of the layer
                unitsConv2D = self.modelStructure['Conv2D']["layers"][l]
                layerConv2D = tf.keras.layers.Conv2D(filters=unitsConv2D,
                                                     kernel_size=self.modelStructure['Conv2D']["kernel_size"],
                                                     strides=self.modelStructure['Conv2D']["strides"],
                                                     padding=self.modelStructure['Conv2D']["padding"],
                                                     activation=self.modelStructure['Conv2D']["activation"])
                layerPooling = tf.keras.layers.MaxPooling2D(pool_size=self.modelStructure['Conv2D']["pool_size"],
                                                            padding=self.modelStructure['Conv2D']["padding"])
                layerUpsample = tf.keras.layers.UpSampling2D(size=self.modelStructure['Conv2D']["pool_size"])
                # 1.3. Finally, add the layers to the model
                modelBuilder = tf.keras.layers.TimeDistributed(layerConv2D)(modelBuilder)
                modelBuilder = tf.keras.layers.TimeDistributed(layerPooling)(modelBuilder)
                modelBuilder = tf.keras.layers.TimeDistributed(layerUpsample)(modelBuilder)

        return modelBuilder

    # Function to create a Layer to connect CNN and RNN
    def ensure_3d(self, modelBuilder):
        def reshape_fn(modelBuilder):
            s = tf.shape(modelBuilder)
            B, T, C, S = s[0], s[1], s[2], s[3]
            modelBuilder = tf.transpose(modelBuilder, [0, 3, 1, 2])
            modelBuilder = tf.reshape(modelBuilder, (B * S, T, C))
            return modelBuilder

        return tf.keras.layers.Lambda(reshape_fn)(modelBuilder)

    # Generalized method to create a Model with custom layers
    def createModelArchitecture(self, dropout_FF=None, mode="sequential"):

        # 0. Initialize tf model object
        if mode=="sequential":
            # 0. Initialize tf model object
            model = tf.keras.Sequential()

            prev_block = ""
            if "Conv2D" in self.modelStructure.keys():
                self.createConv2DLayer(model)
                prev_block = "Conv2D"

            if ("LSTM" in self.modelStructure.keys()) | ("Bidirectional" in self.modelStructure.keys()):
                if prev_block == "Conv2D":
                    model.add(self.ensure_3d())
                self.createRecurrentLayer(model)
                prev_block = "LSTM"

            if "FF" in self.modelStructure.keys():
                self.createFeedForwardLayer(model, dropout_FF=dropout_FF)

            return model

        # Functional API implementation
        elif mode == "functional":
            modelBuilder = tf.keras.Input(shape=(None, None, None))
            prev_block = ""
            if "Conv2D" in self.modelStructure.keys():
                modelBuilder = self.createConv2DLayer(modelBuilder)
                prev_block = "Conv2D"

            if ("LSTM" in self.modelStructure.keys()) | ("Bidirectional" in self.modelStructure.keys()):
                if prev_block == "Conv2D":
                    modelBuilder = self.ensure_3d(modelBuilder)
                modelBuilder = self.createRecurrentLayer(modelBuilder, mode=mode)
                prev_block = "LSTM"

            if "FF" in self.modelStructure.keys():
                modelBuilder = self.createFeedForwardLayer(modelBuilder, mode=mode)

            return modelBuilder

        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Super-generalized function to have a Regression Model
    def createRegressionModelArchitecture(self, mode="sequential", dropout_FF=None):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for regression...")
        modelInfo = {}
        modelBuilder = self.createModelArchitecture(mode=mode, dropout_FF=dropout_FF)
        if mode=="sequential":
            modelBuilder.add(tf.keras.layers.Dense(1, activation='linear'))
        elif mode=="functional":
            modelBuilder = tf.keras.layers.Dense(1, activation='linear')(modelBuilder)
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

