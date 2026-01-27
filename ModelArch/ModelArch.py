"""Class to create the Model Architecture according user params"""
import tensorflow as tf
from tensorflow.keras import backend as K

class ModelArch:

    def __init__(self, modelStructure):
        self.modelStructure = modelStructure
        pass

    # Function to create and add to model FF layers
    def createFeedForwardLayer(self, model, dropout_FF=None):

        # 1. Iterate for the FF layers specified by the user
        for l in range(len(self.modelStructure['FF']["layers"])):
            # 1.1. extract the Feed-Forward units and the nodes for each one of the layer
            unitsFF = self.modelStructure['FF']["layers"][l]
            layerFF = tf.keras.layers.Dense(unitsFF, activation=self.modelStructure['FF']["activation"])
            # 1.2. Add the Dropout layer for FF
            if dropout_FF is not None:
                model.add(tf.keras.layers.Dropout(dropout_FF))
            # 1.3. Finally, add the FF layer to the model
            model.add(layerFF)

        return model

    # Function to create and add to model LSTM layers
    def createRecurrentLayer(self, model):

        # 1. Iterate for the FF layers specified by the user
        if 'LSTM' in self.modelStructure.keys():
            for l in range(len(self.modelStructure['LSTM']["layers"])):
                # 1.1. extract the LSTM units and the nodes for each one of the layer
                unitsLSTM = self.modelStructure['LSTM']["layers"][l]
                layerLSTM = tf.keras.layers.LSTM(unitsLSTM, activation=self.modelStructure['LSTM']["activation"],
                                                 return_sequences=True, dropout=self.modelStructure['LSTM']["dropout"])
                # 1.3. Finally, add the FF layer to the model
                model.add(layerLSTM)

        # 2. Add option for Bidirectional Layer

        return model

    # Function to add a con2D layer
    def createConv2DLayer (self, model):

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
                model.add(layerConv2D)
                model.add(layerPooling)
                model.add(layerUpsample)

        return model

    # Function to create a Layer to connect CNN and RNN
    def ensure_3d (self):
        def reshape_fn(x):
            s = K.int_shape(x)
            # s = (batch, time, d1, d2, ..., dn)

            time_dim = s[1]
            feature_dim = 1
            for d in s[2:]:
                feature_dim *= d

            return tf.reshape(x, (-1, time_dim, feature_dim))

        return tf.keras.layers.Lambda(reshape_fn)

    # Generalized method to create a Model with custom layers
    def createModelArchitecture(self, dropout_FF=None):

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

