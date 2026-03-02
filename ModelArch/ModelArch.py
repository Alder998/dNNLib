"""Class to create the Model Architecture according user params"""
import tensorflow as tf
from tensorflow.keras import backend as K
from ModelArch.CustomLayers import GraphConv as gnn
from ModelArch.CustomLayers import GraphGRU as gru
from ModelArch.CustomLayers import TimeSpaceReshape as tsrh
from ModelArch.CustomLayers import TimeSpaceRestore as tsrt
from ModelArch.CustomLayers import SeasonalGatedConv1D as pc
from ModelArch.CustomLayers import MultiSeasonConv1D as mpc
from ModelArch.CustomLosses import PeakAwareMSEWithSlope as pal

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
    def createLSTMLayer(self, model=None, modelBuilder=None, mode="sequential"):

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

    # Function to create a GraphConv Layer with TensorFlowService
    def createGraphConvLayer(self, model=None, modelBuilder=None, mode="sequential", adjacency_matrix=None):

        # 1. Iterate for the GConv layers specified by the user
        if 'GConv' in self.modelStructure.keys():
            for l in range(len(self.modelStructure['GConv']["layers"])):
                # 1.1. extract the LSTM units and the nodes for each one of the layer
                unitsGConv = self.modelStructure['GConv']["layers"][l]
                layerGConv = gnn.GraphConv(unitsGConv,
                                           adjacency_matrix,
                                           self.modelStructure['GConv']["activation"])
                # 1.3. Finally, include the time distributed into the created layer
                def time_distributed(layer, x):
                    if len(x.shape) == 4:
                            return tf.keras.layers.TimeDistributed(layer)(x)
                    return layer(x)

                if mode == "functional":
                    modelBuilder = time_distributed(layerGConv, modelBuilder)

                elif mode == "sequential":
                    model.add(layerGConv)
                else:
                    raise Exception("Mode " + str(mode) + " not recognised!")

        # 2. Add option for Bidirectional Layer

        if mode == "functional":
            return modelBuilder
        elif mode == "sequential":
            return model
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Function to create a GraphConv Layer with TensorFlowService
    def createGraphGRULayer(self, model=None, modelBuilder=None, mode="sequential", adjacency_matrix=None):

        # 1. Iterate for the GConv layers specified by the user
        if 'GraphGRU' in self.modelStructure.keys():
            for l in range(len(self.modelStructure['GraphGRU']["layers"])):
                # 1.1. extract the LSTM units and the nodes for each one of the layer
                unitsGRU = self.modelStructure['GraphGRU']["layers"][l]
                layerGRU = gru.GraphGRU(unitsGRU,
                                        adjacency_matrix,
                                        self.modelStructure['GraphGRU']["activation"])
                if mode == "functional":
                    modelBuilder = layerGRU(modelBuilder)

                elif mode == "sequential":
                    model.add(layerGRU)
                else:
                    raise Exception("Mode " + str(mode) + " not recognised!")

        # 2. Add option for Bidirectional Layer

        if mode == "functional":
            return modelBuilder
        elif mode == "sequential":
            return model
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Function to create a Conv1D layer gated Layer with TensorFlowService
    def createConv1DGatedLayer (self, model=None, modelBuilder=None, mode="sequential"):

        # 1. Iterate for the GConv layers specified by the user
        if 'Conv1DGated' in self.modelStructure.keys():
            for l in range(len(self.modelStructure['Conv1DGated']["layers"])):

                unitsPC = self.modelStructure['Conv1DGated']["layers"][l]

                conv_layer = pc.SeasonalGatedConv1D(
                    units=unitsPC,
                    kernel_size=self.modelStructure['Conv1DGated']["kernel_size"]
                )

                if mode == "functional":
                    modelBuilder = conv_layer(modelBuilder)

                elif mode == "sequential":
                    model.add(conv_layer)

        if mode == "functional":
            return modelBuilder
        elif mode == "sequential":
            return model
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Function to add a multi season Conv1D layer
    def createMultiSeasonConv1DGatedLayer (self, model=None, modelBuilder=None, mode="sequential"):

        # 1. Iterate for the GConv layers specified by the user
        if 'MultiSeasonConv1DGated' in self.modelStructure.keys():
            for l in range(len(self.modelStructure['MultiSeasonConv1DGated']["layers"])):

                unitsMPC = self.modelStructure['MultiSeasonConv1DGated']["layers"][l]

                sconv_layer = mpc.MultiSeasonalGatedConv1D(
                    units_per_cycle=unitsMPC,
                    cycles=self.modelStructure['MultiSeasonConv1DGated']["cycles"],
                    mix_units=self.modelStructure['MultiSeasonConv1DGated']["mix_units"],
                    use_layer_norm=self.modelStructure['MultiSeasonConv1DGated']["use_layer_norm"],
                    baseline_kernel=self.modelStructure['MultiSeasonConv1DGated']["baseline_kernel"],
                    use_seasonal_memory=self.modelStructure['MultiSeasonConv1DGated']["use_seasonal_memory"],
                    use_cross_cycle_attention=self.modelStructure['MultiSeasonConv1DGated']["use_cross_cycle_attention"],
                )

                if mode == "functional":
                    modelBuilder = sconv_layer(modelBuilder)

                elif mode == "sequential":
                    model.add(sconv_layer)

        if mode == "functional":
            return modelBuilder
        elif mode == "sequential":
            return model
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Generalized method to create a Model with custom layers
    def createModelArchitecture(self, dropout_FF=None, mode="sequential", adjacency_matrix=None, input_shape=(None, None, None, None)):

        # 0. Initialize tf model object
        if mode=="sequential":
            # 0. Initialize tf model object
            model = tf.keras.Sequential()

            # 1. Add the recurrent layer, if required by the user
            if "LSTM" in self.modelStructure.keys():
                self.createLSTMLayer(model)

            # 2. Add the FF layer, if required by the user
            if "FF" in self.modelStructure.keys():
                self.createFeedForwardLayer(model, dropout_FF=dropout_FF)

            return model, None

        # Functional API implementation
        elif mode == "functional":
            inputs = tf.keras.Input(shape=input_shape)
            modelBuilder = inputs

            previous=""
            if "GConv" in self.modelStructure.keys():
                modelBuilder = self.createGraphConvLayer(modelBuilder=modelBuilder, mode=mode, adjacency_matrix=adjacency_matrix)
                previous="GConv"

            if 'GraphGRU' in self.modelStructure.keys():
                modelBuilder = self.createGraphGRULayer(modelBuilder=modelBuilder, mode=mode, adjacency_matrix=adjacency_matrix)
                previous="GraphGRU"

            if "Conv1DGated" in self.modelStructure.keys():
                modelBuilder = self.createConv1DGatedLayer(modelBuilder=modelBuilder, mode=mode)
                previous="Conv1DGated"

            if "MultiSeasonConv1DGated" in self.modelStructure.keys():

                if previous == "GConv":
                    # Reshape after Gconv
                    modelBuilder = tsrh.TimeSpaceReshape()(modelBuilder)

                modelBuilder = self.createMultiSeasonConv1DGatedLayer(modelBuilder=modelBuilder, mode=mode)

                if previous == "GConv":
                    # Restore the dimensions with the custom layer to keep the space dimension
                    modelBuilder = tsrt.TimeSpaceRestore(N=input_shape[1])(modelBuilder)
                    modelBuilder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(modelBuilder)  # (B, T, N, 1)

                    if "residual_output_shape" in self.modelStructure['GConv'].keys():
                        # Create and define the spatial embedding
                        spatial_embedding = tf.keras.layers.Embedding(
                            input_dim=input_shape[1],
                            output_dim=self.modelStructure['GConv']["residual_output_shape"],
                            name="spatial_embedding"
                        )
                        # Create Range for nodes embedding
                        node_ids = tf.range(input_shape[1])
                        node_bias = spatial_embedding(node_ids)  # (N, 1)

                        # Create dimensions
                        node_bias = tf.keras.layers.Lambda(
                            lambda x: tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
                        )(node_bias)  # (1, 1, N, 1)

                        # Sum the residual to the model
                        modelBuilder = tf.keras.layers.Add()([modelBuilder, node_bias])
                #previous="MultiSeasonConv1DGated"

            if "LSTM" in self.modelStructure.keys():

                if previous == "GConv":
                    # Reshape after Gconv
                    modelBuilder = tsrh.TimeSpaceReshape()(modelBuilder)

                modelBuilder = self.createLSTMLayer(modelBuilder=modelBuilder, mode=mode)

                if previous == "GConv":
                    # Restore the dimensions with the custom layer to keep the space dimension
                    modelBuilder = tsrt.TimeSpaceRestore(N=input_shape[1])(modelBuilder)
                    modelBuilder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(modelBuilder)  # (B, T, N, 1)

                    if "residual_output_shape" in self.modelStructure['GConv'].keys():
                        # Create and define the spatial embedding
                        spatial_embedding = tf.keras.layers.Embedding(
                            input_dim=input_shape[1],
                            output_dim=self.modelStructure['GConv']["residual_output_shape"],
                            name="spatial_embedding"
                        )
                        # Create Range for nodes embedding
                        node_ids = tf.range(input_shape[1])
                        node_bias = spatial_embedding(node_ids)  # (N, 1)

                        # Create dimensions
                        node_bias = tf.keras.layers.Lambda(
                            lambda x: tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
                        )(node_bias)  # (1, 1, N, 1)

                        # Sum the residual to the model
                        modelBuilder = tf.keras.layers.Add()([modelBuilder, node_bias])

            if "FF" in self.modelStructure.keys():
                modelBuilder = self.createFeedForwardLayer(modelBuilder=modelBuilder, mode=mode)

            return modelBuilder, inputs

        else:
            raise Exception("Mode " + str(mode) + " not recognised!")

    # Super-generalized function to have a Regression Model
    def createRegressionModelArchitecture(self, mode="sequential", dropout_FF=None, adjacency_matrix=None, input_shape=(None, None, None, None), loss="MSE",
                                          peak_aware_loss_params={"alpha": 3.0, "beta":2.0, "gamma":1.0}):

        # Logging
        print("INFO - MODEL ARCHITECTURE: creating model Architecture for regression...")
        modelInfo = {}
        modelBuilder, inputs = self.createModelArchitecture(mode=mode, dropout_FF=dropout_FF, adjacency_matrix=adjacency_matrix, input_shape=input_shape)
        if mode=="sequential":
            modelBuilder.add(tf.keras.layers.Dense(1, activation='linear'))
        elif mode=="functional":
            outputs = tf.keras.layers.Dense(1, activation='linear')(modelBuilder)

            # Add the Permute layer in case of time-space prediction
            if ("GConv" in self.modelStructure.keys()) & ("LSTM" in self.modelStructure.keys()):
                modelBuilder = tf.keras.layers.Permute((1, 3, 2))(modelBuilder)  # (B, T, 1, N)

            modelBuilder = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            raise Exception("Mode " + str(mode) + " not recognised!")
        modelInfo["model"] = modelBuilder

        # Add the typical loss used for regression problems (MSE)
        if loss == "MSE":
            loss = tf.keras.losses.MeanSquaredError()
        elif loss == "peak-aware-MSE":
            loss = pal.PeakAwareMSEWithSlopeRecall(alpha=peak_aware_loss_params["alpha"],
                                                   beta=peak_aware_loss_params["beta"],
                                                   gamma=peak_aware_loss_params["gamma"],
                                                   delta=peak_aware_loss_params["delta"],
                                                   peak_threshold=peak_aware_loss_params["peak_threshold"]
                                                   )
        else:
            raise Exception("Loss " + str(loss) + " not recognised!")
        # Add the loss to the model
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

