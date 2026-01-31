"""Class to train the model that has been built before, according to training params from user"""

import tensorflow as tf
from VectorModule import VectorModule as vector

class ModelTraining:

    def __init__(self, model):
        self.model = model

    def trainModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, validation_split, standardize, batch_size, epochs, time_window, split_method="random", seasonal_splits=10):

        # 0. Make pandas dataFrame array, to be used for training
        print("INFO - MODEL TRAINING: Vectorizing the data from a DataFrame format...")
        features_train, features_test, target_train, target_test, feature_scaler, target_scaler = vector.VectorModule(modelStructure=self.model["modelStructure"]).processDataFrame(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits)
        modelTrainingInfo = {}
        print("INFO - MODEL TRAINING: Compilation and training...")
        # 1. Compile the model
        self.model["model"].compile(optimizer='adam',
                           loss=self.model["loss"],
                           metrics=['mse'])

        # 2. Train + add to the JSON for evaluation
        self.model["model"].fit(features_train, target_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        modelTrainingInfo["model"] = self.model["model"]

        # 3. Add all the prediction needed features to the model
        modelTrainingInfo["test_set"] = features_test
        modelTrainingInfo["test_labels"] = target_test
        modelTrainingInfo["time_window"] = time_window
        modelTrainingInfo["var_to_predict"] = target_variables
        modelTrainingInfo["params"] = feature_variables
        modelTrainingInfo["modelStructure"] = self.model["modelStructure"]
        modelTrainingInfo["feature_scaler"] = feature_scaler
        modelTrainingInfo["target_scaler"] = target_scaler

        # 4. Return for Evaluation
        return modelTrainingInfo

    def trainGeospatialModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, validation_split,
                              standardize, batch_size, epochs, time_window, space_variables, split_method="random", seasonal_splits=10):

        # 0. Make pandas dataFrame array, to be used for training
        print("INFO - MODEL TRAINING: Vectorizing the data from a DataFrame format...")
        features_train, features_test, target_train, target_test, feature_scaler, target_scaler = vector.VectorModule(modelStructure=self.model["modelStructure"]).processDataFrame(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, timeSpace=True, space_variables=space_variables)
        modelTrainingInfo = {}
        print("INFO - MODEL TRAINING: Compilation and training...")
        # 1. Compile the model
        self.model["model"].compile(optimizer='adam',
                           loss=self.model["loss"],
                           metrics=['mse'])

        # 2. Train + add to the JSON for evaluation
        self.model["model"].fit(features_train.transpose(1, 0, 2, 3), target_train.transpose(1, 0, 2), epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        modelTrainingInfo["model"] = self.model["model"]

        # 3. Add all the prediction needed features to the model
        modelTrainingInfo["test_set"] = features_test
        modelTrainingInfo["test_labels"] = target_test
        modelTrainingInfo["time_window"] = time_window
        modelTrainingInfo["var_to_predict"] = target_variables
        modelTrainingInfo["params"] = feature_variables
        modelTrainingInfo["modelStructure"] = self.model["modelStructure"]
        modelTrainingInfo["feature_scaler"] = feature_scaler
        modelTrainingInfo["target_scaler"] = target_scaler
        modelTrainingInfo["space_variables"] = dataInDataFrameFormat[space_variables]

        # 4. Return for Evaluation
        return modelTrainingInfo