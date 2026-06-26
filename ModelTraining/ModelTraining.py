"""Class to train the model that has been built before, according to training params from user"""

from VectorModule import VectorModule as vector

class ModelTraining:

    def __init__(self, model):
        self.model = model

    def trainModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, validation_split,
                    standardize, batch_size, epochs, time_window, split_method="random", seasonal_splits=10, target_division=1,
                    lag_series=[], scaler="std"):

        # 0. Make pandas dataFrame array, to be used for training
        print("INFO - MODEL TRAINING: Vectorizing the data from a DataFrame format...")
        features_train, features_test, target_train, target_test, feature_scaler, target_scaler = vector.VectorModule(modelStructure=self.model["modelStructure"]).processDataFrame(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, target_division=target_division, lag_series=lag_series, scaler=scaler)
        print("INFO - MODEL TRAINING: features shape for train set: ", features_train.shape)
        modelTrainingInfo = {}
        print("INFO - MODEL TRAINING: Compilation and training...")
        # 1. Compile the model
        # 1.1. If the problem type is set as classification, use the accuracy as metrics
        if "classification" in self.model["problem"]:
            self.model["model"].compile(optimizer='adam',
                               loss=self.model["loss"],
                               metrics=["accuracy"])
        else:
            self.model["model"].compile(optimizer='adam',
                               loss=self.model["loss"],
                               metrics=["mse"])

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
        modelTrainingInfo["input_shape"] = self.model["input_shape"]
        modelTrainingInfo["loss_name"] = self.model["loss_name"]
        modelTrainingInfo["problem"] = self.model["problem"]
        modelTrainingInfo["peak_aware_loss_params"] = self.model["peak_aware_loss_params"]
        modelTrainingInfo["mode"] = self.model["mode"]
        modelTrainingInfo["adjacency_matrix"] = self.model["adjacency_matrix"]

        # 4. Return for Evaluation
        return modelTrainingInfo

    def trainGeospatialModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, validation_split,
                              standardize, batch_size, epochs, time_window, space_variables, split_method="random", seasonal_splits=10, target_division=1, lag_series=[], scaler="std"):

        # 0. Make pandas dataFrame array, to be used for training
        print("INFO - MODEL TRAINING: Vectorizing the data from a DataFrame format...")
        features_train, features_test, target_train, target_test, feature_scaler, target_scaler = vector.VectorModule(modelStructure=self.model["modelStructure"]).processDataFrame(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, timeSpace=True, space_variables=space_variables, target_division=target_division, lag_series=lag_series, scaler=scaler)
        modelTrainingInfo = {}
        print("INFO - MODEL TRAINING: Compilation and training...")
        # 1. Compile the model
        self.model["model"].compile(optimizer='adam',
                           loss=self.model["loss"],
                           metrics=['mse'])

        # 2. Train + add to the JSON for evaluation
        self.model["model"].fit(features_train.transpose(0, 1, 3, 2), target_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
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
        modelTrainingInfo["input_shape"] = self.model["input_shape"]
        modelTrainingInfo["loss_name"] = self.model["loss_name"]
        modelTrainingInfo["problem"] = self.model["problem"]
        modelTrainingInfo["peak_aware_loss_params"] = self.model["peak_aware_loss_params"]
        modelTrainingInfo["mode"] = self.model["mode"]
        modelTrainingInfo["adjacency_matrix"] = self.model["adjacency_matrix"]
        modelTrainingInfo["space_variables"] = dataInDataFrameFormat[space_variables]
        modelTrainingInfo["space_variables_list"] = space_variables
        modelTrainingInfo["target_division"] = target_division

        # 4. Return for Evaluation
        return modelTrainingInfo