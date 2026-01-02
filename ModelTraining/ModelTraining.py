"""Class to train the model that has been built before, according to training params from user"""

import tensorflow as tf
from VectorModule import VectorModule as vector

class ModelTraining:

    def __init__(self, model):
        self.model = model

    def trainModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, validation_split, batch_size, epochs):

        # 0. Make pandas dataFrame array, to be used for training
        print("INFO - MODEL TRAINING: Vectorizing the data from a DataFrame format...")
        features_train, features_test, target_train, target_test = vector.VectorModule(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                       modelStructure=self.model["modelStructure"]).processDataFrame(feature_variables, target_variables, test_size)
        modelTrainingInfo = {}
        print("INFO - MODEL TRAINING: Compilation and training...")
        # 1. Compile the model
        self.model["model"].compile(optimizer='adam',
                           loss=self.model["loss"],
                           metrics=['mse'])

        # 2. Train + add to the JSON for evaluation
        self.model["model"].fit(features_train, target_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        modelTrainingInfo["model"] = self.model["model"]

        # 3. Add test set and test labels to the model object
        modelTrainingInfo["test_set"] = features_test
        modelTrainingInfo["test_labels"] = target_test

        # 4. Return for Evaluation
        return modelTrainingInfo