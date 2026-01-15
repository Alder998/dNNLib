"""Class to process the data in DataFrame format to be feed into the model """

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class VectorModule:

    def __init__(self, dataInDataFrameFormat, modelStructure):
        self.dataInDataFrameFormat = dataInDataFrameFormat
        self.modelStructure = modelStructure
        pass

    # Function to standardize data
    def standardizeData (self, dataInDataFrameFormat, feature_variables, target_variables):

        # 0. Instantiate the scaler
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        # 1. Scale the feature variables
        dataInDataFrameFormat_scaled_features = dataInDataFrameFormat.copy()
        dataInDataFrameFormat_scaled_features[feature_variables] = feature_scaler.fit_transform(dataInDataFrameFormat[feature_variables])

        # 1. Scale the feature variables
        dataInDataFrameFormat_scaled = dataInDataFrameFormat_scaled_features.copy()
        dataInDataFrameFormat_scaled[target_variables] = target_scaler.fit_transform(pd.DataFrame(dataInDataFrameFormat_scaled_features[target_variables]))

        # 2. Return the scaler + the data scaled
        return feature_scaler, target_scaler, dataInDataFrameFormat_scaled

    # Data Processing for feed forward (easiest one)
    def processDataForFF (self, feature_variables, target_variables, test_size, standardize = False, split_method="random"):

        # 0.0. Standardize the data
        if standardize:
            feature_scaler, target_scaler, self.dataInDataFrameFormat = self.standardizeData(dataInDataFrameFormat=self.dataInDataFrameFormat,
                                                                                             feature_variables=feature_variables,
                                                                                             target_variables=target_variables)
        else:
            feature_scaler = None
            target_scaler = None

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(self.dataInDataFrameFormat[target_variables])

        # 1. For the features it is needed an array of shape (index, features number)
        features_array = np.array(self.dataInDataFrameFormat[feature_variables])

        # 2. Split for train, test, validation
        if split_method == "random":
            features_train, features_test, target_train, target_test = train_test_split(features_array, target_array, test_size=test_size,
                                                                                        random_state=1893)
        # 2.1. implement a split for time-series (only first % is train, the rest is test)
        elif split_method == "time-series":
            train_index = int(features_array.shape[0] * test_size)
            features_train = features_array[0:train_index]
            features_test = features_array[train_index:]
            target_train = target_array[0:train_index]
            target_test = target_array[train_index:]
        else:
            raise Exception("The split method " + split_method + " is invalid!")

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

    # Data Processing for recurrent NN
    def processDataForRecurrentNet (self, feature_variables, target_variables, test_size, time_window, standardize=False, split_method="random"):

        # 0.0. Standardize the data
        if standardize:
            feature_scaler, target_scaler, self.dataInDataFrameFormat = self.standardizeData(dataInDataFrameFormat=self.dataInDataFrameFormat,
                                                                                             feature_variables=feature_variables,
                                                                                             target_variables=target_variables)
        else:
            feature_scaler = None
            target_scaler = None

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(self.dataInDataFrameFormat[target_variables])

        # 1. For the features it is needed an array of shape (batch, time_steps, features) while now is (time_steps, features)
        features_array = np.array(self.dataInDataFrameFormat[feature_variables])
        batch_size_LSTM = int(features_array.shape[0] / time_window)
        fabs = []
        fabst = []
        for i in range(batch_size_LSTM):
            fab = features_array[time_window*i : time_window*(i+1), :]
            fabt = target_array[time_window*i : time_window*(i+1)]
            fabs.append(fab)
            fabst.append(fabt)
        features_array = np.stack(fabs, axis=0)
        target_array = np.stack(fabst, axis=0)

        # 2. Split for train, test, validation
        if split_method == "random":
            features_train, features_test, target_train, target_test = train_test_split(features_array, target_array, test_size=test_size,
                                                                                        random_state=1893)
        # 2.1. implement a split for time-series (only first % is train, the rest is test)
        elif split_method == "time-series":
            train_index = int(features_array.shape[0] * (1-test_size))
            features_train = features_array[0:train_index]
            features_test = features_array[train_index:]
            target_train = target_array[0:train_index]
            target_test = target_array[train_index:]
        else:
            raise Exception("The split method " + split_method + " is invalid!")

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

    # Main function for data processing
    def processDataFrame (self, feature_variables, target_variables, test_size, time_window, standardize=False, split_method="random"):

        # 0. initialize
        features_train = None
        features_test = None
        target_train = None
        target_test = None
        feature_scaler = None
        target_scaler = None

        # Process according model Structure
        if "FF" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForFF(feature_variables, target_variables, test_size, standardize, split_method)
        if "LSTM" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForRecurrentNet(feature_variables, target_variables, test_size, time_window, standardize, split_method)

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler
