"""Class to process the data in DataFrame format to be feed into the model """

import numpy as np
from sklearn.model_selection import train_test_split

class VectorModule:

    def __init__(self, dataInDataFrameFormat, modelStructure):
        self.dataInDataFrameFormat = dataInDataFrameFormat
        self.modelStructure = modelStructure
        pass

    # Data Processing for feed forward (easiest one)
    def processDataForFF (self, feature_variables, target_variables, test_size, split_method="random"):

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

        return features_train, features_test, target_train, target_test

    # Data Processing for recurrent NN
    def processDataForRecurrentNet (self, feature_variables, target_variables, test_size, time_window, split_method="random"):

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

        return features_train, features_test, target_train, target_test

    # Main function for data processing
    def processDataFrame (self, feature_variables, target_variables, test_size, time_window, split_method="random"):

        # 0. initialize
        features_train = None
        features_test = None
        target_train = None
        target_test = None

        # Process according model Structure
        if "FF" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test = self.processDataForFF(feature_variables, target_variables, test_size, split_method)
        if "LSTM" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test = self.processDataForRecurrentNet(feature_variables, target_variables, test_size, time_window, split_method)

        return features_train, features_test, target_train, target_test
