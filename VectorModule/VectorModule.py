"""Class to process the data in DataFrame format to be feed into the model """

import numpy as np
from sklearn.model_selection import train_test_split

class VectorModule:

    def __init__(self, dataInDataFrameFormat):
        self.dataInDataFrameFormat = dataInDataFrameFormat
        pass

    # Data Processing for feed forward (easiest one)
    def processDataForFF (self, feature_variables, target_variables, test_size):

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(self.dataInDataFrameFormat[target_variables])

        # 1. For the features it is needed an array of shape (index, features number)
        features_array = np.array(self.dataInDataFrameFormat[feature_variables])

        # 2. Split for train, test, validation
        features_train, features_train, target_train, target_test = train_test_split(features_array, target_array, test_size=test_size,
                                                                                     stratify=target_variables, random_state=1893)

        return features_train, features_train, target_train, target_test

    # Main function for data processing
    def processDataFrame (self, modelStructure):

        return 0
