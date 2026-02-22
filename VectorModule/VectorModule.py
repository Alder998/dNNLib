"""Class to process the data in DataFrame format to be feed into the model """

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd

class VectorModule:

    def __init__(self, modelStructure):
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
    def processDataForFF (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, standardize = False, split_method="random",
                          seasonal_splits=10):

        # 0.0. Standardize the data
        if standardize:
            feature_scaler, target_scaler, dataInDataFrameFormat = self.standardizeData(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                             feature_variables=feature_variables,
                                                                                             target_variables=target_variables)
        else:
            feature_scaler = None
            target_scaler = None

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(dataInDataFrameFormat[target_variables])

        # 1. For the features it is needed an array of shape (index, features number)
        features_array = np.array(dataInDataFrameFormat[feature_variables])

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
        # 2.2. Implement seasonal split to account for different levels, but to maintain the temporal order
        elif split_method == "seasonal-time-series":
            train_index = int(features_array.shape[0] * (1-test_size))
            train_list = list(int(i) for i in np.linspace(0, features_array.shape[0], seasonal_splits))
            tfl = []
            ttl = []
            indexes = []
            for t in train_list[:-1]:
                tf = features_array[t:min(t+int(train_index/seasonal_splits), features_array.shape[0]-1)]
                tt = target_array[t:min(t+int(train_index/seasonal_splits), target_array.shape[0]-1)]
                tfl.append(tf)
                ttl.append(tt)
                indexes.append(list(range(t, min(t+int(train_index/seasonal_splits), features_array.shape[0]-1))))
            features_train = np.stack(tfl, axis=0)
            features_train = features_train.reshape(features_train.shape[0] * features_train.shape[1], features_train.shape[2])
            target_train = np.stack(ttl, axis=0)
            target_train = target_train.reshape(target_train.shape[0] * target_train.shape[1])
            indexes = [x for sub in indexes for x in sub]
            features_test = np.delete(features_array, indexes, axis=0)
            target_test = np.delete(target_array, indexes, axis=0)
        else:
            raise Exception("The split method " + split_method + " is invalid!")

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

    # Data Processing for recurrent NN
    def processDataForRecurrentNet (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize=False,
                                    split_method="random", seasonal_splits=10, prediction=False):

        # 0.0. Standardize the data
        if standardize:
            feature_scaler, target_scaler, dataInDataFrameFormat = self.standardizeData(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                             feature_variables=feature_variables,
                                                                                             target_variables=target_variables)
        else:
            feature_scaler = None
            target_scaler = None

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(dataInDataFrameFormat[target_variables])

        # 1. For the features it is needed an array of shape (batch, time_steps, features) while now is (time_steps, features)
        features_array = np.array(dataInDataFrameFormat[feature_variables])
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

        # 1.1. if prediction, preserve the data
        if prediction:
            return features_array, target_array, feature_scaler, target_scaler

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
        # 2.2. Implement seasonal split to account for different levels, but to maintain the temporal order
        elif split_method == "seasonal-time-series":
            train_index = int(features_array.shape[0] * (1-test_size))
            train_list = list(int(i) for i in np.linspace(0, features_array.shape[0], seasonal_splits))
            tfl = []
            ttl = []
            indexes = []
            for t in train_list[:-1]:
                tf = features_array[t:min(t+int(train_index/seasonal_splits), features_array.shape[0]-1)]
                tt = target_array[t:min(t+int(train_index/seasonal_splits), target_array.shape[0]-1)]
                tfl.append(tf)
                ttl.append(tt)
                indexes.append(list(range(t, min(t+int(train_index/seasonal_splits), features_array.shape[0]-1))))
            features_train = np.stack(tfl, axis=0)
            features_train = features_train.reshape(features_train.shape[0] * features_train.shape[1], features_train.shape[2], features_train.shape[3])
            target_train = np.stack(ttl, axis=0)
            target_train = target_train.reshape(target_train.shape[0] * target_train.shape[1], target_train.shape[2])
            indexes = [x for sub in indexes for x in sub]
            features_test = np.delete(features_array, indexes, axis=0)
            target_test = np.delete(target_array, indexes, axis=0)
        else:
            raise Exception("The split method " + split_method + " is invalid!")

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

    # Data Processing for geo-spatial Model
    def processDataForGeospatialModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, space_variables, standardize=False,
                                       split_method="random", seasonal_splits=10, prediction=False):

        # 0. Isolate single time-series for each coord
        space_col = "_".join(space_variables) if len(space_variables) > 1 else space_variables[0]
        dataInDataFrameFormat[space_col] = dataInDataFrameFormat[space_variables].astype(str).agg("_".join, axis=1) if len(space_variables) > 1 else dataInDataFrameFormat[space_variables[0]]

        if not prediction:
            features_train = []
            features_test = []
            target_train = []
            target_test = []
            feature_scaler = []
            target_scaler = []
            for uniqueCoord in dataInDataFrameFormat[space_col].unique():
                dfc = dataInDataFrameFormat[dataInDataFrameFormat[space_col] == uniqueCoord].reset_index(drop=True)
                features_train_i, features_test_i, target_train_i, target_test_i, feature_scaler_i, target_scaler_i = self.processDataForRecurrentNet(dfc, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits)
                features_train.append(features_train_i)
                features_test.append(features_test_i)
                target_train.append(target_train_i)
                target_test.append(target_test_i)
                feature_scaler.append(feature_scaler_i)
                target_scaler.append(target_scaler_i)
            features_train = np.stack(features_train, axis = 3)
            features_test = np.stack(features_test, axis = 3)
            target_train = np.stack(target_train, axis = 2)
            target_test = np.stack(target_test, axis = 2)
            if feature_scaler[0] is not None:
                feature_scaler = np.stack(feature_scaler, axis = 3)
                target_scaler = np.stack(target_scaler, axis = 3)

            return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

        else:
            features_array = []
            target_array = []
            feature_scaler = []
            target_scaler = []
            for uniqueCoord in dataInDataFrameFormat[space_col].unique():
                dfc = dataInDataFrameFormat[dataInDataFrameFormat[space_col] == uniqueCoord].reset_index(drop=True)
                features_array_i, target_array_i, feature_scaler_i, target_scaler_i = self.processDataForRecurrentNet(
                    dfc, feature_variables, target_variables, test_size, time_window, standardize, split_method,
                    seasonal_splits, prediction=True)
                features_array.append(features_array_i)
                target_array.append(target_array_i)
                feature_scaler.append(feature_scaler_i)
                target_scaler.append(target_scaler_i)
            features_array = np.stack(features_array, axis=3)
            target_array = np.stack(target_array, axis=2)
            if feature_scaler[0] is not None:
                feature_scaler = np.stack(feature_scaler, axis=3)
                target_scaler = np.stack(target_scaler, axis=3)

            return features_array, target_array, feature_scaler, target_scaler

    # Utils-like function to create the Adjacency Matrix from a DataFrame
    def createAdjacencyMatrixFromDataFrame (self, dataInDataFrameFormat, space_variables, target_variables, sigma=None):

        # 0. Very primitive Adjacency Matrix - mean of the target variable difference (standardized)
        space_col = "_".join(space_variables) if len(space_variables) > 1 else space_variables[0]
        dataInDataFrameFormat[space_col] = dataInDataFrameFormat[space_variables].astype(str).agg("_".join, axis=1) if len(space_variables) > 1 else dataInDataFrameFormat[space_variables[0]]

        # 1. Group data by mean
        data_grouped = dataInDataFrameFormat[[space_col, target_variables]].groupby(space_col, as_index=False).mean()
        data_grouped = data_grouped[target_variables].values.astype(float)
        diff_matrix = data_grouped[None, :] - data_grouped[:, None]
        dist2 = diff_matrix ** 2
        if sigma is None:
            sigma = np.std(data_grouped)
        A = np.exp(-dist2 / (sigma ** 2))

        # Fill diagonal with 0
        A = self.normalize_adjacency(A)
        np.fill_diagonal(A, 0)

        # Show the density of adjacency matrix
        print("ADJACENCY MATRIX - Density: " + str(round((np.count_nonzero(A) / A.size) * 100, 2)) + " %")

        return A

    # Function to standardize adjacency Matrix
    def normalize_adjacency(self, A):
        A = A + np.eye(A.shape[0])
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
        return A_hat.astype("float32")

    # Main function for data processing
    def processDataFrame (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize=False,
                          split_method="random", seasonal_splits=10, timeSpace=False, space_variables=None, prediction=False):

        # 0. initialize
        features_train = None
        features_test = None
        target_train = None
        target_test = None
        feature_scaler = None
        target_scaler = None

        # Process according model Structure
        if "FF" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForFF(dataInDataFrameFormat, feature_variables, target_variables, test_size, standardize, split_method)
        if "LSTM" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForRecurrentNet(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, prediction=prediction)
        if "Conv2D" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForRecurrentNet(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, prediction=prediction)

        # Ad-hoc config for time-space
        if timeSpace:
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForGeospatialModel(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, space_variables, standardize, split_method, seasonal_splits, prediction=prediction)

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler
