"""Class to process the data in DataFrame format to be feed into the model """

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

class VectorModule:

    def __init__(self, modelStructure):
        self.modelStructure = modelStructure
        pass

    # Function to standardize data
    def standardizeData (self, dataInDataFrameFormat, feature_variables, target_variables, model="std"):

        # 0. Instantiate the scaler
        if model == "std":
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
        elif model == "min-max":
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        else:
            raise Exception("Standardizer " + str() + " not implemented.")

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
                          seasonal_splits=10, target_division=1, lag_series=[], scaler="std"):

        # S. if lags are present, add them to the features
        if len(lag_series) != 0:
            for tc in target_variables:
                lag_features = [tc + "_" + str(lag_number) + "_lag" for lag_number in lag_series]
                for l in lag_features:
                    feature_variables.append(l)

        # 0.0. Standardize the data
        if standardize:
            feature_scaler, target_scaler, dataInDataFrameFormat = self.standardizeData(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                        feature_variables=feature_variables,
                                                                                        target_variables=target_variables,
                                                                                        model=scaler)
        else:
            feature_scaler = None
            target_scaler = None

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(dataInDataFrameFormat[target_variables] / target_division)

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
            target_train = target_train.reshape(target_train.shape[0] * target_train.shape[1], target_train.shape[2])
            indexes = [x for sub in indexes for x in sub]
            features_test = np.delete(features_array, indexes, axis=0)
            target_test = np.delete(target_array, indexes, axis=0)
        else:
            raise Exception("The split method " + split_method + " is invalid!")

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

    # Data Processing for recurrent NN
    def processDataForRecurrentNet (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize=False,
                                    split_method="random", seasonal_splits=10, prediction=False, target_division=1, lag_series=[], scaler="std"):

        # S. if lags are present, add them to the features
        if len(lag_series) != 0:
            for tc in target_variables:
                lag_features = [tc + "_" + str(lag_number) + "_lag" for lag_number in lag_series]
                for l in lag_features:
                    if l not in feature_variables:
                        feature_variables.append(l)

        # 0.0. Standardize the data
        if standardize:
            feature_scaler, target_scaler, dataInDataFrameFormat = self.standardizeData(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                        feature_variables=feature_variables,
                                                                                        target_variables=target_variables,
                                                                                        model=scaler)
        else:
            feature_scaler = None
            target_scaler = None

        # 0. It is needed an array of shape (index,) for target variable
        target_array = np.array(dataInDataFrameFormat[target_variables] / target_division)

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
            # 2.2.1. Define the problem, if time-space, the dimensions are +1 wrt time series
            # 2.2.2. Process
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
            target_train = np.stack(ttl, axis=0)
            # Align dimensions if not aligned between the feature and the target sets (they must have 4 dimensions)
            if len(target_train.shape) < len(features_train.shape):
                target_train = np.expand_dims(target_train, axis=3)
            features_train = features_train.reshape(features_train.shape[0] * features_train.shape[1], features_train.shape[2], features_train.shape[3])
            target_train = target_train.reshape(target_train.shape[0] * target_train.shape[1], target_train.shape[2], target_train.shape[3])
            indexes = [x for sub in indexes for x in sub]
            features_test = np.delete(features_array, indexes, axis=0)
            target_test = np.delete(target_array, indexes, axis=0)
        else:
            raise Exception("The split method " + split_method + " is invalid!")

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler

    # Data Processing for geo-spatial Model
    def processDataForGeospatialModel (self, dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, space_variables, standardize=False,
                                       split_method="random", seasonal_splits=10, prediction=False, target_division=1, lag_series=[], scaler="std"):

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
                features_train_i, features_test_i, target_train_i, target_test_i, feature_scaler_i, target_scaler_i = self.processDataForRecurrentNet(dfc, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, target_division=target_division, lag_series=lag_series, scaler=scaler)
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
            # It may happen that the target_test lacks the last dimension. In case, add it
            if len(target_test.shape) < len(target_train.shape):
                target_test = np.expand_dims(target_test, axis=len(target_train.shape)-1)
            #if feature_scaler[0] is not None:
            #    feature_scaler = np.stack(feature_scaler, axis = 3)
            #   target_scaler = np.stack(target_scaler, axis = 3)

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
                    seasonal_splits, prediction=True, target_division=target_division, lag_series=lag_series, scaler=scaler)
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
    def createAdjacencyMatrixFromDataFrame (self, dataInDataFrameFormat, space_variables, target_variables, date_column, radius=2):

        # 0. Very primitive Adjacency Matrix - mean of the target variable difference (standardized)
        print("ADJACENCY MATRIX - Creating the mutual-information adjacency matrix (radius: " + str(radius) + ")...")
        space_col = "_".join(space_variables) if len(space_variables) > 1 else space_variables[0]
        dataInDataFrameFormat[space_col] = dataInDataFrameFormat[space_variables].astype(str).agg("_".join, axis=1) if len(space_variables) > 1 else dataInDataFrameFormat[space_variables[0]]

        # 1. Isolate the space points
        n_lat = len(dataInDataFrameFormat[space_variables[0]].unique())
        n_lon = len(dataInDataFrameFormat[space_variables[1]].unique())
        S = n_lat * n_lon
        radius = radius

        # 2. Get the candidate pairs for each one of the coordinates
        candidate_pairs = []
        for r in range(n_lat):
            for c in range(n_lon):
                i = r * n_lon + c
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < n_lat and 0 <= cc < n_lon:
                            j = rr * n_lon + cc
                            if i < j:
                                candidate_pairs.append((i, j))

        # 3. Build the climate Matrix (for each one of the variable)
        spaces = dataInDataFrameFormat[space_col].unique()
        feature_matrices = {}
        for variable in target_variables:
            M = (dataInDataFrameFormat.pivot(index=date_column, columns=space_col, values=variable).reindex(columns=spaces))
            feature_matrices[variable] = M.to_numpy(dtype=np.float32)

        # 4. Launch the mutual Information algorithm
        A_features = {}
        for variable, X in feature_matrices.items():
            A = np.zeros((S, S), dtype=np.float32)
            for i, j in candidate_pairs:
                x = X[:, i]
                y = X[:, j]
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 100:
                    continue
                mi = mutual_info_regression(x[mask].reshape(-1, 1), y[mask], random_state=42)[0]
                A[i, j] = mi
                A[j, i] = mi
            A_features[variable] = A

        # 5. Normalize the values to uniform the scale of the variables
        normalized = []
        for variable in target_variables:
            A = A_features[variable]
            max_value = A.max()
            if max_value > 0:
                A = A / max_value
            normalized.append(A)
        # Mean of the normalized matrix
        A = np.mean(normalized, axis=0)
        np.fill_diagonal(A, 0)

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
                          split_method="random", seasonal_splits=10, timeSpace=False, space_variables=None, prediction=False, target_division=1, lag_series=[], scaler="std"):

        # 0. initialize
        features_train = None
        features_test = None
        target_train = None
        target_test = None
        feature_scaler = None
        target_scaler = None

        # Process according model Structure
        if "FF" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForFF(dataInDataFrameFormat, feature_variables, target_variables, test_size, standardize, split_method, target_division=target_division, lag_series=lag_series, scaler=scaler)
        if "LSTM" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForRecurrentNet(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, prediction=prediction, target_division=target_division, lag_series=lag_series, scaler=scaler)
        if "Conv2D" in self.modelStructure.keys():
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForRecurrentNet(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, standardize, split_method, seasonal_splits, prediction=prediction, target_division=target_division, lag_series=lag_series, scaler=scaler)

        # Ad-hoc config for time-space
        if timeSpace:
            features_train, features_test, target_train, target_test, feature_scaler, target_scaler = self.processDataForGeospatialModel(dataInDataFrameFormat, feature_variables, target_variables, test_size, time_window, space_variables, standardize, split_method, seasonal_splits, prediction=prediction, target_division=target_division, lag_series=lag_series, scaler=scaler)

        return features_train, features_test, target_train, target_test, feature_scaler, target_scaler
